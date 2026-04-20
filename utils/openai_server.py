from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from openai import OpenAI

_SYNC_CONTEXTS: dict[tuple[str, str, int], "WeightSyncContext"] = {}
_SYNC_LOCK = threading.Lock()

_NCCL_INVALID_USAGE_MESSAGE = (
    "NCCL weight sync failed with 'invalid usage'. This repo supports NCCL weight sync only for "
    "a trainer with a single-GPU vLLM server, and every trainer rank must participate "
    "in the sync call. If you need a fallback, restart the vLLM server with "
    "`./setup/start_vllm.sh <model> <max_model_len> <host> <port> ipc` or pass "
    "`--vllm-weight-sync-backend ipc` to the trainer so both sides match."
)
_NCCL_SERVER_WORLD_SIZE_MESSAGE = (
    "NCCL weight sync in this repo currently supports only a single-GPU vLLM server. "
    "Restart vLLM with one worker/GPU or switch both sides to the IPC backend."
)
_NCCL_DISTRIBUTED_MESSAGE = (
    "NCCL weight sync requires every trainer rank to participate. Launch the trainer with "
    "`torchrun` so `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` are initialized."
)


def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))


def _get_text_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "output_text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)
    return str(message_content or "")


@dataclass
class OpenAICompatibleRolloutClient:
    host: str
    port: int
    model_name: str
    api_key: str = "EMPTY"
    request_timeout: float = 300.0
    max_parallel_requests: int = 8

    def __post_init__(self) -> None:
        self.server_base_url = f"http://{self.host}:{self.port}"
        self.client = OpenAI(
            base_url=f"{self.server_base_url}/v1",
            api_key=self.api_key,
            timeout=self.request_timeout,
        )

    def _generate_one(
        self,
        tokenizer,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_generations: int,
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            n=num_generations,
        )

        all_input_ids = []
        all_completion_masks = []
        all_texts = []
        for choice in response.choices:
            text = _get_text_content(choice.message.content).strip()
            completion_ids = tokenizer.encode(text, add_special_tokens=False)
            seq_ids = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            all_input_ids.append(seq_ids)
            all_completion_masks.append(mask)
            all_texts.append(text)

        return all_input_ids, all_completion_masks, all_texts

    def generate_completions(
        self,
        tokenizer,
        prompt: str | list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_generations: int,
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            return [], [], []

        all_input_ids: list[list[int]] = []
        all_completion_masks: list[list[int]] = []
        all_texts: list[str] = []

        max_workers = min(self.max_parallel_requests, len(prompts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_one,
                    tokenizer,
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_generations=num_generations,
                )
                for prompt_text in prompts
            ]

            for future in futures:
                input_ids, completion_masks, texts = future.result()
                all_input_ids.extend(input_ids)
                all_completion_masks.extend(completion_masks)
                all_texts.extend(texts)

        return all_input_ids, all_completion_masks, all_texts


@dataclass
class WeightSyncContext:
    base_url: str
    backend: str
    trainer_world_size: int
    group: Any | None = None
    initialized: bool = False


@dataclass
class _HTTPPostAsyncResult:
    response: dict[str, Any] | None = None
    error: Exception | None = None


def _get_world_size(base_url: str, timeout: float) -> int:
    payload = _http_get_json(f"{base_url}/get_world_size", timeout)
    world_size = payload.get("world_size")
    if not isinstance(world_size, int):
        raise ValueError(f"Invalid get_world_size response from {base_url}: {payload}")
    return world_size


def _pause_generation(base_url: str, timeout: float) -> None:
    _http_post_json(f"{base_url}/pause", None, timeout)


def _resume_generation(base_url: str, timeout: float) -> None:
    _http_post_json(f"{base_url}/resume", None, timeout)


def _init_ipc_transfer(base_url: str, timeout: float) -> None:
    _http_post_json(f"{base_url}/init_weight_transfer_engine", {"init_info": {}}, timeout)


def _send_ipc_weights(base_url: str, model) -> None:
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCTrainerSendWeightsArgs,
        IPCWeightTransferEngine,
    )

    trainer_args = IPCTrainerSendWeightsArgs(mode="http", url=base_url)
    IPCWeightTransferEngine.trainer_send_weights(
        iterator=model.named_parameters(),
        trainer_args=trainer_args,
    )


def _update_nccl_metadata(
    base_url: str,
    names: list[str],
    dtype_names: list[str],
    shapes: list[list[int]],
    timeout: float,
) -> None:
    _http_post_json(
        f"{base_url}/update_weights",
        {
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": True,
            }
        },
        timeout,
    )


def _spawn_http_post_thread(
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[threading.Thread, _HTTPPostAsyncResult]:
    result = _HTTPPostAsyncResult()

    def runner() -> None:
        try:
            result.response = _http_post_json(url, payload, timeout)
        except Exception as exc:  # pragma: no cover - propagated on join
            result.error = exc

    thread = threading.Thread(target=runner)
    thread.start()
    return thread, result


def _join_http_post_thread(
    thread: threading.Thread,
    result: _HTTPPostAsyncResult,
) -> dict[str, Any] | None:
    thread.join()
    if result.error is not None:
        raise result.error
    return result.response


def _dist_barrier(trainer_world_size: int) -> None:
    if trainer_world_size > 1:
        dist.barrier()


def _broadcast_objects(values: list[Any], trainer_world_size: int) -> list[Any]:
    if trainer_world_size > 1:
        dist.broadcast_object_list(values, src=0)
    return values


def _ensure_nccl_initialized(
    base_url: str,
    timeout: float,
    context: WeightSyncContext,
    *,
    trainer_rank: int,
    trainer_world_size: int,
    is_sync_leader: bool,
) -> None:
    if context.initialized:
        return

    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    if trainer_world_size > 1 and (not dist.is_available() or not dist.is_initialized()):
        raise RuntimeError(_NCCL_DISTRIBUTED_MESSAGE)

    init_values: list[Any] = [None, None, None]
    init_thread: threading.Thread | None = None
    init_result: _HTTPPostAsyncResult | None = None

    if is_sync_leader:
        try:
            inference_world_size = _get_world_size(base_url, timeout)
            if inference_world_size != 1:
                raise RuntimeError(
                    f"{_NCCL_SERVER_WORLD_SIZE_MESSAGE} Server reported world size={inference_world_size}."
                )

            master_address = get_ip()
            master_port = get_open_port()
            init_values = [master_address, master_port, None]

            init_thread, init_result = _spawn_http_post_thread(
                f"{base_url}/init_weight_transfer_engine",
                {
                    "init_info": {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": trainer_world_size,
                        "world_size": trainer_world_size + 1,
                    }
                },
                timeout,
            )
        except Exception as exc:
            init_values = [None, None, str(exc)]

    master_address, master_port, init_error = _broadcast_objects(init_values, trainer_world_size)
    if init_error is not None:
        raise RuntimeError(str(init_error))
    if not isinstance(master_address, str) or not isinstance(master_port, int):
        raise RuntimeError("Failed to initialize NCCL weight sync rendezvous info for trainer ranks.")

    context.group = NCCLWeightTransferEngine._stateless_init_process_group(
        master_address,
        master_port,
        trainer_rank,
        trainer_world_size + 1,
        device=torch.accelerator.current_device_index(),
    )
    _dist_barrier(trainer_world_size)
    join_error: str | None = None
    if init_thread is not None and init_result is not None:
        try:
            _join_http_post_thread(init_thread, init_result)
        except Exception as exc:
            join_error = str(exc)
    join_error = _broadcast_objects([join_error], trainer_world_size)[0]
    if join_error is not None:
        raise RuntimeError(str(join_error))
    context.initialized = True


def sync_server_model_weights(
    *,
    host: str,
    port: int,
    model,
    backend: str,
    timeout: float,
    is_sync_leader: bool,
    trainer_rank: int = 0,
    trainer_world_size: int = 1,
) -> None:
    if backend != "nccl" and not is_sync_leader:
        return
    if backend == "nccl" and trainer_world_size > 1 and not dist.is_initialized():
        raise RuntimeError(_NCCL_DISTRIBUTED_MESSAGE)

    base_url = f"http://{host}:{port}"
    with _SYNC_LOCK:
        context = _SYNC_CONTEXTS.setdefault(
            (base_url, backend, trainer_world_size),
            WeightSyncContext(base_url, backend, trainer_world_size),
        )

    if backend == "nccl":
        try:
            _ensure_nccl_initialized(
                base_url,
                timeout,
                context,
                trainer_rank=trainer_rank,
                trainer_world_size=trainer_world_size,
                is_sync_leader=is_sync_leader,
            )
            assert context.group is not None

            names = []
            dtype_names = []
            shapes = []
            for name, param in model.named_parameters():
                names.append(name)
                dtype_names.append(str(param.dtype).split(".")[-1])
                shapes.append(list(param.shape))

            update_thread: threading.Thread | None = None
            update_result: _HTTPPostAsyncResult | None = None
            setup_error: str | None = None
            if is_sync_leader:
                try:
                    _pause_generation(base_url, timeout)
                    update_thread, update_result = _spawn_http_post_thread(
                        f"{base_url}/update_weights",
                        {
                            "update_info": {
                                "names": names,
                                "dtype_names": dtype_names,
                                "shapes": shapes,
                                "packed": True,
                            }
                        },
                        timeout,
                    )
                except Exception as exc:
                    setup_error = str(exc)
            setup_error = _broadcast_objects([setup_error], trainer_world_size)[0]
            if setup_error is not None:
                raise RuntimeError(str(setup_error))
            _dist_barrier(trainer_world_size)
            try:
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLTrainerSendWeightsArgs,
                    NCCLWeightTransferEngine,
                )

                trainer_args = NCCLTrainerSendWeightsArgs(
                    group=context.group,
                    src=0,
                    packed=True,
                )
                NCCLWeightTransferEngine.trainer_send_weights(
                    iterator=model.named_parameters(),
                    trainer_args=trainer_args,
                )
                _dist_barrier(trainer_world_size)
                finish_error: str | None = None
                if update_thread is not None and update_result is not None:
                    try:
                        _join_http_post_thread(update_thread, update_result)
                    except Exception as exc:
                        finish_error = str(exc)
                finish_error = _broadcast_objects([finish_error], trainer_world_size)[0]
                if finish_error is not None:
                    raise RuntimeError(str(finish_error))
            finally:
                if is_sync_leader:
                    _resume_generation(base_url, timeout)
        except Exception as exc:
            if "NCCL error: invalid usage" in str(exc):
                raise RuntimeError(_NCCL_INVALID_USAGE_MESSAGE) from exc
            raise
    elif backend == "ipc":
        if not context.initialized:
            _init_ipc_transfer(base_url, timeout)
            context.initialized = True
        _pause_generation(base_url, timeout)
        try:
            _send_ipc_weights(base_url, model)
        finally:
            _resume_generation(base_url, timeout)
    else:
        raise ValueError(f"Unsupported weight sync backend: {backend}")
