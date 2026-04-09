from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
from openai import OpenAI

_SYNC_CONTEXTS: dict[tuple[str, str], "WeightSyncContext"] = {}
_SYNC_LOCK = threading.Lock()


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
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_generations: int,
    ) -> tuple[list[list[int]], list[list[int]], list[str]]:
        all_input_ids: list[list[int]] = []
        all_completion_masks: list[list[int]] = []
        all_texts: list[str] = []

        max_workers = max(1, min(self.max_parallel_requests, len(prompts)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_one,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_generations=num_generations,
                )
                for prompt in prompts
            ]
            for future in futures:
                ids, masks, texts = future.result()
                all_input_ids.extend(ids)
                all_completion_masks.extend(masks)
                all_texts.extend(texts)

        return all_input_ids, all_completion_masks, all_texts


@dataclass
class WeightSyncContext:
    base_url: str
    backend: str
    group: Any | None = None
    initialized: bool = False


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


def _ensure_nccl_initialized(
    base_url: str,
    timeout: float,
    context: WeightSyncContext,
) -> None:
    if context.initialized:
        return

    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip, get_open_port

    inference_world_size = _get_world_size(base_url, timeout)
    world_size = inference_world_size + 1
    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1

    init_thread = threading.Thread(
        target=_http_post_json,
        args=(
            f"{base_url}/init_weight_transfer_engine",
            {
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                }
            },
            timeout,
        ),
    )
    init_thread.start()
    context.group = NCCLWeightTransferEngine.trainer_init(
        {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": world_size,
        }
    )
    init_thread.join()
    context.initialized = True


def sync_server_model_weights(
    *,
    host: str,
    port: int,
    model,
    backend: str,
    timeout: float,
    is_sync_leader: bool,
) -> None:
    if not is_sync_leader:
        return

    base_url = f"http://{host}:{port}"
    with _SYNC_LOCK:
        context = _SYNC_CONTEXTS.setdefault((base_url, backend), WeightSyncContext(base_url, backend))

    if backend == "nccl":
        _ensure_nccl_initialized(base_url, timeout, context)
        assert context.group is not None

        names = []
        dtype_names = []
        shapes = []
        for name, param in model.named_parameters():
            names.append(name)
            dtype_names.append(str(param.dtype).split(".")[-1])
            shapes.append(list(param.shape))

        _pause_generation(base_url, timeout)
        try:
            update_thread = threading.Thread(
                target=_update_nccl_metadata,
                args=(base_url, names, dtype_names, shapes, timeout),
            )
            update_thread.start()

            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLTrainerSendWeightsArgs,
                NCCLWeightTransferEngine,
            )

            trainer_args = NCCLTrainerSendWeightsArgs(
                group=context.group,
                packed=True,
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=model.named_parameters(),
                trainer_args=trainer_args,
            )
            update_thread.join()
        finally:
            _resume_generation(base_url, timeout)
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
