from __future__ import annotations

import json
import math
import pickle
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any

import pybase64 as base64
import requests
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor

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
    device_index = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(device_index)
    gpu_uuid = str(props.uuid)

    names = []
    dtype_names = []
    shapes = []
    ipc_handles = []

    for name, tensor in model.named_parameters():
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))

        weight = tensor.detach().contiguous()
        ipc_handle = reduce_tensor(weight)
        ipc_handles.append({gpu_uuid: ipc_handle})

    pickled_handles = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
    response = requests.post(
        f"{base_url}/update_weights",
        json={
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "ipc_handles_pickled": pickled_handles,
            }
        },
        timeout=300,
    )
    response.raise_for_status()


def _spawn_http_post_thread(
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[threading.Thread, _HTTPPostAsyncResult]:
    result = _HTTPPostAsyncResult()

    def runner() -> None:
        try:
            result.response = _http_post_json(url, payload, timeout)
        except Exception as exc:  # pragma: no cover
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


def _get_vllm_network_utils() -> tuple[Any, Any]:
    try:
        from vllm.utils.network_utils import get_ip, get_open_port

        return get_ip, get_open_port
    except Exception:
        import socket
        import warnings

        def fallback_get_ip() -> str:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.connect(("8.8.8.8", 80))
                    return sock.getsockname()[0]
            except Exception:
                warnings.warn(
                    "Failed to determine a routable IP via the trainer-side vLLM shim; using 0.0.0.0.",
                    stacklevel=2,
                )
                return "0.0.0.0"

        def fallback_get_open_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                return int(sock.getsockname()[1])

        return fallback_get_ip, fallback_get_open_port


def _packed_broadcast_producer(
    iterator,
    group: Any,
    src: int,
    *,
    buffer_size_bytes: int = 1024 * 1024 * 1024,
    num_buffers: int = 2,
) -> None:
    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    while True:
        streams[buffer_idx].synchronize()
        with torch.cuda.stream(streams[buffer_idx]):
            try:
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                while True:
                    _name, param = next(iterator)
                    tensor = param.contiguous().view(torch.uint8).view(-1)
                    packing_tensor_list[buffer_idx].append(tensor)
                    packing_tensor_sizes[buffer_idx] += tensor.numel()
                    if packing_tensor_sizes[buffer_idx] > buffer_size_bytes:
                        break

                packed_tensors[buffer_idx] = torch.cat(packing_tensor_list[buffer_idx], dim=0)
                group.broadcast(packed_tensors[buffer_idx], src=src)
                buffer_idx = (buffer_idx + 1) % num_buffers
            except StopIteration:
                if packing_tensor_list[buffer_idx]:
                    packed_tensors[buffer_idx] = torch.cat(packing_tensor_list[buffer_idx], dim=0)
                    group.broadcast(packed_tensors[buffer_idx], src=src)
                break


def _make_nccl_group(
    *,
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: int,
):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    process_group = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
    )
    return PyNcclCommunicator(process_group, device=device)


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

    get_ip, get_open_port = _get_vllm_network_utils()

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

    context.group = _make_nccl_group(
        master_address=master_address,
        master_port=master_port,
        rank=trainer_rank,
        world_size=trainer_world_size + 1,
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
                _packed_broadcast_producer(
                    iterator=iter(model.named_parameters()),
                    group=context.group,
                    src=0,
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
