import os
import sys

import torch
import torch.distributed as dist


def get_env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return int(value)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 1

    rank = get_env_int("RANK")
    local_rank = get_env_int("LOCAL_RANK")
    world_size = get_env_int("WORLD_SIZE")

    if world_size < 2:
        print("WORLD_SIZE must be at least 2 for an NCCL multi-GPU check", file=sys.stderr)
        return 1

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        dist.init_process_group("nccl", device_id=device)

        tensor = torch.tensor([rank + 1.0], device=device)
        expected = world_size * (world_size + 1) / 2

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()

        actual = tensor.item()
        if actual != expected:
            print(
                f"Rank {rank}: NCCL all_reduce mismatch, expected {expected}, got {actual}",
                file=sys.stderr,
            )
            return 1

        print(f"Rank {rank}/{world_size}: NCCL OK on cuda:{local_rank}, all_reduce={actual}")
        return 0
    except Exception as exc:
        print(f"Rank {rank}: NCCL check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
