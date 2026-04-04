import os
import math
import traceback
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"

    # 실제 학습에서 쓰려는 값으로 맞추는 게 중요
    seq_len: int = 2048
    per_device_batch_size: int = 1
    grad_accum_steps: int = 1
    steps: int = 2

    lr: float = 2e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    use_activation_checkpointing: bool = True
    attn_implementation: str = "sdpa"   # "sdpa" or "flash_attention_2" if your env supports it
    dtype = torch.bfloat16

    # true면 optimizer.step까지 수행
    do_optimizer_step: bool = True


CFG = Config()


def setup_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return local_rank, rank, world_size
    

def cleanup_dist():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def bytes_to_gib(x: int) -> float:
    return x / (1024 ** 3)


def print_rank(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank {rank}] {msg}", flush=True)


def mem_report(tag: str, device: int):
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    print_rank(
        f"{tag} | "
        f"alloc={bytes_to_gib(alloc):.2f} GiB | "
        f"reserved={bytes_to_gib(reserved):.2f} GiB | "
        f"peak_alloc={bytes_to_gib(peak_alloc):.2f} GiB | "
        f"peak_reserved={bytes_to_gib(peak_reserved):.2f} GiB"
    )


def reset_peak(device: int):
    torch.cuda.reset_peak_memory_stats(device)


def apply_activation_checkpointing(model: nn.Module):
    from torch.utils.checkpoint import checkpoint

    class CheckpointBlock(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *args, **kwargs):
            def custom_forward(*inputs):
                return self.mod(*inputs, **kwargs)
            return checkpoint(custom_forward, *args, use_reentrant=False)

    for i in range(len(model.model.layers)):
        model.model.layers[i] = CheckpointBlock(model.model.layers[i])

    return model


def apply_fsdp2(model: nn.Module, world_size: int):
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )

    # block-level shard
    for block in model.model.layers:
        fully_shard(
            block,
            mesh=mesh["dp"],
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )

    # root shard
    fully_shard(
        model,
        mesh=mesh["dp"],
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )
    return model


def make_fake_batch(tokenizer, batch_size: int, seq_len: int, device: int):
    # 실제 데이터셋이 없어도 OOM 여부는 거의 같은 경향을 볼 수 있음
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones(
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    local_rank, rank, world_size = setup_dist()
    device = local_rank

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if is_main():
            print("=== OOM check start ===", flush=True)
            print(f"model={CFG.model_name}", flush=True)
            print(f"seq_len={CFG.seq_len}", flush=True)
            print(f"per_device_batch_size={CFG.per_device_batch_size}", flush=True)
            print(f"grad_accum_steps={CFG.grad_accum_steps}", flush=True)
            print(f"steps={CFG.steps}", flush=True)
            print(f"dtype={CFG.dtype}", flush=True)
            print(f"activation_checkpointing={CFG.use_activation_checkpointing}", flush=True)
            print(f"attn_implementation={CFG.attn_implementation}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch.cuda.empty_cache()
        reset_peak(device)

        # 1) real weight load
        model = AutoModelForCausalLM.from_pretrained(
            CFG.model_name,
            torch_dtype=CFG.dtype,
            attn_implementation=CFG.attn_implementation,
        )
        model.config.use_cache = False
        model.train()

        if CFG.use_activation_checkpointing:
            model = apply_activation_checkpointing(model)

        model.cuda(device)
        torch.cuda.synchronize(device)
        mem_report("after model load", device)

        # 2) FSDP2 wrap
        reset_peak(device)
        model = apply_fsdp2(model, world_size)
        torch.cuda.synchronize(device)
        mem_report("after fsdp2 wrap", device)

        # optimizer must be created after sharding
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.lr,
            betas=(0.9, 0.95),
            weight_decay=CFG.weight_decay,
            fused=True,
        )

        torch.cuda.synchronize(device)
        mem_report("after optimizer init", device)

        # warmup fake batch
        batch = make_fake_batch(
            tokenizer=tokenizer,
            batch_size=CFG.per_device_batch_size,
            seq_len=CFG.seq_len,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        # 3) actual train steps
        for step in range(CFG.steps):
            dist.barrier()
            reset_peak(device)

            loss_acc = 0.0

            for ga in range(CFG.grad_accum_steps):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / CFG.grad_accum_steps
                loss_acc += float(loss.detach())
                torch.cuda.synchronize(device)
                mem_report(f"step={step} ga={ga} after forward", device)

                loss.backward()
                torch.cuda.synchronize(device)
                mem_report(f"step={step} ga={ga} after backward", device)

            if CFG.do_optimizer_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    CFG.max_grad_norm,
                )
                torch.cuda.synchronize(device)
                mem_report(f"step={step} after clip_grad_norm", device)

                optimizer.step()
                torch.cuda.synchronize(device)
                mem_report(f"step={step} after optimizer.step", device)

                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                mem_report(f"step={step} after zero_grad", device)

            # all-reduce loss for cleaner logging
            loss_tensor = torch.tensor(loss_acc, device=device, dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

            if is_main():
                print(
                    f"[global] step={step} avg_loss={loss_tensor.item():.4f}",
                    flush=True,
                )

        if is_main():
            print("=== OOM check finished successfully ===", flush=True)

    except torch.cuda.OutOfMemoryError as e:
        print_rank("CUDA OOM detected")
        print_rank(str(e))
        try:
            mem_report("at OOM", device)
        except Exception:
            pass
        raise

    except Exception as e:
        print_rank("Unhandled exception")
        print_rank("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise

    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()