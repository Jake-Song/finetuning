import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import fully_shard


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    train_path: str = "train.jsonl"
    max_length: int = 2048

    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    log_every: int = 10
    save_every_steps: int = 500

    output_dir: str = "./ckpt_llama31_8b_fsdp2_sft"
    seed: int = 42


CFG = TrainConfig()


# -----------------------------
# Distributed helpers
# -----------------------------
def setup_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_dist():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class SFTJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DataCollatorForAssistantOnlyLM:
    """
    - messages -> chat template string
    - labels: assistant response tokens only participate in loss
    """
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _render_full_text(self, messages: List[Dict[str, str]]) -> str:
        # full conversation with assistant answer
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _render_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        # prompt only up to before final assistant content
        # assumes last message is assistant
        prompt_messages = messages[:-1]
        return self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for ex in batch:
            messages = ex["messages"]
            assert messages[-1]["role"] == "assistant", "Last message must be assistant"

            full_text = self._render_full_text(messages)
            prompt_text = self._render_prompt_text(messages)

            full_enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            prompt_enc = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )

            input_ids = full_enc["input_ids"]
            attn_mask = full_enc["attention_mask"]
            prompt_len = len(prompt_enc["input_ids"])

            labels = input_ids.copy()
            # prompt 부분은 loss 제외
            labels[:prompt_len] = [-100] * prompt_len

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attn_mask, dtype=torch.long))

        input_ids = nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
        )

        # 우측 잘림 안전장치
        input_ids = input_ids[:, : self.max_length]
        labels = labels[:, : self.max_length]
        attention_mask = attention_mask[:, : self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# -----------------------------
# FSDP2 wrapping
# -----------------------------
def apply_fsdp2(model: nn.Module):
    """
    FSDP2 minimal wrapping:
    - shard each transformer block
    - then shard root module
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

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

    # LlamaForCausalLM -> model.layers is decoder blocks
    for block in model.model.layers:
        fully_shard(
            block,
            mesh=mesh["dp"],
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )

    fully_shard(
        model,
        mesh=mesh["dp"],
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )

    return model


# -----------------------------
# Activation checkpointing
# -----------------------------
def apply_activation_checkpointing(model: nn.Module):
    """
    Simple per-block checkpoint wrapper.
    """
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


# -----------------------------
# Save
# -----------------------------
def save_model_checkpoint(model, tokenizer, output_dir, step):
    if not is_main_process():
        return

    os.makedirs(output_dir, exist_ok=True)

    # NOTE:
    # For a production checkpoint, gather full state dict properly.
    # This is a minimal placeholder.
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    tokenizer.save_pretrained(ckpt_dir)
    torch.save({"step": step}, os.path.join(ckpt_dir, "trainer_state.pt"))

    print(f"[rank0] Saved lightweight checkpoint metadata to {ckpt_dir}")


# -----------------------------
# Main
# -----------------------------
def main():
    local_rank, rank, world_size = setup_dist()
    set_seed(CFG.seed + rank)

    if is_main_process():
        os.makedirs(CFG.output_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)

    # llama base often has no pad token -> use eos as pad for SFT batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # training
    model.train()

    # optional but strongly recommended on 8B full FT
    model = apply_activation_checkpointing(model)

    # move to local device before sharding
    model.cuda(local_rank)

    # FSDP2
    model = apply_fsdp2(model)

    # optimizer after sharding
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        betas=(0.9, 0.95),
        weight_decay=CFG.weight_decay,
        fused=True,
    )

    # dataset / loader
    dataset = SFTJsonlDataset(CFG.train_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    collator = DataCollatorForAssistantOnlyLM(
        tokenizer=tokenizer,
        max_length=CFG.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG.per_device_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(loader) / CFG.gradient_accumulation_steps
    )
    total_training_steps = num_update_steps_per_epoch * CFG.num_epochs
    warmup_steps = int(total_training_steps * CFG.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(CFG.num_epochs):
        sampler.set_epoch(epoch)

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].cuda(local_rank, non_blocking=True)
            labels = batch["labels"].cuda(local_rank, non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(local_rank, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / CFG.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    CFG.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % CFG.log_every == 0:
                    # averaged loss across ranks for logging
                    reduced_loss = loss.detach().float().clone()
                    dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)

                    if is_main_process():
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"loss={reduced_loss.item() * CFG.gradient_accumulation_steps:.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e} "
                            f"grad_norm={float(grad_norm):.4f}"
                        )

                if global_step % CFG.save_every_steps == 0:
                    save_model_checkpoint(model, tokenizer, CFG.output_dir, global_step)

    if is_main_process():
        print("Training finished.")

    cleanup_dist()


if __name__ == "__main__":
    main()