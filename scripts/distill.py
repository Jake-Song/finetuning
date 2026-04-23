# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.9.1",
#     "transformers",
#     "datasets",
#     "wandb",
#     "pyyaml",
# ]
# ///

"""
Knowledge distillation: train a student model to match a teacher model's
soft probability distributions, combined with standard cross-entropy on
hard labels.

Loss = alpha * KL(student_soft || teacher_soft) + (1 - alpha) * CE(student, labels)

Usage:
  Single GPU:   uv run scripts/distill.py --train-path data.jsonl
  Multi-GPU:    uv run torchrun --nproc_per_node=N scripts/distill.py --train-path data.jsonl
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import yaml
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.attention import load_causal_lm_with_attention
from utils.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    autodetect_device_type,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    teacher_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    train_path: str = "train.jsonl"
    max_length: int = 2048

    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    temperature: float = 2.0
    alpha: float = 0.5  # weight for KL loss; (1 - alpha) weights CE loss

    log_every: int = 10
    save_every: int = 500
    output_dir: str = "./ckpt_distill"
    seed: int = 42


# -----------------------------------------------------------------------------
# Dataset & collator
# -----------------------------------------------------------------------------
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


class AssistantOnlyCollator:
    """Tokenizes chat messages; only assistant tokens participate in loss."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for ex in batch:
            messages = ex["messages"]
            assert messages[-1]["role"] == "assistant", "Last message must be assistant"

            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )

            full_enc = self.tokenizer(
                full_text, truncation=True, max_length=self.max_length,
                padding=False, return_tensors=None,
            )
            prompt_enc = self.tokenizer(
                prompt_text, truncation=True, max_length=self.max_length,
                padding=False, return_tensors=None,
            )

            input_ids = full_enc["input_ids"]
            attn_mask = full_enc["attention_mask"]
            prompt_len = len(prompt_enc["input_ids"])

            labels = input_ids.copy()
            labels[:prompt_len] = [-100] * prompt_len

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attn_mask, dtype=torch.long))

        input_ids = nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id,
        )
        labels = nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100,
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0,
        )

        input_ids = input_ids[:, :self.max_length]
        labels = labels[:, :self.max_length]
        attention_mask = attention_mask[:, :self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation for causal LMs")
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
    parser.add_argument("--config", type=str, default=None, help="YAML config override")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--train-path", type=str, default=None, help="Path to training JSONL")
    parser.add_argument("--teacher", type=str, default=None, help="Teacher model name/path")
    parser.add_argument("--student", type=str, default=None, help="Student model name/path")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()

    # YAML overrides
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))

    # CLI overrides
    if args.train_path:
        cfg.train_path = args.train_path
    if args.teacher:
        cfg.teacher_name = args.teacher
    if args.student:
        cfg.student_name = args.student
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.lr is not None:
        cfg.lr = args.lr

    # Init compute
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    torch.manual_seed(cfg.seed + ddp_rank)

    # wandb
    use_dummy = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy else wandb.init(
        project="distill", name=args.run, config=vars(cfg),
    )

    # Tokenizer (use student's tokenizer — teacher and student must share the same tokenizer/vocab)
    print0(f"Loading tokenizer from {cfg.student_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Teacher model (frozen)
    print0(f"Loading teacher model: {cfg.teacher_name}...")
    teacher, _teacher_attn_implementation = load_causal_lm_with_attention(
        cfg.teacher_name,
        log_prefix="teacher model",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
    )
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student model
    print0(f"Loading student model: {cfg.student_name}...")
    student, _student_attn_implementation = load_causal_lm_with_attention(
        cfg.student_name,
        log_prefix="student model",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
    )
    student.config.use_cache = False
    student.to(device)
    student.train()

    if ddp:
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[ddp_local_rank],
        )
    raw_student = student.module if ddp else student

    # Dataset & dataloader
    dataset = SFTJsonlDataset(cfg.train_path)
    print0(f"Loaded {len(dataset)} training examples from {cfg.train_path}")

    sampler = DistributedSampler(
        dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, drop_last=True,
    ) if ddp else None

    collator = AssistantOnlyCollator(tokenizer=tokenizer, max_length=cfg.max_length)
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay, fused=(device_type == "cuda"),
    )

    num_update_steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def get_lr(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    print0(f"Training config: {cfg}")
    print0(f"Total update steps: {total_steps}, warmup: {warmup_steps}")

    # Training loop
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                # Shift: predict next token. logits[:, :-1] predicts labels[:, 1:]
                teacher_logits = teacher_outputs.logits[:, :-1, :]

            # Student forward
            student_outputs = student(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            student_logits = student_outputs.logits[:, :-1, :]

            # Shifted labels for loss computation
            shift_labels = labels[:, 1:].contiguous()

            # Mask: only compute loss on assistant tokens (where label != -100)
            mask = (shift_labels != -100).float()  # (B, T-1)
            num_valid = mask.sum().clamp(min=1)

            # --- KL divergence loss on soft targets ---
            T = cfg.temperature
            teacher_soft = F.log_softmax(teacher_logits / T, dim=-1)
            student_log_soft = F.log_softmax(student_logits / T, dim=-1)
            # KL(teacher || student) = sum teacher_prob * (log teacher_prob - log student_prob)
            # Using F.kl_div with log_target=True: expects input=student_log, target=teacher_log
            kl_per_token = F.kl_div(
                student_log_soft, teacher_soft, reduction="none", log_target=True,
            ).sum(dim=-1)  # (B, T-1)
            kl_loss = (kl_per_token * mask).sum() / num_valid
            # Scale by T^2 as per Hinton et al. to balance gradient magnitudes
            kl_loss = kl_loss * (T * T)

            # --- Cross-entropy loss on hard labels ---
            ce_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

            # Combined loss
            loss = cfg.alpha * kl_loss + (1 - cfg.alpha) * ce_loss
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation step
            if (micro_step + 1) % cfg.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    # Aggregate loss across ranks for logging
                    log_loss = loss.detach().float() * cfg.gradient_accumulation_steps
                    log_kl = kl_loss.detach().float()
                    log_ce = ce_loss.detach().float()
                    if ddp:
                        for t in (log_loss, log_kl, log_ce):
                            dist.all_reduce(t, op=dist.ReduceOp.AVG)

                    current_lr = scheduler.get_last_lr()[0]
                    print0(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={log_loss.item():.4f} kl={log_kl.item():.4f} ce={log_ce.item():.4f} "
                        f"lr={current_lr:.2e} grad_norm={float(grad_norm):.4f}"
                    )
                    wandb_run.log({
                        "step": global_step,
                        "loss": log_loss.item(),
                        "kl_loss": log_kl.item(),
                        "ce_loss": log_ce.item(),
                        "lr": current_lr,
                        "grad_norm": float(grad_norm),
                    })

                # Save checkpoint
                if master_process and global_step % cfg.save_every == 0:
                    ckpt_dir = os.path.join(cfg.output_dir, f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    raw_student.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print0(f"Saved checkpoint to {ckpt_dir}")

    # Final save
    if master_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        raw_student.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print0(f"Saved final model to {final_dir}")

    wandb_run.finish()
    compute_cleanup()
    print0("Training complete.")


if __name__ == "__main__":
    main()
