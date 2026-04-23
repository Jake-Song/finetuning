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
Multi-Domain On-Policy Distillation (MOPD) training in native PyTorch.

Implements the MOPD algorithm from Nemotron-Cascade 2 (arXiv:2603.19220):
  - Generate rollouts from inference policy (the current model)
  - Compute token-level distillation advantage: a_t = log π_teacher(y_t|s_t) - log π_train(y_t|s_t)
  - Truncated importance weighting for train/inference mismatch
  - Surrogate loss: L = -1/|V(y)| * Σ w_t * sg(a_t) * log π_train(y_t|s_t)

Also supports standard GRPO with outcome-based rewards when no teacher is used.

Usage:
  Single GPU:   uv run scripts/mopd_train.py
  Multi-GPU:    uv run torchrun --nproc_per_node=N scripts/mopd_train.py
  Dry run:      uv run scripts/mopd_train.py --dry-run
  Paper preset: uv run scripts/mopd_train.py --config configs/mopd/nemotron_cascade2_paper.yaml
"""

import argparse
import copy
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import yaml
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
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
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "nvidia/Nemotron-Cascade-2-RL-data"
    dataset_config: str = "MOPD"
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    min_pass_rate: float = 0.0
    attn_implementation: str = "flash_attention_3"

    # MOPD-specific
    use_mopd: bool = True
    teacher_name: str = ""  # if empty, uses a frozen copy of the initial model
    epsilon_low: float = 0.5   # importance weight clipping lower bound
    epsilon_high: float = 2.0  # importance weight clipping upper bound

    # GRPO fallback (used when use_mopd=False)
    num_generations: int = 4
    temperature: float = 1.0
    top_p: float = 1.0

    # Optimization
    batch_size: int = 128        # total prompts per update (across all ranks)
    per_device_batch_size: int = 4
    learning_rate: float = 2e-6
    warmup_steps: int = 30
    init_lr_frac: float = 0.1    # initial LR = init_lr_frac * learning_rate
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    max_steps: int = 50
    output_dir: str = "./ckpt_mopd"
    save_every: int = 25
    log_every: int = 1
    eval_size: int = 100
    seed: int = 42

    # wandb
    run_name: str = "dummy"


# -----------------------------------------------------------------------------
# Reward functions (for GRPO mode)
# -----------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _extract_with_regex(completion: str, output_regex: str) -> str | None:
    if not output_regex:
        return None
    try:
        m = re.search(output_regex, completion, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
    except re.error:
        pass
    return None


def _check_mcqa(completion: str, expected: str, output_regex: str) -> float:
    expected_norm = expected.strip().upper()
    extracted = _extract_with_regex(completion, output_regex)
    if extracted and extracted.strip().upper() == expected_norm:
        return 1.0
    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Z])\)?",
        r"\b([A-Z])\)",
        r"^\s*\(?([A-Z])\)?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, completion, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() == expected_norm:
            return 1.0
    if expected_norm in completion.upper().split():
        return 0.5
    return 0.0


def _check_tool_call(completion: str, ground_truth: list[dict]) -> float:
    if not ground_truth:
        return 0.5
    try:
        parsed = json.loads(completion.strip())
        if isinstance(parsed, dict):
            parsed = [parsed]
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", completion, re.DOTALL)
        if m:
            try:
                parsed = [json.loads(m.group())]
            except (json.JSONDecodeError, ValueError):
                return 0.0
        else:
            return 0.0

    if not parsed:
        return 0.0

    total = len(ground_truth)
    matched = 0
    for gt in ground_truth:
        gt_name = gt.get("name", "")
        gt_args = gt.get("arguments", {})
        if isinstance(gt_args, str):
            try:
                gt_args = json.loads(gt_args)
            except (json.JSONDecodeError, ValueError):
                pass
        for call in parsed:
            call_name = call.get("name", "") or call.get("function", "")
            call_args = call.get("arguments", {}) or call.get("parameters", {})
            if isinstance(call_args, str):
                try:
                    call_args = json.loads(call_args)
                except (json.JSONDecodeError, ValueError):
                    pass
            if _normalize(call_name) == _normalize(gt_name):
                if call_args == gt_args:
                    matched += 1
                else:
                    matched += 0.5
                break
    return matched / total


def _check_structured_output(completion: str, expected: str, output_regex: str) -> float:
    extracted = _extract_with_regex(completion, output_regex)
    if extracted and expected and _normalize(extracted) == _normalize(expected):
        return 1.0

    try:
        parsed = json.loads(completion.strip())
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", completion)
        if m:
            try:
                parsed = json.loads(m.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                return 0.0
        else:
            return 0.0

    score = 0.5
    if expected:
        try:
            expected_parsed = json.loads(expected.strip())
            if parsed == expected_parsed:
                score = 1.0
            elif isinstance(parsed, type(expected_parsed)):
                score = 0.7
        except (json.JSONDecodeError, ValueError):
            if _normalize(str(parsed)) == _normalize(expected):
                score = 1.0
    return score


def _detect_category(category: str) -> str:
    cat = (category or "").lower()
    if "tool" in cat or "function" in cat or "agent" in cat:
        return "tool_call"
    if "struct" in cat or "json" in cat or "schema" in cat:
        return "structured"
    return "mcqa"


def compute_rewards(completions: list[str], examples: list[dict]) -> list[float]:
    rewards = []
    for i, text in enumerate(completions):
        text = text.strip()
        if not text:
            rewards.append(0.0)
            continue
        ex = examples[i]
        cat = _detect_category(ex.get("category", ""))
        expected = ex.get("expected_answer", "") or ""
        gt = ex.get("ground_truth") or []
        regex = ex.get("output_regex", "") or ""

        if cat == "tool_call":
            rewards.append(_check_tool_call(text, gt))
        elif cat == "structured":
            rewards.append(_check_structured_output(text, expected, regex))
        else:
            rewards.append(_check_mcqa(text, expected, regex))
    return rewards


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def load_mopd_dataset(
    name: str, config: str, max_prompt_length: int, tokenizer_name: str,
    min_pass_rate: float = 0.0, eval_size: int = 100,
) -> tuple[Dataset, Dataset | None]:
    ds = load_dataset(name, config, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = []
    skipped = 0
    for example in ds:
        pass_rate = example.get("pass_rate")
        if pass_rate is not None and pass_rate < min_pass_rate:
            skipped += 1
            continue

        prompt_text = example.get("prompt") or example.get("question") or ""
        if not prompt_text:
            continue

        tokens = tokenizer.encode(prompt_text, truncation=True, max_length=max_prompt_length)
        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        tmeta = example.get("template_metadata")
        output_regex = ""
        if isinstance(tmeta, dict):
            output_regex = tmeta.get("output_regex", "") or ""
        elif isinstance(tmeta, list) and tmeta:
            output_regex = tmeta[0].get("output_regex", "") or ""

        rows.append({
            "prompt": prompt_text,
            "expected_answer": example.get("expected_answer", "") or "",
            "category": example.get("category", "") or "",
            "ground_truth": example.get("ground_truth") or [],
            "output_regex": output_regex,
        })

    if skipped:
        print0(f"  Filtered {skipped} samples below min_pass_rate={min_pass_rate}")

    if not rows:
        raise ValueError(f"No usable rows found in dataset {name}/{config}.")

    full = Dataset.from_list(rows)
    if eval_size > 0 and eval_size < len(full):
        splits = full.train_test_split(test_size=eval_size, seed=42)
        return splits["train"], splits["test"]
    return full, None


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_completions(
    model, tokenizer, prompts: list[str], *,
    max_new_tokens: int, temperature: float, top_p: float,
    num_generations: int, device: torch.device, per_device_batch_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Generate completions for a list of prompts. Returns (all_input_ids, all_completion_masks)
    where each entry corresponds to one (prompt, generation) pair.
    Each input_ids is the full sequence (prompt + completion).
    Each completion_mask has 1 for completion tokens, 0 for prompt tokens.
    """
    model.eval()
    all_input_ids = []
    all_completion_masks = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                               max_length=max_new_tokens * 2)
        prompt_ids = prompt_enc["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]

        # Generate in sub-batches to avoid OOM
        for gen_start in range(0, num_generations, per_device_batch_size):
            gen_count = min(per_device_batch_size, num_generations - gen_start)
            batch_prompt = prompt_ids.expand(gen_count, -1)

            outputs = model.generate(
                batch_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            for seq in outputs:
                seq_ids = seq.tolist()
                mask = [0] * prompt_len + [1] * (len(seq_ids) - prompt_len)
                all_input_ids.append(seq_ids)
                all_completion_masks.append(mask)

    model.train()
    return all_input_ids, all_completion_masks


# -----------------------------------------------------------------------------
# MOPD loss computation
# -----------------------------------------------------------------------------
def compute_mopd_loss(
    train_model, teacher_model, inf_model,
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_low: float, epsilon_high: float,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the MOPD surrogate loss (Eq. 4 from arXiv:2603.19220).

    Args:
        train_model: the model being trained (π_train)
        teacher_model: frozen domain teacher (π_teacher)
        inf_model: frozen inference policy that generated the rollouts (π_inf)
        input_ids: (B, T) token ids
        attention_mask: (B, T) attention mask
        completion_mask: (B, T) 1 for completion tokens, 0 for prompt/pad
        epsilon_low, epsilon_high: importance weight clipping bounds

    Returns:
        loss: scalar MOPD loss
        stats: dict with diagnostic values
    """
    # Forward all three models
    train_logits = train_model(
        input_ids=input_ids, attention_mask=attention_mask,
    ).logits  # (B, T, V)

    with torch.no_grad():
        teacher_logits = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits
        inf_logits = inf_model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits

    # Shift: predict next token. Use positions [:-1] to predict tokens at [1:]
    train_logits = train_logits[:, :-1, :]  # (B, T-1, V)
    teacher_logits = teacher_logits[:, :-1, :]
    inf_logits = inf_logits[:, :-1, :]
    targets = input_ids[:, 1:]  # (B, T-1)
    token_mask = completion_mask[:, 1:].float()  # (B, T-1), valid completion tokens

    # Log probabilities of the actually sampled tokens under each policy
    train_log_probs = F.log_softmax(train_logits, dim=-1)  # (B, T-1, V)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    inf_log_probs = F.log_softmax(inf_logits, dim=-1)

    # Gather log probs for the sampled tokens
    # targets: (B, T-1) -> (B, T-1, 1)
    target_idx = targets.unsqueeze(-1)
    train_lp = train_log_probs.gather(dim=-1, index=target_idx).squeeze(-1)  # (B, T-1)
    teacher_lp = teacher_log_probs.gather(dim=-1, index=target_idx).squeeze(-1)
    inf_lp = inf_log_probs.gather(dim=-1, index=target_idx).squeeze(-1)

    # Token-level distillation advantage (Eq. 2):
    # a_t = log π_teacher(y_t|s_t) - log π_train(y_t|s_t)
    advantage = (teacher_lp - train_lp).detach()  # sg[a_t]

    # Importance weight (Eq. 3):
    # r_t = π_train(y_t|s_t) / π_inf(y_t|s_t)
    # w_t = sg[r_t] * 1[ε_low <= r_t <= ε_high]
    log_ratio = train_lp - inf_lp
    ratio = log_ratio.detach().exp()
    in_range = (ratio >= epsilon_low) & (ratio <= epsilon_high)
    weights = ratio * in_range.float()  # sg[r_t] * indicator

    # MOPD surrogate loss (Eq. 4):
    # L = -1/|V(y)| * Σ w_t * sg(a_t) * log π_train(y_t|s_t)
    per_token_loss = weights * advantage * train_lp  # (B, T-1)
    per_token_loss = per_token_loss * token_mask

    num_valid = token_mask.sum().clamp(min=1)
    loss = -per_token_loss.sum() / num_valid

    # Diagnostics
    with torch.no_grad():
        mean_advantage = (advantage * token_mask).sum() / num_valid
        mean_ratio = (ratio * token_mask).sum() / num_valid
        clip_frac = 1.0 - (in_range.float() * token_mask).sum() / num_valid

    stats = {
        "mopd/advantage": mean_advantage.item(),
        "mopd/importance_ratio": mean_ratio.item(),
        "mopd/clip_fraction": clip_frac.item(),
    }
    return loss, stats


# -----------------------------------------------------------------------------
# GRPO loss computation (fallback when no teacher)
# -----------------------------------------------------------------------------
def compute_grpo_loss(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    completion_mask: torch.Tensor, advantages: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Standard GRPO loss (Eq. 1 from arXiv:2603.19220):
    On-policy REINFORCE with group-normalized rewards, token-level loss.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    token_mask = completion_mask[:, 1:].float()

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    token_entropy = -(probs * log_probs).sum(dim=-1)

    # advantages is (B,), broadcast to (B, T-1)
    # Token-level: same advantage for all tokens in a sequence (Eq. 1)
    per_token_loss = advantages.unsqueeze(-1) * token_log_probs * token_mask

    # Normalize by total valid tokens across the group
    num_valid = token_mask.sum().clamp(min=1)
    loss = -per_token_loss.sum() / num_valid
    mean_entropy = (token_entropy * token_mask).sum() / num_valid

    stats = {
        "grpo/mean_advantage": advantages.mean().item(),
        "grpo/entropy": mean_entropy.item(),
    }
    return loss, stats


# -----------------------------------------------------------------------------
# Pad sequences to same length for batched forward pass
# -----------------------------------------------------------------------------
def pad_and_stack(
    sequences: list[list[int]], masks: list[list[int]], pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences. Returns (input_ids, attention_mask, completion_mask)."""
    max_len = max(len(s) for s in sequences)
    input_ids = []
    attention_masks = []
    completion_masks = []
    for seq, mask in zip(sequences, masks):
        pad_len = max_len - len(seq)
        input_ids.append(seq + [pad_id] * pad_len)
        attention_masks.append([1] * len(seq) + [0] * pad_len)
        completion_masks.append(mask + [0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(completion_masks, dtype=torch.long),
    )


# -----------------------------------------------------------------------------
# Dry run
# -----------------------------------------------------------------------------
def dry_run(cfg: TrainConfig):
    print("=" * 60)
    print("MOPD NATIVE PYTORCH - DRY RUN")
    print("=" * 60)

    print(f"\n[Config]")
    print(f"  model:            {cfg.model_name}")
    print(f"  mode:             {'MOPD' if cfg.use_mopd else 'GRPO'}")
    if cfg.use_mopd:
        print(f"  teacher:          {cfg.teacher_name or '(frozen copy of initial model)'}")
        print(f"  epsilon:          [{cfg.epsilon_low}, {cfg.epsilon_high}]")
    print(f"  dataset:          {cfg.dataset_name} ({cfg.dataset_config})")
    print(f"  max_prompt_len:   {cfg.max_prompt_length}")
    print(f"  min_pass_rate:    {cfg.min_pass_rate}")
    print(f"  num_generations:  {cfg.num_generations}")
    print(f"  batch_size:       {cfg.batch_size}")
    print(f"  lr:               {cfg.learning_rate}")
    print(f"  warmup_steps:     {cfg.warmup_steps}")
    print(f"  max_steps:        {cfg.max_steps}")
    print(f"  output_dir:       {cfg.output_dir}")

    print(f"\n[Dataset]")
    train_dataset, eval_dataset = load_mopd_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name,
        cfg.min_pass_rate, cfg.eval_size,
    )
    print(f"  train samples: {len(train_dataset)}")
    print(f"  eval samples:  {len(eval_dataset) if eval_dataset else 0}")

    cats = Counter()
    regex_count = 0
    for row in train_dataset:
        cats[_detect_category(row["category"])] += 1
        if row["output_regex"]:
            regex_count += 1
    print(f"\n  Category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")
    print(f"  Samples with output_regex: {regex_count}")

    first = train_dataset[0]
    print(f"\n  First prompt preview: {first['prompt'][:150]}...")
    print(f"  First expected_answer: {first['expected_answer'][:100]}")
    print(f"  First category: {first['category']}")

    print(f"\n[Reward function test]")
    test_completions = ["", "Answer: B", '{"name": "search", "arguments": {"query": "test"}}']
    test_examples = [
        {"category": "mcqa", "expected_answer": "B", "ground_truth": [], "output_regex": ""},
        {"category": "mcqa", "expected_answer": "B", "ground_truth": [], "output_regex": ""},
        {"category": "tool_call", "expected_answer": "", "ground_truth": [{"name": "search", "arguments": {"query": "test"}}], "output_regex": ""},
    ]
    test_rewards = compute_rewards(test_completions, test_examples)
    labels = ["empty", "correct mcqa", "exact tool call"]
    for label, rew in zip(labels, test_rewards):
        print(f"  {rew:.2f} <- {label}")

    print(f"\n[Tokenizer]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        print(f"  vocab_size:  {tokenizer.vocab_size}")
        print(f"  pad_token:   {tokenizer.pad_token}")
        print(f"  eos_token:   {tokenizer.eos_token}")
    except Exception as e:
        print(f"  Failed to load tokenizer: {e}")

    print(f"\n{'=' * 60}")
    print("DRY RUN COMPLETE")
    print("=" * 60)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MOPD training in native PyTorch")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config override, e.g. configs/mopd/nemotron_cascade2_paper.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--teacher", type=str, default=None, help="Teacher model name/path")
    parser.add_argument("--no-mopd", action="store_true", help="Use GRPO instead of MOPD")
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.run_name = args.run

    # YAML overrides
    if args.config:
        with open(args.config) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))

    # CLI overrides
    if args.teacher:
        cfg.teacher_name = args.teacher
    if args.no_mopd:
        cfg.use_mopd = False
    if args.num_generations is not None:
        cfg.num_generations = args.num_generations
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.attn_implementation is not None:
        cfg.attn_implementation = args.attn_implementation

    if args.dry_run:
        dry_run(cfg)
        return

    # Init compute
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    torch.manual_seed(cfg.seed + ddp_rank)

    # wandb
    use_dummy = cfg.run_name == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy else wandb.init(
        project="mopd", name=cfg.run_name, config=asdict(cfg),
    )

    # Tokenizer
    print0(f"Loading tokenizer from {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Train model (π_train)
    print0(f"Loading train model: {cfg.model_name}...")
    train_model, _resolved_attn_implementation = load_causal_lm_with_attention(
        cfg.model_name,
        log_prefix="train model",
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    train_model.config.use_cache = False
    train_model.to(device)

    if ddp:
        train_model = torch.nn.parallel.DistributedDataParallel(
            train_model, device_ids=[ddp_local_rank],
        )
    raw_train_model = train_model.module if ddp else train_model

    if cfg.use_mopd:
        # Teacher model (π_teacher) — frozen
        if cfg.teacher_name:
            print0(f"Loading teacher model: {cfg.teacher_name}...")
            teacher_model, _teacher_attn_implementation = load_causal_lm_with_attention(
                cfg.teacher_name,
                log_prefix="teacher model",
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
            )
        else:
            print0("Using frozen copy of initial model as teacher...")
            teacher_model = copy.deepcopy(raw_train_model)
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        # Inference model (π_inf) — frozen snapshot, updated each step
        # In fully on-policy mode, π_inf = π_train at the start of each step.
        # We keep a separate frozen copy to compute importance weights.
        print0("Creating inference model snapshot...")
        inf_model = copy.deepcopy(raw_train_model)
        inf_model.to(device)
        inf_model.eval()
        for p in inf_model.parameters():
            p.requires_grad = False

    # Dataset
    print0(f"Loading dataset: {cfg.dataset_name}/{cfg.dataset_config}...")
    train_dataset, eval_dataset = load_mopd_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name,
        cfg.min_pass_rate, cfg.eval_size,
    )
    print0(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset) if eval_dataset else 0}")

    sampler = DistributedSampler(
        train_dataset, num_replicas=ddp_world_size, rank=ddp_rank,
        shuffle=True, drop_last=True,
    ) if ddp else None

    # We load prompts in batches, then generate completions
    prompts_per_rank = cfg.batch_size // ddp_world_size
    loader = DataLoader(
        train_dataset, batch_size=prompts_per_rank, sampler=sampler,
        shuffle=(sampler is None), drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=cfg.learning_rate,
        betas=(0.9, 0.95), weight_decay=cfg.weight_decay,
        fused=(device_type == "cuda"),
    )

    # LR schedule: linear warmup from init_lr_frac * lr to lr, then constant
    def get_lr_lambda(step):
        if step < cfg.warmup_steps:
            return cfg.init_lr_frac + (1.0 - cfg.init_lr_frac) * step / max(cfg.warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    print0(f"Mode: {'MOPD' if cfg.use_mopd else 'GRPO'}")
    print0(f"Prompts per rank: {prompts_per_rank}, Generations per prompt: {cfg.num_generations}")
    print0(f"Max steps: {cfg.max_steps}, Warmup: {cfg.warmup_steps}")

    # Training loop
    global_step = 0
    data_iter = iter(loader)

    while global_step < cfg.max_steps:
        # Get next batch of prompts
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(global_step)
            data_iter = iter(loader)
            batch = next(data_iter)

        prompts = batch["prompt"]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        if cfg.use_mopd:
            # --- MOPD mode ---
            # 1) Sync inference model to current train model weights (on-policy)
            inf_model.load_state_dict(raw_train_model.state_dict())

            # 2) Generate rollouts from π_inf
            all_ids, all_masks = generate_completions(
                inf_model, tokenizer, prompts,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.temperature, top_p=cfg.top_p,
                num_generations=cfg.num_generations,
                device=device, per_device_batch_size=cfg.per_device_batch_size,
            )

            # 3) Forward pass + MOPD loss in sub-batches
            input_ids, attention_mask, completion_mask = pad_and_stack(all_ids, all_masks, pad_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            completion_mask = completion_mask.to(device)

            total_seqs = input_ids.shape[0]
            num_sub_batches = math.ceil(total_seqs / cfg.per_device_batch_size)

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            all_stats = {}

            train_model.train()
            for sb in range(num_sub_batches):
                b0 = sb * cfg.per_device_batch_size
                b1 = min(b0 + cfg.per_device_batch_size, total_seqs)

                loss, stats = compute_mopd_loss(
                    train_model, teacher_model, inf_model,
                    input_ids[b0:b1], attention_mask[b0:b1], completion_mask[b0:b1],
                    cfg.epsilon_low, cfg.epsilon_high,
                )
                loss = loss / num_sub_batches
                loss.backward()
                total_loss += loss.item()

                for k, v in stats.items():
                    all_stats[k] = all_stats.get(k, 0.0) + v / num_sub_batches

        else:
            # --- GRPO mode ---
            # 1) Generate rollouts from current policy
            all_ids, all_masks = generate_completions(
                raw_train_model, tokenizer, prompts,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.temperature, top_p=cfg.top_p,
                num_generations=cfg.num_generations,
                device=device, per_device_batch_size=cfg.per_device_batch_size,
            )

            # 2) Compute rewards
            completions_text = []
            examples_for_reward = []
            for i, (seq, mask) in enumerate(zip(all_ids, all_masks)):
                prompt_idx = i // cfg.num_generations
                comp_start = sum(1 for m in mask if m == 0)
                comp_tokens = seq[comp_start:]
                completions_text.append(tokenizer.decode(comp_tokens, skip_special_tokens=True))
                examples_for_reward.append({
                    "category": batch["category"][prompt_idx],
                    "expected_answer": batch["expected_answer"][prompt_idx],
                    "ground_truth": batch["ground_truth"][prompt_idx],
                    "output_regex": batch["output_regex"][prompt_idx],
                })

            rewards = compute_rewards(completions_text, examples_for_reward)
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)

            # 3) Group-normalize rewards (z-score per prompt group, Eq. 1)
            rewards_grouped = rewards_t.view(-1, cfg.num_generations)
            mu = rewards_grouped.mean(dim=1, keepdim=True)
            std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            advantages = ((rewards_grouped - mu) / std).view(-1)

            # 4) Forward + GRPO loss in sub-batches
            input_ids, attention_mask, completion_mask = pad_and_stack(all_ids, all_masks, pad_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            completion_mask = completion_mask.to(device)

            total_seqs = input_ids.shape[0]
            num_sub_batches = math.ceil(total_seqs / cfg.per_device_batch_size)

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            all_stats = {
                "grpo/mean_reward": rewards_t.mean().item(),
                "grpo/group_reward_std": rewards_grouped.std(dim=1).mean().item(),
            }

            train_model.train()
            for sb in range(num_sub_batches):
                b0 = sb * cfg.per_device_batch_size
                b1 = min(b0 + cfg.per_device_batch_size, total_seqs)

                loss, stats = compute_grpo_loss(
                    train_model, input_ids[b0:b1], attention_mask[b0:b1],
                    completion_mask[b0:b1], advantages[b0:b1],
                )
                loss = loss / num_sub_batches
                loss.backward()
                total_loss += loss.item()

                for k, v in stats.items():
                    all_stats[k] = all_stats.get(k, 0.0) + v / num_sub_batches

        # Gradient step
        grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        # Logging
        if global_step % cfg.log_every == 0:
            loss_tensor = torch.tensor(total_loss, device=device)
            if ddp:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

            current_lr = scheduler.get_last_lr()[0]
            stats_str = " ".join(f"{k}={v:.4f}" for k, v in all_stats.items())
            print0(
                f"step={global_step}/{cfg.max_steps} loss={loss_tensor.item():.4f} "
                f"lr={current_lr:.2e} grad_norm={float(grad_norm):.4f} {stats_str}"
            )
            wandb_run.log({
                "step": global_step,
                "loss": loss_tensor.item(),
                "lr": current_lr,
                "grad_norm": float(grad_norm),
                **all_stats,
            })

        # Save checkpoint
        if master_process and global_step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, f"step_{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            raw_train_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print0(f"Saved checkpoint to {ckpt_dir}")

    # Final save
    if master_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        raw_train_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print0(f"Saved final model to {final_dir}")

    wandb_run.finish()
    compute_cleanup()
    print0("Training complete.")


if __name__ == "__main__":
    main()
