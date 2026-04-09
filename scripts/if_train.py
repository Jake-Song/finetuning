# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.9.1",
#     "transformers",
#     "datasets",
#     "wandb",
#     "pyyaml",
#     "vllm>=0.19.0",
#     "openai",
# ]
# ///

"""
IFEval GRPO training in native PyTorch.

Trains a model to follow instruction constraints (word count, formatting, keywords, etc.)
using Group Relative Policy Optimization with the Nemotron-Cascade-2 IF-RL dataset.

Usage:
  Single GPU:   uv run scripts/if_train.py
  Multi-GPU:    uv run torchrun --nproc_per_node=N scripts/if_train.py
  Dry run:      uv run scripts/if_train.py --dry-run
"""

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    autodetect_device_type,
)
from utils.openai_server import OpenAICompatibleRolloutClient, sync_server_model_weights


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "nvidia/Nemotron-Cascade-2-RL-data"
    dataset_config: str = "IF-RL"
    max_prompt_length: int = 256

    # GRPO
    num_generations: int = 4
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 1.0

    # vLLM server
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_model_name_for_requests: str = ""
    vllm_api_key: str = "EMPTY"
    vllm_request_timeout: float = 300.0
    vllm_sync_timeout: float = 300.0
    vllm_weight_sync_backend: str = "nccl"
    vllm_max_parallel_requests: int = 8

    # Optimization
    batch_size: int = 16
    per_device_batch_size: int = 8
    learning_rate: float = 1e-6
    warmup_steps: int = 10
    init_lr_frac: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    max_steps: int = 500
    output_dir: str = "./ckpt_grpo_ifeval"
    save_every: int = 50
    log_every: int = 1
    eval_size: int = 200
    seed: int = 42

    # wandb
    run_name: str = "dummy"


# -----------------------------------------------------------------------------
# Constraint checkers
# -----------------------------------------------------------------------------
def _compare(value: int, relation: str, target: int) -> bool:
    if relation == "at least":
        return value >= target
    if relation == "at most":
        return value <= target
    if relation == "less than":
        return value < target
    if relation == "more than":
        return value > target
    if relation == "exactly":
        return value == target
    return True


def check_no_comma(text: str, kw: dict) -> bool:
    return "," not in text


def check_number_words(text: str, kw: dict) -> bool:
    relation = kw.get("relation")
    num_words = kw.get("num_words")
    if relation is None or num_words is None:
        return True
    return _compare(len(text.split()), relation, num_words)


def check_number_sentences(text: str, kw: dict) -> bool:
    relation = kw.get("relation")
    num_sentences = kw.get("num_sentences")
    if relation is None or num_sentences is None:
        return True
    count = len(re.findall(r"[.!?]+(?:\s|$)", text))
    return _compare(count, relation, num_sentences)


def check_number_paragraphs(text: str, kw: dict) -> bool:
    num_paragraphs = kw.get("num_paragraphs")
    if num_paragraphs is None:
        return True
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paragraphs) == num_paragraphs


def check_keywords_existence(text: str, kw: dict) -> bool:
    keywords = kw.get("keywords")
    if not keywords:
        return True
    text_lower = text.lower()
    return all(k.lower() in text_lower for k in keywords)


def check_forbidden_words(text: str, kw: dict) -> bool:
    forbidden = kw.get("forbidden_words")
    if not forbidden:
        return True
    text_lower = text.lower()
    return all(w.lower() not in text_lower for w in forbidden)


def check_keyword_frequency(text: str, kw: dict) -> bool:
    keyword = kw.get("keyword")
    frequency = kw.get("frequency")
    relation = kw.get("relation")
    if keyword is None or frequency is None or relation is None:
        return True
    count = text.lower().count(keyword.lower())
    return _compare(count, relation, frequency)


def check_english_lowercase(text: str, kw: dict) -> bool:
    return text == text.lower()


def check_english_capital(text: str, kw: dict) -> bool:
    return text == text.upper()


def check_title(text: str, kw: dict) -> bool:
    return bool(re.match(r"^#\s", text.strip()))


def check_number_highlighted_sections(text: str, kw: dict) -> bool:
    num_highlights = kw.get("num_highlights")
    if num_highlights is None:
        return True
    count = len(re.findall(r"(?m)^\*{2,}[^*]+\*{2,}", text))
    if count == 0:
        count = len(re.findall(r"(?m)^#{1,6}\s", text))
    return _compare(count, "at least", num_highlights)


def check_number_bullet_lists(text: str, kw: dict) -> bool:
    num_bullets = kw.get("num_bullets")
    if num_bullets is None:
        return True
    count = len(re.findall(r"(?m)^[\*\-]\s", text))
    return _compare(count, "at least", num_bullets)


def check_json_format(text: str, kw: dict) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_postscript(text: str, kw: dict) -> bool:
    marker = kw.get("postscript_marker", "P.S.")
    if marker is None:
        marker = "P.S."
    return marker in text


def check_end_checker(text: str, kw: dict) -> bool:
    end_phrase = kw.get("end_phrase")
    if end_phrase is None:
        return True
    return text.strip().endswith(end_phrase)


def check_quotation(text: str, kw: dict) -> bool:
    stripped = text.strip()
    return (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("\u201c") and stripped.endswith("\u201d")
    )


def check_repeat_prompt(text: str, kw: dict) -> bool:
    prompt_to_repeat = kw.get("prompt_to_repeat")
    if prompt_to_repeat is None:
        return True
    return prompt_to_repeat in text


CHECKERS = {
    "punctuation:no_comma": check_no_comma,
    "length_constraints:number_words": check_number_words,
    "length_constraints:number_sentences": check_number_sentences,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "keywords:existence": check_keywords_existence,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:frequency": check_keyword_frequency,
    "change_case:english_lowercase": check_english_lowercase,
    "change_case:english_capital": check_english_capital,
    "detectable_format:title": check_title,
    "detectable_format:number_highlighted_sections": check_number_highlighted_sections,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:json_format": check_json_format,
    "detectable_content:postscript": check_postscript,
    "startend:end_checker": check_end_checker,
    "startend:quotation": check_quotation,
    "combination:repeat_prompt": check_repeat_prompt,
}


# -----------------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------------
def compute_rewards(
    completions: list[str],
    instruction_id_list: list[list[str]],
    kwargs_list: list[list[dict]],
) -> list[float]:
    rewards = []
    for text, ids, kws in zip(completions, instruction_id_list, kwargs_list):
        text = text.strip()
        if not text:
            rewards.append(0.0)
            continue
        if not ids:
            rewards.append(0.5)
            continue

        passed = 0
        for constraint_id, kw in zip(ids, kws):
            checker = CHECKERS.get(constraint_id)
            if checker is None:
                passed += 1
                continue
            if checker(text, kw):
                passed += 1
        rewards.append(passed / len(ids))
    return rewards


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def load_ifeval_dataset(cfg: TrainConfig, eval_size: int = 100) -> tuple[Dataset, Dataset | None]:
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")

    rows = []
    for example in ds:
        rows.append({
            "prompt": example["prompt"],
            "instruction_id_list": json.dumps(example["instruction_id_list"]),
            "kwargs": json.dumps(example["kwargs"]),
        })

    full = Dataset.from_list(rows)
    if eval_size > 0 and eval_size < len(full):
        splits = full.train_test_split(test_size=eval_size, seed=42)
        return splits["train"], splits["test"]
    return full, None


# -----------------------------------------------------------------------------
# Generation (vLLM)
# -----------------------------------------------------------------------------
def generate_completions(
    rollout_client: OpenAICompatibleRolloutClient, tokenizer, prompts: list[str], *,
    max_new_tokens: int, temperature: float, top_p: float,
    num_generations: int,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    return rollout_client.generate_completions(
        tokenizer,
        list(prompts),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_generations=num_generations,
    )


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
# GRPO loss computation
# -----------------------------------------------------------------------------
def compute_grpo_loss(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    completion_mask: torch.Tensor, advantages: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Standard GRPO loss: on-policy REINFORCE with group-normalized rewards, token-level loss.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    token_mask = completion_mask[:, 1:].float()

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    per_token_loss = advantages.unsqueeze(-1) * token_log_probs * token_mask
    num_valid = token_mask.sum().clamp(min=1)
    loss = -per_token_loss.sum() / num_valid

    stats = {
        "grpo/mean_advantage": advantages.mean().item(),
    }
    return loss, stats


# -----------------------------------------------------------------------------
# Dry run
# -----------------------------------------------------------------------------
def dry_run(cfg: TrainConfig):
    print("=" * 60)
    print("IFEval GRPO (Native PyTorch) - DRY RUN")
    print("=" * 60)

    print(f"\n[Config]")
    print(f"  model:            {cfg.model_name}")
    print(f"  num_generations:  {cfg.num_generations}")
    print(f"  batch_size:       {cfg.batch_size}")
    print(f"  per_device_bs:    {cfg.per_device_batch_size}")
    print(f"  max_completion:   {cfg.max_completion_length}")
    print(f"  lr:               {cfg.learning_rate}")
    print(f"  max_steps:        {cfg.max_steps}")
    print(f"  vllm_server:      {cfg.vllm_server_host}:{cfg.vllm_server_port}")
    print(f"  vllm_sync:        {cfg.vllm_weight_sync_backend}")
    print(f"  output_dir:       {cfg.output_dir}")

    print(f"\n[Dataset]")
    train_dataset, eval_dataset = load_ifeval_dataset(cfg, cfg.eval_size)
    print(f"  train samples: {len(train_dataset)}")
    print(f"  eval samples:  {len(eval_dataset) if eval_dataset else 0}")

    # constraint coverage stats
    from collections import Counter
    all_ids = []
    for row in train_dataset:
        all_ids.extend(json.loads(row["instruction_id_list"]))
    counts = Counter(all_ids)
    covered = sum(c for cid, c in counts.items() if cid in CHECKERS)
    total = sum(counts.values())
    print(f"  total constraints: {total}")
    print(f"  covered: {covered} ({100 * covered / total:.1f}%)")
    print(f"  constraint types: {len(counts)}")
    print(f"  implemented: {len(CHECKERS)}")
    print(f"\n  Top constraints:")
    for cid, count in counts.most_common(10):
        tag = "✓" if cid in CHECKERS else "✗"
        print(f"    {tag} {cid}: {count}")

    print(f"\n[Reward function test]")
    first = train_dataset[0]
    first_ids = json.loads(first["instruction_id_list"])
    first_kwargs = json.loads(first["kwargs"])
    print(f"  Prompt: {first['prompt'][:100]}...")
    print(f"  Constraints: {first_ids}")
    test_rewards = compute_rewards(
        ["This is a test response."],
        [first_ids],
        [first_kwargs],
    )
    print(f"  Test reward: {test_rewards[0]:.2f}")

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
    parser = argparse.ArgumentParser(description="IFEval GRPO training (native PyTorch)")
    parser.add_argument("--config", type=str, default=None, help="YAML config override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--vllm-server-host", type=str, default=None)
    parser.add_argument("--vllm-server-port", type=int, default=None)
    parser.add_argument("--vllm-model-name", type=str, default=None)
    parser.add_argument("--vllm-api-key", type=str, default=None)
    parser.add_argument("--vllm-request-timeout", type=float, default=None)
    parser.add_argument("--vllm-sync-timeout", type=float, default=None)
    parser.add_argument("--vllm-weight-sync-backend", type=str, default=None)
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
    if args.num_generations is not None:
        cfg.num_generations = args.num_generations
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.vllm_server_host is not None:
        cfg.vllm_server_host = args.vllm_server_host
    if args.vllm_server_port is not None:
        cfg.vllm_server_port = args.vllm_server_port
    if args.vllm_model_name is not None:
        cfg.vllm_model_name_for_requests = args.vllm_model_name
    if args.vllm_api_key is not None:
        cfg.vllm_api_key = args.vllm_api_key
    if args.vllm_request_timeout is not None:
        cfg.vllm_request_timeout = args.vllm_request_timeout
    if args.vllm_sync_timeout is not None:
        cfg.vllm_sync_timeout = args.vllm_sync_timeout
    if args.vllm_weight_sync_backend is not None:
        cfg.vllm_weight_sync_backend = args.vllm_weight_sync_backend

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
        project="grpo-ifeval", name=cfg.run_name, config=asdict(cfg),
    )

    # Tokenizer
    print0(f"Loading tokenizer from {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training model
    print0(f"Loading training model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.to(device)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_local_rank],
        )
    raw_model = model.module if ddp else model

    request_model_name = cfg.vllm_model_name_for_requests or cfg.model_name
    print0(
        f"Connecting to vLLM server at {cfg.vllm_server_host}:{cfg.vllm_server_port} "
        f"(model={request_model_name})..."
    )
    rollout_client = OpenAICompatibleRolloutClient(
        host=cfg.vllm_server_host,
        port=cfg.vllm_server_port,
        model_name=request_model_name,
        api_key=cfg.vllm_api_key,
        request_timeout=cfg.vllm_request_timeout,
        max_parallel_requests=cfg.vllm_max_parallel_requests,
    )

    # Dataset
    print0(f"Loading dataset: {cfg.dataset_name}/{cfg.dataset_config}...")
    train_dataset, eval_dataset = load_ifeval_dataset(cfg, cfg.eval_size)
    print0(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset) if eval_dataset else 0}")

    sampler = DistributedSampler(
        train_dataset, num_replicas=ddp_world_size, rank=ddp_rank,
        shuffle=True, drop_last=True,
    ) if ddp else None

    prompts_per_rank = cfg.batch_size // ddp_world_size
    loader = DataLoader(
        train_dataset, batch_size=prompts_per_rank, sampler=sampler,
        shuffle=(sampler is None), drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        betas=(0.9, 0.95), weight_decay=cfg.weight_decay,
        fused=(device_type == "cuda"),
    )

    def get_lr_lambda(step):
        if step < cfg.warmup_steps:
            return cfg.init_lr_frac + (1.0 - cfg.init_lr_frac) * step / max(cfg.warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

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

        # 1) Sync training weights into vLLM engine
        sync_server_model_weights(
            host=cfg.vllm_server_host,
            port=cfg.vllm_server_port,
            model=raw_model,
            backend=cfg.vllm_weight_sync_backend,
            timeout=cfg.vllm_sync_timeout,
            is_sync_leader=master_process,
        )
        if ddp:
            dist.barrier()

        # 2) Generate completions via vLLM
        all_ids, all_masks, completions_text = generate_completions(
            rollout_client, tokenizer, prompts,
            max_new_tokens=cfg.max_completion_length,
            temperature=cfg.temperature, top_p=cfg.top_p,
            num_generations=cfg.num_generations,
        )

        # 3) Expand per-prompt metadata to per-completion
        ids_expanded = []
        kwargs_expanded = []
        for i in range(len(all_ids)):
            prompt_idx = i // cfg.num_generations
            ids_expanded.append(json.loads(batch["instruction_id_list"][prompt_idx]))
            kwargs_expanded.append(json.loads(batch["kwargs"][prompt_idx]))

        # 4) Compute rewards
        rewards = compute_rewards(completions_text, ids_expanded, kwargs_expanded)
        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)

        # 5) Group-normalize rewards to advantages (z-score per prompt group)
        rewards_grouped = rewards_t.view(-1, cfg.num_generations)
        mu = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = ((rewards_grouped - mu) / std).view(-1)

        # 6) Forward + GRPO loss in sub-batches
        input_ids, attention_mask, completion_mask = pad_and_stack(all_ids, all_masks, pad_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        completion_mask = completion_mask.to(device)

        total_seqs = input_ids.shape[0]
        num_sub_batches = math.ceil(total_seqs / cfg.per_device_batch_size)

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        all_stats = {"grpo/mean_reward": rewards_t.mean().item()}

        model.train()
        for sb in range(num_sub_batches):
            b0 = sb * cfg.per_device_batch_size
            b1 = min(b0 + cfg.per_device_batch_size, total_seqs)

            loss, stats = compute_grpo_loss(
                model, input_ids[b0:b1], attention_mask[b0:b1],
                completion_mask[b0:b1], advantages[b0:b1],
            )
            loss = loss / num_sub_batches
            loss.backward()
            total_loss += loss.item()

            for k, v in stats.items():
                all_stats[k] = all_stats.get(k, 0.0) + v / num_sub_batches

        # Gradient step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
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
            raw_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print0(f"Saved checkpoint to {ckpt_dir}")

    # Final save
    if master_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        raw_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print0(f"Saved final model to {final_dir}")

    wandb_run.finish()
    compute_cleanup()
    print0("Training complete.")


if __name__ == "__main__":
    main()
