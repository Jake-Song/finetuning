# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.9.1",
#     "transformers",
#     "datasets",
#     "wandb",
#     "pyyaml",
#     "vllm",
#     "python-dotenv>=1.2.2",
# ]
# ///

import argparse
import gc
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from utils.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    print0,
)
from utils.vllm_sync import sync_vllm_model_weights

load_dotenv()


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "nvidia/Nemotron-Cascade-2-RL-data"
    dataset_config: str = "SWE-RL"
    max_prompt_length: int = 2048

    # GRPO
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_completion_length: int = 4096
    learning_rate: float = 1e-6
    epsilon: float = 0.2
    temperature: float = 0.7
    warmup_steps: int = 10
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0

    # vLLM (colocated mode)
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1

    max_steps: int = 500
    output_dir: str = "./ckpt_grpo_swe_rl"
    save_steps: int = 50
    logging_steps: int = 1
    eval_steps: int = 200
    eval_size: int = 100
    seed: int = 42

    # wandb
    report_to: str = "none"
    wandb_project: str = "grpo-swe-rl"
    wandb_entity: str = ""
    wandb_mode: str = "online"


def _extract_changed_lines(patch: str) -> set[str]:
    lines = set()
    for line in patch.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            lines.add(line.strip())
    return lines


def reward_fn(completions: list[str], golden_patch: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, golden_patch):
        text = completion.strip() if isinstance(completion, str) else str(completion).strip()
        gold = gold.strip()

        if not text:
            rewards.append(0.0)
            continue

        score = 0.0

        has_diff_headers = bool(re.search(r"^---\s", text, re.MULTILINE)) and bool(
            re.search(r"^\+\+\+\s", text, re.MULTILINE)
        )
        has_hunks = bool(re.search(r"^@@\s", text, re.MULTILINE))
        if has_diff_headers and has_hunks:
            score += 0.2
        elif has_hunks:
            score += 0.1

        has_file_path = bool(re.search(r"[a-zA-Z_/]+\.\w+", text))
        if has_file_path:
            score += 0.1

        if 10 < len(text) < len(gold) * 5:
            score += 0.1

        gen_lines = _extract_changed_lines(text)
        gold_lines = _extract_changed_lines(gold)
        if gen_lines and gold_lines:
            intersection = len(gen_lines & gold_lines)
            union = len(gen_lines | gold_lines)
            jaccard = intersection / union if union > 0 else 0.0
            score += 0.4 * jaccard

        if text.strip() == gold.strip():
            score += 0.2

        rewards.append(score)
    return rewards


def load_swe_rl_dataset(
    name: str, config: str, max_prompt_length: int, tokenizer_name: str, eval_size: int = 100
) -> tuple[Dataset, Dataset | None]:
    ds = load_dataset(name, config, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = []
    for example in ds:
        prompt_text = example.get("prompt") or example.get("problem_statement")
        golden_patch = example.get("golden_patch") or example.get("patch")
        if not prompt_text or not golden_patch:
            continue

        tokens = tokenizer.encode(prompt_text, truncation=True, max_length=max_prompt_length)
        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        rows.append({
            "prompt": prompt_text,
            "golden_patch": golden_patch,
        })

    if not rows:
        raise ValueError(
            f"No usable rows found in dataset {name}/{config}. "
            "Expected prompt+patch fields like (prompt, golden_patch) or (problem_statement, patch)."
        )

    full = Dataset.from_list(rows)
    if eval_size > 0 and eval_size < len(full):
        splits = full.train_test_split(test_size=eval_size, seed=42)
        return splits["train"], splits["test"]
    return full, None


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _wandb_enabled(report_to: str) -> bool:
    targets = {target.strip().lower() for target in report_to.split(",")}
    return "wandb" in targets


def setup_wandb(cfg: TrainConfig, run_name: str) -> None:
    if not _wandb_enabled(cfg.report_to) or not _is_main_process():
        return

    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "W&B logging is enabled but `wandb` is not installed. Install it with: uv add wandb"
        ) from e

    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.wandb_entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity

    api_key = (os.environ.get("WANDB_API_KEY") or "").strip()
    wandb_mode = cfg.wandb_mode
    if not api_key and str(wandb_mode).lower() == "online":
        print(
            "W&B: no API key (set WANDB_API_KEY); "
            "using offline mode. Logs are written under wandb/ locally."
        )
        wandb_mode = "offline"

    os.environ["WANDB_MODE"] = wandb_mode

    if api_key:
        wandb.login(key=api_key, relogin=True)

    if wandb.run is None:
        init_kwargs: dict[str, Any] = {
            "project": cfg.wandb_project,
            "name": run_name,
            "mode": wandb_mode,
            "config": asdict(cfg),
        }
        if cfg.wandb_entity:
            init_kwargs["entity"] = cfg.wandb_entity
        wandb.init(**init_kwargs)


def finish_wandb() -> None:
    if not _is_main_process():
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()


def generate_completions(
    llm: LLM,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    num_generations: int,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    sampling_params = SamplingParams(
        n=num_generations,
        temperature=max(temperature, 1e-5),
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    formatted = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    outputs = llm.generate(formatted, sampling_params)

    all_input_ids = []
    all_completion_masks = []
    all_texts = []
    for output in outputs:
        prompt_ids = list(output.prompt_token_ids)
        prompt_len = len(prompt_ids)
        for completion in output.outputs:
            comp_ids = list(completion.token_ids)
            seq_ids = prompt_ids + comp_ids
            mask = [0] * prompt_len + [1] * len(comp_ids)
            all_input_ids.append(seq_ids)
            all_completion_masks.append(mask)
            all_texts.append(completion.text)

    return all_input_ids, all_completion_masks, all_texts


def pad_and_stack(
    sequences: list[list[int]], masks: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def compute_swe_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    token_mask = completion_mask[:, 1:].float()

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    per_token_obj = advantages.unsqueeze(-1) * token_log_probs * token_mask
    num_valid = token_mask.sum().clamp(min=1)
    loss = -per_token_obj.sum() / num_valid

    stats = {
        "swe/mean_advantage": advantages.mean().item(),
        "swe/mean_centered_abs_advantage": advantages.abs().mean().item(),
    }
    return loss, stats


def append_eval_log(output_dir: str, step: int, metrics: dict[str, float]) -> None:
    if not _is_main_process():
        return

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval_results.md")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("# Evaluation Results\n\n")
            f.write("| Step | Timestamp | " + " | ".join(sorted(metrics.keys())) + " |\n")
            f.write("|------|-----------|" + "|".join("---" for _ in metrics) + "|\n")
        values = " | ".join(
            f"{metrics[k]:.4f}" if isinstance(metrics[k], float) else str(metrics[k])
            for k in sorted(metrics.keys())
        )
        f.write(f"| {step} | {timestamp} | {values} |\n")


def save_checkpoint(
    checkpoint_dir: str,
    raw_model,
    tokenizer,
    optimizer,
    scheduler,
    cfg: TrainConfig,
    global_step: int,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    raw_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    trainer_state = {
        "global_step": global_step,
        "seed": cfg.seed,
        "config": asdict(cfg),
    }
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)


def load_training_state(
    checkpoint_dir: str,
    optimizer,
    scheduler,
    device: torch.device,
) -> int:
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")

    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    if os.path.exists(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, encoding="utf-8") as f:
            trainer_state = json.load(f)
        return int(trainer_state.get("global_step", 0))
    return 0


@torch.no_grad()
def run_eval(
    cfg: TrainConfig,
    model,
    raw_model,
    llm: LLM,
    tokenizer,
    eval_dataset: Dataset | None,
    device: torch.device,
    ddp: bool,
    ddp_world_size: int,
    ddp_rank: int,
) -> dict[str, float] | None:
    if eval_dataset is None or len(eval_dataset) == 0:
        return None

    model.eval()
    sync_vllm_model_weights(llm, raw_model)

    prompts_per_rank = max(1, cfg.per_device_train_batch_size)
    sampler = (
        DistributedSampler(
            eval_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=False,
            drop_last=False,
        )
        if ddp
        else None
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=prompts_per_rank,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
    )

    reward_sum = torch.zeros(1, dtype=torch.float64, device=device)
    reward_sq_sum = torch.zeros(1, dtype=torch.float64, device=device)
    reward_count = torch.zeros(1, dtype=torch.float64, device=device)
    exact_match_sum = torch.zeros(1, dtype=torch.float64, device=device)

    for batch in loader:
        prompts = list(batch["prompt"])
        golden_patch = list(batch["golden_patch"])
        _, _, completions = generate_completions(
            llm,
            tokenizer,
            prompts,
            max_new_tokens=cfg.max_completion_length,
            temperature=cfg.temperature,
            num_generations=cfg.num_generations,
        )

        expanded_golden = []
        for gold in golden_patch:
            expanded_golden.extend([gold] * cfg.num_generations)

        rewards = reward_fn(completions=completions, golden_patch=expanded_golden)
        reward_tensor = torch.tensor(rewards, dtype=torch.float64, device=device)
        exact_tensor = torch.tensor(
            [float(c.strip() == g.strip()) for c, g in zip(completions, expanded_golden)],
            dtype=torch.float64,
            device=device,
        )

        reward_sum += reward_tensor.sum()
        reward_sq_sum += (reward_tensor * reward_tensor).sum()
        reward_count += reward_tensor.numel()
        exact_match_sum += exact_tensor.sum()

    if ddp:
        dist.all_reduce(reward_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(exact_match_sum, op=dist.ReduceOp.SUM)

    mean_reward = (reward_sum / reward_count.clamp(min=1)).item()
    variance = (reward_sq_sum / reward_count.clamp(min=1)).item() - mean_reward * mean_reward
    std_reward = max(variance, 0.0) ** 0.5

    return {
        "eval/exact_match_rate": (exact_match_sum / reward_count.clamp(min=1)).item(),
        "eval/mean_reward": mean_reward,
        "eval/reward_std": std_reward,
        "eval/num_completions": reward_count.item(),
    }


def cleanup_compute() -> None:
    gc.collect()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def dry_run(cfg: TrainConfig):
    if os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is set")
    else:
        print("WANDB_API_KEY is not set")
    print("=" * 60)
    print("SWE-RL NATIVE PYTORCH DRY RUN")
    print("=" * 60)

    print(f"\n[Config]")
    print(f"  model:            {cfg.model_name}")
    print(f"  dataset:          {cfg.dataset_name} ({cfg.dataset_config})")
    print(f"  max_prompt_len:   {cfg.max_prompt_length}")
    print(f"  num_generations:  {cfg.num_generations}")
    print(f"  batch_size:       {cfg.per_device_train_batch_size}")
    print(f"  grad_accum:       {cfg.gradient_accumulation_steps}")
    print(f"  max_completion:   {cfg.max_completion_length}")
    print(f"  lr:               {cfg.learning_rate}")
    print(f"  max_steps:        {cfg.max_steps}")
    print(f"  output_dir:       {cfg.output_dir}")

    print(f"\n[Dataset]")
    train_dataset, eval_dataset = load_swe_rl_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, model_source, cfg.eval_size
    )
    print(f"  train samples: {len(train_dataset)}")
    print(f"  eval samples:  {len(eval_dataset) if eval_dataset else 0}")

    from collections import Counter

    sources = Counter()
    for row in train_dataset:
        sources[len(row["golden_patch"]) < 500] += 1
    print(f"  short patches (<500 chars): {sources.get(True, 0)}")
    print(f"  long patches (>=500 chars): {sources.get(False, 0)}")

    first = train_dataset[0]
    print(f"\n  First prompt preview: {first['prompt'][:150]}...")
    print(f"  First golden_patch preview: {first['golden_patch'][:150]}...")

    print(f"\n[Reward function test]")
    test_completions = [
        "",
        "Some random text that is not a patch.",
        "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old line\n+new line\n context",
        first["golden_patch"],
    ]
    test_golden = [first["golden_patch"]] * len(test_completions)
    test_rewards = reward_fn(completions=test_completions, golden_patch=test_golden)
    labels = ["empty", "random text", "synthetic diff", "exact match"]
    for label, comp, rew in zip(labels, test_completions, test_rewards):
        print(f"  {rew:.2f} <- {label}: {repr(comp[:60])}")

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


def main():
    parser = argparse.ArgumentParser(description="SWE-RL training for coding agent (native PyTorch)")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset/tokenizer without training")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--run", nargs="?", const="", metavar="PROJECT", help="Enable W&B logging")
    parser.add_argument("--eval-steps", type=int, default=None, help="Run eval every N steps (default: 50)")
    parser.add_argument("--eval-size", type=int, default=None, help="Number of eval examples (default: 100)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    args = parser.parse_args()

    cfg = TrainConfig()

    overrides = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
    if args.run is not None:
        overrides["report_to"] = "wandb"
        if args.run:
            overrides["wandb_project"] = args.run
    if args.eval_steps is not None:
        overrides["eval_steps"] = args.eval_steps
    if args.eval_size is not None:
        overrides["eval_size"] = args.eval_size

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, type(getattr(cfg, k))(v))

    if args.dry_run:
        dry_run(cfg)
        return

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    if device_type != "cuda":
        raise ValueError("This native rewrite keeps vLLM rollouts and requires CUDA.")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    torch.manual_seed(cfg.seed + ddp_rank)

    prompts_per_rank = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    if prompts_per_rank < 1:
        raise ValueError("per_device_train_batch_size * gradient_accumulation_steps must be at least 1.")

    model_source = args.resume_from_checkpoint or cfg.model_name
    print0(f"Loading tokenizer from {model_source}...")
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print0(f"Loading training model: {model_source}...")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.to(device)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    print0(f"Loading vLLM engine: {model_source}...")
    llm = LLM(
        model=model_source,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )

    train_dataset, eval_dataset = load_swe_rl_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name, cfg.eval_size
    )
    print0(f"Train dataset: {len(train_dataset)} examples")
    if eval_dataset:
        print0(f"Eval dataset: {len(eval_dataset)} examples (every {cfg.eval_steps} steps)")

    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=True,
            drop_last=True,
        )
        if ddp
        else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=prompts_per_rank,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=(device.type == "cuda"),
    )

    def get_lr_lambda(step: int) -> float:
        if cfg.lr_scheduler_type != "linear":
            return 1.0
        if step < cfg.warmup_steps:
            return (step + 1) / max(cfg.warmup_steps, 1)
        remaining_steps = max(cfg.max_steps - cfg.warmup_steps, 1)
        decay_step = min(step - cfg.warmup_steps, remaining_steps)
        return max(0.0, 1.0 - decay_step / remaining_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = load_training_state(args.resume_from_checkpoint, optimizer, scheduler, device)
        print0(f"Resumed optimizer/scheduler state from {args.resume_from_checkpoint} at step {global_step}")

    model_short = cfg.model_name.split("/")[-1]
    run_name = (
        f"swe_rl_{model_short}"
        f"_g{cfg.num_generations}"
        f"_bs{cfg.per_device_train_batch_size}"
        f"_ga{cfg.gradient_accumulation_steps}"
        f"_lr{cfg.learning_rate}"
    )

    setup_wandb(cfg, run_name)
    wandb_run = DummyWandb()
    if _wandb_enabled(cfg.report_to) and master_process:
        import wandb

        wandb_run = wandb

    print0(f"Prompts per rank: {prompts_per_rank}, Generations per prompt: {cfg.num_generations}")
    print0(f"Max steps: {cfg.max_steps}, Warmup: {cfg.warmup_steps}")

    data_iter = iter(loader)

    try:
        while global_step < cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(global_step)
                data_iter = iter(loader)
                batch = next(data_iter)

            prompts = list(batch["prompt"])
            golden_patch = list(batch["golden_patch"])
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            sync_vllm_model_weights(llm, raw_model)

            all_ids, all_masks, completions_text = generate_completions(
                llm,
                tokenizer,
                prompts,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.temperature,
                num_generations=cfg.num_generations,
            )

            expanded_golden = []
            for gold in golden_patch:
                expanded_golden.extend([gold] * cfg.num_generations)

            rewards = reward_fn(completions=completions_text, golden_patch=expanded_golden)
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
            rewards_grouped = rewards_t.view(-1, cfg.num_generations)
            advantages = (rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)).view(-1)

            input_ids, attention_mask, completion_mask = pad_and_stack(all_ids, all_masks, pad_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            completion_mask = completion_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            all_stats = {
                "swe/mean_reward": rewards_t.mean().item(),
                "swe/max_reward": rewards_t.max().item(),
            }

            model.train()
            for micro_step in range(cfg.gradient_accumulation_steps):
                prompt_start = micro_step * cfg.per_device_train_batch_size
                prompt_end = prompt_start + cfg.per_device_train_batch_size
                seq_start = prompt_start * cfg.num_generations
                seq_end = prompt_end * cfg.num_generations

                loss, stats = compute_swe_loss(
                    model,
                    input_ids[seq_start:seq_end],
                    attention_mask[seq_start:seq_end],
                    completion_mask[seq_start:seq_end],
                    advantages[seq_start:seq_end],
                )
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()

                for k, v in stats.items():
                    all_stats[k] = all_stats.get(k, 0.0) + v / cfg.gradient_accumulation_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % cfg.logging_steps == 0:
                loss_tensor = torch.tensor(total_loss, device=device)
                reward_mean = torch.tensor(all_stats["swe/mean_reward"], device=device)
                if ddp:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(reward_mean, op=dist.ReduceOp.AVG)
                all_stats["swe/mean_reward"] = reward_mean.item()

                current_lr = scheduler.get_last_lr()[0]
                stats_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(all_stats.items()))
                print0(
                    f"step={global_step}/{cfg.max_steps} loss={loss_tensor.item():.4f} "
                    f"lr={current_lr:.2e} grad_norm={float(grad_norm):.4f} {stats_str}"
                )
                if master_process:
                    wandb_run.log({
                        "step": global_step,
                        "loss": loss_tensor.item(),
                        "lr": current_lr,
                        "grad_norm": float(grad_norm),
                        **all_stats,
                    })

            if eval_dataset and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                metrics = run_eval(
                    cfg,
                    model,
                    raw_model,
                    llm,
                    tokenizer,
                    eval_dataset,
                    device,
                    ddp,
                    ddp_world_size,
                    ddp_rank,
                )
                if metrics is not None and master_process:
                    print0(" ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))
                    append_eval_log(cfg.output_dir, global_step, metrics)
                    wandb_run.log({"step": global_step, **metrics})

            if master_process and global_step % cfg.save_steps == 0:
                ckpt_dir = os.path.join(cfg.output_dir, f"step_{global_step}")
                save_checkpoint(ckpt_dir, raw_model, tokenizer, optimizer, scheduler, cfg, global_step)
                print0(f"Saved checkpoint to {ckpt_dir}")

        if master_process:
            final_dir = os.path.join(cfg.output_dir, "final")
            save_checkpoint(final_dir, raw_model, tokenizer, optimizer, scheduler, cfg, global_step)
            print0(f"Saved final model to {final_dir}")
    finally:
        finish_wandb()
        cleanup_compute()
        compute_cleanup()


if __name__ == "__main__":
    main()
