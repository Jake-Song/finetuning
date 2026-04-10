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
#     "python-dotenv>=1.2.2",
# ]
# ///

"""
Multi-Domain RL training in native PyTorch.

Usage:
  Single GPU:   uv run scripts/multi_domain_rl_train.py
  Multi-GPU:    uv run torchrun --nproc_per_node=N scripts/multi_domain_rl_train.py
  Dry run:      uv run scripts/multi_domain_rl_train.py --dry-run
  Paper preset: uv run scripts/multi_domain_rl_train.py --config configs/multi_domain_rl/nemotron_cascade2_paper.yaml
"""

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
from utils.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    print0,
)
from utils.openai_server import OpenAICompatibleRolloutClient, sync_server_model_weights

load_dotenv()


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "nvidia/Nemotron-Cascade-2-RL-data"
    dataset_config: str = "multi-domain-RL"
    max_prompt_length: int = 256

    # GRPO
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_completion_length: int = 256
    learning_rate: float = 3e-6
    epsilon: float = 0.2
    temperature: float = 0.7
    warmup_steps: int = 10
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0

    # vLLM server
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_model_name_for_requests: str = ""
    vllm_api_key: str = "EMPTY"
    vllm_request_timeout: float = 300.0
    vllm_sync_timeout: float = 300.0
    vllm_weight_sync_backend: str = "nccl"
    vllm_max_parallel_requests: int = 8

    max_steps: int = 500
    output_dir: str = "./ckpt_grpo_multi_domain"
    save_steps: int = 50
    logging_steps: int = 1
    eval_steps: int = 200
    eval_size: int = 100
    seed: int = 42

    # wandb
    report_to: str = "none"
    wandb_project: str = "grpo-multi-domain"
    wandb_entity: str = ""
    wandb_mode: str = "online"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _check_mcqa(completion: str, expected: str) -> float:
    expected_norm = expected.strip().upper()
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


def _check_structured_output(completion: str, expected: str) -> float:
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


def reward_fn(
    completions: list[str],
    expected_answer: list[str],
    category: list[str],
    ground_truth: list,
    **kwargs,
) -> list[float]:
    rewards = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        text = text.strip()

        if not text:
            rewards.append(0.0)
            continue

        cat = _detect_category(category[i])
        expected = expected_answer[i] or ""
        gt = ground_truth[i] or []

        if cat == "tool_call":
            rewards.append(_check_tool_call(text, gt))
        elif cat == "structured":
            rewards.append(_check_structured_output(text, expected))
        else:
            rewards.append(_check_mcqa(text, expected))

    return rewards


def load_multi_domain_dataset(
    name: str, config: str, max_prompt_length: int, tokenizer_name: str, eval_size: int = 100
) -> tuple[Dataset, Dataset | None]:
    ds = load_dataset(name, config, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = []
    for example in ds:
        prompt_text = example.get("prompt") or example.get("question") or ""
        if not prompt_text:
            continue

        tokens = tokenizer.encode(prompt_text, truncation=True, max_length=max_prompt_length)
        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        rows.append({
            "prompt": prompt_text,
            "expected_answer": example.get("expected_answer", "") or "",
            "category": example.get("category", "") or "",
            "ground_truth": json.dumps(example.get("ground_truth") or []),
        })

    if not rows:
        raise ValueError(
            f"No usable rows found in dataset {name}/{config}. "
            "Expected at least a prompt or question field."
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
    rollout_client: OpenAICompatibleRolloutClient,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    num_generations: int,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    return rollout_client.generate_completions(
        tokenizer,
        list(prompts),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        num_generations=num_generations,
    )


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


def compute_grpo_loss(
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
    probs = log_probs.exp()
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    token_entropy = -(probs * log_probs).sum(dim=-1)

    per_token_loss = advantages.unsqueeze(-1) * token_log_probs * token_mask
    num_valid = token_mask.sum().clamp(min=1)
    loss = -per_token_loss.sum() / num_valid
    mean_entropy = (token_entropy * token_mask).sum() / num_valid

    stats = {
        "grpo/mean_advantage": advantages.mean().item(),
        "grpo/entropy": mean_entropy.item(),
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
    rollout_client: OpenAICompatibleRolloutClient,
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
    sync_server_model_weights(
        host=cfg.vllm_server_host,
        port=cfg.vllm_server_port,
        model=raw_model,
        backend=cfg.vllm_weight_sync_backend,
        timeout=cfg.vllm_sync_timeout,
        is_sync_leader=_is_main_process(),
        trainer_rank=ddp_rank,
        trainer_world_size=ddp_world_size,
    )
    if ddp:
        dist.barrier()

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

    for batch in loader:
        prompts = list(batch["prompt"])
        expected = list(batch["expected_answer"])
        categories = list(batch["category"])
        ground_truth = [json.loads(item) for item in batch["ground_truth"]]

        _, _, completions = generate_completions(
            rollout_client,
            tokenizer,
            prompts,
            max_new_tokens=cfg.max_completion_length,
            temperature=cfg.temperature,
            num_generations=cfg.num_generations,
        )

        expanded_expected = []
        expanded_categories = []
        expanded_ground_truth = []
        for i in range(len(prompts)):
            expanded_expected.extend([expected[i]] * cfg.num_generations)
            expanded_categories.extend([categories[i]] * cfg.num_generations)
            expanded_ground_truth.extend([ground_truth[i]] * cfg.num_generations)

        rewards = reward_fn(
            completions=completions,
            expected_answer=expanded_expected,
            category=expanded_categories,
            ground_truth=expanded_ground_truth,
        )
        reward_tensor = torch.tensor(rewards, dtype=torch.float64, device=device)
        reward_sum += reward_tensor.sum()
        reward_sq_sum += (reward_tensor * reward_tensor).sum()
        reward_count += reward_tensor.numel()

    if ddp:
        dist.all_reduce(reward_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reward_count, op=dist.ReduceOp.SUM)

    mean_reward = (reward_sum / reward_count.clamp(min=1)).item()
    variance = (reward_sq_sum / reward_count.clamp(min=1)).item() - mean_reward * mean_reward
    std_reward = max(variance, 0.0) ** 0.5

    return {
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
    print("=" * 60)
    print("Multi-Domain RL NATIVE PYTORCH DRY RUN")
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
    print(f"  vllm_server:      {cfg.vllm_server_host}:{cfg.vllm_server_port}")
    print(f"  vllm_sync:        {cfg.vllm_weight_sync_backend}")

    print(f"\n[Dataset]")
    train_dataset, eval_dataset = load_multi_domain_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name, cfg.eval_size
    )
    print(f"  train samples: {len(train_dataset)}")
    print(f"  eval samples:  {len(eval_dataset) if eval_dataset else 0}")

    from collections import Counter

    cats = Counter()
    for row in train_dataset:
        cats[_detect_category(row["category"])] += 1
    print(f"\n  Category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count}")

    first = train_dataset[0]
    print(f"\n  First prompt preview: {first['prompt'][:150]}...")
    print(f"  First expected_answer: {first['expected_answer'][:100]}")
    print(f"  First category: {first['category']}")

    print(f"\n[Reward function test]")
    test_completions = ["", "Answer: B", '{"name": "search", "arguments": {"query": "test"}}']
    test_expected = ["B", "B", ""]
    test_cats = ["mcqa", "mcqa", "tool_call"]
    test_gt: list = [[], [], [{"name": "search", "arguments": {"query": "test"}}]]
    test_rewards = reward_fn(
        completions=test_completions,
        expected_answer=test_expected,
        category=test_cats,
        ground_truth=test_gt,
    )
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


def main():
    parser = argparse.ArgumentParser(description="Multi-Domain RL training (native PyTorch)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config override, e.g. configs/multi_domain_rl/nemotron_cascade2_paper.yaml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset/tokenizer without training")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--run", nargs="?", const="", metavar="PROJECT", help="Enable W&B logging")
    parser.add_argument("--eval-steps", type=int, default=None, help="Run eval every N steps")
    parser.add_argument("--eval-size", type=int, default=None, help="Number of eval examples")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--vllm-server-host", type=str, default=None)
    parser.add_argument("--vllm-server-port", type=int, default=None)
    parser.add_argument("--vllm-model-name", type=str, default=None)
    parser.add_argument("--vllm-api-key", type=str, default=None)
    parser.add_argument("--vllm-request-timeout", type=float, default=None)
    parser.add_argument("--vllm-sync-timeout", type=float, default=None)
    parser.add_argument("--vllm-weight-sync-backend", type=str, default=None)
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
    if args.vllm_server_host is not None:
        overrides["vllm_server_host"] = args.vllm_server_host
    if args.vllm_server_port is not None:
        overrides["vllm_server_port"] = args.vllm_server_port
    if args.vllm_model_name is not None:
        overrides["vllm_model_name_for_requests"] = args.vllm_model_name
    if args.vllm_api_key is not None:
        overrides["vllm_api_key"] = args.vllm_api_key
    if args.vllm_request_timeout is not None:
        overrides["vllm_request_timeout"] = args.vllm_request_timeout
    if args.vllm_sync_timeout is not None:
        overrides["vllm_sync_timeout"] = args.vllm_sync_timeout
    if args.vllm_weight_sync_backend is not None:
        overrides["vllm_weight_sync_backend"] = args.vllm_weight_sync_backend

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

    request_model_name = cfg.vllm_model_name_for_requests or model_source
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

    train_dataset, eval_dataset = load_multi_domain_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, model_source, cfg.eval_size
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
        f"multi_domain_{model_short}"
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
            expected_answer = list(batch["expected_answer"])
            category = list(batch["category"])
            ground_truth = [json.loads(item) for item in batch["ground_truth"]]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

            sync_server_model_weights(
                host=cfg.vllm_server_host,
                port=cfg.vllm_server_port,
                model=raw_model,
                backend=cfg.vllm_weight_sync_backend,
                timeout=cfg.vllm_sync_timeout,
                is_sync_leader=master_process,
                trainer_rank=ddp_rank,
                trainer_world_size=ddp_world_size,
            )
            if ddp:
                dist.barrier()

            all_ids, all_masks, completions_text = generate_completions(
                rollout_client,
                tokenizer,
                prompts,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.temperature,
                num_generations=cfg.num_generations,
            )

            expanded_expected = []
            expanded_category = []
            expanded_ground_truth = []
            for i in range(len(prompts)):
                expanded_expected.extend([expected_answer[i]] * cfg.num_generations)
                expanded_category.extend([category[i]] * cfg.num_generations)
                expanded_ground_truth.extend([ground_truth[i]] * cfg.num_generations)

            rewards = reward_fn(
                completions=completions_text,
                expected_answer=expanded_expected,
                category=expanded_category,
                ground_truth=expanded_ground_truth,
            )
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
            rewards_grouped = rewards_t.view(-1, cfg.num_generations)
            mu = rewards_grouped.mean(dim=1, keepdim=True)
            std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            advantages = ((rewards_grouped - mu) / std).view(-1)

            input_ids, attention_mask, completion_mask = pad_and_stack(all_ids, all_masks, pad_id)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            completion_mask = completion_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            all_stats = {
                "grpo/mean_reward": rewards_t.mean().item(),
                "grpo/group_reward_std": rewards_grouped.std(dim=1).mean().item(),
                "grpo/max_reward": rewards_t.max().item(),
            }

            model.train()
            for micro_step in range(cfg.gradient_accumulation_steps):
                prompt_start = micro_step * cfg.per_device_train_batch_size
                prompt_end = prompt_start + cfg.per_device_train_batch_size
                seq_start = prompt_start * cfg.num_generations
                seq_end = prompt_end * cfg.num_generations

                loss, stats = compute_grpo_loss(
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
                reward_mean = torch.tensor(all_stats["grpo/mean_reward"], device=device)
                reward_group_std = torch.tensor(all_stats["grpo/group_reward_std"], device=device)
                entropy = torch.tensor(all_stats["grpo/entropy"], device=device)
                if ddp:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(reward_mean, op=dist.ReduceOp.AVG)
                    dist.all_reduce(reward_group_std, op=dist.ReduceOp.AVG)
                    dist.all_reduce(entropy, op=dist.ReduceOp.AVG)
                all_stats["grpo/mean_reward"] = reward_mean.item()
                all_stats["grpo/group_reward_std"] = reward_group_std.item()
                all_stats["grpo/entropy"] = entropy.item()

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
                    rollout_client,
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
