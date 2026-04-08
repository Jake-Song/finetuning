# /// script
# requires-python = ">=3.10"
# dependencies = [
#      "torch==2.9.1",
#     "trl[vllm]",
#     "aiohttp",
#     "omegaconf",
#     "pyyaml",
#     "datasets",
#     "accelerate",
#     "huggingface-hub",
#     "wandb",
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
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, TrainerCallback

from trl import GRPOConfig, GRPOTrainer

from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "nvidia/Nemotron-Cascade-2-RL-data"
    dataset_config: str = "MOPD"
    max_prompt_length: int = 2048
    min_pass_rate: float = 0.0  # filter out samples below this pass_rate

    # GRPO
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_completion_length: int = 2048
    learning_rate: float = 3e-6
    epsilon: float = 0.2
    temperature: float = 0.7
    warmup_steps: int = 10
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0

    # vLLM (colocated mode)
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1

    max_steps: int = 500
    output_dir: str = "./ckpt_grpo_mopd"
    save_steps: int = 50
    logging_steps: int = 1
    eval_steps: int = 200
    eval_size: int = 100
    seed: int = 42

    # wandb
    report_to: str = "none"
    wandb_project: str = "grpo-mopd"
    wandb_entity: str = ""
    wandb_mode: str = "online"


# -----------------------------
# Reward
# -----------------------------
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _extract_with_regex(completion: str, output_regex: str) -> str | None:
    """Extract answer from completion using the dataset's output_regex."""
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
    # use output_regex from template_metadata if available
    extracted = _extract_with_regex(completion, output_regex)
    if extracted and extracted.strip().upper() == expected_norm:
        return 1.0
    # fallback: pattern matching
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
    # try regex extraction first
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

    score = 0.5  # valid JSON

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
    output_regex: list[str],
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
        regex = output_regex[i] or ""

        if cat == "tool_call":
            rewards.append(_check_tool_call(text, gt))
        elif cat == "structured":
            rewards.append(_check_structured_output(text, expected, regex))
        else:
            rewards.append(_check_mcqa(text, expected, regex))

    return rewards


# -----------------------------
# Dataset
# -----------------------------
def load_mopd_dataset(
    name: str, config: str, max_prompt_length: int, tokenizer_name: str,
    min_pass_rate: float = 0.0, eval_size: int = 100,
) -> tuple[Dataset, Dataset]:
    ds = load_dataset(name, config, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = []
    skipped_pass_rate = 0
    for example in ds:
        # filter by pass_rate quality signal
        pass_rate = example.get("pass_rate")
        if pass_rate is not None and pass_rate < min_pass_rate:
            skipped_pass_rate += 1
            continue

        prompt_text = example.get("prompt") or example.get("question") or ""
        if not prompt_text:
            continue

        # truncate long prompts
        tokens = tokenizer.encode(prompt_text, truncation=True, max_length=max_prompt_length)
        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        # extract output_regex from template_metadata
        tmeta = example.get("template_metadata")
        output_regex = ""
        if isinstance(tmeta, dict):
            output_regex = tmeta.get("output_regex", "") or ""
        elif isinstance(tmeta, list) and tmeta:
            output_regex = tmeta[0].get("output_regex", "") or ""

        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "expected_answer": example.get("expected_answer", "") or "",
            "category": example.get("category", "") or "",
            "ground_truth": example.get("ground_truth") or [],
            "output_regex": output_regex,
        })

    if skipped_pass_rate:
        print(f"  Filtered {skipped_pass_rate} samples below min_pass_rate={min_pass_rate}")

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


# -----------------------------
# wandb helpers
# -----------------------------
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


class EvalLogCallback(TrainerCallback):
    """Appends evaluation metrics to a markdown file after each eval run."""

    def __init__(self, output_dir: str):
        self.log_path = os.path.join(output_dir, "eval_results.md")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or int(os.environ.get("RANK", "0")) != 0:
            return
        step = state.global_step
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if write_header:
                f.write("# Evaluation Results\n\n")
                f.write("| Step | Timestamp | " + " | ".join(sorted(metrics.keys())) + " |\n")
                f.write("|------|-----------|" + "|".join("---" for _ in metrics) + "|\n")
            values = " | ".join(
                f"{metrics[k]:.4f}" if isinstance(metrics[k], float) else str(metrics[k])
                for k in sorted(metrics.keys())
            )
            f.write(f"| {step} | {timestamp} | {values} |\n")


def cleanup_compute() -> None:
    gc.collect()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# -----------------------------
# Dry run
# -----------------------------
def dry_run(cfg: TrainConfig):
    print("=" * 60)
    print("MOPD GRPO DRY RUN")
    print("=" * 60)

    print(f"\n[Config]")
    print(f"  model:            {cfg.model_name}")
    print(f"  dataset:          {cfg.dataset_name} ({cfg.dataset_config})")
    print(f"  max_prompt_len:   {cfg.max_prompt_length}")
    print(f"  min_pass_rate:    {cfg.min_pass_rate}")
    print(f"  num_generations:  {cfg.num_generations}")
    print(f"  batch_size:       {cfg.per_device_train_batch_size}")
    print(f"  grad_accum:       {cfg.gradient_accumulation_steps}")
    print(f"  max_completion:   {cfg.max_completion_length}")
    print(f"  lr:               {cfg.learning_rate}")
    print(f"  max_steps:        {cfg.max_steps}")
    print(f"  output_dir:       {cfg.output_dir}")

    print(f"\n[Dataset]")
    train_dataset, eval_dataset = load_mopd_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name,
        cfg.min_pass_rate, cfg.eval_size,
    )
    print(f"  train samples: {len(train_dataset)}")
    print(f"  eval samples:  {len(eval_dataset) if eval_dataset else 0}")

    # category distribution
    from collections import Counter
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
    print(f"\n  First prompt preview: {first['prompt'][0]['content'][:150]}...")
    print(f"  First expected_answer: {first['expected_answer'][:100]}")
    print(f"  First category: {first['category']}")
    print(f"  First output_regex: {first['output_regex'][:100] if first['output_regex'] else '(none)'}")

    print(f"\n[Reward function test]")
    test_completions = ["", "Answer: B", '{"name": "search", "arguments": {"query": "test"}}']
    test_expected = ["B", "B", ""]
    test_cats = ["mcqa", "mcqa", "tool_call"]
    test_gt: list = [[], [], [{"name": "search", "arguments": {"query": "test"}}]]
    test_regex = ["", "", ""]
    test_rewards = reward_fn(
        completions=test_completions,
        expected_answer=test_expected,
        category=test_cats,
        ground_truth=test_gt,
        output_regex=test_regex,
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MOPD (Multi-Domain On-Policy Distillation) GRPO training")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset/tokenizer without training")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--run", nargs="?", const="", metavar="PROJECT", help="Enable W&B logging")
    parser.add_argument("--eval-steps", type=int, default=None, help="Run eval every N steps")
    parser.add_argument("--eval-size", type=int, default=None, help="Number of eval examples")
    args = parser.parse_args()

    cfg = TrainConfig()

    overrides = {}
    if args.config:
        with open(args.config) as f:
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

    # load dataset
    train_dataset, eval_dataset = load_mopd_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name,
        cfg.min_pass_rate, cfg.eval_size,
    )
    print(f"Train dataset: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Eval dataset: {len(eval_dataset)} examples (every {cfg.eval_steps} steps)")

    # training config
    model_short = cfg.model_name.split("/")[-1]
    run_name = (
        f"mopd_{model_short}"
        f"_g{cfg.num_generations}"
        f"_bs{cfg.per_device_train_batch_size}"
        f"_ga{cfg.gradient_accumulation_steps}"
        f"_lr{cfg.learning_rate}"
    )

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        run_name=run_name,
        use_vllm=True,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_completion_length=cfg.max_completion_length,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        epsilon=cfg.epsilon,
        temperature=cfg.temperature,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        gradient_checkpointing=True,
        loss_type="grpo",
        mask_truncated_completions=True,
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg.eval_steps if eval_dataset else None,
        report_to=cfg.report_to,
        seed=cfg.seed,
        log_completions=True,
        model_init_kwargs={"torch_dtype": "auto"},
    )

    setup_wandb(cfg, run_name)

    trainer = GRPOTrainer(
        model=cfg.model_name,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[EvalLogCallback(cfg.output_dir)] if eval_dataset else None,
    )

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    finally:
        finish_wandb()
        cleanup_compute()


if __name__ == "__main__":
    main()
