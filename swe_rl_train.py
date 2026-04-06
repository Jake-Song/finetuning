import argparse
import gc
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


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "nvidia/Nemotron-Cascade-2-RL-data"
    dataset_config: str = "SWE-RL"
    max_prompt_length: int = 8192

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


# -----------------------------
# Reward
# -----------------------------
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

        # format reward (0.2): looks like a unified diff
        has_diff_headers = bool(re.search(r"^---\s", text, re.MULTILINE)) and bool(
            re.search(r"^\+\+\+\s", text, re.MULTILINE)
        )
        has_hunks = bool(re.search(r"^@@\s", text, re.MULTILINE))
        if has_diff_headers and has_hunks:
            score += 0.2
        elif has_hunks:
            score += 0.1

        # structure reward (0.1): contains file paths and hunk markers
        has_file_path = bool(re.search(r"[a-zA-Z_/]+\.\w+", text))
        if has_file_path:
            score += 0.1

        # length reward (0.1): non-trivial but not excessively long
        if 10 < len(text) < len(gold) * 5:
            score += 0.1

        # overlap reward (0.4): Jaccard similarity of changed lines
        gen_lines = _extract_changed_lines(text)
        gold_lines = _extract_changed_lines(gold)
        if gen_lines and gold_lines:
            intersection = len(gen_lines & gold_lines)
            union = len(gen_lines | gold_lines)
            jaccard = intersection / union if union > 0 else 0.0
            score += 0.4 * jaccard

        # exact match bonus (0.2)
        if text.strip() == gold.strip():
            score += 0.2

        rewards.append(score)
    return rewards


# -----------------------------
# Dataset
# -----------------------------
def load_swe_rl_dataset(
    name: str, config: str, max_prompt_length: int, tokenizer_name: str, eval_size: int = 100
) -> tuple[Dataset, Dataset]:
    ds = load_dataset(name, config, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = []
    for example in ds:
        prompt_text = example["prompt"]

        # truncate long prompts to max_prompt_length tokens
        tokens = tokenizer.encode(prompt_text, truncation=True, max_length=max_prompt_length)
        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "golden_patch": example["golden_patch"],
        })

    full = Dataset.from_list(rows)
    if eval_size > 0 and eval_size < len(full):
        splits = full.train_test_split(test_size=eval_size, seed=42)
        return splits["train"], splits["test"]
    return full, None


# -----------------------------
# wandb helpers
# -----------------------------
ENV_YAML_PATH = os.path.join(os.path.dirname(__file__), "env.yaml")


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _wandb_enabled(report_to: str) -> bool:
    targets = {target.strip().lower() for target in report_to.split(",")}
    return "wandb" in targets


def _load_env_yaml() -> dict:
    if not os.path.exists(ENV_YAML_PATH):
        return {}
    with open(ENV_YAML_PATH) as f:
        return yaml.safe_load(f) or {}


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
    os.environ["WANDB_MODE"] = cfg.wandb_mode

    env = _load_env_yaml()
    api_key = env.get("wandb_api_key", "") or os.environ.get("WANDB_API_KEY", "")
    login_kwargs: dict[str, Any] = {"relogin": True}
    if api_key:
        login_kwargs["key"] = api_key
    wandb.login(**login_kwargs)

    if wandb.run is None:
        init_kwargs: dict[str, Any] = {
            "project": cfg.wandb_project,
            "name": run_name,
            "mode": cfg.wandb_mode,
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

        # write header on first eval
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
    print("SWE-RL GRPO DRY RUN")
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
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name, cfg.eval_size
    )
    print(f"  train samples: {len(train_dataset)}")
    print(f"  eval samples:  {len(eval_dataset) if eval_dataset else 0}")
    dataset = train_dataset

    # source distribution
    from collections import Counter
    sources = Counter()
    for row in dataset:
        # source info is lost after processing, show golden_patch stats instead
        patch = row["golden_patch"]
        sources[len(patch) < 500] += 1
    print(f"  short patches (<500 chars): {sources.get(True, 0)}")
    print(f"  long patches (>=500 chars): {sources.get(False, 0)}")

    first = dataset[0]
    prompt_preview = first["prompt"][0]["content"][:150]
    print(f"\n  First prompt preview: {prompt_preview}...")
    print(f"  First golden_patch preview: {first['golden_patch'][:150]}...")

    print(f"\n[Reward function test]")
    # test with a synthetic diff-like completion
    test_completions = [
        "",
        "Some random text that is not a patch.",
        "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old line\n+new line\n context",
        first["golden_patch"],  # exact match test
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SWE-RL GRPO training for coding agent")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset/tokenizer without training")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--run", nargs="?", const="", metavar="PROJECT", help="Enable W&B logging")
    parser.add_argument("--eval-steps", type=int, default=None, help="Run eval every N steps (default: 50)")
    parser.add_argument("--eval-size", type=int, default=None, help="Number of eval examples (default: 100)")
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
    train_dataset, eval_dataset = load_swe_rl_dataset(
        cfg.dataset_name, cfg.dataset_config, cfg.max_prompt_length, cfg.model_name, cfg.eval_size
    )
    print(f"Train dataset: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Eval dataset: {len(eval_dataset)} examples (every {cfg.eval_steps} steps)")

    # training config
    model_short = cfg.model_name.split("/")[-1]
    run_name = (
        f"swe_rl_{model_short}"
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
