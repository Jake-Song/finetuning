import argparse
import gc
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
import yaml
from datasets import Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = "data/instruct_train.jsonl"

    # GRPO
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_completion_length: int = 1024
    learning_rate: float = 1e-6
    epsilon: float = 0.2
    temperature: float = 0.7
    warmup_steps: int = 10
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0

    # vLLM (colocated mode)
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1

    max_steps: int = 200
    output_dir: str = "./ckpt_grpo_instruct"
    save_steps: int = 50
    logging_steps: int = 1
    seed: int = 42

    # wandb
    report_to: str = "none"
    wandb_project: str = "grpo-instruct"
    wandb_entity: str = ""
    wandb_mode: str = "online"


# -----------------------------
# Reward
# -----------------------------
def reward_fn(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for text in completions:
        text = text.strip()
        if not text:
            rewards.append(0.0)
            continue

        score = 0.2  # non-empty baseline

        # sufficient length
        if len(text) >= 50:
            score += 0.3

        # ends with sentence-ending punctuation (complete response)
        if re.search(r"[.!?]$", text):
            score += 0.2

        # structural elements (paragraphs, lists, code blocks)
        if "\n" in text or re.search(r"[-*]\s", text) or "```" in text:
            score += 0.3

        rewards.append(score)
    return rewards


# -----------------------------
# Dataset
# -----------------------------
def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # normalize to conversational prompt format
            if "prompt" in item:
                prompt = item["prompt"]
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
            elif "instruction" in item:
                prompt = [{"role": "user", "content": item["instruction"]}]
            else:
                raise ValueError(f"Expected 'prompt' or 'instruction' field, got keys: {list(item.keys())}")

            data.append({"prompt": prompt})

    return Dataset.from_list(data)


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
    print("GRPO DRY RUN (standalone)")
    print("=" * 60)

    print(f"\n[Config]")
    print(f"  model:            {cfg.model_name}")
    print(f"  dataset:          {cfg.dataset_path}")
    print(f"  num_generations:  {cfg.num_generations}")
    print(f"  batch_size:       {cfg.per_device_train_batch_size}")
    print(f"  grad_accum:       {cfg.gradient_accumulation_steps}")
    print(f"  max_completion:   {cfg.max_completion_length}")
    print(f"  lr:               {cfg.learning_rate}")
    print(f"  max_steps:        {cfg.max_steps}")
    print(f"  output_dir:       {cfg.output_dir}")
    print(f"  vllm_gpu_mem:     {cfg.vllm_gpu_memory_utilization}")
    print(f"  vllm_tp_size:     {cfg.vllm_tensor_parallel_size}")
    print(f"  report_to:        {cfg.report_to}")
    if _wandb_enabled(cfg.report_to):
        print(f"  wandb_project:    {cfg.wandb_project}")
        print(f"  wandb_entity:     {cfg.wandb_entity or '(default)'}")

    print(f"\n[Dataset]")
    if os.path.exists(cfg.dataset_path):
        dataset = load_dataset_from_jsonl(cfg.dataset_path)
        print(f"  samples: {len(dataset)}")
        first_prompt = dataset[0]["prompt"]
        if isinstance(first_prompt, list) and first_prompt:
            content = first_prompt[-1].get("content", "")
            print(f"  first prompt preview: {content[:120]}...")
    else:
        print(f"  {cfg.dataset_path} not found.")

    print(f"\n[Tokenizer]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        print(f"  vocab_size:  {tokenizer.vocab_size}")
        print(f"  pad_token:   {tokenizer.pad_token}")
        print(f"  eos_token:   {tokenizer.eos_token}")
    except Exception as e:
        print(f"  Failed to load tokenizer: {e}")

    print(f"\n[Reward function test]")
    test_completions = [
        "",
        "Yes.",
        "Here is a detailed explanation of the concept.\n\nFirst, let's consider the basics.",
    ]
    test_rewards = reward_fn(test_completions)
    for comp, rew in zip(test_completions, test_rewards):
        print(f"  {rew:.1f} <- {repr(comp[:60])}")

    print(f"\n{'=' * 60}")
    print("DRY RUN COMPLETE")
    print("=" * 60)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Standalone GRPO training for instruction following")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/dataset/tokenizer without training")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--run", nargs="?", const="", metavar="PROJECT", help="Enable W&B logging")
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

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, type(getattr(cfg, k))(v))

    if args.dry_run:
        dry_run(cfg)
        return

    # load dataset
    dataset = load_dataset_from_jsonl(cfg.dataset_path)
    print(f"Train dataset: {len(dataset)} examples")

    # training config
    model_short = cfg.model_name.split("/")[-1]
    run_name = (
        f"instruct_{model_short}"
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
        report_to=cfg.report_to,
        seed=cfg.seed,
        log_completions=True,
        model_init_kwargs={"torch_dtype": "auto"},
    )

    setup_wandb(cfg, run_name)

    trainer = GRPOTrainer(
        model=cfg.model_name,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=training_args,
    )

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    finally:
        finish_wandb()
        cleanup_compute()


if __name__ == "__main__":
    main()
