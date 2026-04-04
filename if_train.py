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
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # GRPO
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_completion_length: int = 2048
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
    output_dir: str = "./ckpt_grpo_ifeval"
    save_steps: int = 50
    logging_steps: int = 1
    seed: int = 42

    # wandb
    report_to: str = "none"
    wandb_project: str = "grpo-ifeval"
    wandb_entity: str = ""
    wandb_mode: str = "online"


# -----------------------------
# Constraint checkers
# -----------------------------
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


# -----------------------------
# Reward
# -----------------------------
def reward_fn(*, completions, instruction_id_list, kwargs, **_ignored) -> list[float]:
    rewards = []
    for i, completion in enumerate(completions):
        # extract text from conversational format
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        text = text.strip()

        if not text:
            rewards.append(0.0)
            continue

        ids = instruction_id_list[i]
        kws = kwargs[i]
        if not ids:
            rewards.append(0.5)
            continue

        passed = 0
        for constraint_id, kw in zip(ids, kws):
            checker = CHECKERS.get(constraint_id)
            if checker is None:
                passed += 1  # skip unknown constraints
                continue
            if checker(text, kw):
                passed += 1

        rewards.append(passed / len(ids))
    return rewards


# -----------------------------
# Dataset
# -----------------------------
def load_ifeval_dataset() -> Dataset:
    ds = load_dataset("google/IFEval", split="train")

    rows = []
    for example in ds:
        rows.append({
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "instruction_id_list": example["instruction_id_list"],
            "kwargs": example["kwargs"],
        })

    return Dataset.from_list(rows)


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
    print("IFEval GRPO DRY RUN")
    print("=" * 60)

    print(f"\n[Config]")
    print(f"  model:            {cfg.model_name}")
    print(f"  num_generations:  {cfg.num_generations}")
    print(f"  batch_size:       {cfg.per_device_train_batch_size}")
    print(f"  grad_accum:       {cfg.gradient_accumulation_steps}")
    print(f"  max_completion:   {cfg.max_completion_length}")
    print(f"  lr:               {cfg.learning_rate}")
    print(f"  max_steps:        {cfg.max_steps}")
    print(f"  output_dir:       {cfg.output_dir}")

    print(f"\n[Dataset]")
    dataset = load_ifeval_dataset()
    print(f"  samples: {len(dataset)}")

    # constraint coverage stats
    from collections import Counter
    all_ids = []
    for ids in dataset["instruction_id_list"]:
        all_ids.extend(ids)
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
    # test with a synthetic completion against the first example
    first = dataset[0]
    print(f"  Prompt: {first['prompt'][0]['content'][:100]}...")
    print(f"  Constraints: {first['instruction_id_list']}")
    test_completions = [[{"role": "assistant", "content": "This is a test response."}]]
    test_rewards = reward_fn(
        completions=test_completions,
        instruction_id_list=[first["instruction_id_list"]],
        kwargs=[first["kwargs"]],
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="IFEval GRPO training for instruction following")
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
    dataset = load_ifeval_dataset()
    print(f"Train dataset: {len(dataset)} examples")

    # training config
    model_short = cfg.model_name.split("/")[-1]
    run_name = (
        f"ifeval_{model_short}"
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
