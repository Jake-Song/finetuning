import argparse
import json
import math
import time
from collections import Counter

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from mopd_train import TrainConfig, compute_rewards, load_mopd_dataset


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex/HumanEval paper."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_config(config_path: str | None) -> TrainConfig:
    cfg = TrainConfig()
    if not config_path:
        return cfg

    with open(config_path, encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, type(getattr(cfg, key))(value))
    return cfg


def load_tokenizer_with_safe_mistral_fix(path: str):
    try:
        return AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(path)


def load_eval_examples(
    dataset_name: str,
    dataset_config: str,
    max_prompt_length: int,
    tokenizer_name: str,
    min_pass_rate: float,
    eval_size: int,
) -> list[dict]:
    train_dataset, eval_dataset = load_mopd_dataset(
        dataset_name,
        dataset_config,
        max_prompt_length,
        tokenizer_name,
        min_pass_rate,
        eval_size,
    )
    ds = eval_dataset if eval_dataset is not None else train_dataset
    return list(ds)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: torch.device) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tokenizer_kwargs = {"return_tensors": "pt"}
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_positions, int) and max_positions > max_new_tokens:
        tokenizer_kwargs["truncation"] = True
        tokenizer_kwargs["max_length"] = max_positions - max_new_tokens

    inputs = tokenizer(text, **tokenizer_kwargs)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def main():
    default_cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Evaluate a trained model on MOPD tasks")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Model checkpoint path (default: base model {default_cfg.model_name})",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (defaults to checkpoint)")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=f"HF dataset id (default: {default_cfg.dataset_name})",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help=f"HF dataset config name (default: {default_cfg.dataset_config})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=f"Max generated tokens (default: {default_cfg.max_completion_length})",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples to evaluate")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples per example for pass@k")
    parser.add_argument("--pass-k", type=str, default="1", help="Comma-separated k values for pass@k (e.g. 1,5,10)")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Save per-example results to JSONL")
    parser.add_argument("--min-pass-rate", type=float, default=None, help="Minimum dataset pass_rate filter")
    parser.add_argument("--eval-size", type=int, default=None, help="Eval split size (default from TrainConfig)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    args = parser.parse_args()

    cfg = build_config(args.config)
    model_path = args.checkpoint or cfg.model_name
    tokenizer_path = args.tokenizer or model_path
    ds_name = args.dataset_name or cfg.dataset_name
    ds_config = args.dataset_config or cfg.dataset_config
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.max_completion_length
    num_samples = args.num_samples if args.num_samples is not None else cfg.num_generations
    temperature = args.temperature if args.temperature is not None else cfg.temperature
    min_pass_rate = args.min_pass_rate if args.min_pass_rate is not None else cfg.min_pass_rate
    eval_size = args.eval_size if args.eval_size is not None else cfg.eval_size

    k_values = [int(k) for k in args.pass_k.split(",")]

    if num_samples > 1 and temperature == 0.0:
        temperature = 0.7
        print(f"Note: temperature set to {temperature} for multi-sample generation")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading model from {model_path} (device={device}, dtype={dtype})")
    tokenizer = load_tokenizer_with_safe_mistral_fix(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    print(f"Loading eval data: {ds_name} ({ds_config})")
    examples = load_eval_examples(
        ds_name,
        ds_config,
        cfg.max_prompt_length,
        tokenizer_path,
        min_pass_rate,
        eval_size,
    )
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Evaluating on {len(examples)} examples, {num_samples} sample(s) each")

    reward_sum = 0.0
    reward_passes = 0
    total_examples = 0
    category_reward_sums = Counter()
    category_pass_sums = Counter()
    category_totals = Counter()
    pass_k_sums = {k: 0.0 for k in k_values}
    results = []
    started = time.time()

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        category = ex.get("category", "") or "unknown"

        completions = []
        rewards = []
        n_correct = 0
        for _ in range(num_samples):
            completion = generate(model, tokenizer, prompt, max_new_tokens, temperature, device)
            reward = compute_rewards([completion], [ex])[0]
            completions.append(completion)
            rewards.append(reward)
            if reward >= 1.0:
                n_correct += 1

        first_reward = rewards[0]
        best_reward = max(rewards) if rewards else 0.0
        total_examples += 1
        reward_sum += first_reward
        reward_passes += int(first_reward >= 1.0)
        category_reward_sums[category] += first_reward
        category_pass_sums[category] += int(first_reward >= 1.0)
        category_totals[category] += 1

        for k in k_values:
            pass_k_sums[k] += pass_at_k(num_samples, n_correct, k)

        result = {
            "prompt": prompt,
            "category": category,
            "expected_answer": ex.get("expected_answer", ""),
            "ground_truth": ex.get("ground_truth", []),
            "output_regex": ex.get("output_regex", ""),
            "num_samples": num_samples,
            "best_reward": best_reward,
            "n_correct": n_correct,
        }
        if num_samples == 1:
            result["completion"] = completions[0]
            result["reward"] = first_reward
        else:
            result["samples"] = [
                {"completion": completion, "reward": reward}
                for completion, reward in zip(completions, rewards)
            ]
        results.append(result)

        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            elapsed = time.time() - started
            avg_time = elapsed / (i + 1)
            avg_reward = reward_sum / max(total_examples, 1)
            pass_k_str = "  ".join(f"pass@{k}={100*pass_k_sums[k]/total_examples:.1f}%" for k in k_values)
            print(
                f"  [{i+1}/{len(examples)}] avg_reward={avg_reward:.3f}  "
                f"pass={reward_passes}/{total_examples} ({100*reward_passes/max(total_examples,1):.1f}%)  "
                f"{pass_k_str}  "
                f"{avg_time:.1f}s/example"
            )

    elapsed = time.time() - started
    print(f"\n{'='*60}")
    print(f"MOPD Results ({ds_name}/{ds_config}): {model_path}")
    print(f"{'='*60}")
    print(f"  Examples:        {total_examples}")
    print(f"  Samples/example: {num_samples}")
    print(f"  Mean reward:     {reward_sum/max(total_examples,1):.3f} (first sample)")
    print(
        f"  Strict pass:     {reward_passes}/{total_examples} "
        f"({100*reward_passes/max(total_examples,1):.1f}%) (reward=1.0 on first sample)"
    )

    print(f"\n  pass@k:")
    for k in k_values:
        avg = pass_k_sums[k] / max(total_examples, 1)
        print(f"    pass@{k}: {100*avg:.1f}%")

    print(f"\n  Time:            {elapsed:.1f}s ({elapsed/max(total_examples,1):.1f}s/example)")

    print(f"\n  Per-category breakdown (first sample):")
    for category in sorted(category_totals, key=lambda item: category_totals[item], reverse=True):
        total = category_totals[category]
        mean_reward = category_reward_sums[category] / total
        passed = category_pass_sums[category]
        print(
            f"    {category}: mean_reward={mean_reward:.3f}  "
            f"pass={passed}/{total} ({100*passed/total:.0f}%)"
        )

    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\n  Results saved to {args.output_jsonl}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
