import argparse
import json
import math
import time
from collections import Counter

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from if_train import CHECKERS, TrainConfig


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex/HumanEval paper."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_eval_examples(dataset_name: str | None = None, dataset_config: str | None = None) -> list[dict]:
    cfg = TrainConfig()
    name = dataset_name or cfg.dataset_name
    config = dataset_config or cfg.dataset_config
    ds = load_dataset(name, config, split="train")
    return list(ds)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: torch.device) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

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


def evaluate_constraints(text: str, instruction_ids: list[str], kwargs_list: list[dict]) -> dict:
    results = {}
    for constraint_id, kw in zip(instruction_ids, kwargs_list):
        checker = CHECKERS.get(constraint_id)
        if checker is None:
            results[constraint_id] = None  # unknown
        else:
            results[constraint_id] = checker(text, kw)
    return results


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


def main():
    default_cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on instruction-following constraints (Nemotron IF-RL by default)"
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
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
        "--checkpoint",
        type=str,
        default=None,
        help=f"Model checkpoint path (default: base model {default_cfg.model_name})",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (defaults to checkpoint)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=f"Max generated tokens (default: {default_cfg.max_completion_length})",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples to evaluate")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per example for pass@k")
    parser.add_argument("--pass-k", type=str, default="1", help="Comma-separated k values for pass@k (e.g. 1,5,10)")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Save per-example results to JSONL")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    args = parser.parse_args()
    cfg = build_config(args.config)

    k_values = [int(k) for k in args.pass_k.split(",")]
    num_samples = args.num_samples
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.max_completion_length

    # force sampling when generating multiple samples
    if num_samples > 1 and temperature == 0.0:
        temperature = 0.7
        print(f"Note: temperature set to {temperature} for multi-sample generation")

    # resolve device/dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # load model
    model_path = args.checkpoint or cfg.model_name
    print(f"Loading model from {model_path} (device={device}, dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # load dataset
    ds_name = args.dataset_name or cfg.dataset_name
    ds_config = args.dataset_config or cfg.dataset_config
    print(f"Loading eval data: {ds_name} ({ds_config})")
    examples = load_eval_examples(ds_name, ds_config)
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Evaluating on {len(examples)} examples, {num_samples} sample(s) each")

    # run eval
    total_constraints = 0
    passed_constraints = 0
    total_examples = 0
    passed_examples = 0  # examples where ALL constraints pass (best of n)
    constraint_stats = Counter()
    constraint_totals = Counter()
    pass_k_sums = {k: 0.0 for k in k_values}
    results = []
    started = time.time()

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        instruction_ids = ex["instruction_id_list"]
        kwargs_list = ex["kwargs"]

        samples = []
        n_correct = 0
        for s in range(num_samples):
            completion = generate(model, tokenizer, prompt, max_new_tokens, temperature, device)
            constraint_results = evaluate_constraints(completion, instruction_ids, kwargs_list)
            known_results = {k: v for k, v in constraint_results.items() if v is not None}
            n_passed = sum(known_results.values())
            n_total = len(known_results)
            all_passed = n_total > 0 and n_passed == n_total

            if all_passed:
                n_correct += 1

            # aggregate constraint stats from first sample (for backward compat)
            if s == 0:
                total_constraints += n_total
                passed_constraints += n_passed
                for cid, passed in known_results.items():
                    constraint_totals[cid] += 1
                    if passed:
                        constraint_stats[cid] += 1

            samples.append({
                "completion": completion,
                "constraint_results": constraint_results,
                "score": n_passed / n_total if n_total > 0 else 0.0,
                "all_passed": all_passed,
            })

        total_examples += 1
        if n_correct > 0:
            passed_examples += 1

        for k in k_values:
            pass_k_sums[k] += pass_at_k(num_samples, n_correct, k)

        result = {
            "prompt": prompt,
            "instruction_id_list": instruction_ids,
            "n_correct": n_correct,
            "num_samples": num_samples,
        }
        if num_samples == 1:
            result["completion"] = samples[0]["completion"]
            result["constraint_results"] = samples[0]["constraint_results"]
            result["score"] = samples[0]["score"]
        else:
            result["samples"] = samples
        results.append(result)

        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            elapsed = time.time() - started
            avg = elapsed / (i + 1)
            pass_k_str = "  ".join(f"pass@{k}={100*pass_k_sums[k]/total_examples:.1f}%" for k in k_values)
            print(f"  [{i+1}/{len(examples)}] constraint_acc={passed_constraints}/{total_constraints} "
                  f"({100*passed_constraints/max(total_constraints,1):.1f}%)  "
                  f"{pass_k_str}  "
                  f"{avg:.1f}s/example")

    # final report
    elapsed = time.time() - started
    print(f"\n{'='*60}")
    print(f"Results ({ds_name}/{ds_config}): {model_path}")
    print(f"{'='*60}")
    print(f"  Examples:           {total_examples}")
    print(f"  Samples/example:    {num_samples}")
    print(f"  Constraint accuracy: {passed_constraints}/{total_constraints} "
          f"({100*passed_constraints/max(total_constraints,1):.1f}%) (first sample)")
    print(f"  Strict accuracy:    {passed_examples}/{total_examples} "
          f"({100*passed_examples/max(total_examples,1):.1f}%) (any sample passes all)")

    print(f"\n  pass@k:")
    for k in k_values:
        avg = pass_k_sums[k] / max(total_examples, 1)
        print(f"    pass@{k}: {100*avg:.1f}%")

    print(f"\n  Time:               {elapsed:.1f}s ({elapsed/max(total_examples,1):.1f}s/example)")

    print(f"\n  Per-constraint breakdown (first sample):")
    for cid in sorted(constraint_totals, key=lambda c: constraint_totals[c], reverse=True):
        total = constraint_totals[cid]
        passed = constraint_stats[cid]
        print(f"    {cid}: {passed}/{total} ({100*passed/total:.0f}%)")

    # save results
    if args.output_jsonl:
        with open(args.output_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  Results saved to {args.output_jsonl}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
