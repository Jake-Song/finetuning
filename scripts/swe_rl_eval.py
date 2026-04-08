import argparse
import json
import math
import time
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from swe_rl_train import _extract_changed_lines


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex/HumanEval paper."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_tokenizer_with_safe_mistral_fix(path: str):
    """Load tokenizer and enable Mistral regex fix when supported."""
    try:
        return AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)
    except TypeError:
        # Older transformers versions may not support this kwarg.
        return AutoTokenizer.from_pretrained(path)


def load_swe_rl_examples(name: str, config: str) -> list[dict]:
    ds = load_dataset(name, config, split="train")
    rows = []
    for ex in ds:
        prompt = ex.get("prompt") or ex.get("problem_statement")
        golden_patch = ex.get("golden_patch") or ex.get("patch")
        if not prompt or not golden_patch:
            continue

        rows.append({
            **ex,
            "prompt": prompt,
            "golden_patch": golden_patch,
        })

    if not rows:
        raise ValueError(
            f"No usable rows found in dataset {name}/{config}. "
            "Expected prompt+patch fields like (prompt, golden_patch) or (problem_statement, patch)."
        )
    return rows


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: torch.device) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - max_new_tokens)
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


def score_patch(generated: str, golden: str) -> dict:
    """Score a generated patch against the golden patch. Returns individual metric scores."""
    gen = generated.strip()
    gold = golden.strip()

    if not gen:
        return {"format": 0.0, "structure": 0.0, "length": 0.0, "overlap": 0.0, "exact_match": 0.0, "total": 0.0}

    import re

    # format (0-1): valid unified diff structure
    has_diff_headers = bool(re.search(r"^---\s", gen, re.MULTILINE)) and bool(
        re.search(r"^\+\+\+\s", gen, re.MULTILINE)
    )
    has_hunks = bool(re.search(r"^@@\s", gen, re.MULTILINE))
    if has_diff_headers and has_hunks:
        fmt = 1.0
    elif has_hunks:
        fmt = 0.5
    else:
        fmt = 0.0

    # structure: contains file paths
    has_file_path = bool(re.search(r"[a-zA-Z_/]+\.\w+", gen))
    structure = 1.0 if has_file_path else 0.0

    # length: non-trivial but not excessively long
    length = 1.0 if 10 < len(gen) < len(gold) * 5 else 0.0

    # overlap: Jaccard similarity of changed lines
    gen_lines = _extract_changed_lines(gen)
    gold_lines = _extract_changed_lines(gold)
    if gen_lines and gold_lines:
        intersection = len(gen_lines & gold_lines)
        union = len(gen_lines | gold_lines)
        overlap = intersection / union if union > 0 else 0.0
    else:
        overlap = 0.0

    # exact match
    exact = 1.0 if gen == gold else 0.0

    # weighted total (same weights as training reward)
    total = 0.2 * fmt + 0.1 * structure + 0.1 * length + 0.4 * overlap + 0.2 * exact

    return {
        "format": fmt,
        "structure": structure,
        "length": length,
        "overlap": overlap,
        "exact_match": exact,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on SWE-RL patch generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (defaults to checkpoint)")
    parser.add_argument("--dataset-name", type=str, default="nvidia/Nemotron-Cascade-2-RL-data")
    parser.add_argument("--dataset-config", type=str, default="SWE-RL")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples to evaluate")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per example for pass@k")
    parser.add_argument("--pass-k", type=str, default="1", help="Comma-separated k values for pass@k (e.g. 1,5,10)")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Save per-example results to JSONL")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    args = parser.parse_args()

    k_values = [int(k) for k in args.pass_k.split(",")]
    num_samples = args.num_samples
    temperature = args.temperature

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
    print(f"Loading model from {args.checkpoint} (device={device}, dtype={dtype})")
    tokenizer = load_tokenizer_with_safe_mistral_fix(args.tokenizer or args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # load dataset
    examples = load_swe_rl_examples(args.dataset_name, args.dataset_config)
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Evaluating on {len(examples)} examples, {num_samples} sample(s) each")

    # run eval
    metric_sums = Counter()
    total_examples = 0
    exact_matches = 0
    source_scores = {}
    pass_k_strict_sums = {k: 0.0 for k in k_values}  # exact match
    pass_k_loose_sums = {k: 0.0 for k in k_values}   # total >= 0.5
    results = []
    started = time.time()

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        golden_patch = ex["golden_patch"]
        source = ex.get("source", "unknown")

        samples = []
        n_exact = 0
        n_loose = 0
        best_scores = None

        for s in range(num_samples):
            completion = generate(model, tokenizer, prompt, args.max_new_tokens, temperature, device)
            scores = score_patch(completion, golden_patch)

            if scores["exact_match"] == 1.0:
                n_exact += 1
            if scores["total"] >= 0.5:
                n_loose += 1

            if best_scores is None or scores["total"] > best_scores["total"]:
                best_scores = scores

            # aggregate metrics from first sample (for backward compat)
            if s == 0:
                for metric, val in scores.items():
                    metric_sums[metric] += val
                if scores["exact_match"] == 1.0:
                    exact_matches += 1

            samples.append({
                "completion": completion[:2000],
                "scores": scores,
            })

        total_examples += 1

        for k in k_values:
            pass_k_strict_sums[k] += pass_at_k(num_samples, n_exact, k)
            pass_k_loose_sums[k] += pass_at_k(num_samples, n_loose, k)

        # track per-source
        if source not in source_scores:
            source_scores[source] = {"total_score": 0.0, "count": 0, "exact": 0}
        source_scores[source]["total_score"] += best_scores["total"]
        source_scores[source]["count"] += 1
        if n_exact > 0:
            source_scores[source]["exact"] += 1

        result = {
            "instance_id": ex.get("instance_id", ""),
            "source": source,
            "prompt": prompt[:500],
            "golden_patch": golden_patch[:2000],
            "n_exact": n_exact,
            "n_loose": n_loose,
            "num_samples": num_samples,
            "best_scores": best_scores,
        }
        if num_samples == 1:
            result["completion"] = samples[0]["completion"]
            result["scores"] = samples[0]["scores"]
        else:
            result["samples"] = samples
        results.append(result)

        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            elapsed = time.time() - started
            avg_time = elapsed / (i + 1)
            avg_total = metric_sums["total"] / total_examples
            strict_str = "  ".join(
                f"pass@{k}={100*pass_k_strict_sums[k]/total_examples:.1f}%" for k in k_values
            )
            print(f"  [{i+1}/{len(examples)}] avg_score={avg_total:.3f}  "
                  f"{strict_str}  "
                  f"{avg_time:.1f}s/example")

    # final report
    elapsed = time.time() - started
    print(f"\n{'='*60}")
    print(f"SWE-RL Results: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"  Examples:       {total_examples}")
    print(f"  Samples/example: {num_samples}")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/max(total_examples,1):.1f}s/example)")

    print(f"\n  Metrics (averaged, first sample):")
    for metric in ["total", "format", "structure", "length", "overlap", "exact_match"]:
        avg = metric_sums[metric] / max(total_examples, 1)
        print(f"    {metric:15s}: {avg:.3f}")

    print(f"\n  pass@k (strict = exact match):")
    for k in k_values:
        avg = pass_k_strict_sums[k] / max(total_examples, 1)
        print(f"    pass@{k}: {100*avg:.1f}%")

    print(f"\n  pass@k (loose = score >= 0.5):")
    for k in k_values:
        avg = pass_k_loose_sums[k] / max(total_examples, 1)
        print(f"    pass@{k}: {100*avg:.1f}%")

    if source_scores:
        print(f"\n  Per-source breakdown (best of {num_samples}):")
        for source in sorted(source_scores, key=lambda s: source_scores[s]["count"], reverse=True):
            info = source_scores[source]
            avg = info["total_score"] / info["count"]
            print(f"    {source}: avg_score={avg:.3f}  exact={info['exact']}/{info['count']}  n={info['count']}")

    # save results
    if args.output_jsonl:
        with open(args.output_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  Results saved to {args.output_jsonl}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
