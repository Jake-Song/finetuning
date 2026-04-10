import argparse
import json
import math
from collections import Counter
from pathlib import Path


def pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    if k <= 0:
        raise ValueError("k must be positive")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_results(jsonl_path: Path) -> list[dict]:
    results = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            results.append(record)

    if not results:
        raise ValueError("Input JSONL is empty")
    return results


def get_first_sample_constraint_results(record: dict) -> dict:
    if isinstance(record.get("constraint_results"), dict):
        return record["constraint_results"]

    samples = record.get("samples")
    if isinstance(samples, list) and samples:
        first_sample = samples[0]
        if isinstance(first_sample, dict) and isinstance(first_sample.get("constraint_results"), dict):
            return first_sample["constraint_results"]

    raise ValueError("Record is missing constraint results")


def get_num_samples(record: dict) -> int:
    num_samples = record.get("num_samples")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("Record is missing a valid num_samples value")
    return num_samples


def get_n_correct(record: dict) -> int:
    n_correct = record.get("n_correct")
    if not isinstance(n_correct, int) or n_correct < 0:
        raise ValueError("Record is missing a valid n_correct value")
    return n_correct


def summarize_results(results: list[dict]) -> dict:
    total_examples = len(results)
    samples_per_example = get_num_samples(results[0])
    strict_passes = 0
    first_sample_passed_constraints = 0
    first_sample_total_constraints = 0
    constraint_passes = Counter()
    constraint_totals = Counter()

    for index, record in enumerate(results, start=1):
        num_samples = get_num_samples(record)
        if num_samples != samples_per_example:
            raise ValueError(
                f"Inconsistent num_samples at record {index}: expected {samples_per_example}, got {num_samples}"
            )

        n_correct = get_n_correct(record)
        if n_correct > num_samples:
            raise ValueError(
                f"Invalid n_correct at record {index}: n_correct={n_correct} exceeds num_samples={num_samples}"
            )
        if n_correct > 0:
            strict_passes += 1

        constraint_results = get_first_sample_constraint_results(record)
        known_results = {cid: passed for cid, passed in constraint_results.items() if passed is not None}

        first_sample_total_constraints += len(known_results)
        first_sample_passed_constraints += sum(bool(passed) for passed in known_results.values())
        for cid, passed in known_results.items():
            constraint_totals[cid] += 1
            if passed:
                constraint_passes[cid] += 1

    pass_at_k_metrics = {
        k: sum(pass_at_k(samples_per_example, get_n_correct(record), k) for record in results) / total_examples
        for k in range(1, samples_per_example + 1)
    }

    return {
        "total_examples": total_examples,
        "samples_per_example": samples_per_example,
        "strict_passes": strict_passes,
        "first_sample_passed_constraints": first_sample_passed_constraints,
        "first_sample_total_constraints": first_sample_total_constraints,
        "constraint_passes": constraint_passes,
        "constraint_totals": constraint_totals,
        "pass_at_k_metrics": pass_at_k_metrics,
    }


def format_percent(numerator: int | float, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{100 * numerator / denominator:.1f}%"


def render_markdown(input_path: Path, summary: dict) -> str:
    total_examples = summary["total_examples"]
    samples_per_example = summary["samples_per_example"]
    strict_passes = summary["strict_passes"]
    first_sample_passed_constraints = summary["first_sample_passed_constraints"]
    first_sample_total_constraints = summary["first_sample_total_constraints"]
    constraint_passes = summary["constraint_passes"]
    constraint_totals = summary["constraint_totals"]
    pass_at_k_metrics = summary["pass_at_k_metrics"]

    lines = [
        "# IF Eval Results",
        "",
        f"Source JSONL: `{input_path}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Examples | {total_examples} |",
        f"| Samples/example | {samples_per_example} |",
        f"| Strict accuracy | {strict_passes}/{total_examples} ({format_percent(strict_passes, total_examples)}) |",
        (
            "| First-sample constraint accuracy | "
            f"{first_sample_passed_constraints}/{first_sample_total_constraints} "
            f"({format_percent(first_sample_passed_constraints, first_sample_total_constraints)}) |"
        ),
        "",
        "## Pass@k",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]

    for k, value in pass_at_k_metrics.items():
        lines.append(f"| pass@{k} | {100 * value:.1f}% |")

    lines.extend([
        "",
        "## Per-Constraint Breakdown",
        "",
        "| Constraint ID | Passed | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ])

    for cid in sorted(constraint_totals, key=lambda item: (-constraint_totals[item], item)):
        passed = constraint_passes[cid]
        total = constraint_totals[cid]
        lines.append(f"| `{cid}` | {passed} | {total} | {format_percent(passed, total)} |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Markdown report from scripts/if_eval.py JSONL output")
    parser.add_argument("input_jsonl", type=Path, help="Path to the JSONL results file produced by scripts/if_eval.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_jsonl
    if input_path.suffix != ".jsonl":
        output_path = input_path.with_name(f"{input_path.name}.md")
    else:
        output_path = input_path.with_suffix(".md")

    results = load_results(input_path)
    summary = summarize_results(results)
    markdown = render_markdown(input_path, summary)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Markdown report saved to {output_path}")


if __name__ == "__main__":
    main()
