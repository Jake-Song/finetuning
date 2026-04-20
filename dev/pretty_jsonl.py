#!/usr/bin/env python3
"""Pretty-print each line of a JSONL file as indented JSON."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path


def parse_step(value: str) -> int:
    try:
        return int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid step: {value!r}") from e


def parse_record_generation(value: str) -> tuple[int, int]:
    try:
        record_text, generation_text = value.split(":", 1)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid record generation mapping: {value!r} (expected RECORD:GEN)"
        ) from e

    try:
        record = int(record_text)
        generation = int(generation_text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid record generation mapping: {value!r} (expected integer RECORD:GEN)"
        ) from e

    if record < 1:
        raise argparse.ArgumentTypeError(
            f"invalid record generation mapping: {value!r} (record must be >= 1)"
        )
    return record, generation


def build_record_generation_map(
    mappings: list[tuple[int, int]] | None,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for record, generation in mappings or []:
        existing = result.get(record)
        if existing is not None and existing != generation:
            raise argparse.ArgumentTypeError(
                f"conflicting generation mappings for record {record}: {existing} vs {generation}"
            )
        result[record] = generation
    return result


def select_generation_for_record(
    obj: dict,
    record_number: int,
    generation_map: dict[int, int],
    *,
    src_label: str,
) -> dict | None:
    target_generation = generation_map.get(record_number)
    if target_generation is None:
        return obj

    if isinstance(obj.get("samples"), list):
        selected_samples = [
            sample
            for sample in obj["samples"]
            if isinstance(sample, dict)
            and sample.get("generation_index") == target_generation
        ]
        if not selected_samples:
            raise ValueError(
                f"{src_label}:{record_number}: no sample with generation_index={target_generation}"
            )
        selected = dict(obj)
        selected["samples"] = selected_samples
        return selected

    if "generation_index" in obj:
        if obj.get("generation_index") == target_generation:
            return obj
        return None

    raise ValueError(
        f"{src_label}:{record_number}: record-generation mapping requires either top-level "
        f"'generation_index' or a 'samples' array"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="JSONL file (omit to read from stdin)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write here instead of stdout",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2)",
    )
    p.add_argument(
        "--step",
        dest="steps",
        action="append",
        type=parse_step,
        default=None,
        help="Only print records whose top-level 'step' matches this value. Repeatable.",
    )
    p.add_argument(
        "--record-generation",
        dest="record_generations",
        action="append",
        type=parse_record_generation,
        default=None,
        help="For a specific 1-based JSONL record, select generation_index RECORD:GEN. Repeatable.",
    )
    args = p.parse_args()

    if args.path is not None and not args.path.is_file():
        p.error(f"not a file or missing: {args.path.resolve()}")

    src_label = str(args.path) if args.path is not None else "<stdin>"
    step_filter = set(args.steps) if args.steps is not None else None
    try:
        record_generation_map = build_record_generation_map(args.record_generations)
    except argparse.ArgumentTypeError as e:
        p.error(str(e))

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        ctx = (
            args.path.open(encoding="utf-8")
            if args.path is not None
            else nullcontext(sys.stdin)
        )
        with ctx as f:
            printed = 0
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"{src_label}:{i}: JSON decode error: {e}", file=sys.stderr)
                    sys.exit(1)
                if step_filter is not None and obj.get("step") not in step_filter:
                    continue
                try:
                    obj = select_generation_for_record(
                        obj,
                        i,
                        record_generation_map,
                        src_label=src_label,
                    )
                except ValueError as e:
                    print(str(e), file=sys.stderr)
                    sys.exit(1)
                if obj is None:
                    continue
                if printed > 0:
                    out.write("\n")
                out.write(f"===== record {i} =====\n")
                out.write(
                    json.dumps(obj, indent=args.indent, ensure_ascii=False) + "\n"
                )
                printed += 1
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
