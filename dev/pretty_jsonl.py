#!/usr/bin/env python3
"""Pretty-print JSONL records, optionally selecting a sample block by row size."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path


def parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from e
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"value must be >= 0: {value!r}")
    return parsed


def should_print_record(record_index: int, *, sample: int | None, rows: int) -> bool:
    if sample is None:
        return True
    start = sample * rows
    end = start + rows
    return start <= record_index < end


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
        "--row",
        type=parse_non_negative_int,
        default=16,
        help="Number of rows in each sample block (default: 16)",
    )
    p.add_argument(
        "--sample",
        type=parse_non_negative_int,
        default=None,
        help="0-based sample block index for the configured row size",
    )
    args = p.parse_args()

    if args.row < 1:
        p.error("--row must be >= 1")

    if args.path is not None and not args.path.is_file():
        p.error(f"not a file or missing: {args.path.resolve()}")

    src_label = str(args.path) if args.path is not None else "<stdin>"
    out = sys.stdout

    ctx = (
        args.path.open(encoding="utf-8")
        if args.path is not None
        else nullcontext(sys.stdin)
    )
    with ctx as f:
        printed = 0
        data_index = 0
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{src_label}:{line_number}: JSON decode error: {e}", file=sys.stderr)
                sys.exit(1)

            if not should_print_record(data_index, sample=args.sample, rows=args.row):
                data_index += 1
                continue

            if printed > 0:
                out.write("\n")
            out.write(f"===== record {line_number} =====\n")
            out.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
            printed += 1
            data_index += 1


if __name__ == "__main__":
    main()
