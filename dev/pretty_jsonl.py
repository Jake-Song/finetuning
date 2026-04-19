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
    args = p.parse_args()

    if args.path is not None and not args.path.is_file():
        p.error(f"not a file or missing: {args.path.resolve()}")

    src_label = str(args.path) if args.path is not None else "<stdin>"
    step_filter = set(args.steps) if args.steps is not None else None

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
