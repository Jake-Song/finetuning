#!/usr/bin/env python3
"""Pretty-print each line of a JSONL file as indented JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("sample.jsonl"),
        help="JSONL file (default: sample.jsonl in cwd)",
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
    args = p.parse_args()

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        with args.path.open(encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"{args.path}:{i}: JSON decode error: {e}", file=sys.stderr)
                    sys.exit(1)
                if i > 1:
                    out.write("\n")
                out.write(f"===== record {i} =====\n")
                out.write(
                    json.dumps(obj, indent=args.indent, ensure_ascii=False) + "\n"
                )
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
