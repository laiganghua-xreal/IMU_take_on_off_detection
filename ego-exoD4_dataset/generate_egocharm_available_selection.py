#!/usr/bin/env python3
"""Filter the official EgoCHARM selection to complete local VRS files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egocharm.data.selection import build_available_selection


def parse_args() -> argparse.Namespace:
    dataset_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    parser.add_argument(
        "--source",
        type=Path,
        default=dataset_root / "metadata" / "egocharm_official_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_root / "metadata" / "egocharm_available_selection.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    available = build_available_selection(source, args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(available, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {available['num_takes']} complete takes "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
