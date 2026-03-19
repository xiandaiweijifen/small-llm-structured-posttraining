"""Profile a mapped dataset and write summary statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_jsonl
from src.data.profiling import profile_dataset
from src.evaluation.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile mapped dataset quality statistics.")
    parser.add_argument("--input", required=True, help="Path to mapped dataset jsonl.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output summary json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    summary = profile_dataset(records)
    write_json_report(args.output, summary)
    print(f"Profile written to {args.output}")


if __name__ == "__main__":
    main()
