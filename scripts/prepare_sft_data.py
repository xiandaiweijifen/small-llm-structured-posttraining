"""Convert dataset splits into generic SFT chat-format jsonl files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.training.formatters import convert_to_sft_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT-format data from dataset splits.")
    parser.add_argument("--input", required=True, help="Path to dataset split jsonl.")
    parser.add_argument("--output", required=True, help="Path to output SFT jsonl.")
    parser.add_argument(
        "--include-schema-definition",
        action="store_true",
        help="Embed the JSON schema definition in the user prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input)
    records = convert_to_sft_records(
        samples,
        include_schema_definition=args.include_schema_definition,
    )
    dump_jsonl(args.output, records)
    print(f"SFT data written to {args.output}")


if __name__ == "__main__":
    main()
