"""Run field-level error analysis for a prediction file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_jsonl
from src.evaluation.field_analysis import analyze_field_errors
from src.evaluation.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze field-level errors for structured-output predictions.")
    parser.add_argument("--gold", required=True, help="Path to gold dataset jsonl.")
    parser.add_argument("--pred", required=True, help="Path to prediction jsonl.")
    parser.add_argument("--output", required=True, help="Path to output analysis json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_records = load_jsonl(args.gold)
    pred_records = load_jsonl(args.pred)
    report = analyze_field_errors(gold_records, pred_records)
    write_json_report(args.output, report)
    print(f"Field analysis written to {args.output}")


if __name__ == "__main__":
    main()
