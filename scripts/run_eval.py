"""Evaluate structured-output predictions against ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_jsonl
from src.evaluation.metrics import evaluate_sample, summarize_results
from src.evaluation.reporting import write_json_report
from src.schemas.ticket_schema import TICKET_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for structured-output predictions.")
    parser.add_argument("--gold", required=True, help="Path to gold dataset jsonl.")
    parser.add_argument("--pred", required=True, help="Path to prediction jsonl.")
    parser.add_argument(
        "--output",
        default="results/metrics/eval_report.json",
        help="Path to write aggregated evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_records = load_jsonl(args.gold)
    pred_records = load_jsonl(args.pred)

    predictions_by_id = {record["sample_id"]: record for record in pred_records}
    sample_results = []

    for gold_record in gold_records:
        sample_id = gold_record["sample_id"]
        pred_record = predictions_by_id.get(sample_id, {})
        sample_results.append(
            evaluate_sample(
                sample_id=sample_id,
                prediction_text=pred_record.get("prediction_text"),
                prediction_json=pred_record.get("prediction_json"),
                target_json=gold_record["target_json"],
                schema=TICKET_SCHEMA,
            )
        )

    summary = summarize_results(sample_results)
    report = {
        "gold_path": str(Path(args.gold)),
        "pred_path": str(Path(args.pred)),
        "summary": summary,
        "per_sample": [result.__dict__ for result in sample_results],
    }
    write_json_report(args.output, report)
    print(f"Evaluation report written to {args.output}")


if __name__ == "__main__":
    main()
