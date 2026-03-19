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
from src.evaluation.reporting import group_sample_results, write_json_report
from src.schemas.registry import get_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for structured-output predictions.")
    parser.add_argument("--gold", required=True, help="Path to gold dataset jsonl.")
    parser.add_argument("--pred", required=True, help="Path to prediction jsonl.")
    parser.add_argument(
        "--schema-name",
        default=None,
        help="Optional schema override. Defaults to each gold record's schema_name.",
    )
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
        schema_name = args.schema_name or gold_record["schema_name"]
        schema = get_schema(schema_name)
        sample_eval = evaluate_sample(
            sample_id=sample_id,
            prediction_text=pred_record.get("prediction_text"),
            prediction_json=pred_record.get("prediction_json"),
            target_json=gold_record["target_json"],
            schema=schema,
        )
        sample_results.append(
            {
                **sample_eval.__dict__,
                "schema_name": schema_name,
                "complexity_bucket": gold_record.get("complexity_bucket", "unknown"),
                "schema_seen_status": gold_record.get(
                    "schema_seen_status",
                    gold_record.get("metadata", {}).get("schema_seen_status", "unknown"),
                ),
            }
        )

    summary = summarize_results_from_dicts(sample_results)
    grouped_by_complexity = summarize_grouped_results(sample_results, "complexity_bucket")
    grouped_by_schema = summarize_grouped_results(sample_results, "schema_name")
    grouped_by_seen_status = summarize_grouped_results(sample_results, "schema_seen_status")
    report = {
        "gold_path": str(Path(args.gold)),
        "pred_path": str(Path(args.pred)),
        "summary": summary,
        "grouped_summary": {
            "by_complexity_bucket": grouped_by_complexity,
            "by_schema_name": grouped_by_schema,
            "by_schema_seen_status": grouped_by_seen_status,
        },
        "per_sample": sample_results,
    }
    write_json_report(args.output, report)
    print(f"Evaluation report written to {args.output}")


def summarize_results_from_dicts(sample_results: list[dict]) -> dict:
    evaluations = [
        evaluate_dict_to_object(sample_result)
        for sample_result in sample_results
    ]
    return summarize_results(evaluations)


def summarize_grouped_results(sample_results: list[dict], group_field: str) -> dict[str, dict]:
    grouped = group_sample_results(sample_results, group_field)
    return {
        group_name: summarize_results_from_dicts(group_items)
        for group_name, group_items in grouped.items()
    }


def evaluate_dict_to_object(sample_result: dict):
    from src.evaluation.metrics import SampleEvaluation

    return SampleEvaluation(
        sample_id=sample_result["sample_id"],
        valid_json=sample_result["valid_json"],
        schema_compliant=sample_result["schema_compliant"],
        field_exact_match=sample_result["field_exact_match"],
        exact_match=sample_result["exact_match"],
        primary_error=sample_result["primary_error"],
    )


if __name__ == "__main__":
    main()
