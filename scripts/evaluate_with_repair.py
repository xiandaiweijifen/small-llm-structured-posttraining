"""Repair a prediction file and evaluate the repaired outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.evaluation.metrics import evaluate_sample, summarize_results
from src.evaluation.reporting import write_json_report
from src.evaluation.field_analysis import analyze_field_errors
from src.evaluation.metrics import try_parse_prediction_text
from src.inference.repair import repair_prediction
from src.schemas.registry import get_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair predictions and evaluate repaired outputs.")
    parser.add_argument("--gold", required=True, help="Path to gold dataset jsonl.")
    parser.add_argument("--pred", required=True, help="Path to raw prediction jsonl.")
    parser.add_argument("--schema-name", default="ticket_schema_v1", help="Schema name for repair/eval.")
    parser.add_argument("--repaired-output", required=True, help="Path to repaired prediction jsonl.")
    parser.add_argument("--report-output", required=True, help="Path to repaired evaluation report json.")
    parser.add_argument("--field-output", required=True, help="Path to repaired field analysis json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = get_schema(args.schema_name)
    gold_records = load_jsonl(args.gold)
    pred_records = load_jsonl(args.pred)

    repaired_records = []
    for record in pred_records:
        prediction_json = record.get("prediction_json")
        prediction_text = record.get("prediction_text")
        if prediction_json is None and isinstance(prediction_text, str):
            _, prediction_json = try_parse_prediction_text(prediction_text)

        repaired_json, repaired = repair_prediction(prediction_json, schema)
        repaired_record = dict(record)
        repaired_record["prediction_json"] = repaired_json
        metadata = dict(record.get("metadata", {}))
        metadata["repaired"] = repaired
        repaired_record["metadata"] = metadata
        repaired_records.append(repaired_record)

    dump_jsonl(args.repaired_output, repaired_records)

    predictions_by_id = {record["sample_id"]: record for record in repaired_records}
    sample_results = []
    for gold_record in gold_records:
        pred_record = predictions_by_id.get(gold_record["sample_id"], {})
        sample_eval = evaluate_sample(
            sample_id=gold_record["sample_id"],
            prediction_text=pred_record.get("prediction_text"),
            prediction_json=pred_record.get("prediction_json"),
            target_json=gold_record["target_json"],
            schema=schema,
        )
        sample_results.append(
            {
                **sample_eval.__dict__,
                "schema_name": gold_record.get("schema_name", args.schema_name),
                "complexity_bucket": gold_record.get("complexity_bucket", "unknown"),
            }
        )

    report = {
        "gold_path": args.gold,
        "pred_path": args.pred,
        "repaired_pred_path": args.repaired_output,
        "summary": summarize_results_from_dicts(sample_results),
        "per_sample": sample_results,
    }
    write_json_report(args.report_output, report)
    write_json_report(args.field_output, analyze_field_errors(gold_records, repaired_records))
    print(f"Repaired predictions written to {args.repaired_output}")
    print(f"Repaired report written to {args.report_output}")
    print(f"Repaired field analysis written to {args.field_output}")


def summarize_results_from_dicts(sample_results: list[dict]) -> dict:
    from src.evaluation.metrics import SampleEvaluation

    evaluations = [
        SampleEvaluation(
            sample_id=item["sample_id"],
            valid_json=item["valid_json"],
            schema_compliant=item["schema_compliant"],
            field_exact_match=item["field_exact_match"],
            exact_match=item["exact_match"],
            primary_error=item["primary_error"],
        )
        for item in sample_results
    ]
    return summarize_results(evaluations)


if __name__ == "__main__":
    main()
