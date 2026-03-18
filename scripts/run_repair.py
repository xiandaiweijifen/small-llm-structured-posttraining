"""Apply schema-aware repair to prediction jsonl files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.evaluation.metrics import try_parse_prediction_text
from src.inference.repair import repair_prediction
from src.schemas.ticket_schema import TICKET_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair prediction jsonl using schema-aware rules.")
    parser.add_argument("--input", required=True, help="Path to prediction jsonl.")
    parser.add_argument("--output", required=True, help="Path to repaired prediction jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_jsonl(args.input)
    repaired_records = []

    for record in predictions:
        prediction_json = record.get("prediction_json")
        prediction_text = record.get("prediction_text")

        if prediction_json is None and isinstance(prediction_text, str):
            _, prediction_json = try_parse_prediction_text(prediction_text)

        repaired_json, repaired = repair_prediction(prediction_json, TICKET_SCHEMA)
        repaired_record = dict(record)
        repaired_record["prediction_json"] = repaired_json
        metadata = dict(record.get("metadata", {}))
        metadata["repaired"] = repaired
        repaired_record["metadata"] = metadata
        repaired_records.append(repaired_record)

    dump_jsonl(args.output, repaired_records)
    print(f"Repaired predictions written to {args.output}")


if __name__ == "__main__":
    main()
