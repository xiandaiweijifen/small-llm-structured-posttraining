"""Build seen/unseen schema evaluation datasets for phase-1 reduced schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.data.schema_variants import build_seen_unseen_reduced_sets
from src.evaluation.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase-1 seen/unseen schema eval datasets.")
    parser.add_argument(
        "--input",
        default="data/reduced/phase1_test_reduced.jsonl",
        help="Base reduced-schema test set.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/generalization",
        help="Directory for seen/unseen schema eval outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    seen_records, unseen_records = build_seen_unseen_reduced_sets(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_path = output_dir / "phase1_test_seen_reduced.jsonl"
    unseen_path = output_dir / "phase1_test_unseen_reduced.jsonl"
    combined_path = output_dir / "phase1_test_seen_unseen_reduced.jsonl"
    summary_path = output_dir / "schema_generalization_summary.json"

    combined_records = seen_records + unseen_records

    dump_jsonl(seen_path, seen_records)
    dump_jsonl(unseen_path, unseen_records)
    dump_jsonl(combined_path, combined_records)

    summary = {
        "input_path": str(Path(args.input)),
        "seen_output_path": str(seen_path),
        "unseen_output_path": str(unseen_path),
        "combined_output_path": str(combined_path),
        "counts": {
            "input_records": len(records),
            "seen_records": len(seen_records),
            "unseen_records": len(unseen_records),
            "combined_records": len(combined_records),
        },
        "schemas": {
            "seen_schema_name": "ticket_schema_v1_reduced",
            "unseen_schema_name": "ticket_schema_v1_reduced_1_1",
        },
    }
    write_json_report(summary_path, summary)
    print(f"Seen eval dataset written to {seen_path}")
    print(f"Unseen eval dataset written to {unseen_path}")
    print(f"Combined eval dataset written to {combined_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
