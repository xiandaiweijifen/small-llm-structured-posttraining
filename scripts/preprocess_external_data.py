"""Convert external exported datasets into project sample jsonl files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.data.mappers import (
    map_console_ai_record,
    map_kameronb_record,
    set_record_sample_id,
)
from src.data.validation import ensure_unique_sample_ids
from src.evaluation.reporting import write_json_report


SOURCE_CONFIG = {
    "console_ai_it_helpdesk_tickets": {
        "input_path": "data/raw/exports/console_ai_it_helpdesk_tickets/train.jsonl",
        "output_path": "data/samples/console_ai_mapped.jsonl",
        "mapper": map_console_ai_record,
    },
    "kameronb_it_callcenter_tickets": {
        "input_path": "data/raw/exports/kameronb_it_callcenter_tickets/train.jsonl",
        "output_path": "data/samples/kameronb_mapped.jsonl",
        "mapper": map_kameronb_record,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess external datasets into project samples.")
    parser.add_argument(
        "--source-id",
        choices=tuple(SOURCE_CONFIG.keys()) + ("all",),
        default="all",
        help="Which configured source to preprocess.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_ids = (
        list(SOURCE_CONFIG.keys()) if args.source_id == "all" else [args.source_id]
    )

    summaries = {}
    for source_id in selected_ids:
        source = SOURCE_CONFIG[source_id]
        raw_records = load_jsonl(source["input_path"])
        mapped_records = []
        for index, record in enumerate(raw_records):
            mapped_record = source["mapper"](record)
            mapped_record = set_record_sample_id(
                mapped_record,
                build_unique_sample_id(source_id, index, mapped_record["target_json"]["ticket_id"]),
            )
            mapped_records.append(mapped_record)
        ensure_unique_sample_ids(mapped_records)
        dump_jsonl(source["output_path"], mapped_records)

        summaries[source_id] = {
            "input_path": source["input_path"],
            "output_path": source["output_path"],
            "num_records": len(mapped_records),
            "complexity_counts": count_field(mapped_records, "complexity_bucket"),
            "source_type_counts": count_metadata_field(mapped_records, "source_type"),
        }
        print(f"Preprocessed {source_id} -> {source['output_path']}")

    summary_path = "data/samples/external_preprocess_summary.json"
    write_json_report(summary_path, summaries)
    print(f"Preprocess summary written to {summary_path}")


def count_field(records: list[dict], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field_name, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_metadata_field(records: list[dict], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("metadata", {}).get(field_name, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_unique_sample_id(source_id: str, index: int, ticket_id: str | None) -> str:
    suffix = ticket_id if ticket_id else f"row{index:06d}"
    return f"{source_id}-{index:06d}-{suffix}"


if __name__ == "__main__":
    main()
