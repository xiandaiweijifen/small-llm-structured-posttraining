"""Build reduced-schema train/val/test and SFT files from phase-1 splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.data.reduced_schema import convert_records_to_reduced_schema
from src.data.dataset_builder import validate_records
from src.schemas.registry import get_schema
from src.training.formatters import convert_to_sft_records
from src.evaluation.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reduced-schema dataset files from phase-1 splits.")
    parser.add_argument("--input-dir", default="data/samples", help="Directory containing phase1 train/val/test jsonl.")
    parser.add_argument("--output-dir", default="data/reduced", help="Directory to write reduced-schema files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    schema = get_schema("ticket_schema_v1_reduced")

    for split_name in ("train", "val", "test"):
        input_path = input_dir / f"phase1_{split_name}.jsonl"
        records = load_jsonl(input_path)
        reduced_records = convert_records_to_reduced_schema(records)
        validate_records(
            reduced_records,
            schema=schema,
            task_name="ticket_structured_output",
            schema_name="ticket_schema_v1_reduced",
        )

        reduced_path = output_dir / f"phase1_{split_name}_reduced.jsonl"
        dump_jsonl(reduced_path, reduced_records)

        if split_name in {"train", "val"}:
            sft_records = convert_to_sft_records(reduced_records)
            sft_path = output_dir / f"phase1_sft_{split_name}_reduced.jsonl"
            dump_jsonl(sft_path, sft_records)

        summary[split_name] = {
            "input_records": len(records),
            "output_records": len(reduced_records),
            "output_path": str(reduced_path),
        }

    write_json_report(output_dir / "reduced_schema_summary.json", summary)
    print(f"Reduced-schema dataset written to {output_dir}")


if __name__ == "__main__":
    main()
