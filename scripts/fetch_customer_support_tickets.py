"""Fetch `gorkemsevinc/customer_support_tickets` and save project-local copies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hf_datasets import (
    export_dataset_to_jsonl,
    load_hf_dataset,
    save_hf_dataset_to_disk,
    summarize_dataset_columns,
)
from src.evaluation.reporting import write_json_report


DATASET_NAME = "gorkemsevinc/customer_support_tickets"
LOCAL_DIR = PROJECT_ROOT / "data" / "raw" / "external" / "gorkemsevinc_customer_support_tickets"
EXPORT_DIR = PROJECT_ROOT / "data" / "raw" / "exports" / "gorkemsevinc_customer_support_tickets"
SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "exports" / "gorkemsevinc_customer_support_tickets_fetch_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch gorkemsevinc/customer_support_tickets and save local copies."
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help="Hugging Face dataset name. Defaults to gorkemsevinc/customer_support_tickets.",
    )
    parser.add_argument(
        "--local-dir",
        default=str(LOCAL_DIR),
        help="Directory for save_to_disk output.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(EXPORT_DIR),
        help="Directory for exported JSONL files.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(SUMMARY_PATH),
        help="Path for the fetch summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_hf_dataset(args.dataset_name)

    local_dir = Path(args.local_dir)
    export_dir = Path(args.export_dir)
    summary_path = Path(args.summary_path)

    save_hf_dataset_to_disk(dataset, local_dir / "hf_saved")
    exported_files = export_dataset_to_jsonl(dataset, export_dir)

    summary = {
        "hf_dataset_name": args.dataset_name,
        "local_dir": str(local_dir),
        "export_dir": str(export_dir),
        "exported_files": exported_files,
        "dataset_summary": summarize_dataset_columns(dataset),
    }
    write_json_report(summary_path, summary)

    print(f"Fetched {args.dataset_name}")
    print(f"Saved local dataset copy to {local_dir / 'hf_saved'}")
    print(f"Exported JSONL files to {export_dir}")
    print(f"Fetch summary written to {summary_path}")


if __name__ == "__main__":
    main()
