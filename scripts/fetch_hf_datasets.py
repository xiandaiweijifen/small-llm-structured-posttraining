"""Download Hugging Face datasets and save project-local copies."""

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
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch configured Hugging Face datasets and save local project copies."
    )
    parser.add_argument(
        "--config",
        default="configs/data_sources/hf_sources.yaml",
        help="Path to data source config yaml.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Optional source_id filter. If omitted, fetch all configured sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    sources = config.get("sources", [])
    selected_sources = [
        source for source in sources if args.source_id is None or source["source_id"] == args.source_id
    ]

    if not selected_sources:
        raise ValueError("No matching data sources found in config.")

    summaries = {}
    for source in selected_sources:
        dataset = load_hf_dataset(source["hf_dataset_name"])
        save_hf_dataset_to_disk(dataset, Path(source["local_dir"]) / "hf_saved")
        exported_files = export_dataset_to_jsonl(dataset, source["export_dir"])
        summaries[source["source_id"]] = {
            "hf_dataset_name": source["hf_dataset_name"],
            "local_dir": source["local_dir"],
            "export_dir": source["export_dir"],
            "exported_files": exported_files,
            "dataset_summary": summarize_dataset_columns(dataset),
        }
        print(f"Fetched {source['hf_dataset_name']}")
        print(f"Saved local dataset copy to {source['local_dir']}\\hf_saved")

    write_json_report("data/raw/exports/hf_fetch_summary.json", summaries)
    print("Fetch summary written to data/raw/exports/hf_fetch_summary.json")


if __name__ == "__main__":
    main()
