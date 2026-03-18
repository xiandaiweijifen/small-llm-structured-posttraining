"""Helpers for Hugging Face dataset download and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def load_hf_dataset(dataset_name: str) -> Dataset | DatasetDict:
    return load_dataset(dataset_name)


def save_hf_dataset_to_disk(dataset: Dataset | DatasetDict, output_dir: str | Path) -> None:
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))


def load_local_hf_dataset(dataset_dir: str | Path) -> Dataset | DatasetDict:
    return load_from_disk(str(dataset_dir))


def export_dataset_to_jsonl(
    dataset: Dataset | DatasetDict,
    output_dir: str | Path,
) -> list[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    exported_files: list[str] = []

    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            file_path = output_path / f"{split_name}.jsonl"
            split_dataset.to_json(str(file_path), orient="records", lines=True, force_ascii=False)
            exported_files.append(str(file_path))
        return exported_files

    file_path = output_path / "data.jsonl"
    dataset.to_json(str(file_path), orient="records", lines=True, force_ascii=False)
    exported_files.append(str(file_path))
    return exported_files


def summarize_dataset_columns(dataset: Dataset | DatasetDict) -> dict[str, Any]:
    if isinstance(dataset, DatasetDict):
        return {
            split_name: {
                "num_rows": split_dataset.num_rows,
                "columns": split_dataset.column_names,
            }
            for split_name, split_dataset in dataset.items()
        }
    return {
        "default": {
            "num_rows": dataset.num_rows,
            "columns": dataset.column_names,
        }
    }
