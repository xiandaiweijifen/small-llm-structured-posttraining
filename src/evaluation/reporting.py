"""Reporting helpers for evaluation outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def group_sample_results(
    sample_results: list[dict[str, Any]],
    group_field: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in sample_results:
        group_value = result.get(group_field, "unknown")
        grouped[str(group_value)].append(result)
    return dict(grouped)
