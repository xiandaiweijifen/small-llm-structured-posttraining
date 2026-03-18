"""Reporting helpers for evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
