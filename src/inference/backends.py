"""Inference backends used for local pipeline testing and future model integration."""

from __future__ import annotations

import json
from typing import Any


def generate_prediction(record: dict[str, Any], backend: str) -> tuple[str, dict[str, Any] | None]:
    if backend == "oracle":
        prediction_json = record["target_json"]
        return json.dumps(prediction_json, ensure_ascii=False), prediction_json

    if backend == "empty_json":
        prediction_json: dict[str, Any] = {}
        return "{}", prediction_json

    if backend == "invalid_json":
        return '{"broken": true', None

    raise ValueError(f"Unsupported backend: {backend}")
