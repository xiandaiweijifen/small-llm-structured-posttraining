"""Complexity-bucket scoring and relabeling helpers."""

from __future__ import annotations

from typing import Any


def compute_complexity_features(target_json: dict[str, Any]) -> dict[str, int]:
    reporter = target_json.get("reporter", {})
    constraints = target_json.get("constraints", {})
    affected_systems = target_json.get("affected_systems", [])
    actions_requested = target_json.get("actions_requested", [])

    return {
        "has_reporter_name": int(reporter.get("name") is not None),
        "has_reporter_team": int(reporter.get("team") is not None),
        "has_environment": int(constraints.get("environment") is not None),
        "has_blocking": int(constraints.get("blocking") is not None),
        "num_affected_systems": len(affected_systems),
        "num_actions_requested": len(actions_requested),
    }


def infer_complexity_bucket(target_json: dict[str, Any]) -> str:
    features = compute_complexity_features(target_json)

    if is_simple_case(target_json, features):
        return "simple"
    if is_complex_case(features):
        return "complex"
    return "medium"


def is_simple_case(target_json: dict[str, Any], features: dict[str, int]) -> bool:
    summary = target_json.get("summary", "")
    return (
        features["has_reporter_team"] == 0
        and features["has_environment"] == 0
        and features["has_blocking"] == 0
        and features["num_affected_systems"] <= 1
        and features["num_actions_requested"] <= 1
        and len(summary) <= 80
    )


def is_complex_case(features: dict[str, int]) -> bool:
    score = 0
    score += features["has_reporter_name"]
    score += features["has_reporter_team"]
    score += features["has_environment"]
    score += features["has_blocking"]
    score += min(features["num_affected_systems"], 2)
    score += min(features["num_actions_requested"], 2)
    return score >= 5


def relabel_record_complexity(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    updated["metadata"] = dict(record.get("metadata", {}))
    updated["complexity_bucket"] = infer_complexity_bucket(record["target_json"])
    return updated
