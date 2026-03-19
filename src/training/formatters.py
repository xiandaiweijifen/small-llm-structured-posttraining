"""Training-data formatters for SFT and related experiments."""

from __future__ import annotations

import json
from typing import Any

from src.schemas.registry import get_schema


DEFAULT_SYSTEM_PROMPT = (
    "You are an information extraction model. "
    "Return only JSON that matches the requested schema. "
    "Do not add explanations or markdown."
)


def build_user_prompt(
    input_text: str,
    task_name: str,
    schema_name: str,
    include_schema_definition: bool = False,
) -> str:
    prompt = (
        f"Task: extract a structured record for {task_name}.\n"
        f"Schema name: {schema_name}\n"
    )
    if include_schema_definition:
        schema_definition = json.dumps(get_schema(schema_name), ensure_ascii=False, separators=(",", ":"))
        prompt += (
            "Schema definition:\n"
            f"{schema_definition}\n"
        )
    prompt += (
        "Return a JSON object only.\n"
        "Input text:\n"
        f"{input_text}"
    )
    return prompt


def to_sft_record(sample: dict[str, Any], include_schema_definition: bool = False) -> dict[str, Any]:
    assistant_content = json.dumps(sample["target_json"], ensure_ascii=False)
    return {
        "sample_id": sample["sample_id"],
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    input_text=sample["input_text"],
                    task_name=sample["task_name"],
                    schema_name=sample["schema_name"],
                    include_schema_definition=include_schema_definition,
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "task_name": sample["task_name"],
            "schema_name": sample["schema_name"],
            "complexity_bucket": sample["complexity_bucket"],
            "source_type": sample["metadata"]["source_type"],
            "split": sample["metadata"].get("split"),
            "is_synthetic": sample["metadata"]["is_synthetic"],
            "schema_conditioned_prompt": include_schema_definition,
        },
    }


def convert_to_sft_records(
    samples: list[dict[str, Any]],
    include_schema_definition: bool = False,
) -> list[dict[str, Any]]:
    return [
        to_sft_record(sample, include_schema_definition=include_schema_definition)
        for sample in samples
    ]
