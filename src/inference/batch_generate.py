from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def batched_generate_texts(
    model,
    tokenizer,
    records: Sequence[dict[str, Any]],
    build_messages: Callable[[dict[str, Any]], list[dict[str, str]]],
    generation_kwargs: dict[str, Any],
    batch_size: int = 16,
) -> list[str]:
    texts: list[str] = []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            prompt_texts = [
                tokenizer.apply_chat_template(
                    build_messages(record),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in batch_records
            ]
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
            outputs = model.generate(**inputs, **generation_kwargs)
            prompt_token_count = inputs["input_ids"].shape[1]
            for row in outputs:
                generated_tokens = row[prompt_token_count:]
                texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = original_padding_side

    return texts
