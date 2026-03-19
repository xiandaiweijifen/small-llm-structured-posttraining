"""Schema registry for dataset, inference, and evaluation."""

from __future__ import annotations

from src.schemas.ticket_schema import (
    TICKET_SCHEMA_V1,
    TICKET_SCHEMA_V1_1,
    TICKET_SCHEMA_V1_REDUCED,
    TICKET_SCHEMA_V1_REDUCED_1_1,
)

SCHEMA_REGISTRY = {
    "ticket_schema_v1": TICKET_SCHEMA_V1,
    "ticket_schema_v1_1": TICKET_SCHEMA_V1_1,
    "ticket_schema_v1_reduced": TICKET_SCHEMA_V1_REDUCED,
    "ticket_schema_v1_reduced_1_1": TICKET_SCHEMA_V1_REDUCED_1_1,
}


def get_schema(schema_name: str) -> dict:
    if schema_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema_name: {schema_name}")
    return SCHEMA_REGISTRY[schema_name]
