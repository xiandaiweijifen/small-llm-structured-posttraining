"""Phase-1 schema definitions."""

from __future__ import annotations

from copy import deepcopy

TICKET_SCHEMA_V1 = {
    "type": "object",
    "properties": {
        "ticket_id": {"type": ["string", "null"]},
        "summary": {"type": "string"},
        "category": {
            "type": "string",
            "enum": ["bug", "feature", "question", "incident", "task"],
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "urgent"],
        },
        "requires_followup": {"type": "boolean"},
        "reporter": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "team": {"type": ["string", "null"]},
            },
            "required": ["name", "team"],
            "additionalProperties": False,
        },
        "affected_systems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "component": {"type": ["string", "null"]},
                },
                "required": ["name", "component"],
                "additionalProperties": False,
            },
        },
        "actions_requested": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "deadline": {"type": ["string", "null"]},
                },
                "required": ["action", "owner", "deadline"],
                "additionalProperties": False,
            },
        },
        "constraints": {
            "type": "object",
            "properties": {
                "environment": {
                    "type": ["string", "null"],
                    "enum": ["prod", "staging", "dev", None],
                },
                "blocking": {"type": ["boolean", "null"]},
            },
            "required": ["environment", "blocking"],
            "additionalProperties": False,
        },
    },
    "required": [
        "ticket_id",
        "summary",
        "category",
        "priority",
        "requires_followup",
        "reporter",
        "affected_systems",
        "actions_requested",
        "constraints",
    ],
    "additionalProperties": False,
}

TICKET_SCHEMA_V1_1 = deepcopy(TICKET_SCHEMA_V1)
TICKET_SCHEMA_V1_1["properties"]["customer_impact"] = {"type": ["string", "null"]}
TICKET_SCHEMA_V1_1["required"] = TICKET_SCHEMA_V1["required"] + ["customer_impact"]

TICKET_SCHEMA_V1_REDUCED = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "category": {
            "type": "string",
            "enum": ["bug", "feature", "question", "incident", "task"],
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "urgent"],
        },
        "requires_followup": {"type": "boolean"},
        "affected_systems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "component": {"type": ["string", "null"]},
                },
                "required": ["name", "component"],
                "additionalProperties": False,
            },
        },
        "actions_requested": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "deadline": {"type": ["string", "null"]},
                },
                "required": ["action", "owner", "deadline"],
                "additionalProperties": False,
            },
        },
        "constraints": {
            "type": "object",
            "properties": {
                "environment": {
                    "type": ["string", "null"],
                    "enum": ["prod", "staging", "dev", None],
                },
                "blocking": {"type": ["boolean", "null"]},
            },
            "required": ["environment", "blocking"],
            "additionalProperties": False,
        },
    },
    "required": [
        "summary",
        "category",
        "priority",
        "requires_followup",
        "affected_systems",
        "actions_requested",
        "constraints",
    ],
    "additionalProperties": False,
}

TICKET_SCHEMA_V1_REDUCED_1_1 = deepcopy(TICKET_SCHEMA_V1_REDUCED)
TICKET_SCHEMA_V1_REDUCED_1_1["properties"]["customer_impact"] = {"type": ["string", "null"]}
TICKET_SCHEMA_V1_REDUCED_1_1["required"] = TICKET_SCHEMA_V1_REDUCED["required"] + ["customer_impact"]

TICKET_SCHEMA = TICKET_SCHEMA_V1
