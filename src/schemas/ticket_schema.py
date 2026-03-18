"""Phase-1 schema definition."""

TICKET_SCHEMA = {
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
