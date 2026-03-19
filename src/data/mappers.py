"""Map external datasets into the project's canonical sample format."""

from __future__ import annotations

import re
from typing import Any

from src.data.complexity import infer_complexity_bucket


def map_console_ai_record(record: dict[str, Any]) -> dict[str, Any]:
    subject = normalize_text(record.get("subject"))
    description = normalize_text(record.get("description"))
    raw_category = normalize_text(record.get("category"))
    priority = normalize_priority(record.get("priority"))
    category = map_console_category(raw_category, subject, description)
    reporter_name = email_to_name(record.get("requesterEmail"))
    environment = infer_environment(subject, description)
    blocking = infer_blocking(subject, description, priority)

    affected_system_name = choose_affected_system_name(raw_category, subject)
    action_text = build_action_text(category, subject, description)

    target_json = {
        "ticket_id": value_or_none(record.get("id")),
        "summary": choose_summary(subject, description),
        "category": category,
        "priority": priority,
        "requires_followup": True,
        "reporter": {
            "name": reporter_name,
            "team": None,
        },
        "affected_systems": [
            {
                "name": affected_system_name,
                "component": raw_category.lower() if raw_category else None,
            }
        ],
        "actions_requested": [
            {
                "action": action_text,
                "owner": None,
                "deadline": None,
            }
        ],
        "constraints": {
            "environment": environment,
            "blocking": blocking,
        },
    }

    return build_project_record(
        sample_id=f"console-ai-{record['id']}",
        input_text=compose_input_text(subject, description),
        target_json=target_json,
        source_type="email",
        source_name="console_ai_it_helpdesk_tickets",
    )


def map_kameronb_record(record: dict[str, Any]) -> dict[str, Any]:
    short_description = normalize_text(record.get("short_description"))
    content = normalize_text(record.get("content"))
    raw_type = normalize_text(record.get("type"))
    raw_category = normalize_text(record.get("category"))
    raw_subcategory = normalize_text(record.get("subcategory"))
    issue_request = normalize_text(record.get("issue/request"))
    software_system = normalize_text(record.get("software/system"))
    assignment_group = normalize_text(record.get("assignment_group"))

    category = map_kameronb_category(raw_type, raw_category, raw_subcategory, short_description, content)
    priority = infer_priority_from_kameronb(raw_type, raw_subcategory, short_description, content)
    environment = infer_environment(short_description, content)
    blocking = infer_blocking(short_description, content, priority)

    component = raw_subcategory.lower() if raw_subcategory else None
    affected_name = choose_affected_system_name(software_system or raw_category, issue_request or short_description)
    action_text = build_action_text(category, issue_request or short_description, content)

    target_json = {
        "ticket_id": value_or_none(record.get("number")),
        "summary": choose_summary(short_description, issue_request, content),
        "category": category,
        "priority": priority,
        "requires_followup": True,
        "reporter": {
            "name": normalize_person_name(record.get("customer")),
            "team": assignment_group,
        },
        "affected_systems": [
            {
                "name": affected_name,
                "component": component,
            }
        ],
        "actions_requested": [
            {
                "action": action_text,
                "owner": None,
                "deadline": None,
            }
        ],
        "constraints": {
            "environment": environment,
            "blocking": blocking,
        },
    }

    source_type = map_contact_type(record.get("contact_type"))
    input_text = compose_input_text(short_description, content)
    return build_project_record(
        sample_id=f"kameronb-{record['number']}",
        input_text=input_text,
        target_json=target_json,
        source_type=source_type,
        source_name="kameronb_it_callcenter_tickets",
    )


def build_project_record(
    sample_id: str,
    input_text: str,
    target_json: dict[str, Any],
    source_type: str,
    source_name: str,
) -> dict[str, Any]:
    complexity_bucket = infer_complexity_bucket(target_json)
    return {
        "sample_id": sample_id,
        "task_name": "ticket_structured_output",
        "schema_name": "ticket_schema_v1",
        "complexity_bucket": complexity_bucket,
        "input_text": input_text,
        "target_json": target_json,
        "metadata": {
            "source_type": source_type,
            "is_synthetic": False,
            "raw_source": source_name,
        },
    }


def set_record_sample_id(record: dict[str, Any], sample_id: str) -> dict[str, Any]:
    updated = dict(record)
    updated["metadata"] = dict(record["metadata"])
    updated["sample_id"] = sample_id
    return updated


def choose_summary(*parts: str | None) -> str:
    for part in parts:
        text = normalize_text(part)
        if text:
            return truncate_text(text, 160)
    return "No summary provided."


def compose_input_text(subject: str | None, description: str | None) -> str:
    pieces = []
    if normalize_text(subject):
        pieces.append(f"Subject: {normalize_text(subject)}")
    if normalize_text(description):
        pieces.append(f"Description: {normalize_text(description)}")
    return "\n".join(pieces).strip()


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def value_or_none(value: Any) -> str | None:
    text = normalize_text(value)
    return text if text else None


def normalize_priority(value: Any) -> str:
    text = (normalize_text(value) or "").lower()
    mapping = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "urgent": "urgent",
        "critical": "urgent",
    }
    return mapping.get(text, "medium")


def map_console_category(raw_category: str | None, subject: str | None, description: str | None) -> str:
    text = " ".join(filter(None, [raw_category, subject, description])).lower()
    if any(keyword in text for keyword in ("question", "how to", "can you", "whether")):
        return "question"
    if any(keyword in text for keyword in ("outage", "down", "cannot access", "can't access")):
        return "incident"
    if any(keyword in text for keyword in ("request", "setup", "provision", "enable", "access")):
        return "task"
    if any(keyword in text for keyword in ("upgrade", "new feature", "support")):
        return "feature"
    if any(keyword in text for keyword in ("error", "issue", "crash", "disconnect", "fail")):
        return "bug"

    raw = (raw_category or "").lower()
    if raw in {"account", "security", "training"}:
        return "task"
    return "bug"


def map_kameronb_category(
    raw_type: str | None,
    raw_category: str | None,
    raw_subcategory: str | None,
    short_description: str | None,
    content: str | None,
) -> str:
    text = " ".join(
        filter(None, [raw_type, raw_category, raw_subcategory, short_description, content])
    ).lower()
    if "incident" in text and any(keyword in text for keyword in ("error", "failure", "crash", "malfunction")):
        return "incident"
    if any(keyword in text for keyword in ("upgrade", "installation", "request")):
        return "feature" if "upgrade" in text or "installation" in text else "task"
    if any(keyword in text for keyword in ("access", "activation", "deactivation", "renewal")):
        return "task"
    if any(keyword in text for keyword in ("question", "clarify")):
        return "question"
    if any(keyword in text for keyword in ("error", "issue", "malfunction", "compatibility", "bypass")):
        return "bug"
    return "task" if (raw_type or "").lower() == "request" else "bug"


def infer_priority_from_kameronb(
    raw_type: str | None,
    raw_subcategory: str | None,
    short_description: str | None,
    content: str | None,
) -> str:
    text = " ".join(filter(None, [raw_type, raw_subcategory, short_description, content])).lower()
    if any(keyword in text for keyword in ("outage", "critical", "blocking", "cannot", "can't")):
        return "urgent"
    if any(keyword in text for keyword in ("error", "failure", "crash", "malfunction")):
        return "high"
    if (raw_type or "").lower() == "request":
        return "medium"
    return "medium"


def infer_environment(*parts: str | None) -> str | None:
    text = " ".join(filter(None, parts)).lower()
    if "production" in text or "prod" in text:
        return "prod"
    if "staging" in text:
        return "staging"
    if "development" in text or re.search(r"\bdev\b", text):
        return "dev"
    return None


def infer_blocking(*parts: str | None) -> bool | None:
    text = " ".join(filter(None, [str(part) if part is not None else None for part in parts])).lower()
    if any(keyword in text for keyword in ("blocking", "outage", "cannot access", "can't access", "down")):
        return True
    if any(keyword in text for keyword in ("low", "medium")):
        return None
    return None


def choose_affected_system_name(primary_hint: str | None, fallback_text: str | None) -> str:
    primary = normalize_text(primary_hint)
    if primary:
        return primary
    fallback = normalize_text(fallback_text)
    if fallback:
        return truncate_text(fallback, 80)
    return "unknown_system"


def build_action_text(category: str, title: str | None, description: str | None) -> str:
    anchor = normalize_text(title) or normalize_text(description) or "the reported issue"
    if category == "question":
        return f"Answer and clarify: {truncate_text(anchor, 80)}"
    if category == "feature":
        return f"Review and plan request: {truncate_text(anchor, 80)}"
    if category == "incident":
        return f"Investigate and mitigate incident: {truncate_text(anchor, 80)}"
    if category == "task":
        return f"Handle request: {truncate_text(anchor, 80)}"
    return f"Investigate issue: {truncate_text(anchor, 80)}"


def email_to_name(email: Any) -> str | None:
    text = normalize_text(email)
    if not text or "@" not in text:
        return None
    local = text.split("@", 1)[0]
    local = local.replace(".", " ").replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in local.split()) or None


def normalize_person_name(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    if "," in text:
        last, first = [item.strip() for item in text.split(",", 1)]
        text = f"{first} {last}".strip()
    return text


def map_contact_type(value: Any) -> str:
    text = (normalize_text(value) or "").lower()
    if text in {"email"}:
        return "email"
    if text in {"chat", "phone", "self-service"}:
        return "task"
    return "task"

