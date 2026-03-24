from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
from pathlib import Path
import json
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_jsonl

INTERNAL_TRAIN = PROJECT_ROOT / "data" / "reduced" / "phase1_train_reduced.jsonl"
EXTERNAL_TRAIN = PROJECT_ROOT / "data" / "external_generalization" / "gorkemsevinc_customer_support_tickets_train_reduced.jsonl"
EXTERNAL_TEST = PROJECT_ROOT / "data" / "external_generalization" / "gorkemsevinc_customer_support_tickets_test_reduced.jsonl"

OUTPUT_MD = PROJECT_ROOT / "docs" / "results" / "internal_external_mismatch_audit.md"
OUTPUT_JSON = PROJECT_ROOT / "docs" / "results" / "internal_external_mismatch_audit.json"


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def action_template(action: str | None) -> str | None:
    if not action:
        return None
    prefix = action.split(":", 1)[0].strip().lower()
    return prefix.replace(" ", "_")


def normalized_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 1 or len(counter) <= 1:
        return 0.0
    probs = [count / total for count in counter.values()]
    entropy = -sum(p * log2(p) for p in probs if p > 0)
    return entropy / log2(len(counter))


def extract_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        target = record["target_json"]
        system = (target.get("affected_systems") or [{}])[0]
        action_obj = (target.get("actions_requested") or [{}])[0]
        row = {
            "sample_id": record["sample_id"],
            "summary": target.get("summary"),
            "category": target.get("category"),
            "priority": target.get("priority"),
            "requires_followup": target.get("requires_followup"),
            "name": system.get("name"),
            "component": system.get("component"),
            "action": action_obj.get("action"),
            "action_template": action_template(action_obj.get("action")),
            "blocking": (target.get("constraints") or {}).get("blocking"),
            "environment": (target.get("constraints") or {}).get("environment"),
            "raw_source": record.get("metadata", {}).get("raw_source"),
        }
        rows.append(row)
    return rows


def label_counter(rows: list[dict], field: str) -> Counter[str]:
    return Counter(str(row[field]) for row in rows if row.get(field) is not None)


def top_items(counter: Counter[str], k: int = 10) -> list[dict]:
    return [{"label": label, "count": count} for label, count in counter.most_common(k)]


def overlap_stats(internal: Counter[str], external: Counter[str]) -> dict:
    internal_labels = set(internal)
    external_labels = set(external)
    overlap = internal_labels & external_labels
    external_only = external_labels - internal_labels
    external_total = sum(external.values())
    external_overlap_count = sum(external[label] for label in overlap)
    external_only_count = sum(external[label] for label in external_only)
    return {
        "internal_vocab": len(internal_labels),
        "external_vocab": len(external_labels),
        "overlap_vocab": len(overlap),
        "external_only_vocab": len(external_only),
        "external_overlap_mass": safe_div(external_overlap_count, external_total),
        "external_only_mass": safe_div(external_only_count, external_total),
        "external_only_top": top_items(Counter({label: external[label] for label in external_only}), 12),
    }


def mapping_purity(rows: list[dict], key_field: str, value_field: str, min_count: int = 3) -> dict:
    mapping: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = row.get(key_field)
        value = row.get(value_field)
        if key is None or value is None:
            continue
        mapping[str(key)][str(value)] += 1

    eligible = {key: counter for key, counter in mapping.items() if sum(counter.values()) >= min_count}
    if not eligible:
        return {
            "eligible_keys": 0,
            "weighted_purity": 0.0,
            "median_key_purity": 0.0,
            "top_ambiguous": [],
        }

    total = 0
    correct = 0
    purities: list[float] = []
    ambiguity_rows: list[tuple[float, int, str, list[dict]]] = []
    for key, counter in eligible.items():
        key_total = sum(counter.values())
        key_top = counter.most_common(1)[0][1]
        purity = key_top / key_total
        purities.append(purity)
        total += key_total
        correct += key_top
        ambiguity_rows.append(
            (
                purity,
                key_total,
                key,
                [{"label": label, "count": count} for label, count in counter.most_common(4)],
            )
        )

    ambiguity_rows.sort(key=lambda item: (item[0], -item[1], item[2]))
    return {
        "eligible_keys": len(eligible),
        "weighted_purity": safe_div(correct, total),
        "median_key_purity": statistics.median(purities),
        "top_ambiguous": [
            {
                "key": key,
                "purity": purity,
                "support": support,
                "labels": labels,
            }
            for purity, support, key, labels in ambiguity_rows[:12]
        ],
    }


def source_mix(rows: list[dict]) -> list[dict]:
    counter = Counter(str(row["raw_source"]) for row in rows if row.get("raw_source") is not None)
    total = sum(counter.values())
    return [
        {"source": source, "count": count, "share": safe_div(count, total)}
        for source, count in counter.most_common()
    ]


def render_overlap_block(field: str, stats: dict) -> list[str]:
    lines = [
        f"### {field}",
        "",
        f"- internal vocab: `{stats['internal_vocab']}`",
        f"- external vocab: `{stats['external_vocab']}`",
        f"- overlap vocab: `{stats['overlap_vocab']}`",
        f"- external-only vocab: `{stats['external_only_vocab']}`",
        f"- external mass on overlapping labels: `{stats['external_overlap_mass']:.4f}`",
        f"- external mass on external-only labels: `{stats['external_only_mass']:.4f}`",
    ]
    if stats["external_only_top"]:
        lines.extend(["", "Top external-only labels:"])
        for item in stats["external_only_top"]:
            lines.append(f"- `{item['label']}`: `{item['count']}`")
    lines.append("")
    return lines


def render_mapping_block(title: str, stats: dict) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"- eligible keys: `{stats['eligible_keys']}`",
        f"- weighted purity: `{stats['weighted_purity']:.4f}`",
        f"- median key purity: `{stats['median_key_purity']:.4f}`",
    ]
    if stats["top_ambiguous"]:
        lines.extend(["", "Most ambiguous keys:"])
        for item in stats["top_ambiguous"][:8]:
            labels = ", ".join(f"{entry['label']} ({entry['count']})" for entry in item["labels"])
            lines.append(
                f"- `{item['key']}`: purity `{item['purity']:.3f}`, support `{item['support']}`, labels `{labels}`"
            )
    lines.append("")
    return lines


def build_summary(internal_rows: list[dict], external_train_rows: list[dict], external_test_rows: list[dict]) -> dict:
    fields = ["category", "priority", "component", "action_template", "name"]
    internal_counters = {field: label_counter(internal_rows, field) for field in fields}
    external_train_counters = {field: label_counter(external_train_rows, field) for field in fields}
    external_test_counters = {field: label_counter(external_test_rows, field) for field in fields}

    summary = {
        "datasets": {
            "internal_train": {
                "num_rows": len(internal_rows),
                "source_mix": source_mix(internal_rows),
                "category_entropy": normalized_entropy(internal_counters["category"]),
                "priority_entropy": normalized_entropy(internal_counters["priority"]),
                "component_entropy": normalized_entropy(internal_counters["component"]),
            },
            "external_train": {
                "num_rows": len(external_train_rows),
                "source_mix": source_mix(external_train_rows),
                "category_entropy": normalized_entropy(external_train_counters["category"]),
                "priority_entropy": normalized_entropy(external_train_counters["priority"]),
                "component_entropy": normalized_entropy(external_train_counters["component"]),
            },
            "external_test": {
                "num_rows": len(external_test_rows),
                "category_entropy": normalized_entropy(external_test_counters["category"]),
                "priority_entropy": normalized_entropy(external_test_counters["priority"]),
                "component_entropy": normalized_entropy(external_test_counters["component"]),
            },
        },
        "overlap": {
            field: overlap_stats(internal_counters[field], external_train_counters[field])
            for field in ["category", "priority", "component", "action_template", "name"]
        },
        "top_labels": {
            "internal": {field: top_items(internal_counters[field], 12) for field in ["category", "priority", "component"]},
            "external_train": {field: top_items(external_train_counters[field], 12) for field in ["category", "priority", "component"]},
        },
        "mapping_purity": {
            "internal_name_to_component": mapping_purity(internal_rows, "name", "component", min_count=3),
            "external_name_to_component": mapping_purity(external_train_rows, "name", "component", min_count=3),
            "internal_summary_to_category": mapping_purity(internal_rows, "summary", "category", min_count=3),
            "external_summary_to_category": mapping_purity(external_train_rows, "summary", "category", min_count=3),
        },
    }
    return summary


def build_markdown(summary: dict) -> str:
    internal = summary["datasets"]["internal_train"]
    external_train = summary["datasets"]["external_train"]
    external_test = summary["datasets"]["external_test"]
    lines: list[str] = [
        "# Internal vs External Mismatch Audit",
        "",
        "## Purpose",
        "",
        "This audit explains why mapped external performance plateaus even after external few-shot adaptation restores schema completeness.",
        "",
        "It focuses on label-space mismatch, field entropy, and mapping purity differences between the in-domain reduced training set and the mapped external customer-support dataset.",
        "",
        "## Dataset Snapshot",
        "",
        f"- internal train rows: `{internal['num_rows']}`",
        f"- external train rows: `{external_train['num_rows']}`",
        f"- external test rows: `{external_test['num_rows']}`",
        "",
        "Internal source mix:",
    ]
    for item in internal["source_mix"]:
        lines.append(f"- `{item['source']}`: `{item['count']}` (`{item['share']:.3f}`)")

    lines.extend(
        [
            "",
            "Entropy comparison:",
            f"- category entropy: internal `{internal['category_entropy']:.4f}` vs external train `{external_train['category_entropy']:.4f}`",
            f"- priority entropy: internal `{internal['priority_entropy']:.4f}` vs external train `{external_train['priority_entropy']:.4f}`",
            f"- component entropy: internal `{internal['component_entropy']:.4f}` vs external train `{external_train['component_entropy']:.4f}`",
            "",
            "## Label-Space Overlap",
            "",
        ]
    )

    for field in ["category", "priority", "component", "action_template", "name"]:
        lines.extend(render_overlap_block(field, summary["overlap"][field]))

    lines.extend(
        [
            "## Top Label Distribution",
            "",
            "### Internal train",
            "",
        ]
    )
    for field in ["category", "priority", "component"]:
        lines.append(f"- `{field}`: " + ", ".join(f"`{item['label']}` ({item['count']})" for item in summary["top_labels"]["internal"][field][:8]))

    lines.extend(["", "### External train", ""])
    for field in ["category", "priority", "component"]:
        lines.append(f"- `{field}`: " + ", ".join(f"`{item['label']}` ({item['count']})" for item in summary["top_labels"]["external_train"][field][:8]))

    lines.extend(["", "## Mapping Purity", ""])
    lines.extend(render_mapping_block("Internal `name -> component`", summary["mapping_purity"]["internal_name_to_component"]))
    lines.extend(render_mapping_block("External `name -> component`", summary["mapping_purity"]["external_name_to_component"]))
    lines.extend(render_mapping_block("Internal `summary -> category`", summary["mapping_purity"]["internal_summary_to_category"]))
    lines.extend(render_mapping_block("External `summary -> category`", summary["mapping_purity"]["external_summary_to_category"]))

    lines.extend(
        [
            "## Main Conclusions",
            "",
            "- `category` and `priority` mostly share the same coarse label space across domains; the problem is not raw label absence but changed conditional semantics.",
            "- `component` is the main taxonomy mismatch field: external carries substantial mass on labels and label usages that are weakly supported or differently structured relative to the in-domain data.",
            "- `action_template` mismatch is downstream of `category`; when category semantics shift, canonical action also shifts with it.",
            "- `name -> component` mappings are much less reusable externally, which explains why in-domain deterministic `component <- name` rules do not transfer cleanly.",
            "- external adaptation solves completeness, but the remaining plateau is best explained by conditional label mismatch rather than by formatting or missing fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    internal_rows = extract_rows(load_jsonl(INTERNAL_TRAIN))
    external_train_rows = extract_rows(load_jsonl(EXTERNAL_TRAIN))
    external_test_rows = extract_rows(load_jsonl(EXTERNAL_TEST))

    summary = build_summary(internal_rows, external_train_rows, external_test_rows)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(summary), encoding="utf-8")

    print(f"internal_train = {INTERNAL_TRAIN}")
    print(f"external_train = {EXTERNAL_TRAIN}")
    print(f"external_test = {EXTERNAL_TEST}")
    print(f"summary_json = {OUTPUT_JSON}")
    print(f"summary_md = {OUTPUT_MD}")


if __name__ == "__main__":
    main()
