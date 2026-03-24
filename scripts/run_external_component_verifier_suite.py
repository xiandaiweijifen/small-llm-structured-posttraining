from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.evaluation.field_analysis import analyze_field_errors
from src.evaluation.metrics import SampleEvaluation, evaluate_sample, summarize_results
from src.evaluation.reporting import group_sample_results, write_json_report
from src.schemas.registry import get_schema

SCHEMA_NAME = "ticket_schema_v1_reduced"
SOURCE_EXPERIMENT = "qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external_generalization"
SOURCE_NAME = "gorkemsevinc_customer_support_tickets"
EXTERNAL_TRAIN = EXTERNAL_DIR / f"{SOURCE_NAME}_train_reduced.jsonl"
EXTERNAL_TEST = EXTERNAL_DIR / f"{SOURCE_NAME}_test_reduced.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "docs" / "results" / "external_component_verifier_batch_summary.md"

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

RUN_PRESETS = [
    "guarded_name_majority_p80",
    "component_nb_text_name",
    "component_nb_text_name_pred",
    "hybrid_guarded_or_nb",
    "hybrid_guarded_vote_nb",
]


def tokenize(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.lower())


def build_output_paths(experiment_name: str) -> dict[str, Path]:
    return {
        "prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test.jsonl",
        "repaired_prediction_path": PROJECT_ROOT / "results" / "predictions" / f"{experiment_name}_test_repaired.jsonl",
        "raw_report_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_report.json",
        "repaired_report_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_report.json",
        "raw_field_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_field_analysis.json",
        "repaired_field_path": PROJECT_ROOT / "results" / "metrics" / f"{experiment_name}_test_repaired_field_analysis.json",
    }


def component_value(target_json: dict) -> str | None:
    systems = target_json.get("affected_systems")
    if isinstance(systems, list) and systems and isinstance(systems[0], dict):
        value = systems[0].get("component")
        if isinstance(value, str):
            return value
    return None


def name_value(obj: dict) -> str | None:
    systems = obj.get("affected_systems")
    if isinstance(systems, list) and systems and isinstance(systems[0], dict):
        value = systems[0].get("name")
        if isinstance(value, str):
            return value
    return None


def set_component(obj: dict, component: str) -> None:
    systems = obj.get("affected_systems")
    if isinstance(systems, list) and systems and isinstance(systems[0], dict):
        systems[0]["component"] = component


def feature_tokens(record_or_prediction: dict, include_pred: bool) -> list[str]:
    if "target_json" in record_or_prediction:
        target = record_or_prediction["target_json"]
        input_text = record_or_prediction.get("input_text")
        name = name_value(target)
        category = target.get("category")
        priority = target.get("priority")
    else:
        target = record_or_prediction
        input_text = None
        name = name_value(target)
        category = target.get("category")
        priority = target.get("priority")

    tokens: list[str] = []
    tokens.extend(tokenize(input_text))
    tokens.extend(f"name_{token}" for token in tokenize(name))
    tokens.extend(f"summary_{token}" for token in tokenize(target.get("summary")))
    if include_pred and isinstance(category, str):
        tokens.append(f"predcat_{category}")
    if include_pred and isinstance(priority, str):
        tokens.append(f"predprio_{priority}")
    return tokens


def build_name_majority_map(records: list[dict], min_purity: float, min_support: int = 3) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        name = name_value(record["target_json"])
        component = component_value(record["target_json"])
        if name and component:
            counts[name][component] += 1

    mapping: dict[str, str] = {}
    for name, counter in counts.items():
        support = sum(counter.values())
        component, top = counter.most_common(1)[0]
        purity = top / support
        if support >= min_support and purity >= min_purity:
            mapping[name] = component
    return mapping


class MultinomialNBVerifier:
    def __init__(self, records: list[dict], include_pred: bool, alpha: float = 0.5) -> None:
        self.include_pred = include_pred
        self.alpha = alpha
        self.label_doc_counts: Counter[str] = Counter()
        self.label_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.label_total_tokens: Counter[str] = Counter()
        self.vocab: set[str] = set()

        for record in records:
            label = component_value(record["target_json"])
            if label is None:
                continue
            self.label_doc_counts[label] += 1
            tokens = feature_tokens(record, include_pred=include_pred)
            self.vocab.update(tokens)
            self.label_token_counts[label].update(tokens)
            self.label_total_tokens[label] += len(tokens)

        self.labels = sorted(self.label_doc_counts)
        self.total_docs = sum(self.label_doc_counts.values())
        self.vocab_size = max(len(self.vocab), 1)

    def predict(self, prediction_json: dict) -> str | None:
        if not self.labels:
            return None
        tokens = feature_tokens(prediction_json, include_pred=self.include_pred)
        best_label = None
        best_score = -float("inf")
        for label in self.labels:
            prior = math.log(self.label_doc_counts[label] / self.total_docs)
            denom = self.label_total_tokens[label] + self.alpha * self.vocab_size
            score = prior
            counter = self.label_token_counts[label]
            for token in tokens:
                score += math.log((counter.get(token, 0) + self.alpha) / denom)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label


def apply_variant(prediction_json: dict | None, preset_name: str, strict_map: dict[str, str], nb_plain: MultinomialNBVerifier, nb_pred: MultinomialNBVerifier) -> dict | None:
    if not isinstance(prediction_json, dict):
        return prediction_json

    updated = deepcopy(prediction_json)
    name = name_value(updated)
    strict_component = strict_map.get(name) if isinstance(name, str) else None
    nb_plain_component = nb_plain.predict(updated)
    nb_pred_component = nb_pred.predict(updated)

    if preset_name == "guarded_name_majority_p80":
        if strict_component is not None:
            set_component(updated, strict_component)
    elif preset_name == "component_nb_text_name":
        if nb_plain_component is not None:
            set_component(updated, nb_plain_component)
    elif preset_name == "component_nb_text_name_pred":
        if nb_pred_component is not None:
            set_component(updated, nb_pred_component)
    elif preset_name == "hybrid_guarded_or_nb":
        if strict_component is not None:
            set_component(updated, strict_component)
        elif nb_pred_component is not None:
            set_component(updated, nb_pred_component)
    elif preset_name == "hybrid_guarded_vote_nb":
        candidates = [value for value in [strict_component, nb_plain_component, nb_pred_component] if value is not None]
        if candidates:
            component = Counter(candidates).most_common(1)[0][0]
            set_component(updated, component)
    return updated


def canonicalize_record(record: dict) -> dict:
    return deepcopy(record)


def evaluate_prediction_records(gold_records: list[dict], pred_records: list[dict], schema: dict) -> dict:
    by_id = {record["sample_id"]: record for record in pred_records}
    sample_results: list[dict] = []
    for gold in gold_records:
        pred = by_id.get(gold["sample_id"], {})
        sample_eval = evaluate_sample(
            sample_id=gold["sample_id"],
            prediction_text=pred.get("prediction_text"),
            prediction_json=pred.get("prediction_json"),
            target_json=gold["target_json"],
            schema=schema,
        )
        sample_results.append(
            {
                **sample_eval.__dict__,
                "schema_name": gold["schema_name"],
                "complexity_bucket": gold.get("complexity_bucket", "unknown"),
            }
        )

    report = {
        "summary": summarize_results([SampleEvaluation(**{k: item[k] for k in SampleEvaluation.__dataclass_fields__}) for item in sample_results]),
        "grouped_summary": {
            "by_complexity_bucket": {
                name: summarize_results([SampleEvaluation(**{k: item[k] for k in SampleEvaluation.__dataclass_fields__}) for item in items])
                for name, items in group_sample_results(sample_results, "complexity_bucket").items()
            }
        },
        "per_sample": sample_results,
    }
    return report


def write_summary(batch_results: list[dict]) -> None:
    lines = [
        "# External Component Verifier Batch Summary",
        "",
        f"- source experiment: `{SOURCE_EXPERIMENT}`",
        f"- external train: `{EXTERNAL_TRAIN}`",
        f"- external test: `{EXTERNAL_TEST}`",
        "",
        "| Preset | Field Exact Match | End-to-End Exact Match | Main Note |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in batch_results:
        summary = item["raw_summary"]
        lines.append(
            f"| {item['preset_name']} | {summary['field_exact_match']:.4f} | {summary['end_to_end_exact_match']:.4f} | {item['note']} |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    print("project_root =", PROJECT_ROOT)
    print("source_experiment =", SOURCE_EXPERIMENT)
    print("scheduled_presets =", RUN_PRESETS)

    schema = get_schema(SCHEMA_NAME)
    train_records = load_jsonl(EXTERNAL_TRAIN)
    test_records = [canonicalize_record(record) for record in load_jsonl(EXTERNAL_TEST)]
    source_predictions = load_jsonl(PROJECT_ROOT / "results" / "predictions" / f"{SOURCE_EXPERIMENT}_test.jsonl")

    strict_map = build_name_majority_map(train_records, min_purity=0.8, min_support=3)
    nb_plain = MultinomialNBVerifier(train_records, include_pred=False)
    nb_pred = MultinomialNBVerifier(train_records, include_pred=True)

    print("guarded_name_majority_keys =", len(strict_map))
    print("test_samples =", len(test_records))

    batch_results: list[dict] = []
    for preset_name in RUN_PRESETS:
        experiment_name = f"qwen25_3b_stage18_{preset_name}"
        paths = build_output_paths(experiment_name)
        print("\n" + "=" * 80)
        print("running preset =", preset_name)
        print("=" * 80)

        predictions = []
        for record in source_predictions:
            prediction_json = apply_variant(record.get("prediction_json"), preset_name, strict_map, nb_plain, nb_pred)
            predictions.append(
                {
                    **record,
                    "prediction_json": prediction_json,
                    "metadata": {
                        **record.get("metadata", {}),
                        "source_experiment": SOURCE_EXPERIMENT,
                        "verifier_preset": preset_name,
                    },
                }
            )
        dump_jsonl(paths["prediction_path"], predictions)
        raw_report = evaluate_prediction_records(test_records, predictions, schema)
        write_json_report(paths["raw_report_path"], raw_report)
        write_json_report(paths["raw_field_path"], analyze_field_errors(test_records, predictions))
        dump_jsonl(paths["repaired_prediction_path"], predictions)
        write_json_report(paths["repaired_report_path"], raw_report)
        write_json_report(paths["repaired_field_path"], analyze_field_errors(test_records, predictions))

        note = {
            "guarded_name_majority_p80": "high-purity external `name -> component` only",
            "component_nb_text_name": "NB verifier on input text, summary, and name tokens",
            "component_nb_text_name_pred": "NB verifier plus predicted category and priority features",
            "hybrid_guarded_or_nb": "use guarded majority first, then NB fallback",
            "hybrid_guarded_vote_nb": "majority vote over guarded mapping and both NB variants",
        }[preset_name]

        batch_results.append(
            {
                "preset_name": preset_name,
                "experiment_name": experiment_name,
                "raw_summary": raw_report["summary"],
                "note": note,
            }
        )
        print(json.dumps(batch_results[-1], indent=2))

    write_summary(batch_results)
    print("summary_path =", SUMMARY_PATH)


if __name__ == "__main__":
    run()
