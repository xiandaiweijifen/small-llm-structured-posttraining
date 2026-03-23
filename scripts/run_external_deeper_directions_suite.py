from __future__ import annotations

import copy
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import dump_jsonl, load_jsonl
from src.schemas.registry import get_schema

spec = importlib.util.spec_from_file_location('external_frontier_common', PROJECT_ROOT / 'scripts' / 'run_external_frontier_suite.py')
common = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(common)

BASE_MODEL_NAME = common.BASE_MODEL_NAME
SCHEMA_NAME = common.SCHEMA_NAME
STAGE7_CHECKPOINT_DIR = common.BASELINE_CHECKPOINT_DIR
STAGE14_CHECKPOINT_DIR = PROJECT_ROOT / 'results' / 'checkpoints' / 'qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5'
EXTERNAL_TRAIN = common.EXTERNAL_TRAIN
EXTERNAL_VAL = common.EXTERNAL_VAL
EXTERNAL_TEST = common.EXTERNAL_TEST
ARTIFACT_DIR = PROJECT_ROOT / 'data' / 'stage17_external_deeper_directions'
SUMMARY_PATH = PROJECT_ROOT / 'docs' / 'results' / 'external_deeper_directions_batch_summary.md'
SKIP_COMPLETED = True
INFERENCE_BATCH_SIZE = 32
IMPORTANT_FIELDS = [
    'actions_requested[0].action',
    'affected_systems[0].component',
    'category',
    'priority',
]

TARGET_REDESIGN_PRESETS = {
    'redesign2048_c80_cat60_epoch3_lr1e4': {
        'experiment_name': 'qwen25_3b_stage17_redesign2048_c80_cat60_epoch3_lr1e4',
        'sample_count': 2048,
        'component_threshold': 0.8,
        'category_threshold': 0.6,
        'learning_rate': 1e-4,
        'epochs': 3,
        'batch_size': 16,
        'grad_accum': 4,
        'seed': 42,
    },
    'redesign4096_c80_cat60_epoch2_lr1e4': {
        'experiment_name': 'qwen25_3b_stage17_redesign4096_c80_cat60_epoch2_lr1e4',
        'sample_count': 4096,
        'component_threshold': 0.8,
        'category_threshold': 0.6,
        'learning_rate': 1e-4,
        'epochs': 2,
        'batch_size': 16,
        'grad_accum': 4,
        'seed': 42,
    },
    'redesignfull_c80_cat60_epoch2_lr5e5': {
        'experiment_name': 'qwen25_3b_stage17_redesignfull_c80_cat60_epoch2_lr5e5',
        'sample_count': None,
        'component_threshold': 0.8,
        'category_threshold': 0.6,
        'learning_rate': 5e-5,
        'epochs': 2,
        'batch_size': 16,
        'grad_accum': 4,
        'seed': 42,
    },
    'redesignfull_c90_cat70_epoch2_lr5e5': {
        'experiment_name': 'qwen25_3b_stage17_redesignfull_c90_cat70_epoch2_lr5e5',
        'sample_count': None,
        'component_threshold': 0.9,
        'category_threshold': 0.7,
        'learning_rate': 5e-5,
        'epochs': 2,
        'batch_size': 16,
        'grad_accum': 4,
        'seed': 42,
    },
}

RESIDUAL_PRESETS = {
    'residual_component_only_x8_epoch1_lr5e5': {
        'experiment_name': 'qwen25_3b_stage17_residual_component_only_x8_epoch1_lr5e5',
        'subset_name': 'component_only',
        'repeat': 8,
        'learning_rate': 5e-5,
        'epochs': 1,
        'batch_size': 14,
        'grad_accum': 4,
        'seed': 42,
    },
    'residual_priority_only_x4_epoch1_lr5e5': {
        'experiment_name': 'qwen25_3b_stage17_residual_priority_only_x4_epoch1_lr5e5',
        'subset_name': 'priority_only',
        'repeat': 4,
        'learning_rate': 5e-5,
        'epochs': 1,
        'batch_size': 14,
        'grad_accum': 4,
        'seed': 42,
    },
    'residual_component_focused_x2_epoch1_lr5e5': {
        'experiment_name': 'qwen25_3b_stage17_residual_component_focused_x2_epoch1_lr5e5',
        'subset_name': 'component_focused',
        'repeat': 2,
        'learning_rate': 5e-5,
        'epochs': 1,
        'batch_size': 14,
        'grad_accum': 4,
        'seed': 42,
    },
    'residual_component_priority_x2_epoch1_lr5e5': {
        'experiment_name': 'qwen25_3b_stage17_residual_component_priority_x2_epoch1_lr5e5',
        'subset_name': 'component_priority_residual',
        'repeat': 2,
        'learning_rate': 5e-5,
        'epochs': 1,
        'batch_size': 14,
        'grad_accum': 4,
        'seed': 42,
    },
    'residual_component_only_x8_epoch1_lr3e5': {
        'experiment_name': 'qwen25_3b_stage17_residual_component_only_x8_epoch1_lr3e5',
        'subset_name': 'component_only',
        'repeat': 8,
        'learning_rate': 3e-5,
        'epochs': 1,
        'batch_size': 14,
        'grad_accum': 4,
        'seed': 42,
    },
}


def extract_field(obj: dict | None, field_name: str):
    if obj is None:
        return None
    current = obj
    for part in field_name.split('.'):
        if '[' in part and part.endswith(']'):
            key, index_text = part[:-1].split('[')
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if not isinstance(current, list):
                return None
            index = int(index_text)
            if index >= len(current):
                return None
            current = current[index]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
    return current


def build_confident_maps(train_records: list[dict], component_threshold: float, category_threshold: float):
    component_counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    category_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for record in train_records:
        target = record['target_json']
        summary = target['summary']
        name = target['affected_systems'][0]['name']
        category = target['category']
        component = target['affected_systems'][0]['component']
        component_counts[(summary, name, category)][component] += 1
        category_counts[(summary, name)][category] += 1

    component_map = {}
    for key, counter in component_counts.items():
        value, count = counter.most_common(1)[0]
        ratio = count / sum(counter.values())
        if ratio >= component_threshold:
            component_map[key] = value

    category_map = {}
    for key, counter in category_counts.items():
        value, count = counter.most_common(1)[0]
        ratio = count / sum(counter.values())
        if ratio >= category_threshold:
            category_map[key] = value
    return component_map, category_map


def canonicalize_record(record: dict, component_map: dict, category_map: dict) -> tuple[dict, dict[str, bool]]:
    updated = copy.deepcopy(record)
    target = updated['target_json']
    summary = target['summary']
    name = target['affected_systems'][0]['name']
    category_key = (summary, name)
    changed = {'category': False, 'component': False, 'action': False}

    new_category = category_map.get(category_key)
    if new_category is not None and new_category != target['category']:
        target['category'] = new_category
        changed['category'] = True

    component_key = (summary, name, target['category'])
    new_component = component_map.get(component_key)
    if new_component is not None and new_component != target['affected_systems'][0]['component']:
        target['affected_systems'][0]['component'] = new_component
        changed['component'] = True

    common.refresh_action_from_category(target)
    if target['actions_requested'][0]['action'] != record['target_json']['actions_requested'][0]['action']:
        changed['action'] = True
    return updated, changed


def build_redesigned_subset(records: list[dict], config: dict) -> tuple[list[dict], dict]:
    subset = common.build_train_subset(records, config['sample_count'], int(config['seed']))
    component_map, category_map = build_confident_maps(subset, float(config['component_threshold']), float(config['category_threshold']))
    redesigned = []
    change_counts = Counter()
    for record in subset:
        updated, changed = canonicalize_record(record, component_map, category_map)
        redesigned.append(updated)
        for name, is_changed in changed.items():
            if is_changed:
                change_counts[name] += 1
    summary = {
        'input_count': len(subset),
        'component_map_size': len(component_map),
        'category_map_size': len(category_map),
        'change_counts': dict(change_counts),
    }
    return redesigned, summary


def load_or_create_stage14_train_predictions(train_records: list[dict]) -> list[dict]:
    cached_path = PROJECT_ROOT / 'data' / 'stage15_external_overnight_optimization' / 'external_train_predictions.jsonl'
    if cached_path.exists():
        return load_jsonl(cached_path)
    tokenizer = common.load_tokenizer()
    model = common.load_trainable_adapter_model(STAGE14_CHECKPOINT_DIR)
    try:
        predictions = common.generate_predictions(model, tokenizer, train_records)
        dump_jsonl(cached_path, predictions)
        return predictions
    finally:
        common.cleanup_model(model, tokenizer)


def build_residual_subsets(train_records: list[dict], predictions: list[dict]) -> dict[str, list[dict]]:
    by_id = {record['sample_id']: record for record in train_records}
    pred_by_id = {record['sample_id']: record for record in predictions}
    subsets = {
        'component_only': [],
        'priority_only': [],
        'component_focused': [],
        'component_priority_residual': [],
    }
    for sample_id, record in by_id.items():
        pred_json = pred_by_id[sample_id].get('prediction_json')
        mismatches = {
            field
            for field in IMPORTANT_FIELDS
            if extract_field(record['target_json'], field) != extract_field(pred_json, field)
        }
        if mismatches == {'affected_systems[0].component'}:
            subsets['component_only'].append(record)
        if mismatches == {'priority'}:
            subsets['priority_only'].append(record)
        if 'affected_systems[0].component' in mismatches and len(mismatches) <= 2:
            subsets['component_focused'].append(record)
        if mismatches and mismatches.issubset({'affected_systems[0].component', 'priority'}) and len(mismatches) <= 2:
            subsets['component_priority_residual'].append(record)
    return subsets


def train_and_eval(experiment_name: str, train_records: list[dict], config: dict, checkpoint_dir: Path, val_records: list[dict], test_records: list[dict], schema: dict):
    paths = common.build_output_paths(experiment_name)
    if SKIP_COMPLETED and common.outputs_complete(paths):
        raw_report = json.loads(paths['raw_report_path'].read_text(encoding='utf-8'))
        repaired_report = json.loads(paths['repaired_report_path'].read_text(encoding='utf-8'))
        return raw_report['summary'], repaired_report['summary'], 'skipped_existing'

    train_sft_path = ARTIFACT_DIR / f'{experiment_name}_train.jsonl'
    val_sft_path = ARTIFACT_DIR / 'external_val_sft.jsonl'
    common.write_sft_split(train_records, train_sft_path)
    common.write_sft_split(val_records, val_sft_path)

    tokenizer = common.load_tokenizer()
    model = common.load_trainable_adapter_model(checkpoint_dir)
    trainer = None
    try:
        dataset = common.load_chat_dataset(train_sft_path, val_sft_path, tokenizer)
        output_root = paths['checkpoint_dir']
        output_root.mkdir(parents=True, exist_ok=True)
        trainer = common.build_trainer(model, dataset, tokenizer, config, output_root)
        trainer.train()
        trainer.save_model(str(output_root))
        predictions = common.generate_predictions(model, tokenizer, test_records)
        raw_report, repaired_report = common.evaluate_and_write(test_records, predictions, schema, paths)
        return raw_report['summary'], repaired_report['summary'], 'completed'
    finally:
        common.cleanup_model(trainer, model, tokenizer)


def run():
    print('project_root =', PROJECT_ROOT)
    print('base_model =', BASE_MODEL_NAME)
    print('stage7_checkpoint =', STAGE7_CHECKPOINT_DIR)
    print('stage14_checkpoint =', STAGE14_CHECKPOINT_DIR)
    print('target_redesign_presets =', list(TARGET_REDESIGN_PRESETS))
    print('residual_presets =', list(RESIDUAL_PRESETS))
    print('skip_completed =', SKIP_COMPLETED)
    print('inference_batch_size =', INFERENCE_BATCH_SIZE)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    schema = get_schema(SCHEMA_NAME)
    external_train = load_jsonl(EXTERNAL_TRAIN)
    external_val = load_jsonl(EXTERNAL_VAL)
    external_test = load_jsonl(EXTERNAL_TEST)

    redesign_results = []
    for preset_name, config in TARGET_REDESIGN_PRESETS.items():
        print('\n' + '=' * 80)
        print('running target redesign preset =', preset_name)
        print('=' * 80)
        redesigned_records, redesign_summary = build_redesigned_subset(external_train, config)
        raw_summary, repaired_summary, status = train_and_eval(
            config['experiment_name'],
            redesigned_records,
            config,
            STAGE7_CHECKPOINT_DIR,
            external_val,
            external_test,
            schema,
        )
        redesign_results.append({
            'preset_name': preset_name,
            'experiment_name': config['experiment_name'],
            'status': status,
            'train_count': len(redesigned_records),
            'component_threshold': config['component_threshold'],
            'category_threshold': config['category_threshold'],
            'redesign_summary': redesign_summary,
            'raw_summary': raw_summary,
            'repaired_summary': repaired_summary,
        })

    stage14_train_predictions = load_or_create_stage14_train_predictions(external_train)
    residual_subsets = build_residual_subsets(external_train, stage14_train_predictions)
    residual_results = []
    for preset_name, config in RESIDUAL_PRESETS.items():
        print('\n' + '=' * 80)
        print('running residual preset =', preset_name)
        print('=' * 80)
        subset = residual_subsets[config['subset_name']]
        train_records = list(external_train) + subset * int(config['repeat'])
        random.Random(int(config['seed'])).shuffle(train_records)
        raw_summary, repaired_summary, status = train_and_eval(
            config['experiment_name'],
            train_records,
            config,
            STAGE14_CHECKPOINT_DIR,
            external_val,
            external_test,
            schema,
        )
        residual_results.append({
            'preset_name': preset_name,
            'experiment_name': config['experiment_name'],
            'status': status,
            'subset_name': config['subset_name'],
            'subset_size': len(subset),
            'mixed_train_size': len(train_records),
            'repeat': config['repeat'],
            'raw_summary': raw_summary,
            'repaired_summary': repaired_summary,
        })

    lines = [
        '# External Deeper Directions Batch Summary',
        '',
        f'- stage7 checkpoint: `{STAGE7_CHECKPOINT_DIR}`',
        f'- stage14 checkpoint: `{STAGE14_CHECKPOINT_DIR}`',
        f'- external train: `{EXTERNAL_TRAIN}`',
        f'- external val: `{EXTERNAL_VAL}`',
        f'- external test: `{EXTERNAL_TEST}`',
        f'- skip completed: `{SKIP_COMPLETED}`',
        '',
        '## Field-Level Target Redesign Runs',
        '',
    ]
    for item in redesign_results:
        raw = item['raw_summary']
        repaired = item['repaired_summary']
        lines.extend([
            f"### {item['preset_name']}",
            '',
            f"- experiment: `{item['experiment_name']}`",
            f"- status: `{item['status']}`",
            f"- train count: `{item['train_count']}`",
            f"- component threshold: `{item['component_threshold']}`",
            f"- category threshold: `{item['category_threshold']}`",
            f"- redesign summary: `{json.dumps(item['redesign_summary'], ensure_ascii=False)}`",
            f"- raw field exact match: `{raw['field_exact_match']:.4f}`",
            f"- raw end-to-end exact match: `{raw['end_to_end_exact_match']:.4f}`",
            f"- repaired field exact match: `{repaired['field_exact_match']:.4f}`",
            f"- repaired end-to-end exact match: `{repaired['end_to_end_exact_match']:.4f}`",
            '',
        ])

    lines.extend(['## Residual Curriculum Runs', ''])
    for item in residual_results:
        raw = item['raw_summary']
        repaired = item['repaired_summary']
        lines.extend([
            f"### {item['preset_name']}",
            '',
            f"- experiment: `{item['experiment_name']}`",
            f"- status: `{item['status']}`",
            f"- subset: `{item['subset_name']}`",
            f"- subset size: `{item['subset_size']}`",
            f"- repeat: `{item['repeat']}`",
            f"- mixed train size: `{item['mixed_train_size']}`",
            f"- raw field exact match: `{raw['field_exact_match']:.4f}`",
            f"- raw end-to-end exact match: `{raw['end_to_end_exact_match']:.4f}`",
            f"- repaired field exact match: `{repaired['field_exact_match']:.4f}`",
            f"- repaired end-to-end exact match: `{repaired['end_to_end_exact_match']:.4f}`",
            '',
        ])
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('summary_path =', SUMMARY_PATH)


if __name__ == '__main__':
    run()
