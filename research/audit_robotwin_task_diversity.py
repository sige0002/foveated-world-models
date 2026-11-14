"""CPU-only task coverage and endpoint-label audit; no video decoding."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path('/media/sadasue/SSD1/foveated-world-models')
GROUP = ROOT / 'runs/robotwin-data-step-continuation-b4-20261114'
AUDIT = ROOT / 'research/official-data-contract-audit-20261113'
XYZ = [0, 1, 2, 8, 9, 10]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def spectrum(x):
    x = x.astype(np.float64)
    values = np.linalg.eigvalsh(np.cov(x, rowvar=False, bias=True))[::-1]
    proportions = values / values.sum()
    return {'eigenvalues_descending': values.tolist(), 'variance_fraction_descending': proportions.tolist(),
        'participation_ratio': float(1 / (proportions**2).sum()),
        'dimensions_for_95pct_variance': int(np.searchsorted(proportions.cumsum(), .95) + 1),
        'coordinate_std': x.std(0).tolist(), 'mean': x.mean(0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = load(GROUP / 'manifest.json')
    run = Path(manifest['source_run'])
    contract = load(run / 'data_contract.json')
    data = Path(contract['data_root'])
    qfix = load(data / 'qfix/audit.json')
    assert digest(data / 'qfix/audit.json') == contract['qfix_audit_sha256']
    split = load(run / 'episode_split.json')['ids']
    assert digest(run / 'episode_split.json') == contract['split_sha256']
    tasks_path = data / 'meta/tasks.parquet'
    assert digest(tasks_path) == qfix['source_meta_sha256']['meta/tasks.parquet']
    tasks = pq.read_table(tasks_path, columns=['task_index', '__index_level_0__']).to_pylist()
    texts = {r['task_index']: r['__index_level_0__'] for r in tasks}
    task_ids = {v: k for k, v in texts.items()}
    assert len(texts) == len(task_ids) == 23607
    columns = ['episode_index', 'dataset_from_index', 'dataset_to_index', 'length', 'tasks',
               'data/chunk_index', 'data/file_index', 'stats/task_index/min', 'stats/task_index/max']
    metadata, stale_task_stats = {}, []
    for relative, expected in contract['episode_metadata_sha256'].items():
        path = data / relative
        assert digest(path) == expected
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096, use_threads=False):
            for row in batch.to_pylist():
                assert len(row['tasks']) == 1
                row['task_index'] = task_ids[row['tasks'][0]]
                assert row['stats/task_index/min'] == row['stats/task_index/max']
                if row['stats/task_index/min'] != [row['task_index']]:
                    stale_task_stats.append(row['episode_index'])
                metadata[row['episode_index']] = row
    assert len(metadata) == 27500
    arms = {}
    for name, entry in manifest['arms'].items():
        assert digest(entry['path']) == entry['sha256']
        arms[name] = load(entry['path'])['rows']
    original = sorted(arms['seed43_replay'], key=lambda r: r['dataset_index'])
    assert [r['dataset_index'] for r in original] == list(range(12000))
    for condition in ('replay', 'new'):
        pairs = lambda rows: Counter((r['episode'], r['base']) for r in rows)
        assert pairs(arms[f'seed43_{condition}']) == pairs(arms[f'seed44_{condition}'])
    validation_path = run.parent / 'evaluations/final3000-highest-fp32-v2/kl_mse/results.json'
    validation = load(validation_path)['rows']
    val_rows = []
    for row in validation:
        m = metadata[row['episode']]
        val_rows.append({'episode': row['episode'], 'base': row['base'], 'future': min(row['base'] + 48, m['length'] - 1),
            'absolute_current': m['dataset_from_index'] + row['base'],
            'absolute_future': m['dataset_from_index'] + min(row['base'] + 48, m['length'] - 1)})
    pools = {'original12000': original, 'replay12000': arms['seed43_replay'], 'new12000': arms['seed43_new'], 'val512': val_rows}
    for name, rows in pools.items():
        allowed = set(split['val' if name == 'val512' else 'train'])
        for row in rows:
            m = metadata[row['episode']]
            assert row['episode'] in allowed
            assert row['future'] == min(row['base'] + 48, m['length'] - 1)
            assert row['absolute_current'] == m['dataset_from_index'] + row['base']
            assert row['absolute_future'] == m['dataset_from_index'] + row['future']
    # Only requested numeric endpoints are retained. Project five parquet columns
    # in bounded batches rather than opening videos or loading all state rows.
    wanted = np.array(sorted({r[k] for rows in pools.values() for r in rows for k in ('absolute_current', 'absolute_future')}))
    endpoints, scanned = {}, []
    for file in sorted((data / 'data').glob('**/*.parquet')):
        relative = str(file.relative_to(data))
        assert digest(file) == qfix['output_data_sha256'][relative]
        for batch in pq.ParquetFile(file).iter_batches(batch_size=65536, columns=['index', 'episode_index', 'frame_index', 'task_index', 'observation.state'], use_threads=False):
            indices = batch.column(0).to_numpy()
            assert (np.diff(indices) == 1).all()
            candidates = wanted[np.searchsorted(wanted, indices[0]):np.searchsorted(wanted, indices[-1], side='right')]
            if not len(candidates):
                continue
            positions = np.searchsorted(indices, candidates)
            assert np.array_equal(indices[positions], candidates)
            selected = batch.take(pa.array(positions))
            for row in selected.to_pylist():
                m = metadata[row['episode_index']]
                assert row['index'] == m['dataset_from_index'] + row['frame_index']
                assert row['task_index'] == m['task_index']
                endpoints[row['index']] = np.array(row['observation.state'], dtype=np.float32)
        scanned.append(relative)
    assert len(endpoints) == len(wanted)
    stats_path = data / 'meta/stats_gr00t.json'
    stats = load(stats_path)['observation.state']
    minimum, maximum = [np.array(stats[k], dtype=np.float32)[XYZ] for k in ('min', 'max')]
    full_counts = {subset: Counter(metadata[ep]['task_index'] for ep in ids) for subset, ids in split.items()}
    frame_counts = {subset: Counter() for subset in split}
    for subset, ids in split.items():
        for ep in ids:
            frame_counts[subset][metadata[ep]['task_index']] += metadata[ep]['length']
    summaries, count_tables = {}, {}
    for name, rows in pools.items():
        counts = Counter(metadata[r['episode']]['task_index'] for r in rows)
        count_tables[name] = counts
        p = np.array(list(counts.values()), dtype=np.float64) / len(rows)
        current = np.stack([endpoints[r['absolute_current']] for r in rows])[:, XYZ]
        future = np.stack([endpoints[r['absolute_future']] for r in rows])[:, XYZ]
        raw_delta = future - current
        normalized_delta = (2 * (future - minimum) / (maximum - minimum) - 1) - (2 * (current - minimum) / (maximum - minimum) - 1)
        motion = np.linalg.norm(raw_delta.reshape(-1, 2, 3), axis=2)
        subset = 'val' if name == 'val512' else 'train'
        total_frames = sum(frame_counts[subset].values())
        expected_p = np.array(list(frame_counts[subset].values()), dtype=np.float64) / total_frames
        expected_unique = float((-np.expm1(len(rows) * np.log1p(-expected_p))).sum())
        summaries[name] = {'draws': len(rows), 'unique_pairs': len({(r['episode'], r['base']) for r in rows}),
            'unique_episodes': len({r['episode'] for r in rows}), 'unique_instruction_task_ids': len(counts),
            'instruction_entropy_effective_count': float(np.exp(-(p * np.log(p)).sum())),
            'largest_instruction_fraction': float(max(counts.values()) / len(rows)),
            'top10_instruction_fraction': float(sum(v for _, v in counts.most_common(10)) / len(rows)),
            'top10_instructions': [{'task_index': k, 'draws': v, 'instruction': texts[k]} for k, v in counts.most_common(10)],
            'expected_unique_instruction_ids_under_uniform_frame_sampling': expected_unique,
            'raw_XYZ_delta_spectrum': spectrum(raw_delta), 'normalized_XYZ_delta_spectrum': spectrum(normalized_delta),
            'maximum_arm_displacement_cm_percentiles': np.quantile(motion.max(1) * 100, [0, .25, .5, .75, .95, 1]).tolist(),
            'moving_over_5cm_fraction': float((motion.max(1) > .05).mean()),
            'zero_horizon_count': sum(r['future'] == r['base'] for r in rows),
            'short_horizon_count': sum(r['future'] - r['base'] < 48 for r in rows)}
        if name == 'val512':
            # Validate derived labels against the exact cached probe labels.
            import torch
            code_path = ROOT / 'runs/robotwin-qfix-delta2-b4-20261114/evaluations/idm-state-probe-20261114/codes.pt'
            cached = torch.load(code_path, weights_only=True, map_location='cpu')['val']
            assert [(r['episode'], r['base']) for r in rows] == [(r['episode'], r['base']) for r in cached['rows']]
            error = float(np.max(np.abs(normalized_delta - cached['delta'].numpy()[:, XYZ])))
            assert error <= 2e-7, error
            summaries[name]['cached_label_max_absolute_difference'] = error
    source = AUDIT / 'LaWAM'
    mixture_file = source / 'starVLA/dataloader/gr00t_lerobot/mixtures.py'
    module = ast.parse(mixture_file.read_text())
    registry = ast.literal_eval(next(node.value for node in module.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DATASET_NAMED_MIXTURES' for t in node.targets)))
    import yaml
    release_config = ROOT / 'models/lawam-lam-bd993da/dino_large_vae.yaml'
    released_mix = yaml.safe_load(release_config.read_text())['data']['data_mix']
    source_sha = manifest['input_file_sha256'][str(AUDIT / 'LaWAM-robotwin-delta2/starVLA/dataloader/gr00t_lerobot/mixtures.py')]
    assert digest(mixture_file) == source_sha
    evidence = {'official_source_commit': contract['source_commit'], 'mixture_file_sha256': source_sha,
        'released_yaml_sha256': digest(release_config), 'released_mix_name': released_mix,
        'released_mix_registered_in_pinned_source': released_mix in registry,
        'local_single_dataset_entry': registry['robotwin_merged'],
        'author_dataset_id': 'jialei02/robotwin_merged', 'author_dataset_revision_in_saved_API': '560f9e9fecc7dbf826afd6042b418d632e319df1',
        'conclusion': 'Official-author dataset and upstream single-dataset registration are verified. Membership/weights in the released lam_plus_human training mixture cannot be verified because its definition is absent from the pinned source.'}
    result = {'complete': True, 'scope': 'Read-only CPU metadata and projected numeric-parquet audit; no video decode, GPU, fitting or training changes.',
        'task_definition': 'task_index identifies an exact natural-language instruction. No canonical physical-task taxonomy is exported here; distinct instructions must not be reported as distinct manipulation task families.',
        'metadata_note': {'episode_stats_task_index_differs_from_global_task_table_count': len(stale_task_stats),
            'first_examples': stale_task_stats[:5],
            'handling': 'Merged per-episode task-index statistics retain source-local IDs. Use episode instruction text mapped to the global tasks table; verify against actual parquet task_index at every selected endpoint. Stage1 does not input language/task IDs.'},
        'full_dataset': {'episodes': len(metadata), 'frames': sum(m['length'] for m in metadata.values()), 'instruction_task_ids': len(tasks)},
        'split': {subset: {'episodes': len(ids), 'frames': sum(frame_counts[subset].values()), 'instruction_task_ids': len(full_counts[subset])} for subset, ids in split.items()},
        'pools': summaries, 'official_mixture_evidence': evidence,
        'sampling': 'Upstream sample_step is uniform over active-split frames, not uniform over instruction/task names. Expected unique count is with-replacement; new pool additionally excludes prior and duplicate pairs.',
        'endpoints_retained': len(endpoints), 'data_columns_scanned': ['index', 'episode_index', 'frame_index', 'task_index', 'observation.state'],
        'input_sha256': {'manifest': digest(GROUP / 'manifest.json'), 'tasks': digest(tasks_path), 'stats': digest(stats_path), 'validation': digest(validation_path), 'qfix_audit': digest(data / 'qfix/audit.json')},
        'data_shards_verified': scanned, 'script_sha256': digest(__file__)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'instruction-distribution.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task_index', 'instruction', 'full_train_episodes', 'full_train_frames', *pools])
        for task, text in texts.items():
            writer.writerow([task, text, full_counts['train'][task], frame_counts['train'][task], *[count_tables[k][task] for k in pools]])
    print(json.dumps({'complete': True, 'output': str(args.output / 'analysis.json'), 'pools': {k: {f: v[f] for f in ('draws', 'unique_episodes', 'unique_instruction_task_ids', 'largest_instruction_fraction')} for k, v in summaries.items()}}), flush=True)


if __name__ == '__main__':
    main()
