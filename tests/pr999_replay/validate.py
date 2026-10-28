#!/usr/bin/env python3
"""Check the frozen private transfer and optional model, without inference.

This verifies the already-validated freeze, not reconstruction from raw sessions
or runtime rendering equivalence. No transcript text is printed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data, model=None):
    pins = json.loads(Path(__file__).with_name('pins.json').read_text())
    package = json.loads((data / 'package.json').read_text())
    require(package['schema'] == 'pr999-private-transfer-v1', 'Unexpected transfer schema')
    for name, expected in [('requests.jsonl', pins['dataset_sha256']), ('manifest.json', pins['manifest_sha256'])]:
        require(sha_file(data / name) == expected == package['files'][name]['sha256'], f'{name}: hash mismatch')
        require((data / name).stat().st_size == package['files'][name]['bytes'], f'{name}: size mismatch')
    manifest = json.loads((data / 'manifest.json').read_text())
    require(manifest['status'] == 'frozen', 'Dataset was not frozen successfully')
    require(manifest['request_jsonl_sha256'] == pins['dataset_sha256'], 'Manifest dataset mismatch')
    require(manifest['model_repository'] == pins['model']['repository'] and
            manifest['model_revision'] == pins['model']['revision'], 'Model revision mismatch')
    records = [json.loads(line) for line in (data / 'requests.jsonl').read_text().splitlines()]
    require(len(records) == manifest['request_count'], 'Request count mismatch')
    grouped = defaultdict(list)
    ids = set()
    for row in records:
        identity = row['id']
        require(identity == row['request_id'] and identity not in ids, 'Duplicate or inconsistent request ID')
        require(row['schema'] == 'pr999-recorded-replay-request-v1', 'Unexpected request schema')
        ids.add(identity)
        body = row['body']
        require(hashlib.sha256(canonical(body)).hexdigest() == row['body_sha256'], f'{identity}: body hash mismatch')
        require(hashlib.sha256(row['rendered_prompt'].encode()).hexdigest() == row['rendered_prompt_sha256'], f'{identity}: prompt hash mismatch')
        require(all(body[k] == v for k, v in manifest['generation'].items()), f'{identity}: generation settings mismatch')
        require(body['model'] == pins['model']['repository'], f'{identity}: model mismatch')
        require(body['chat_template_kwargs'] == {k: v for k, v in manifest['template_kwargs'].items() if k != 'add_generation_prompt'}, f'{identity}: template settings mismatch')
        require(0 < row['input_tokens'] <= manifest['input_token_ceiling'] and
                row['input_tokens'] + body['max_tokens'] <= manifest['context_limit'], f'{identity}: token budget exceeded')
        grouped[row['episode_id']].append(row)
    require(set(grouped) == {e['episode_id'] for e in manifest['episodes']}, 'Episode set mismatch')
    for episode in manifest['episodes']:
        group = grouped[episode['episode_id']]
        require([r['id'] for r in group] == episode['request_ids'] and
                [r['ordinal'] for r in group] == list(range(len(group))), 'Episode order mismatch')
    if model is not None:
        for name, identity in pins['model']['files'].items():
            path = model / name
            require(path.stat().st_size == identity['bytes'] and sha_file(path) == identity['sha256'], f'Model file mismatch: {name}')
    return {'validated_requests': len(records), 'dataset_sha256': pins['dataset_sha256'],
            'model_files_verified': len(pins['model']['files']) if model is not None else 0,
            'episodes': [{'id': e['episode_id'], 'requests': e['request_count']} for e in manifest['episodes']],
            'control_conditions': manifest['control_conditions'],
            'limits': 'Frozen-byte integrity only; raw-source reconstruction was validated before packaging. Runtime rendering parity and M5 inference remain unverified.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.data, args.model), indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Validation failed: {error}\n')
