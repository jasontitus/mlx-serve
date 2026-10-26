#!/usr/bin/env python3
"""Bounded, sequential DFlash/MTP diagnostic. Never builds; tests are opt-in.

--family dflash|mtp selects the engine for local/reuse controls. Reuse compares
legacy/local/corpus/persistent against identical immutable published fixtures.
--scenario baselines compares legacy DFlash with legacy MTP, keeping native copy
and prompt-lookup defaults. Two alternating boot pairs diagnose; --pairs 4 gives
eight boots and 72 requests with one warmup plus two timed requests per case.
--timed-per-case 1 declares a smaller complete schedule; never trims on deadline.
MTP engagement uses spec-stats, not round receipt coverage: the installed binary
may only receipt selected lookup rounds. No full MTP round-cost claim is made.
All scenarios require exact text/tokens/usage/finish and a single power phase.
--plan-only prepares fixtures/declaration without starting a model.
--preflight-tests explicitly opts into existing binaries within the same 290s.
No autochaining, canonical speedup, or automatic promotion.
"""
import argparse
from contextlib import contextmanager
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import shutil
import struct
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
KIT = Path.home() / '.mlx-serve/bench/lookup-staircut'
PRIOR = KIT / 'runs/20261024-223815-fixed-policy-battery'
sys.path.insert(0, str(ROOT / 'tests'))
import bench_lookup_staircut as r
from prepare_lookup_offline import DOC_SEP, MAX_ORDER, write_npy_i32
from lookup_bench_evidence import transfer_measurement_errors

ARMS = {
    'legacy': {'mode': 'dflash-legacy', 'selection': 'adaptive', 'actual': 'legacy',
               'meaning': 'existing DFlash; lookup disabled'},
    'adaptive': {'mode': 'dflash-local', 'selection': 'adaptive', 'actual': 'adaptive',
                 'meaning': 'local lookup; adaptive assistant admission and adaptive width'},
    'cut': {'mode': 'dflash-local', 'selection': 'cut', 'actual': 'cut',
            'meaning': 'always assistant; merge local lookup then adaptive width cut'},
    'fixed8': {'mode': 'dflash-local', 'selection': 'fixed:8', 'actual': 'fixed',
               'meaning': 'always assistant; merge local lookup then fixed width 8'},
    'local': {'mode': 'dflash-local', 'selection': 'adaptive', 'actual': 'adaptive',
              'meaning': 'adaptive local lookup; no external tier'},
    'corpus': {'mode': 'dflash-corpus', 'selection': 'adaptive', 'actual': 'adaptive',
               'meaning': 'adaptive local lookup plus frozen corpus; persistent empty'},
    'persistent': {'mode': 'dflash-persistent', 'selection': 'adaptive', 'actual': 'adaptive',
                   'meaning': 'adaptive local lookup plus frozen persistent history; corpus absent'},
}
SCENARIO_ARMS = {'local': ('legacy', 'adaptive', 'cut', 'fixed8'),
                 'reuse': ('legacy', 'local', 'corpus', 'persistent'),
                 'baselines': ('dflash-legacy', 'mtp-legacy')}
REUSE_CASES = ('lookup-edit', 'prose')
ROUTE_KEYS = {'skip', 'reason', 'assistant_ms', 'alternative_ms', 'skipped', 'interval'}
REASONS = {'unmeasured', 'invalid_alternative', 'assistant', 'alternative', 'recovery',
           'native_copy', 'native_copy_unmeasured'}


def arm_definitions(scenario, family, arms):
    if scenario == 'baselines':
        return {arm: {'mode': arm, 'family': arm.split('-')[0], 'selection': 'adaptive',
                      'actual': 'legacy', 'meaning': 'unmodified lookup-disabled ' + arm}
                for arm in arms}
    definitions = {}
    for arm in arms:
        spec = dict(ARMS[arm])
        spec.update(mode=family + '-' + spec['mode'].split('-', 1)[1], family=family)
        if family == 'mtp':
            spec['meaning'] = {
                'legacy': 'existing MTP with native prompt lookup; tiered lookup disabled',
                'adaptive': 'local lookup; adaptive chain selection versus native MTP',
                'cut': 'local lookup; forced chain cut when available; native MTP on miss',
                'fixed8': 'local lookup; fixed width 8 capped by legal chain; native MTP on miss',
                'local': 'adaptive local lookup versus native MTP; no external tier',
                'corpus': 'adaptive local lookup plus frozen corpus; persistent empty',
                'persistent': 'adaptive local lookup plus frozen persistent history; corpus absent',
            }[arm]
        definitions[arm] = spec
    return definitions


def declare_numerical_mode(definitions, row_exact):
    mode = 'row-exact' if row_exact else 'stock'
    for spec in definitions.values():
        spec['numerical_mode'] = mode
        if row_exact:
            spec['meaning'] = ('Numerical-mode-controlled, explicit row-exact; not unchanged fastest '
                               'default: ' + spec['meaning'].removeprefix('unmodified '))
    return mode


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def load_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    r.atomic_json(Path(path), value)


def read_rows(path):
    rows, errors = [], []
    if Path(path).exists():
        for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                errors.append({'line': line_number, 'error': str(exc), 'raw': line})
    return rows, errors


def content_hash(value):
    return hashlib.sha256(r.canonical(value).encode()).hexdigest()


def tree_hashes(directory):
    return {str(path.relative_to(directory)): r.digest(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def freeze(directory):
    for path in directory.rglob('*'):
        path.chmod(0o500 if path.is_dir() else 0o400)
    directory.chmod(0o500)


def prepare_reuse(out, model, cases):
    """Build both exact loader formats only from the frozen published legacy run."""
    source = PRIOR / 'observations.jsonl'
    prior_plan = load_json(PRIOR / 'plan.json')
    if prior_plan['cases'] != cases:
        raise r.BenchError('published seed requests differ from declared requests')
    tokenizer = Path(model['model_dir']) / 'tokenizer.json'
    tokenizer_sha = r.digest(tokenizer)
    if tokenizer_sha != model['tokenizer_sha256']:
        raise r.BenchError('current tokenizer differs from staged model identity')
    published, parse_errors = read_rows(source)
    if parse_errors:
        raise r.BenchError('invalid published seed observations')
    seed_dir = out / 'seed-fixtures'
    seed_dir.mkdir()
    commits, evidence = [], []
    for case_id in REUSE_CASES:
        candidates = [(index, row) for index, row in enumerate(published, 1)
                      if row.get('mode') == 'dflash-legacy' and row.get('case') == case_id]
        if not candidates:
            raise r.BenchError('no preexisting legacy seed for ' + case_id)
        index, row = candidates[0]
        commit = row['commit']
        reference = {'commits': [commit], 'response': row['response']}
        if any(item.get('errors') or identity({'commits': [item['commit']],
                                              'response': item['response']}) != identity(reference)
               for _, item in candidates):
            raise r.BenchError('published legacy seed outputs are invalid or inconsistent: ' + case_id)
        if commit.get('version') != 1 or any(
                not isinstance(commit.get(key), list) or not commit[key] or
                any(type(token) is not int or not 0 <= token < DOC_SEP for token in commit[key])
                for key in ('prompt_ids', 'generated_ids')):
            raise r.BenchError('invalid published seed token IDs')
        launch_path = PRIOR / (str(row['pair']) + '-dflash-legacy') / 'launch.json'
        launch = load_json(launch_path)
        if launch['command'][launch['command'].index('--model') + 1] != model['model_dir']:
            raise r.BenchError('published legacy seed used a different model')
        commits.append(commit)
        evidence.append({'case': case_id, 'source_line': index, 'pair': row['pair'], 'rep': row['rep'],
                         'source_launch': str(launch_path), 'source_launch_sha256': r.digest(launch_path),
                         'commit_sha256': content_hash(commit), 'identity': identity(reference)})
    tokens = [token for commit in commits
              for token in commit['prompt_ids'] + commit['generated_ids'] + [DOC_SEP]]
    corpus, store = seed_dir / 'corpus', seed_dir / 'persistent'
    corpus.mkdir()
    store.mkdir()
    suffix_array = sorted(range(len(tokens)), key=lambda pos: tokens[pos:pos + MAX_ORDER])
    write_npy_i32(corpus / 'tokens.npy', tokens)
    write_npy_i32(corpus / 'sa.npy', suffix_array)
    save(corpus / 'meta.json', {
        'tokenizer_sha256': tokenizer_sha, 'max_order': MAX_ORDER, 'doc_sep': DOC_SEP,
        'n_tokens': len(tokens), 'documents': len(commits),
        'tokens_sha256': r.digest(corpus / 'tokens.npy'), 'sa_sha256': r.digest(corpus / 'sa.npy'),
        'commits_sha256': content_hash(commits)})
    (store / 'tokens.bin').write_bytes(b'MLXLOOK1' + bytes.fromhex(tokenizer_sha) +
                                     struct.pack('<' + 'I' * len(tokens), *tokens))
    (store / 'store.lock').touch(exist_ok=False)
    if not (store / 'store.lock').is_file():
        raise r.BenchError('persistent snapshot has no regular shared-lock file')
    persistent_validation = r.validate_seed_store(store, tokenizer_sha, commits)
    declaration = {
        'classification': 'cross-request exact-repeat cache hits; NOT held-out task generalization',
        'seeded_cases': list(REUSE_CASES), 'unseeded_control': 'copy-holdout',
        'control_policy': 'unseeded does not imply miss; classify actual selected origins and acceptance',
        'source': str(source), 'source_sha256': r.digest(source),
        'source_plan_sha256': r.digest(PRIOR / 'plan.json'),
        'source_validation_sha256': r.digest(PRIOR / 'response-validation.json'),
        'tokenizer_sha256': tokenizer_sha, 'manifest_sha256': r.digest(KIT / 'manifest.json'),
        'tokenizer_identity_basis': 'staged manifest hash equals current tokenizer; published launch matches model directory; historical commit receipts contain no tokenizer digest',
        'seed_selection': 'first published legacy observation per case; all published legacy repetitions must match',
        'seed_identity': evidence, 'commits': commits, 'documents': len(commits),
        'measured_outputs_used_for_seed': False, 'gpu_tokenization': False,
        'corpus_format': 'flat tokens.npy + sa.npy + meta.json; existing write_npy_i32 helper and MAX_ORDER suffix sort',
        'persistent_validation': persistent_validation,
        'paths': {'corpus': str(corpus), 'persistent': str(store)},
        'hashes': {'corpus': tree_hashes(corpus), 'persistent': tree_hashes(store)}}
    save(seed_dir / 'declaration.json', declaration)
    freeze(seed_dir)
    return declaration


def prepare_boot_assets(folder, plan, arm):
    tier = plan.get('arm_definitions', {}).get(arm, {}).get('tier', arm)
    store = folder / 'server-home' / 'lookup-store'
    store.parent.mkdir(parents=True)
    corpus = None
    if tier == 'persistent':
        shutil.copytree(plan['reuse']['paths']['persistent'], store)
    else:
        store.mkdir()
    if tier == 'corpus':
        corpus = folder / 'corpus'
        shutil.copytree(plan['reuse']['paths']['corpus'], corpus)
        freeze(corpus)
    freeze(store)
    before = {'store': tree_hashes(store), 'corpus': tree_hashes(corpus) if corpus else {}}
    if tier in ('corpus', 'persistent'):
        key = 'store' if tier == 'persistent' else 'corpus'
        if before[key] != plan['reuse']['hashes'][tier]:
            raise r.BenchError('boot fixture does not match frozen seed')
    save(folder / 'assets-before.json', before)
    save(folder / 'storage-before.json', {'store': r.storage_size(store),
         'corpus': r.storage_size(corpus) if corpus else None,
         'scope': 'runtime lookup assets only; excludes model/drafter weights and benchmark artifacts'})
    return store, corpus, before


def finish_boot_assets(folder, plan, store, corpus, before):
    after = {'store': tree_hashes(store), 'corpus': tree_hashes(corpus) if corpus else {}}
    seeds = ({tier: tree_hashes(Path(path)) for tier, path in plan['reuse']['paths'].items()}
             if plan.get('reuse') else None)
    evidence = {'before': before, 'after': after, 'unchanged': before == after,
                'seed_hashes_after': seeds,
                'seeds_unchanged': seeds == plan['reuse']['hashes'] if seeds is not None else True}
    save(folder / 'storage-after.json', {'store': r.storage_size(store),
         'corpus': r.storage_size(corpus) if corpus else None,
         'scope': 'runtime lookup assets only; write-disabled frozen benchmarks do not measure live growth'})
    save(folder / 'assets-after.json', evidence)
    return evidence


@contextmanager
def owned_boot(out, folder, plan, args, pair, arm, telemetry=None):
    spec = plan['arm_definitions'][arm]
    store, corpus, before = prepare_boot_assets(folder, plan, arm)
    machine_before = {'timestamp': time.time(), **r.machine_state()}
    save(folder / 'machine-before.json', machine_before)
    experiment = dict(spec.get('experiment', {}))
    experiment.setdefault('MLX_SERVE_LOOKUP_RECEIPTS', '1')
    experiment.update(MLX_SERVE_LOOKUP_STORE_WRITE='0', MLX_SERVE_LOOKUP_SELECTION=spec['selection'])
    if plan.get('numerical_mode', 'stock') == 'row-exact':
        experiment['MLX_SERVE_ROW_EXACT'] = '1'
    if plan.get('price_trace'):
        experiment['MLX_SERVE_LOOKUP_PRICE_TRACE'] = '1'
    server = r.Server(args, plan['binary'], plan['model'], spec['mode'], folder / 'server.log',
                      corpus=corpus, store=store, experiment=experiment)
    save(folder / 'launch.json', {'command': server.command, 'arm': arm, 'environment': r.evidence_environment(server.env),
                                 'numerical_mode': plan.get('numerical_mode', 'stock'),
                                 'fresh_home': server.env['HOME'], 'fresh_store': str(store),
                                 'corpus_clone': str(corpus) if corpus else None})
    completed = False
    if telemetry is not None:
        telemetry.phase(pair, arm, 'prelaunch')
    try:
        with server:
            if server.numerical_mode != plan.get('numerical_mode', 'stock'):
                raise r.BenchError('owned server numerical mode differs from plan')
            save(folder / 'numerical-mode.json', {'numerical_mode': server.numerical_mode,
                'explicit_activation_observed': server.numerical_mode == 'row-exact'})
            save(folder / 'memory-loaded.json', r.memory(server.proc.pid))
            save(folder / 'allocator-loaded.json', r.allocator_memory(server))
            if telemetry is not None:
                telemetry.phase(pair, arm, 'ready')
            try:
                yield server
            finally:
                save(folder / 'memory-final.json', r.memory(server.proc.pid))
                save(folder / 'allocator-final.json', r.allocator_memory(server))
                if telemetry is not None:
                    telemetry.phase(pair, arm, 'pre_shutdown')
        completed = True
    finally:
        if telemetry is not None:
            telemetry.phase(pair, arm, 'post_shutdown')
        assets = finish_boot_assets(folder, plan, store, corpus, before)
        machine_after = {'timestamp': time.time(), **r.machine_state()}
        save(folder / 'machine-after.json', machine_after)
        r.append(out / 'boots.jsonl', {'pair': pair, 'arm': arm, 'completed': completed,
                                     'assets_preserved': assets['unchanged'] and assets['seeds_unchanged'],
                                     'numerical_mode': server.numerical_mode,
                                     'power_before': machine_before['power_source'],
                                     'power_after': machine_after['power_source'],
                                     'machine_before': machine_before, 'machine_after': machine_after,
                                     'server_returncode': server.proc.returncode if server.proc else None})
        if not assets['unchanged'] or not assets['seeds_unchanged']:
            raise r.BenchError('immutable lookup assets changed during boot')


def origin_evidence(receipts, tier):
    selected = sum(receipt.get('selected_' + tier, 0) for receipt in receipts)
    # Receipts do not name the accepted path's origins. Pigeonhole lower bound:
    # even if every non-tier draft was accepted, this many tier nodes were too.
    lower_bound = sum(max(0, receipt.get('accepted', 0) -
                         max(0, receipt.get('rows', 1) - 1 - receipt.get('selected_' + tier, 0)))
                      for receipt in receipts)
    return {'selected_nodes': selected, 'accepted_origin_lower_bound': lower_bound,
            'selected_rounds': sum(receipt.get('selected_' + tier, 0) > 0 for receipt in receipts),
            'classification': ('selected_positive_accepted_witness' if lower_bound else
                               'selected_acceptance_unproven' if selected else 'no_selected_origin')}




def native_copy_receipt_errors(receipt):
    """A distinct baseline copy is not a subset of the bounded lookup offer."""
    errors = []
    route = receipt.get('route')
    decision = route.get('decision', {}) if isinstance(route, dict) else {}
    if not isinstance(decision, dict):
        return ['malformed_native_copy_decision'] if receipt.get('native_copy') is not None else []
    copy = receipt.get('native_copy')
    keys = {'native_copy', 'native_copy_ms', 'native_copy_tokens'}
    selected = decision.get('native_copy', False)
    source = route.get('alternative_source') if isinstance(route, dict) else None
    reason = decision.get('reason')
    copy_reason = reason in ('native_copy', 'native_copy_unmeasured')
    if keys.intersection(decision):
        if not keys.issubset(decision) or type(selected) is not bool:
            errors.append('incomplete_native_copy_decision')
    if copy is None:
        if selected or source == 'native_copy' or copy_reason or any(
                decision.get(key) is not None for key in ('native_copy_ms', 'native_copy_tokens')):
            errors.append('missing_native_copy_identity')
        return errors
    required = {'rows', 'backing_length', 'geometry_hash', 'selected', 'ms_per_token', 'expected_tokens'}
    if not isinstance(copy, dict) or not required.issubset(copy):
        return errors + ['malformed_native_copy_identity']
    if (receipt.get('family') != 'dflash' or receipt.get('selection') != 'adaptive'
            or receipt.get('lookup_candidates') is not True or receipt.get('requested_rows') != 0):
        errors.append('native_copy_outside_adaptive_suffix_route')
    rows = copy['rows']
    valid_rows = type(rows) is int and 2 <= rows <= 16
    if not valid_rows or type(copy['backing_length']) is not int or copy['backing_length'] < 24:
        errors.append('invalid_native_copy_eligibility')
    if type(copy['geometry_hash']) is not int or not 0 <= copy['geometry_hash'] < 2**64:
        errors.append('invalid_native_copy_geometry')
    if not keys.issubset(decision) or type(copy['selected']) is not bool or copy['selected'] is not selected:
        errors.append('native_copy_selection_mismatch')
    ms, tokens = copy['ms_per_token'], copy['expected_tokens']
    if (ms is None) != (tokens is None) or (
            ms is not None and (not finite(ms) or ms <= 0 or not finite(tokens)
                               or not valid_rows or not 1 <= tokens <= rows)):
        errors.append('invalid_native_copy_economics')
    if decision.get('native_copy_ms') != ms or decision.get('native_copy_tokens') != tokens:
        errors.append('native_copy_economics_mismatch')
    if selected:
        if (not copy_reason or source != 'native_copy' or decision.get('skip') is not True
                or receipt.get('assistant') is not False or receipt.get('rows') != rows
                or receipt.get('executed_geometry_hash') != copy['geometry_hash']):
            errors.append('native_copy_execution_mismatch')
        if (reason == 'native_copy_unmeasured') is not (ms is None):
            errors.append('native_copy_known_state_mismatch')
        if receipt.get('predicted_tokens') != tokens:
            errors.append('native_copy_prediction_mismatch')
        if decision.get('alternative_ms') != ms:
            errors.append('native_copy_alternative_price_mismatch')
        if (receipt.get('selected_local') != rows - 1 if valid_rows else True) or any(
                receipt.get('selected_' + origin) != 0 for origin in ('neural', 'corpus', 'persistent')):
            errors.append('native_copy_origin_mismatch')
    elif copy_reason or source == 'native_copy':
        errors.append('unselected_native_copy_claimed')
    return errors


def check_sample(row):
    errors = list(row.get('errors', []))
    family = row['family']
    spec = row['arm_definition']
    response, commits, rounds = row['response'], row['commits'], row['rounds']
    if row.get('numerical_mode', 'stock') != spec.get('numerical_mode', 'stock'):
        errors.append('numerical_mode_mismatch')
    if response.get('error'):
        errors.append('HTTP/SSE: ' + str(response['error']))
    if response.get('http_status') != 200:
        errors.append('http_status_not_200')
    if not isinstance(response.get('text'), str) or not response['text']:
        errors.append('missing_HTTP_text')
    if response.get('finish_reason') not in ('stop', 'length') or response.get('finish_details'):
        errors.append('abnormal_HTTP_finish')
    usage = response.get('usage')
    if not isinstance(usage, dict):
        errors.append('missing_HTTP_usage')
        usage = {}
    if len(commits) != 1:
        errors.append('expected_exactly_one_commit')
    else:
        commit = commits[0]
        for name in ('prompt_ids', 'generated_ids'):
            ids = commit.get(name)
            if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
                errors.append('invalid_' + name)
        prompt, generated = commit.get('prompt_ids'), commit.get('generated_ids')
        if isinstance(prompt, list) and isinstance(generated, list):
            expected = {'prompt_tokens': len(prompt), 'completion_tokens': len(generated),
                        'total_tokens': len(prompt) + len(generated)}
            for key, count in expected.items():
                if type(usage.get(key)) is not int or usage[key] != count:
                    errors.append('usage_receipt_mismatch:' + key)
            if len(generated) > row['body']['max_tokens']:
                errors.append('output_token_cap_exceeded')
        if commit.get('finish_reason') != response.get('finish_reason'):
            errors.append('HTTP_receipt_finish_mismatch')
    if not finite(row.get('seconds')) or row['seconds'] == 0:
        errors.append('invalid_latency')
    if family == 'mtp':
        stats = row.get('stats', {})
        if not stats.get('family_engaged'):
            errors.append('mtp_not_engaged_in_spec_stats')
    elif family == 'dflash' and not rounds:
        errors.append('missing_dflash_round_receipts')
    elif family == 'serial':
        if any(row['body'].get(key) is not False for key in ('enable_mtp', 'enable_drafter', 'enable_pld')):
            errors.append('serial_speculation_not_disabled')
        if rounds or any(int(stat.get('attempts', 0)) > 0 for stat in row.get('stats', {}).get('spec_stats', [])):
            errors.append('serial_speculation_observed')
    for index, receipt in enumerate(rounds):
        prefix = 'round_' + str(index) + ':'
        errors.extend(prefix + error for error in native_copy_receipt_errors(receipt))
        native_mtp = family == 'mtp' and receipt.get('selection') == 'legacy'
        if not native_mtp and receipt.get('selection') != spec['actual']:
            errors.append(prefix + 'actual_selection_mismatch')
        if receipt.get('timing_scope') == 'native_entry_unsettled':
            if (family != 'pld' or spec['actual'] not in ('adaptive', 'cut')
                    or spec.get('experiment', {}).get('MLX_SERVE_LOOKUP_BACKOFF', '1') != '1'
                    or receipt.get('lookup_candidates') is not True
                    or receipt.get('candidate_rows') != 1 or receipt.get('legal_rows') != 1
                    or receipt.get('requested_rows') != 0 or receipt.get('rows') != 1
                    or receipt.get('accepted') != 0
                    or type(receipt.get('committed')) is not int or not 0 <= receipt['committed'] <= 1
                    or receipt.get('verify_commit_ms') is not None
                    or receipt.get('predicted_verify_ms') is not None
                    or receipt.get('priced') is not False):
                errors.append(prefix + 'invalid_unsettled_native_entry')
        if row['arm'] == 'fixed8' and not native_mtp:
            if receipt.get('requested_rows') != 8:
                errors.append(prefix + 'fixed8_requested_rows_mismatch')
            legal = receipt.get('legal_rows')
            if type(legal) is not int or receipt.get('rows') != min(8, legal):
                errors.append(prefix + 'fixed8_actual_width_mismatch')
        if receipt.get('family') != family or receipt.get('kind') != ('tree' if family == 'dflash' else 'chain'):
            errors.append(prefix + 'actual_family_or_kind_mismatch')
        for key in ('selected_local', 'selected_corpus', 'selected_persistent', 'selected_neural', 'accepted'):
            if type(receipt.get(key)) is not int or not 0 <= receipt[key] <= max(0, receipt.get('rows', 0) - 1):
                errors.append(prefix + 'invalid_' + key)
        for key in ('lookup_order', 'lookup_confidence', 'selection_overhead_ms'):
            if key not in receipt:
                errors.append(prefix + 'missing_' + key)
        route = receipt.get('route')
        if family == 'mtp' and route is not None:
            errors.append(prefix + 'mtp_route_not_null')
        if family == 'dflash' and 'route' not in receipt:
            errors.append(prefix + 'missing_route')
        if family == 'dflash' and spec['actual'] != 'adaptive':
            if route is not None:
                errors.append(prefix + 'forced_or_legacy_route_not_null')
            if row['arm'] in ('cut', 'fixed8') and receipt.get('assistant') is not True:
                errors.append(prefix + 'forced_missing_assistant')
        if family == 'dflash' and route is not None:
            if not isinstance(route, dict) or not isinstance(route.get('decision'), dict):
                errors.append(prefix + 'malformed_route')
                continue
            decision = route['decision']
            if not ROUTE_KEYS.issubset(decision):
                errors.append(prefix + 'incomplete_route_decision')
            if decision.get('reason') not in REASONS or type(decision.get('skip')) is not bool:
                errors.append(prefix + 'invalid_route_decision')
            for key in ('assistant_ms', 'alternative_ms'):
                if decision.get(key) is not None and not finite(decision[key]):
                    errors.append(prefix + 'invalid_' + key)
            for key in ('skipped', 'interval'):
                if type(decision.get(key)) is not int or decision[key] < 0:
                    errors.append(prefix + 'invalid_' + key)
            if route.get('alternative_source') not in (None, 'plain', 'lookup', 'native_copy'):
                errors.append(prefix + 'invalid_alternative_source')
            if type(decision.get('skip')) is bool and receipt.get('assistant') is not (not decision['skip']):
                errors.append(prefix + 'route_assistant_mismatch')
        overhead = receipt.get('selection_overhead_ms')
        if overhead is not None and (not finite(overhead) or not finite(receipt.get('draft_ms'))
                                     or overhead > receipt['draft_ms'] + .05):
            errors.append(prefix + 'invalid_selection_timing')
    return sorted(set(errors))


def identity(row):
    response = row['response']
    commit = row['commits'][0] if len(row['commits']) == 1 else {}
    return {'text': response.get('text'), 'usage': response.get('usage'),
            'finish_reason': response.get('finish_reason'),
            'prompt_ids': commit.get('prompt_ids'), 'generated_ids': commit.get('generated_ids')}


def summarize(out, plan, final=False, supervisor=None):
    rows, parse_errors = read_rows(out / 'observations.jsonl')
    fatal, fatal_parse = read_rows(out / 'errors.jsonl')
    errors = parse_errors + fatal_parse + fatal
    baseline_arm = plan['baseline_arm']
    timed = plan['timed_per_case_per_arm_per_boot']
    refs = {}
    for row in rows:
        if row['arm'] == baseline_arm and not row['errors']:
            refs.setdefault(row['case'], identity(row))
    validation = []
    for row in rows:
        failures = list(row['errors'])
        if row.get('numerical_mode', 'stock') != plan.get('numerical_mode', 'stock'):
            failures.append('numerical_mode_mismatch')
        reference = refs.get(row['case'])
        if reference is None:
            failures.append('missing_valid_legacy_reference')
        elif identity(row) != reference:
            failures += ['legacy_exact_mismatch:' + key for key, value in identity(row).items()
                         if value != reference[key]]
        if plan.get('reuse') and row['case'] in REUSE_CASES:
            seed = next(x for x in plan['reuse']['seed_identity'] if x['case'] == row['case'])
            if identity(row) != seed['identity']:
                failures.append('published_seed_exact_identity_mismatch')
        validation.append({'sample': row['sample'], 'errors': failures})
    invalid = {v['sample'] for v in validation if v['errors']}
    expected = plan['total_requests']
    completed_boots, boot_parse = read_rows(out / 'boots.jsonl')
    errors += boot_parse
    power_sources = {boot.get(key, 'unknown') for boot in completed_boots
                     for key in ('power_before', 'power_after')}
    same_power_phase = len(power_sources) == 1 and 'unknown' not in power_sources
    worker_complete = (out / 'worker-complete.json').exists()
    complete = bool(worker_complete and len(rows) == expected and len(completed_boots) ==
                    plan['pairs'] * len(plan['arms']) and not errors and all(
                        boot.get('completed') and boot.get('assets_preserved')
                        and boot.get('numerical_mode', 'stock') == plan.get('numerical_mode', 'stock')
                        for boot in completed_boots))
    descriptive_valid = complete and not invalid and same_power_phase
    qualification_errors = transfer_measurement_errors(completed_boots, plan.get('measurement_isolation'))
    qualification_errors.extend(boot_parse)
    comparison_valid = descriptive_valid and not qualification_errors
    if supervisor and (supervisor.get('timed_out') or supervisor.get('exit_code') != 0):
        complete = False
        descriptive_valid = False
        comparison_valid = False
    summaries = []
    for arm in plan['arms']:
        if arm == baseline_arm:
            continue
        for case in plan['cases']:
            pairs = []
            for pair in range(plan['pairs']):
                group = [x for x in rows if x['pair'] == pair and x['case'] == case['id']
                         and x['arm'] in (baseline_arm, arm)]
                if len(group) != 2 * (timed + 1) or any(x['sample'] in invalid for x in group):
                    continue
                baseline = statistics.median(x['seconds'] for x in group if x['arm'] == baseline_arm and x['rep'] >= 0)
                feature = statistics.median(x['seconds'] for x in group if x['arm'] == arm and x['rep'] >= 0)
                pairs.append({'pair': pair, 'legacy_seconds': baseline, 'feature_seconds': feature,
                              'latency_reduction_percent': 100 * (1 - feature / baseline)})
            summaries.append({'arm': arm, 'case': case['id'], 'valid_pairs': len(pairs), 'pairs': pairs,
                              'baseline_arm': baseline_arm, 'speed_comparison_valid': comparison_valid,
                              'median_paired_latency_reduction_percent': statistics.median(
                                  p['latency_reduction_percent'] for p in pairs) if pairs else None})
    arm_pass = {}
    for arm in plan['arms']:
        if arm == baseline_arm:
            continue
        checks = [x for x in summaries if x['arm'] == arm]
        arm_pass[arm] = bool(plan['scenario'] != 'baselines' and comparison_valid and len(checks) == 3 and all(
            x['valid_pairs'] == plan['pairs'] and all(
                p['latency_reduction_percent'] >= (
                    (-5 if x['case'] == 'copy-holdout' else 5) if plan['scenario'] == 'reuse'
                    else (-5 if x['case'] == 'prose' else 5))
                for p in x['pairs']) for x in checks))
    coverage = {}
    for arm in plan['arms']:
        arm_rows = [row for row in rows if row['arm'] == arm]
        family = plan['arm_definitions'][arm]['family']
        rounds = [receipt for row in rows if row['arm'] == arm for receipt in row['rounds']]
        routes = [receipt['route'] for receipt in rounds if isinstance(receipt.get('route'), dict)]
        coverage[arm] = {
            'family': family, 'requests': len(arm_rows),
            'spec_stats_engaged_requests': sum(row.get('stats', {}).get('family_engaged', False)
                                               for row in arm_rows),
            'spec_stats_attempts': sum(row.get('stats', {}).get('attempts', 0) for row in arm_rows),
            'spec_stats_lookup_rounds': sum(row.get('stats', {}).get('lookup_rounds') or 0
                                           for row in arm_rows),
            'rounds': len(rounds),
            'receipt_scope': ('DFlash tree-round receipts' if family == 'dflash' else
                              'only emitted MTP chain receipts; baseline may emit none and feature misses may emit none'),
            'full_round_cost_coverage_claim': False,
            'assistant_rounds': sum(x.get('assistant') is True for x in rounds),
            'lookup_only_rounds': sum(x.get('assistant') is False and x.get('selected_local', 0) > 0 for x in rounds),
            'selected_local_rounds': sum(x.get('selected_local', 0) > 0 for x in rounds),
            'route_decisions': len(routes) if family == 'dflash' else None,
            'reasons': ({reason: sum(x.get('decision', {}).get('reason') == reason for x in routes)
                         for reason in sorted(REASONS)} if family == 'dflash' else None)}
    activation = {}
    for arm in plan['arms']:
        if arm == baseline_arm or plan['scenario'] == 'baselines':
            continue
        tier = arm if arm in ('corpus', 'persistent') else 'local'
        evidence = []
        for pair in range(plan['pairs']):
            for case in plan['cases']:
                group = [row for row in rows if row['arm'] == arm and row['pair'] == pair
                         and row['case'] == case['id'] and row['rep'] >= 0]
                receipts = [receipt for row in group for receipt in row['rounds']]
                evidence.append({'pair': pair, 'case': case['id'],
                                 'valid_timed_samples': len(group) == timed and all(
                                     row['sample'] not in invalid for row in group),
                                 'origins': {source: origin_evidence(receipts, source)
                                             for source in ('local', 'corpus', 'persistent')}})
        required = [item for item in evidence if item['case'] in (
            REUSE_CASES if arm in ('corpus', 'persistent') else ('lookup-edit', 'copy-holdout'))]
        selected = all(item['origins'][tier]['selected_nodes'] > 0 for item in required)
        accepted = all(item['origins'][tier]['accepted_origin_lower_bound'] > 0 for item in required)
        valid = all(item['valid_timed_samples'] for item in required)
        assets_preserved = len([boot for boot in completed_boots if boot['arm'] == arm
                                and boot.get('assets_preserved')]) == plan['pairs']
        activated = bool(complete and valid and selected and assets_preserved and
                         (accepted if plan['scenario'] == 'reuse' else True))
        activation[arm] = {'required_tier': tier, 'source_required_at_summary_only': True,
                           'activated': activated, 'assets_preserved': assets_preserved,
                           'selected_positive_accepted_witness': accepted, 'by_boot_case': evidence}
        arm_pass[arm] = arm_pass[arm] and activated
        if (plan['arm_definitions'][arm]['family'] == 'dflash' and
                plan['arm_definitions'][arm]['actual'] == 'adaptive' and coverage[arm]['route_decisions'] == 0):
            arm_pass[arm] = False
    baseline_comparison = None
    if plan['scenario'] == 'baselines':
        baseline_comparison = {
            'valid': comparison_valid, 'reference': baseline_arm, 'candidate': 'mtp-legacy',
            'same_current_binary': plan['binary_sha256'], 'same_target_model': plan['model'],
            'power_sources': sorted(power_sources), 'same_power_phase': same_power_phase,
            'native_defaults_preserved': plan['native_defaults'],
            'fastest_observed_by_case': {}}
        for item in summaries:
            reductions = [pair['latency_reduction_percent'] for pair in item['pairs']]
            winner = None
            if comparison_valid and len(reductions) == plan['pairs']:
                if all(value > 0 for value in reductions):
                    winner = 'mtp-legacy'
                elif all(value < 0 for value in reductions):
                    winner = 'dflash-legacy'
            baseline_comparison['fastest_observed_by_case'][item['case']] = winner
    result = {'complete': complete, 'final': final, 'samples': len(rows), 'expected_samples': expected,
              'errors': errors, 'sample_validation': validation, 'invalid_samples': len(invalid),
              'summaries': summaries, 'coverage': coverage, 'activation': activation,
              'scenario': plan['scenario'], 'reuse_declaration': plan.get('reuse'),
              'family': plan['family'], 'baseline_arm': baseline_arm,
              'numerical_mode': plan.get('numerical_mode', 'stock'),
              'descriptive_timing_valid': descriptive_valid,
              'measurement_qualified': not qualification_errors,
              'measurement_exclusions': qualification_errors,
              'speed_comparison_valid': comparison_valid, 'same_power_phase': same_power_phase,
              'power_sources': sorted(power_sources), 'baseline_comparison': baseline_comparison,
              'arm_performance_pass': arm_pass,
              'performance_pass': (any(arm_pass.get(arm, False) for arm in ('corpus', 'persistent'))
                                   if plan['scenario'] == 'reuse' else arm_pass.get('adaptive', False)),
              'promotion_pass': False,
              'canonical_speedup_claim': False,
              'scope': 'boot-paired HTTP latency only; two pairs diagnose, four pairs confirm this schedule, neither promotes; MTP receipts do not cover all native rounds',
              'pass_rule': plan['pass_rule'],
              'supervisor': supervisor}
    save(out / 'summary.json', result)
    return result


class RecordingResponse:
    def __init__(self, response, stream):
        self.response, self.stream = response, stream
        self.status = response.status
    def __enter__(self):
        self.response.__enter__()
        return self
    def __exit__(self, *args):
        return self.response.__exit__(*args)
    def __iter__(self):
        for raw in self.response:
            self.stream.write(raw)
            self.stream.flush()
            yield raw


class RecordingHTTP:
    def __init__(self, opener, stream, meta):
        self.opener, self.stream, self.meta = opener, stream, meta
    def open(self, *args, **kwargs):
        response = self.opener.open(*args, **kwargs)
        save(self.meta, {'status': response.status, 'headers': list(response.headers.items())})
        return RecordingResponse(response, self.stream)


def preflight_tests(out):
    results = {'complete': False, 'passed': False, 'process_group': os.getpgrp(), 'tests': []}
    save(out / 'preflight-results.json', results)
    for name in ('test', 'mlx-stream-test'):
        binary = ROOT / 'zig-out' / 'tests' / name
        entry = {'command': [str(binary)], 'timeout_seconds': 60, 'started_unix': time.time(),
                 'stdout': str(out / (name + '-stdout.log')),
                 'stderr': str(out / (name + '-stderr.log')),
                 'returncode': None, 'timed_out': False, 'passed': False}
        results['tests'].append(entry)
        save(out / 'preflight-results.json', results)
        started = time.monotonic()
        try:
            r.absolute_asset(binary)
            entry['binary_sha256'] = r.digest(binary)
            with Path(entry['stdout']).open('wb') as stdout, Path(entry['stderr']).open('wb') as stderr:
                run = subprocess.run([str(binary)], cwd=ROOT, stdout=stdout, stderr=stderr,
                                     stdin=subprocess.DEVNULL, timeout=60)
            entry['returncode'] = run.returncode
            entry['passed'] = run.returncode == 0
        except subprocess.TimeoutExpired:
            entry['timed_out'] = True
        except BaseException as exc:
            entry['error'] = str(exc)
            raise
        finally:
            entry['elapsed_seconds'] = time.monotonic() - started
            save(out / 'preflight-results.json', results)
        if not entry['passed']:
            raise r.BenchError('preflight test failed; no benchmark: ' + name)
    results.update(complete=True, passed=True)
    save(out / 'preflight-results.json', results)


def record_sample(out, folder, server, args, spec, pair, arm, case, rep, telemetry=None):
    """One owned request with raw HTTP, settled receipts and native-family validation."""
    sample = f'{pair}-{arm}-{case["id"]}-{rep}'
    sample_dir = folder / (case['id'] + '-' + str(rep))
    sample_dir.mkdir()
    body = r.request_body(case, spec['mode'], None)
    save(sample_dir / 'request.json', body)
    server.assert_owned()
    offset = server.log_path.stat().st_size
    r.append(out / 'sample-starts.jsonl', {'sample': sample, 'pair': pair, 'arm': arm,
              'case': case['id'], 'rep': rep, 'log_offset': offset, 'time': time.time()})
    original_http = r.HTTP
    memory_before = r.memory(server.proc.pid)
    boundary = None
    with (sample_dir / 'response.sse').open('wb') as raw:
        try:
            r.HTTP = RecordingHTTP(original_http, raw, sample_dir / 'http.json')
            if telemetry is not None:
                boundary = telemetry.start_request(sample)
            request_error = None
            try:
                response = r.request(server, body, args.request_timeout)
                request_error = response.get('error')
            except BaseException as exc:
                request_error = type(exc).__name__ + ': ' + str(exc)
                raise
            finally:
                if boundary is not None:
                    telemetry.finish_request(boundary, request_error)
                    save(sample_dir / 'request-boundary.json', boundary)
        finally:
            r.HTTP = original_http
    memory_after = r.memory(server.proc.pid)
    save(sample_dir / 'response.json', response)
    log = r.settled_log(server, offset, timeout=2)
    (sample_dir / 'receipts.log').write_text(log)
    errors = []
    try:
        stats = r.parse_stats(log, spec['family'])
        commits, rounds = stats['commits'], stats['rounds']
    except (ValueError, TypeError) as exc:
        stats, commits, rounds = {}, [], []
        errors.append('stats_or_receipt_parse: ' + str(exc))
    row = {'sample': sample, 'pair': pair, 'arm': arm, 'case': case['id'], 'rep': rep,
           'family': spec['family'], 'mode': spec['mode'], 'arm_definition': spec, 'stats': stats,
           'numerical_mode': server.numerical_mode,
           'warmup': rep < 0, 'body': body, 'response': response, 'commits': commits,
           'rounds': rounds, 'seconds': response.get('total_seconds'), 'errors': errors,
           'memory_before': memory_before, 'memory_after': memory_after}
    if boundary is not None:
        row['request_boundary'] = boundary
    row['errors'] = check_sample(row)
    return row


def worker(out):
    plan = load_json(out / 'plan.json')
    args = argparse.Namespace(**plan['server_args'])
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if plan.get('preflight_tests'):
            preflight_tests(out)
        for pair, order in enumerate(plan['orders']):
            for arm in order:
                folder = out / (str(pair) + '-' + arm)
                folder.mkdir()
                spec = plan['arm_definitions'][arm]
                with owned_boot(out, folder, plan, args, pair, arm) as server:
                    save(folder / 'owned.json', {'pid': server.proc.pid, 'process_group': os.getpgrp()})
                    for case in plan['cases']:
                        for rep in range(-1, plan['timed_per_case_per_arm_per_boot']):
                            row = record_sample(out, folder, server, args, spec, pair, arm, case, rep)
                            r.append(out / 'observations.jsonl', row)
                            summarize(out, plan)
                            print(json.dumps({'sample': row['sample'], 'seconds': row['seconds'], 'errors': row['errors']}), flush=True)
                            if row['response'].get('error'):
                                raise r.BenchError('request failed; no retry, inspect ' + row['sample'])
        save(out / 'worker-complete.json', {'completed': True})
        summarize(out, plan)
        return 0
    except BaseException as exc:
        r.append(out / 'errors.jsonl', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        summarize(out, plan)
        traceback.print_exc()
        return 1


def kill_group(pid, sig):
    if pid is not None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass


def supervise(args, prepare, script, summary, *, deadline_seconds=290, worker_stop_seconds=285):
    """Own the worker group; only explicit None/None removes the default bounds."""
    if not (deadline_seconds is None and worker_stop_seconds is None):
        if (type(deadline_seconds) not in (int, float) or type(worker_stop_seconds) not in (int, float)
                or not math.isfinite(deadline_seconds) or not math.isfinite(worker_stop_seconds)
                or not 0 < worker_stop_seconds < deadline_seconds):
            raise r.BenchError('supervisor requires positive worker < deadline budgets or explicit None/None')
    start = time.monotonic()
    deadline = None if deadline_seconds is None else start + deadline_seconds
    def cleanup_timeout():
        return 1 if deadline is None else min(1, max(.001, deadline - time.monotonic()))
    state = {'pid': None}
    out = args.output or KIT / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-route-diagnostic-' + str(os.getpid()))
    out = out.expanduser().resolve()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    def hard_stop(signum, _frame):
        kill_group(state['pid'], signal.SIGKILL)
        try:
            r.append(out / 'deadline.jsonl', {'signal': signum, 'elapsed_seconds': time.monotonic() - start,
                                           'performance_pass': False, 'complete': False})
        finally:
            os._exit(124)
    signal.signal(signal.SIGALRM, hard_stop)
    if deadline is not None:
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    def supervisor_interrupt(signum, _frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGTERM, supervisor_interrupt)
    signal.signal(signal.SIGINT, supervisor_interrupt)
    child = None
    supervisor = {'exit_code': None, 'timed_out': False,
                  'deadline_seconds': deadline_seconds, 'worker_stop_seconds': worker_stop_seconds}
    try:
        plan = prepare(out)
        if any(plan.get(key, value) != value for key, value in
               (('deadline_seconds', deadline_seconds), ('worker_stop_seconds', worker_stop_seconds))):
            raise r.BenchError('prepared plan and supervisor time budgets differ')
        summary(out, plan)
        print('Evidence: ' + str(out), flush=True)
        if args.plan_only:
            return 0
        with (out / 'performance.log').open('wb') as log:
            child = subprocess.Popen([sys.executable, str(Path(script).resolve()), '--worker', str(out)],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            state['pid'] = child.pid
            supervisor['process_group'] = child.pid
            save(out / 'bounded-run.json', supervisor)
            try:
                child.wait(timeout=None if worker_stop_seconds is None
                           else max(.001, start + worker_stop_seconds - time.monotonic()))
            except subprocess.TimeoutExpired:
                supervisor['timed_out'] = True
                kill_group(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=cleanup_timeout())
                except subprocess.TimeoutExpired:
                    pass
            finally:
                kill_group(child.pid, signal.SIGKILL)
                child.wait(timeout=cleanup_timeout())
                supervisor['exit_code'] = child.returncode
                supervisor['elapsed_seconds'] = time.monotonic() - start
                save(out / 'bounded-run.json', supervisor)
        result = summary(out, plan, final=True, supervisor=supervisor)
        print(json.dumps({'complete': result['complete'], 'performance_pass': result['performance_pass'],
                          'invalid_samples': result['invalid_samples'], 'samples': result['samples']}), flush=True)
        return 0 if result['complete'] and not result['invalid_samples'] else 1
    except BaseException as exc:
        kill_group(state['pid'], signal.SIGKILL)
        r.append(out / 'errors.jsonl', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        if (out / 'plan.json').exists():
            summary(out, load_json(out / 'plan.json'), final=True, supervisor=supervisor)
        traceback.print_exc()
        return 1
    finally:
        kill_group(state['pid'], signal.SIGKILL)
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, choices=(2, 4), default=2)
    parser.add_argument('--scenario', choices=tuple(SCENARIO_ARMS), default='local')
    parser.add_argument('--family', choices=('dflash', 'mtp'), default='dflash',
                        help='engine for local/reuse; baselines always compares both')
    parser.add_argument('--timed-per-case', type=int, choices=(1, 2), default=2,
                        help='predeclare one or two timed requests after one warmup; no runtime trimming')
    parser.add_argument('--arms', help='comma-separated scenario arms; defaults to all, baseline first')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=11279)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--isolation', choices=('declared', 'contended'), default='contended',
                        help='Declared confirms no other GPU inference; otherwise timings are descriptive only')
    parser.add_argument('--row-exact', action='store_true',
                        help='Explicit model-lifetime row-exact mode on every arm; not stock arithmetic')
    parser.add_argument('--preflight-tests', action='store_true',
                        help='run existing host/plugin test binaries before boots within the 290s cap')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker:
        return worker(args.worker)
    arms = args.arms.split(',') if args.arms else list(SCENARIO_ARMS[args.scenario])
    baseline_arm = SCENARIO_ARMS[args.scenario][0]
    if (len(arms) < 2 or arms[0] != baseline_arm or len(set(arms)) != len(arms)
            or any(arm not in SCENARIO_ARMS[args.scenario] for arm in arms)):
        parser.error('--arms requires ' + baseline_arm + ' first and unique arms from --scenario')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    def prepare(out):
        manifest = load_json(KIT / 'manifest.json')
        cases = load_json(PRIOR / 'declared-cases.json')
        if [case['id'] for case in cases] != ['lookup-edit', 'prose', 'copy-holdout']:
            raise r.BenchError('frozen prior case set changed')
        model = next(m for m in manifest['models'] if m['id'] == 'qwen27b')
        definitions = arm_definitions(args.scenario, args.family, arms)
        numerical_mode = declare_numerical_mode(definitions, args.row_exact)
        if any(spec['family'] == 'mtp' for spec in definitions.values()) and not model.get('native_mtp'):
            raise r.BenchError('staged target has no native MTP head')
        binary = r.absolute_asset(manifest['binary'])
        orders = [arms[:] if pair % 2 == 0 else list(reversed(arms)) for pair in range(args.pairs)]
        reuse = prepare_reuse(out, model, cases) if args.scenario == 'reuse' else None
        plan = {'version': 3, 'scenario': args.scenario, 'reuse': reuse,
                'measurement_isolation': args.isolation,
                'family': 'both' if args.scenario == 'baselines' else args.family,
                'numerical_mode': numerical_mode,
                'baseline_arm': baseline_arm,
                'binary': binary, 'binary_sha256': r.digest(binary), 'model': model,
                'pairs': args.pairs, 'arms': arms, 'arm_definitions': definitions, 'orders': orders,
                'cases': cases, 'frozen_cases_source': str(PRIOR / 'declared-cases.json'),
                'frozen_cases_sha256': r.digest(PRIOR / 'declared-cases.json'),
                'source_script_sha256': r.digest(Path(__file__)),
                'warmup_per_case_per_arm_per_boot': 1,
                'timed_per_case_per_arm_per_boot': args.timed_per_case,
                'total_requests': args.pairs * len(arms) * len(cases) * (1 + args.timed_per_case),
                'native_defaults': ('Numerical-mode-controlled references, explicit row-exact on all arms; not unchanged fastest defaults. '
                                    if args.row_exact else 'Stock numerical defaults; ') +
                                   'DFlash copy shortcut and MTP native prompt lookup retain defaults; prefix cache disabled for every arm',
                'power_policy': 'all before/after boot power sources must be known and equal for speed comparisons',
                'family_coverage': 'MTP requires actual spec-stats engagement and exact target token receipts, not native round receipts; lookup misses may have no receipts; route metadata is DFlash-only; no full MTP round-cost claim',
                'server_args': {'port': args.port, 'ctx_size': 2048, 'ready_timeout': 35, 'request_timeout': 25},
                'isolation': 'fresh unique HOME/XDG dirs/store per arm and boot; no inherited MLX settings; offline; store writes off; prefix cache 0; max concurrent 1',
                'deadline_seconds': 290, 'worker_stop_seconds': 285,
                'deadline_policy': 'absolute supervisor alarm 290s kills owned process group; stop work by 285s, TERM then KILL after at most 1s; reserve remaining time for partial summary; no retries or chaining',
                'preflight_tests': args.preflight_tests,
                'preflight_test_policy': 'worker before boots; existing zig-out/tests/test and mlx-stream-test; 60s each; inherited worker process group; failure aborts benchmark; shared 285/290s deadline',
                'cpu_tests': args.preflight_tests, 'build': False, 'canonical': False, 'promotion_pass': False,
                'pass_rule': (
                    'Legacy DFlash versus legacy MTP, same binary/target/workloads and known power phase; '
                    'alternating paired boots; every warmup and timed output exact in text/tokens/usage/finish; '
                    'MTP spec-stats engagement required; incomplete schedules or any disagreement invalidate '
                    'all speed comparisons; report per-case consistent winners only, never promote.'
                    if args.scenario == 'baselines' else
                    'external arm >=5% latency reduction on each seeded edit/prose case in EVERY boot pair; '
                    'unseeded copy control regression <=5% in EVERY pair; selected positive accepted-origin '
                    'witness on both seeded cases in each boot; frozen asset hashes preserved; all outputs '
                    'exact to legacy and published seed; complete schedule. Cross-request exact-repeat '
                    'cache hits only, not held-out task generalization.'
                    if args.scenario == 'reuse' else
                    'adaptive >=5% latency reduction on both edit/copy in every boot pair, prose regression '
                    '<=5% in every pair; summary-only local origin witnesses; all warmup and timed outputs '
                    'exact to legacy; complete run only') +
                    ' Performance/winner qualification requires declared exclusive GPU inference and known, '
                    'unchanged before/after boot power/settings, thermal state, normal memory pressure and '
                    'swap usage/counters without thermal warnings or throttling; otherwise descriptive only.',
                'created_unix': time.time()}
        save(out / 'plan.json', plan)
        (out / 'executed-diagnostic.py').write_bytes(Path(__file__).read_bytes())
        return plan
    return supervise(args, prepare, __file__, summarize)


if __name__ == '__main__':
    sys.exit(main())
