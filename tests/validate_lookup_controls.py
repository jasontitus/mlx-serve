#!/usr/bin/env python3
"""Untimed real-server checks for one-row sampling and budget-limited forced offers."""
import argparse
import json
import os
from pathlib import Path
import sys

import bench_lookup_staircut as bench
from validate_lookup_runtime import one, equivalent, require


def check_budget_geometry(family, rounds, budget):
    require(rounds and any(r['rows'] > 1 for r in rounds),
            'budget check did not exercise root plus a real draft')
    require(sum(r['committed'] for r in rounds) <= budget,
            'published output exceeded token budget')
    if family == 'dflash':
        # Depth is budget-limited; competing siblings need not be. A forced
        # four-row offer may therefore be legal with only two output tokens.
        require(all(r['kind'] == 'tree' and r['requested_rows'] == 4
                    and r['rows'] == min(4, r['legal_rows'])
                    and r['admissible'] == (r['legal_rows'] >= 4) for r in rounds),
                'forced tree width or admissibility disagrees with legal branches')
    else:
        require(all(r['rows'] <= budget and r['legal_rows'] <= budget
                    and r['requested_rows'] == 4 and r['admissible'] is False
                    for r in rounds),
                'oversized fixed chain was called legal or exceeded output budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, default=Path('~/.mlx-serve/bench/lookup-staircut'))
    parser.add_argument('--model', default='qwen27b')
    parser.add_argument('--families', default='dflash,mtp,pld')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=11279)
    parser.add_argument('--ctx-size', type=int, default=2048)
    parser.add_argument('--ready-timeout', type=float, default=1800)
    parser.add_argument('--request-timeout', type=float, default=1800)
    args = parser.parse_args()
    os.umask(0o077)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.kit.expanduser() / 'manifest.json').read_text())
    model = next(m for m in manifest['models'] if m['id'] == args.model)
    binary = bench.absolute_asset(manifest['binary'])
    fixture = json.loads((Path(__file__).parent / 'fixtures/lookup_staircut/validation.json').read_text())
    families = args.families.split(',')
    require(families and len(set(families)) == len(families) and
            all(f in ('dflash', 'mtp', 'pld') for f in families), 'invalid family selection')
    experiments = {'MLX_SERVE_LOOKUP_RECEIPTS': '1', 'MLX_SERVE_LOOKUP_STORE_WRITE': '0'}
    try:
        for family in families:
            mode = family + '-local'
            require(bench.unsupported(model, mode) is None, 'unsupported requested control: ' + mode)
            serial = 'serial-dflash' if family == 'dflash' else 'serial'
            cases = [{'id': 'sample-' + str(seed), 'prompt': fixture['prompt'],
                      'max_tokens': 16, 'temperature': .8, 'seed': seed}
                     for seed in fixture['seeds']]
            cases.append({'id': 'budget', 'prompt': fixture['prompt'],
                          'max_tokens': 2, 'temperature': 0, 'seed': 7})
            references = {}
            with bench.Server(args, binary, model, serial, args.output / (family + '-serial.log'),
                              experiment=experiments) as server:
                for case in cases:
                    references[case['id']] = one(server, bench.request_body(case, 'serial', None))
            with bench.Server(args, binary, model, mode, args.output / (family + '-root.log'),
                              experiment={**experiments, 'MLX_SERVE_LOOKUP_SELECTION': 'fixed:1'}) as server:
                for case in cases[:-1]:
                    actual = one(server, bench.request_body(case, mode, None))
                    equivalent(references[case['id']], actual)
                    require(actual['rounds'] and all(r['rows'] == 1 and r.get('root_no_assistant')
                            and r.get('selected_neural') == 0 and r.get('draft_ms') == 0
                            for r in actual['rounds']), 'fixed:1 was not a real unassisted root')
                    bench.append(args.output / 'gates.jsonl', {'family': family, 'case': case['id'],
                                 'gate': 'sampled-one-row-exact-token-reference', 'status': 'pass',
                                 'evidence': actual})
            with bench.Server(args, binary, model, mode, args.output / (family + '-budget.log'),
                              experiment={**experiments, 'MLX_SERVE_LOOKUP_SELECTION': 'fixed:4'}) as server:
                actual = one(server, bench.request_body(cases[-1], mode, None))
                equivalent(references['budget'], actual)
                check_budget_geometry(family, actual['rounds'], cases[-1]['max_tokens'])
                bench.append(args.output / 'gates.jsonl', {'family': family, 'case': 'budget',
                             'gate': 'production-forced-offer-budget-cap', 'status': 'pass',
                             'evidence': actual})
        bench.atomic_json(args.output / 'summary.json', {'status': 'pass', 'diagnostic_only': True,
                          'families': families, 'sampling_scope': 'fixed:1 only; not general chain distribution proof'})
        return 0
    except (OSError, ValueError, KeyError, bench.BenchError) as exc:
        bench.atomic_json(args.output / 'summary.json', {'status': 'failed-or-blocked', 'error': str(exc)})
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
