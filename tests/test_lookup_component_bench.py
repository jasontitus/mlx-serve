"""CPU-only contracts for frozen independent component ablations; no inference."""
import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import lookup_component_bench as c


def protocol(family='dflash', component='neural-width'):
    definitions, contrasts = c.component_matrix(family, component, 8)
    blocks = 6 if family == 'mtp' else 4
    return {'version': 2, 'family': family, 'component': component, 'blocks': blocks,
            'arms': list(definitions), 'arm_definitions': definitions, 'contrasts': contrasts,
            'orders': c.component_orders(family, component, list(definitions), blocks),
            'cases': [{'id': key} for key in c.CASES]}


def observations(plan):
    rows = [{'sample': f'{block}-{arm}-{case["id"]}-{rep}', 'pair': block, 'arm': arm,
             'case': case['id'], 'rep': rep,
             'seconds': 10 if plan['arm_definitions'][arm].get('comparison_arm', arm).endswith('-native') else 5}
            for block, order in enumerate(plan['orders']) for arm in order
            for case in plan['cases'] for rep in c.EXPOSURES]
    boots = [{'pair': block, 'arm': arm, 'completed': True, 'assets_preserved': True,
              'numerical_mode': 'row-exact'}
             for block, order in enumerate(plan['orders']) for arm in order]
    return rows, boots


def controlled_row(source='neural', selection='cut'):
    definitions, _ = c.component_matrix('dflash', 'neural-width' if source == 'neural' else 'combined-width', 8)
    spec = definitions['dflash-' + source + '-' + selection]
    requested = {'fixed': 8, 'max': 0, 'cut': 0, 'adaptive': 0}[selection]
    receipt = {'lookup_candidates': source != 'neural', 'variant': 'generic',
               'selection': selection, 'candidate_rows': 16, 'legal_rows': 12,
               'rows': 12 if selection == 'max' else 8, 'requested_rows': requested,
               'assistant': True, 'selected_local': 0, 'selected_corpus': 0, 'selected_persistent': 0}
    return {'family': 'dflash', 'arm_definition': spec, 'rounds': [receipt]}


class ComponentMatrix(unittest.TestCase):
    def test_neural_only_never_loads_frozen_lookup_store(self):
        definitions, contrasts = c.component_matrix('dflash', 'neural-width', 8)
        self.assertEqual(len(definitions), 4)
        for name, spec in definitions.items():
            if name.endswith('-native'):
                self.assertEqual(spec['mode'], 'dflash-legacy')
            else:
                self.assertEqual(spec['tier'], 'local')
                self.assertEqual(spec['experiment'], {'MLX_SERVE_LOOKUP_CANDIDATES': '0'})
        self.assertIn({'baseline': 'dflash-neural-max', 'treatment': 'dflash-neural-cut',
                       'mechanism': 'adaptive width versus maximum legal width', 'role': 'primary'}, contrasts)
        self.assertEqual(definitions['dflash-neural-max']['selection'], 'max')

    def test_combined_controls_share_candidates_and_training_source(self):
        definitions, _ = c.component_matrix('dflash', 'combined-width', 8)
        self.assertEqual(len(definitions), 5)
        for name, spec in definitions.items():
            if not name.endswith('-native'):
                self.assertEqual(spec['tier'], 'persistent')
                self.assertEqual(spec['experiment']['MLX_SERVE_LOOKUP_CANDIDATES'], '1')

    def test_lookup_increment_changes_sources_not_width_policy(self):
        definitions, contrasts = c.component_matrix('dflash', 'lookup-increment', 8)
        self.assertEqual(len(definitions), 3)
        contrast = contrasts[-1]
        self.assertEqual(definitions[contrast['baseline']]['selection'], 'cut')
        self.assertEqual(definitions[contrast['treatment']]['selection'], 'cut')
        self.assertNotEqual(definitions[contrast['baseline']]['candidate_source'],
                            definitions[contrast['treatment']]['candidate_source'])

    def test_mtp_lookup_is_not_neural_staircut(self):
        definitions, contrasts = c.component_matrix('mtp', 'lookup', 8)
        self.assertEqual(len(definitions), 3)
        self.assertTrue(all(spec['candidate_source'] != 'neural' for spec in definitions.values()))
        self.assertIn('NOT neural-only', contrasts[0]['mechanism'])
        with self.assertRaises(c.r.BenchError):
            c.component_matrix('mtp', 'neural-width', 8)

    def test_balanced_blocks_cannot_be_shortened_to_diagnostic_pairs(self):
        for count in (0, 2, 3, 5, True):
            with self.subTest(blocks=count), self.assertRaises(c.r.BenchError):
                c.balanced_orders(['a', 'b', 'c'], count)
        orders = c.balanced_orders(['a', 'b', 'c'], 4)
        for left, right in (('a', 'b'), ('a', 'c'), ('b', 'c')):
            self.assertEqual(sum(order.index(left) < order.index(right) for order in orders), 2)

    def test_mtp_six_permutations_balance_positions_precedence_and_adjacency(self):
        plan = protocol('mtp', 'lookup')
        orders, arms = plan['orders'], plan['arms']
        self.assertEqual(len(set(map(tuple, orders))), 6)
        for arm in arms:
            self.assertEqual(Counter(order.index(arm) for order in orders), {0: 2, 1: 2, 2: 2})
        adjacent = Counter(pair for order in orders for pair in zip(order, order[1:]))
        for left in arms:
            for right in arms:
                if left != right:
                    self.assertEqual(sum(order.index(left) < order.index(right) for order in orders), 3)
                    self.assertEqual(adjacent[left, right], 2)
        rows, boots = observations(plan)
        self.assertEqual(len(rows), 144)
        self.assertEqual(c.schedule_errors(plan, rows, boots), [])
        for count in (4, 5, 8, 12, True):
            with self.assertRaises(c.r.BenchError):
                c.component_orders('mtp', 'lookup', arms, count)


class IsolatedWidthQuartets(unittest.TestCase):
    def test_each_primary_has_its_own_adjacent_quartet_and_native_references(self):
        for component in c.WIDTH_QUARTETS:
            plan = protocol(component=component)
            definitions = plan['arm_definitions']
            primary, = [item for item in plan['contrasts'] if item['role'] == 'primary']
            a, b = primary['baseline'], primary['treatment']
            self.assertEqual(len(definitions), 7)
            self.assertEqual(len(plan['contrasts']), 3)
            for block, order in enumerate(plan['orders']):
                self.assertEqual(order[0 if block % 2 == 0 else -1], 'serial')
                logical = [definitions[arm]['comparison_arm'] for arm in order if arm != 'serial']
                self.assertEqual(logical[1:5], [a, b, b, a] if block % 2 == 0 else [b, a, a, b])
                self.assertEqual([logical[0], logical[-1]], ['dflash-native'] * 2)
                self.assertEqual(Counter(order), Counter(list(definitions)))
            rows, boots = observations(plan)
            self.assertEqual(len(rows), 224)
            self.assertEqual(c.schedule_errors(plan, rows, boots), [])
            self.assertTrue(c.schedule_errors(plan, rows[:-1], boots))
            for family in ('mtp', 'pld'):
                with self.assertRaises(c.r.BenchError):
                    c.component_matrix(family, component, 8)

    def test_slot_drift_cancels_without_pooling_native_or_other_contrasts(self):
        for component in c.WIDTH_QUARTETS:
            plan = protocol(component=component)
            primary = plan['contrasts'][0]
            rows, _ = observations(plan)
            for row in rows:
                logical = plan['arm_definitions'][row['arm']].get('comparison_arm', row['arm'])
                slot = plan['orders'][row['pair']].index(row['arm'])
                row['seconds'] = (1 + .2 * slot) * (2 if logical == primary['baseline'] else 1)
                if logical == 'dflash-native':
                    row['seconds'] = 10000 + slot  # Must not contaminate the primary.
            effect = c.contrast_effects(plan, rows)[0]
            self.assertTrue(effect['isolated_width_contrast'])
            self.assertEqual(effect['statistical_unit'], 'complete four-boot ABBA/BAAB quartet')
            for estimate in (effect['full_workload'], *effect['case_exposure'].values()):
                self.assertEqual(estimate['boot_blocks'], 4)
                for ratio in estimate['ratios']:
                    self.assertAlmostEqual(ratio, 2)
            for arm in c.comparison_groups(plan)[primary['treatment']]:
                missing = [row for row in rows if not (row['pair'] == 0 and row['arm'] == arm
                           and row['case'] == c.CASES[0] and row['rep'] == -1)]
                self.assertEqual(c.contrast_effects(plan, missing)[0]['full_workload']['boot_blocks'], 3)

    def test_primary_replicas_share_every_control_except_selection(self):
        for component in c.WIDTH_QUARTETS:
            plan = protocol(component=component)
            primary = plan['contrasts'][0]
            specs = []
            for name in (primary['baseline'], primary['treatment']):
                original = plan['arm_definitions'][name]
                self.assertEqual(original, plan['arm_definitions'][name + '-repeat'])
                specs.append({key: value for key, value in original.items()
                              if key not in ('selection', 'actual', 'comparison_arm')})
            self.assertEqual(*specs)


class ProductionRoutingQuartets(unittest.TestCase):
    def test_production_comparison_does_not_claim_a_forced_ablation(self):
        for family in ('mtp', 'dflash'):
            with self.subTest(family=family):
                _, contrasts = c.component_matrix(family, 'production-routing', 8)
                native, adaptive = family + '-native', family + '-combined-adaptive'
                self.assertEqual(len(contrasts), 1)
                self.assertEqual((contrasts[0]['baseline'], contrasts[0]['treatment'],
                                  contrasts[0]['role']), (native, adaptive, 'production-primary'))
                effects = [dict(contrasts[0], performance_pass=True)]
                claims = c.claim_verdicts(effects, contrasts)
                self.assertTrue(claims['production_performance_pass'])
                self.assertFalse(claims['component_ablation_pass'])
        for family in ('pld', 'serial', 'unknown'):
            with self.subTest(family=family), self.assertRaises(c.r.BenchError):
                c.component_matrix(family, 'production-routing', 8)

    def test_abba_baab_identity_and_logical_slot_balance(self):
        for family in ('mtp', 'dflash'):
            definitions, _ = c.component_matrix(family, 'production-routing', 8)
            arms = list(definitions)
            a0, b0, a1, b1 = arms
            for blocks in (4, 6, 8):
                orders = c.component_orders(family, 'production-routing', arms, blocks)
                with self.subTest(family=family, blocks=blocks):
                    self.assertEqual(orders, [[a0, b0, b1, a1], [b0, a0, a1, b1]] * (blocks // 2))
                    for order in orders:
                        self.assertEqual(Counter(order), Counter(arms))
                    for slot in range(4):
                        logical = Counter(definitions[order[slot]]['comparison_arm'] for order in orders)
                        self.assertEqual(logical, {a0: blocks // 2, b0: blocks // 2})
            for blocks in (0, 1, 2, 3, 5, 7, True, 4.0, '4'):
                with self.subTest(family=family, blocks=blocks), self.assertRaises(c.r.BenchError):
                    c.component_orders(family, 'production-routing', arms, blocks)

    def test_schedule_rejects_missing_duplicate_and_reordered_replicas(self):
        for family in ('mtp', 'dflash'):
            plan = protocol(family, 'production-routing')
            rows, boots = observations(plan)
            self.assertEqual(c.schedule_errors(plan, rows, boots), [])
            repeat = family + '-native-repeat'
            missing = [row for row in rows if not (row['pair'] == 0 and row['arm'] == repeat)]
            reordered = copy.deepcopy(rows)
            for row in reordered:
                if row['pair'] == 0 and row['arm'] in (family + '-native', repeat):
                    row['arm'] = repeat if row['arm'] == family + '-native' else family + '-native'
            for changed in (missing, rows + [rows[-1]], reordered):
                with self.subTest(family=family):
                    self.assertTrue(c.schedule_errors(plan, changed, boots))
            for changed in (boots[:-1], [boots[1], boots[0], *boots[2:]]):
                self.assertTrue(c.schedule_errors(plan, rows, changed))

    def test_linear_slot_drift_cancels_in_each_quartet_not_just_across_blocks(self):
        for family in ('mtp', 'dflash'):
            plan = protocol(family, 'production-routing')
            rows, _ = observations(plan)
            for row in rows:
                slot = plan['orders'][row['pair']].index(row['arm'])
                base = (c.CASES.index(row['case']) + 1) * (3 if row['rep'] == -1 else 7)
                native = plan['arm_definitions'][row['arm']]['comparison_arm'] == family + '-native'
                # A linearly changing shared runtime multiplier, with true speedup 2.
                row['seconds'] = base * (1 + .2 * slot + .1 * row['pair']) * (2 if native else 1)
            effect, = c.contrast_effects(plan, rows)
            self.assertEqual(effect['full_workload']['boot_blocks'], plan['blocks'])
            for estimate in (effect['full_workload'], *effect['case_exposure'].values()):
                self.assertEqual(len(estimate['ratios']), plan['blocks'])
                for ratio in estimate['ratios']:
                    self.assertAlmostEqual(ratio, 2)

    def test_pool_durations_before_ratios_without_request_pseudoreplication(self):
        for family in ('mtp', 'dflash'):
            plan = protocol(family, 'production-routing')
            rows, _ = observations(plan)
            totals = {'native': 0, 'adaptive': 0}
            expected_cells = {}
            for row in rows:
                weight = c.CASES.index(row['case']) + 1
                exposure = 1 if row['rep'] == -1 else 3
                native = plan['arm_definitions'][row['arm']]['comparison_arm'] == family + '-native'
                repeat = row['arm'].endswith('-repeat')
                # Each physical comparison has a different ratio and base duration.
                row['seconds'] = (weight * exposure * (9 if repeat else 1) if native
                                  else (weight + 2) * exposure * (2 if repeat else 3))
                if row['pair'] == 0:
                    totals['native' if native else 'adaptive'] += row['seconds']
                key = row['case'] + ('/first' if row['rep'] == -1 else '/repeat')
                expected_cells[key] = 10 * weight / (5 * (weight + 2))
            effect, = c.contrast_effects(plan, rows)
            workload = effect['full_workload']
            self.assertEqual(workload['boot_blocks'], plan['blocks'])
            self.assertEqual(workload['ratios'], [totals['native'] / totals['adaptive']] * plan['blocks'])
            self.assertEqual(effect['physical_boot_seconds'], [
                {'block': block,
                 'baseline': {family + '-native': 40, family + '-native-repeat': 360},
                 'treatment': {family + '-combined-adaptive': 216,
                               family + '-combined-adaptive-repeat': 144}}
                for block in range(plan['blocks'])])
            self.assertEqual(set(effect['case_exposure']), set(expected_cells))
            for key, estimate in effect['case_exposure'].items():
                self.assertEqual(estimate['boot_blocks'], plan['blocks'])
                self.assertEqual(estimate['ratios'], [expected_cells[key]] * plan['blocks'])
            self.assertFalse(effect['performance_pass'])

    def test_any_missing_replica_cell_excludes_the_whole_block(self):
        for family in ('mtp', 'dflash'):
            plan = protocol(family, 'production-routing')
            rows, _ = observations(plan)
            for arm in plan['arms']:
                for rep in c.EXPOSURES:
                    changed = [row for row in rows if not (
                        row['pair'] == 0 and row['arm'] == arm and row['case'] == c.CASES[0]
                        and row['rep'] == rep)]
                    effect, = c.contrast_effects(plan, changed)
                    with self.subTest(family=family, arm=arm, rep=rep):
                        self.assertEqual(effect['full_workload']['boot_blocks'], plan['blocks'] - 1)
                        self.assertTrue(all(cell['boot_blocks'] == plan['blocks'] - 1
                                            for cell in effect['case_exposure'].values()))
                        self.assertFalse(effect['every_cell_block_within_5pct'])

    def test_activation_requires_every_related_cell_in_both_replicas(self):
        for family in ('mtp', 'dflash'):
            plan = protocol(family, 'production-routing')
            plan['reuse'] = {'seed_identity': [
                {'case': 'training', 'identity': {'prompt_ids': [99], 'generated_ids': [10, 11, 12]}}]}
            rows, _ = observations(plan)
            adaptive = family + '-combined-adaptive'
            for row in rows:
                active = plan['arm_definitions'][row['arm']]['comparison_arm'] == adaptive
                row.update(commits=[{'prompt_ids': [99], 'generated_ids': [10, 11, 12]}],
                           rounds=[{'accepted': 2, 'committed': 3, 'commit_scope': 'published',
                                    'anchor_pos': 1, 'rows': 3, 'selected_persistent': 2}] if active else [])
            evidence = c.mechanism_evidence(plan, rows)
            self.assertEqual(set(evidence), {family + '-native', adaptive})
            self.assertTrue(evidence[adaptive]['source_requirement_pass'])
            self.assertEqual(len(evidence[adaptive]['related_cell_persistent_training_output_lower_bounds']),
                             plan['blocks'] * len(c.s.RELATED) * len(c.EXPOSURES) * 2)
            for arm in (adaptive, adaptive + '-repeat'):
                for rep in c.EXPOSURES:
                    changed = copy.deepcopy(rows)
                    cell = next(row for row in changed if row['arm'] == arm and row['pair'] == 0
                                and row['case'] in c.s.RELATED and row['rep'] == rep)
                    cell['rounds'][0]['selected_persistent'] = 0
                    with self.subTest(family=family, arm=arm, rep=rep):
                        self.assertFalse(c.mechanism_evidence(plan, changed)[adaptive]['source_requirement_pass'])
                        changed.remove(cell)
                        self.assertFalse(c.mechanism_evidence(plan, changed)[adaptive]['source_requirement_pass'])


class MechanismObservation(unittest.TestCase):
    def test_adaptive_reductions_are_observed_without_forced_cut_labels(self):
        plan = protocol('dflash', 'production-routing')
        plan['reuse'] = {}
        arm = 'dflash-combined-adaptive'
        receipts = [
            {'selection': 'adaptive', 'rows': 8, 'legal_rows': 16, 'commit_scope': 'published'},
            {'selection': 'adaptive', 'rows': 16, 'legal_rows': 16, 'commit_scope': 'published'},
            {'selection': 'adaptive', 'rows': 4, 'legal_rows': 16, 'commit_scope': 'discarded'},
        ]
        rows = [{'arm': arm, 'case': 'unrelated-prose', 'rounds': receipts}]
        observed = c.mechanism_evidence(plan, rows)[arm]
        self.assertEqual(observed['published_rounds'], 2)
        self.assertEqual(observed['width_reduced_rounds'], 1)
        self.assertEqual(observed['width_reduced_rounds_by_selection'], {'adaptive': 1})
        self.assertEqual(observed['forced_cut_reduced_rounds'], 0)
        self.assertFalse(observed['width_cut_required'])
        self.assertTrue(observed['width_cut_requirement_pass'])
        self.assertFalse(observed['backoff_observed'])
        self.assertTrue(observed['backoff_requirement_pass'])

    def test_native_copy_is_not_reported_as_staircut_pruning(self):
        plan = protocol('dflash', 'production-routing')
        plan['reuse'] = {}
        arm = 'dflash-combined-adaptive'
        rows = [{'arm': arm, 'case': 'unrelated-prose', 'rounds': [
            {'selection': 'adaptive', 'rows': 8, 'legal_rows': 16, 'commit_scope': 'published',
             'native_copy': {'selected': True}},
            {'selection': 'adaptive', 'rows': 8, 'legal_rows': 16, 'commit_scope': 'published',
             'native_copy': {'selected': False}}]}]
        observed = c.mechanism_evidence(plan, rows)[arm]
        self.assertEqual(observed['native_copy_selected_rounds'], 1)
        self.assertEqual(observed['width_reduced_rounds'], 1)

    def test_fixed_caps_do_not_satisfy_forced_cut_requirement(self):
        plan = protocol('dflash', 'neural-width')
        plan['reuse'] = {}
        arm = 'dflash-neural-cut'
        receipt = {'selection': 'fixed', 'rows': 8, 'legal_rows': 16, 'commit_scope': 'published'}
        rows = [{'arm': arm, 'case': 'unrelated-prose', 'rounds': [receipt]}]
        observed = c.mechanism_evidence(plan, rows)[arm]
        self.assertEqual(observed['width_reduced_rounds'], 1)
        self.assertTrue(observed['width_cut_required'])
        self.assertFalse(observed['width_cut_requirement_pass'])
        receipt['selection'] = 'cut'
        observed = c.mechanism_evidence(plan, rows)[arm]
        self.assertEqual(observed['forced_cut_reduced_rounds'], 1)
        self.assertTrue(observed['width_cut_requirement_pass'])
        receipt['rows'] = 16
        self.assertFalse(c.mechanism_evidence(plan, rows)[arm]['width_cut_requirement_pass'])

    def test_partial_backoff_observation_does_not_satisfy_all_cell_requirement(self):
        plan = protocol('pld', 'suffix-backoff')
        plan['reuse'] = {}
        arm = 'pld-backoff-on'
        rows = [{'arm': arm, 'case': 'related-code', 'rounds': [], 'commits': [],
                 'backoff_events': [{'plain_tokens': 1, 'misses': 1}]}]
        observed = c.mechanism_evidence(plan, rows)[arm]
        self.assertTrue(observed['backoff_observed'])
        self.assertTrue(observed['backoff_required'])
        self.assertFalse(observed['backoff_requirement_pass'])
        native = c.mechanism_evidence(plan, rows)['serial']
        self.assertEqual(native['width_reduced_rounds'], 0)
        self.assertFalse(native['backoff_observed'])
        self.assertFalse(native['backoff_required'])
        self.assertTrue(native['backoff_requirement_pass'])


class FrozenDeclaration(unittest.TestCase):
    def config(self, family='dflash', component='neural-width'):
        config = protocol(family, component)
        contract = c.d.load_json(c.s.CONTRACT)
        training = [c.s.render(contract, item) for item in contract['training']]
        evaluation = [c.s.render(contract, item) for item in contract['evaluation']]
        config.update(fixed_width=8, protocol=copy.deepcopy(c.PROTOCOL),
                      measurement_isolation='declared', contract=contract,
                      training_cases=training,
                      cases=[next(case for case in evaluation if case['id'] == key) for key in c.CASES],
                      request_identities=c.s.unique_requests(training, evaluation))
        config['telemetry'] = dict(c.telemetry.RULES, binary='/macmon', binary_sha256='a' * 64,
            version='macmon frozen', command=['/macmon', 'pipe', '--soc-info', '-i', '1000'],
            interval_ms=1000, max_gap_ms=3000)
        self.seal(config)
        return config

    def seal(self, config):
        config['hash'] = c.d.content_hash({key: value for key, value in config.items() if key != 'hash'})

    def test_changed_hash_and_redeclared_subset_both_fail(self):
        config = self.config()
        c.validate_config(config, current=False)
        config['contrasts'].pop()
        with self.assertRaisesRegex(c.r.BenchError, 'hash'):
            c.validate_config(config, current=False)
        self.seal(config)
        with self.assertRaisesRegex(c.r.BenchError, 'contrasts'):
            c.validate_config(config, current=False)
        config = self.config()
        config['cases'].pop()
        self.seal(config)
        with self.assertRaisesRegex(c.r.BenchError, 'workload'):
            c.validate_config(config, current=False)

    def test_short_deadline_and_candidate_source_overrides_fail(self):
        config = self.config()
        config['protocol']['deadline_seconds'] = 290
        self.seal(config)
        with self.assertRaises(c.r.BenchError):
            c.validate_config(config, current=False)
        config = self.config()
        config['arm_definitions']['dflash-neural-cut']['experiment']['MLX_SERVE_LOOKUP_CANDIDATES'] = '1'
        self.seal(config)
        with self.assertRaises(c.r.BenchError):
            c.validate_config(config, current=False)

    def test_historical_protocol_is_rejected_without_mutation(self):
        config = self.config()
        config['version'] = config['protocol']['version'] = 1
        self.seal(config)
        original = copy.deepcopy(config)
        with self.assertRaisesRegex(c.r.BenchError, 'historical'):
            c.validate_config(config, current=False)
        self.assertEqual(config, original)

    def test_changed_telemetry_command_and_required_fields_fail(self):
        for key, value in (('command', ['/macmon', 'pipe']), ('required_fields', ['gpu_power'])):
            config = self.config()
            config['telemetry'][key] = value
            self.seal(config)
            with self.assertRaises(ValueError):
                c.validate_config(config, current=False)


    def test_production_replica_controls_hash_and_order_are_frozen(self):
        for family in ('mtp', 'dflash'):
            config = self.config(family, 'production-routing')
            c.validate_config(config, current=False)
            changed = copy.deepcopy(config)
            changed['arm_definitions'][family + '-combined-adaptive-repeat']['experiment']['MLX_SERVE_LOOKUP_CANDIDATES'] = '0'
            with self.subTest(family=family), self.assertRaisesRegex(c.r.BenchError, 'hash'):
                c.validate_config(changed, current=False)
            self.seal(changed)
            with self.assertRaises(c.r.BenchError):
                c.validate_config(changed, current=False)
            for field in ('arms', 'orders'):
                changed = copy.deepcopy(config)
                order = changed[field] if field == 'arms' else changed[field][0]
                order[0], order[3] = order[3], order[0]
                self.seal(changed)
                with self.subTest(family=family, field=field), self.assertRaises(c.r.BenchError):
                    c.validate_config(changed, current=False)

class FullWorkloadAndUncertainty(unittest.TestCase):
    def test_full_sequence_measures_first_and_repeat_in_every_boot(self):
        plan = protocol()
        rows, boots = observations(plan)
        self.assertEqual(c.schedule_errors(plan, rows, boots), [])
        self.assertEqual(len(rows), 4 * 4 * 4 * 2)
        for changed in (rows[:-1], [row for row in rows if row['rep'] == 0], rows + [rows[-1]],
                        [rows[1], rows[0], *rows[2:]]):
            self.assertTrue(c.schedule_errors(plan, changed, boots))
        self.assertTrue(c.schedule_errors(plan, rows, boots[:-1]))

    def test_uncertainty_resamples_four_blocks_not_32_requests(self):
        plan = protocol()
        rows, _ = observations(plan)
        effects = c.contrast_effects(plan, rows)
        native = effects[0]
        self.assertEqual(native['full_workload']['boot_blocks'], 4)
        self.assertEqual(native['full_workload']['ratios'], [2] * 4)
        self.assertEqual(len(native['case_exposure']), 8)
        self.assertTrue(all(value['boot_blocks'] == 4 for value in native['case_exposure'].values()))
        self.assertFalse(native['performance_pass'])  # Statistics alone never qualify a claim.

    def test_missing_first_exposure_invalidates_entire_block_estimate(self):
        plan = protocol()
        rows, _ = observations(plan)
        effects = c.contrast_effects(plan, rows[1:])
        self.assertEqual(effects[0]['full_workload']['boot_blocks'], 3)
        self.assertEqual(effects[0]['full_workload']['verdict'], 'inconclusive_insufficient_blocks')
        self.assertFalse(effects[0]['every_cell_block_within_5pct'])

    def test_control_case_regression_cannot_hide_in_workload_average(self):
        plan = protocol()
        rows, _ = observations(plan)
        treatment = plan['contrasts'][0]['treatment']
        next(row for row in rows if row['pair'] == 0 and row['arm'] == treatment
             and row['case'] == 'unrelated-prose' and row['rep'] == -1)['seconds'] = 12
        effect = c.contrast_effects(plan, rows)[0]
        self.assertEqual(effect['full_workload']['verdict'], 'win')
        self.assertFalse(effect['every_cell_block_within_5pct'])


class PrimaryComponentQualification(unittest.TestCase):
    def test_lookup_regression_cannot_pass_through_native_reference_wins(self):
        plan = protocol(component='lookup-increment')
        rows, _ = observations(plan)
        for row in rows:
            row['seconds'] = {'dflash-native': 10, 'dflash-neural-cut': 5, 'dflash-combined-cut': 6}[row['arm']]
        effects = c.contrast_effects(plan, rows)
        for effect in effects:
            effect['performance_pass'] = effect['full_workload']['verdict'] == 'win'
        self.assertTrue(all(effect['performance_pass'] for effect in effects if effect['role'] == 'native-reference'))
        self.assertFalse(c.component_pass(effects, plan['contrasts']))

    def test_width_cut_must_beat_both_fixed_and_maximum(self):
        for component in ('neural-width', 'combined-width'):
            plan = protocol(component=component)
            effects = [dict(item, performance_pass=item['role'] != 'primary') for item in plan['contrasts']]
            self.assertFalse(c.component_pass(effects, plan['contrasts']))
            primary = [effect for effect in effects if effect['role'] == 'primary']
            self.assertEqual(len(primary), 2)
            primary[0]['performance_pass'] = True
            self.assertFalse(c.component_pass(effects, plan['contrasts']))
            primary[1]['performance_pass'] = True
            self.assertTrue(c.component_pass(effects, plan['contrasts']))
            incomplete = [effect for effect in effects if effect is not primary[0]]
            self.assertFalse(c.component_pass(incomplete, plan['contrasts']))

    def test_mtp_production_is_separate_from_failed_forced_controls(self):
        plan = protocol('mtp', 'lookup')
        primary = [item for item in plan['contrasts'] if item['role'] == 'production-primary']
        self.assertEqual([(p['baseline'], p['treatment']) for p in primary],
                         [('mtp-native', 'mtp-combined-adaptive')])
        effects = [dict(item, performance_pass=item['role'] == 'production-primary')
                   for item in plan['contrasts']]
        result = c.claim_verdicts(effects, plan['contrasts'])
        self.assertTrue(result['production_performance_pass'])
        self.assertTrue(result['performance_pass'])
        self.assertFalse(result['component_ablation_pass'])
        self.assertFalse(c.claim_verdicts([], plan['contrasts'])['performance_pass'])
        controls = [dict(item, performance_pass=item['role'] == 'forced-control')
                    for item in plan['contrasts']]
        self.assertFalse(c.claim_verdicts(controls, plan['contrasts'])['production_performance_pass'])


class OfflinePriceContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.price = self.root / 'prices.json'
        self.source = self.root / 'source.json'
        self.table = {'schema': 3, 'context_bucket': 'floor-power-of-two-2048',
                      '2048': {'tree': {'generic': {'8': 1.25}, 'linear_gdn': {'8': .75}}}}
        self.metadata = {'version': 1, 'purpose': 'staircase_calibration', 'phase': 'training',
                         'binary_sha256': 'b' * 64, 'runtime_identity': {'images': 'frozen'}, 'model': {'id': 'target'},
                         'model_content': {'target': {'weights': 'd' * 64}, 'assistant': {'weights': 'e' * 64}},
                         'contract_sha256': c.r.digest(c.s.CONTRACT),
                         'preparation': 'production-neural', 'seed_hashes': {'persistent': {'tokens.bin': 'f' * 64}},
                         'source_windows': [{'path': '/independent-training', 'action': 'assistant8', 'case': 'training'}],
                         'scope': 'same-state first-offer verify/commit only; no throughput claim'}
        c.d.save(self.price, self.table)
        self.metadata['prices_sha256'] = c.r.digest(self.price)
        c.d.save(self.source, self.metadata)

    def price_plan(self, declaration):
        return dict(self.metadata, prices=declaration, reuse={'hashes': self.metadata['seed_hashes']},
                    source_hashes={str(c.s.CONTRACT.resolve()): c.r.digest(c.s.CONTRACT)})

    def bind_current_table(self):
        self.metadata['prices_sha256'] = c.r.digest(self.price)
        c.d.save(self.source, self.metadata)

    def test_frozen_prior_shared_only_by_nonnative_arms(self):
        declaration = c.price_declaration(self.price, self.source)
        plan = protocol()
        c.bind_prices(plan['arm_definitions'], declaration)
        for spec in plan['arm_definitions'].values():
            if spec['candidate_source'] == 'native':
                self.assertNotIn('MLX_SERVE_LOOKUP_PRICES', spec.get('experiment', {}))
            else:
                self.assertEqual(spec['experiment']['MLX_SERVE_LOOKUP_PRICES'], str(self.price.resolve()))
        self.assertEqual(declaration['source']['metadata'], self.metadata)
        self.assertIsNone(c.price_declaration(None))

    def test_bare_or_legacy_table_never_gets_manufactured_provenance(self):
        with self.assertRaisesRegex(c.r.BenchError, 'provenance'):
            c.price_declaration(self.price)
        self.table['schema'] = 2
        c.d.save(self.price, self.table)
        with self.assertRaisesRegex(c.r.BenchError, 'schema3'):
            c.price_declaration(self.price, self.source)

    def test_zero_context_and_optional_step_price_are_preserved(self):
        self.table['0'] = self.table.pop('2048')
        self.table['step_ms'] = 2.5
        c.d.save(self.price, self.table)
        self.bind_current_table()
        declaration = c.price_declaration(self.price, self.source)
        self.assertEqual(declaration['sha256'], c.r.digest(self.price))
        for value in (0, -1, float('nan'), True, '2.5'):
            with self.subTest(value=value):
                self.table['step_ms'] = value
                c.d.save(self.price, self.table)
                with self.assertRaisesRegex(c.r.BenchError, 'step_ms'):
                    c.price_declaration(self.price, self.source)

    def test_price_or_source_mutation_and_foreign_runtime_fail(self):
        declaration = c.price_declaration(self.price, self.source)
        plan = self.price_plan(declaration)
        c.validate_prices(plan)
        plan['runtime_identity'] = {'images': 'different'}
        with self.assertRaisesRegex(c.r.BenchError, 'runtime_identity'):
            c.validate_prices(plan)
        plan['runtime_identity'] = self.metadata['runtime_identity']
        self.table['2048']['tree']['generic']['8'] = 2
        c.d.save(self.price, self.table)
        with self.assertRaisesRegex(c.r.BenchError, 'changed'):
            c.validate_prices(plan)
        c.d.save(self.price, dict(self.table, schema=3))
        self.bind_current_table()
        plan['prices'] = c.price_declaration(self.price, self.source)
        self.metadata['scope'] = 'changed after freezing'
        c.d.save(self.source, self.metadata)
        with self.assertRaisesRegex(c.r.BenchError, 'changed'):
            c.validate_prices(plan)

    def test_evaluation_prices_cannot_be_labeled_training(self):
        self.metadata['phase'] = 'evaluation'
        c.d.save(self.source, self.metadata)
        with self.assertRaisesRegex(c.r.BenchError, 'training-only'):
            c.price_declaration(self.price, self.source)

    def test_different_export_sidecar_is_rejected_at_declaration(self):
        self.metadata['prices_sha256'] = '0' * 64
        c.d.save(self.source, self.metadata)
        with self.assertRaisesRegex(c.r.BenchError, 'prices_sha256'):
            c.price_declaration(self.price, self.source)

    def test_exact_serialized_bytes_are_bound_at_declaration_and_execution(self):
        declaration = c.price_declaration(self.price, self.source)
        plan = self.price_plan(declaration)
        self.price.write_bytes(self.price.read_bytes() + b'\n')
        for action in (lambda: c.price_declaration(self.price, self.source),
                       lambda: c.validate_prices(plan)):
            with self.assertRaisesRegex(c.r.BenchError, 'prices_sha256'):
                action()

    def test_foreign_training_contract_rejected_even_with_matching_runtime(self):
        foreign = c.d.load_json(c.s.CONTRACT)
        foreign['training'].append(foreign['evaluation'].pop(0))
        contract_path = self.root / 'foreign-contract.json'
        c.d.save(contract_path, foreign)
        self.metadata['contract_sha256'] = c.r.digest(contract_path)
        c.d.save(self.source, self.metadata)
        declaration = c.price_declaration(self.price, self.source)
        with self.assertRaisesRegex(c.r.BenchError, 'contract_sha256'):
            c.validate_prices(self.price_plan(declaration))

    def test_changed_target_or_assistant_payload_cannot_reuse_calibration(self):
        model = {'id': 'same-manifest'}
        for role, key in (('target', 'model_dir'), ('assistant', 'drafter_dir')):
            root = self.root / role
            root.mkdir()
            (root / 'weights.safetensors').write_bytes(b'original payload')
            model[key] = str(root)
        self.metadata['model'] = model
        self.metadata['model_content'] = c.calibration.model_content(model)
        c.d.save(self.source, self.metadata)
        declaration = c.price_declaration(self.price, self.source)
        plan = self.price_plan(declaration)
        c.validate_prices(plan)
        for key in ('model_dir', 'drafter_dir'):
            path = Path(model[key]) / 'weights.safetensors'
            path.write_bytes(b'replaced payload')
            plan['model_content'] = c.calibration.model_content(model)
            with self.subTest(key=key), self.assertRaisesRegex(c.r.BenchError, 'model_content'):
                c.validate_prices(plan)
            path.write_bytes(b'original payload')

    def test_required_identity_fields_cannot_be_omitted_or_empty(self):
        original = dict(self.metadata)
        for key in ('prices_sha256', 'contract_sha256', 'binary_sha256', 'runtime_identity', 'model', 'model_content'):
            for value in (None, '', {}):
                with self.subTest(key=key, value=value):
                    self.metadata = dict(original)
                    if value is None:
                        del self.metadata[key]
                    else:
                        self.metadata[key] = value
                    c.d.save(self.source, self.metadata)
                    with self.assertRaisesRegex(c.r.BenchError, key):
                        c.price_declaration(self.price, self.source)

    def test_unknown_provenance_cannot_bypass_binding_or_training_split(self):
        for metadata in ({'purpose': 'legacy'}, {'config': self.metadata},
                         dict(self.metadata, purpose='other_calibration')):
            with self.subTest(metadata=metadata):
                c.d.save(self.source, metadata)
                with self.assertRaisesRegex(c.r.BenchError, 'unsupported price provenance'):
                    c.price_declaration(self.price, self.source)

    def test_matching_profile_validates_with_complete_frozen_contract(self):
        declaration = c.price_declaration(self.price, self.source)
        plan = self.price_plan(declaration)
        c.validate_prices(plan)
        for key in ('binary_sha256', 'runtime_identity', 'model', 'source_hashes'):
            with self.subTest(key=key):
                incomplete = dict(plan)
                del incomplete[key]
                with self.assertRaises(c.r.BenchError):
                    c.validate_prices(incomplete)

    def test_foreign_candidate_preparation_or_seed_cannot_supply_prices(self):
        declaration = c.price_declaration(self.price, self.source)
        for component in ('combined-fixed-cut', 'combined-max-cut'):
            with self.assertRaisesRegex(c.r.BenchError, 'preparation'):
                c.bind_prices(protocol(component=component)['arm_definitions'], declaration)
        plan = self.price_plan(declaration)
        plan['reuse'] = {'hashes': {'persistent': {'tokens.bin': 'different'}}}
        with self.assertRaisesRegex(c.r.BenchError, 'seed_hashes'):
            c.validate_prices(plan)

    def test_runtime_checks_include_price_validation(self):
        plan = {'binary': '/binary', 'runtime_identity': {'images': 'frozen'}}
        with mock.patch.object(c.r, 'runtime_identity', return_value={'images': 'changed'}):
            with self.assertRaisesRegex(c.r.BenchError, 'linked runtime'):
                c.validate_runtime(plan)
        with mock.patch.object(c, 'validate_prices', side_effect=c.r.BenchError('price changed')), \
                mock.patch.object(c.r, 'runtime_identity') as runtime:
            with self.assertRaisesRegex(c.r.BenchError, 'price changed'):
                c.validate_runtime(plan)
            runtime.assert_not_called()


def backoff_trace(windows=(1, 2, 4, 8, 8), reset=True):
    state = dict.fromkeys(('window', 'left', 'plain_tokens', 'native_tokens', 'drain_tokens',
                          'cold_tokens', 'misses', 'resets'), 0)
    events = []
    for width in windows:
        state.update(window=width, left=width, misses=state['misses'] + 1)
        events.append(dict(state, family='pld', event='no_match'))
        for token in range(width):
            event = 'native' if token < width - 1 else 'cold' if width == 1 else 'drain'
            state['left'] -= 1
            state['plain_tokens'] += 1
            state[event + '_tokens'] += 1
            events.append(dict(state, family='pld', event=event))
    if reset:
        state.update(window=0, left=0, resets=state['resets'] + 1)
        events.append(dict(state, family='pld', event='accepted'))
    return events


class PldBackoffContracts(unittest.TestCase):
    def row(self, events=None, enabled=True):
        return {'arm_definition': {'backoff': enabled},
                'backoff_events': backoff_trace() if events is None else events,
                'commits': [{'generated_ids': [1] * 192}]}

    def test_matrix_changes_backoff_only_with_separate_serial_reference(self):
        definitions, contrasts = c.component_matrix('pld', 'suffix-backoff', 8)
        self.assertEqual(definitions['serial']['mode'], 'serial')
        for arm, enabled in (('pld-backoff-off', '0'), ('pld-backoff-on', '1')):
            self.assertEqual(definitions[arm]['mode'], 'pld-persistent')
            self.assertEqual(definitions[arm]['selection'], 'cut')
            self.assertEqual(definitions[arm]['experiment'], {
                'MLX_SERVE_LOOKUP_CANDIDATES': '1', 'MLX_SERVE_LOOKUP_BACKOFF': enabled})
        primary = [item for item in contrasts if item['role'] == 'primary']
        self.assertEqual([(item['baseline'], item['treatment']) for item in primary],
                         [('pld-backoff-off', 'pld-backoff-on')])
        self.assertTrue(all(item['baseline'] == 'serial' for item in contrasts if item['role'] != 'primary'))

    def test_backoff_quartets_cancel_slot_drift_without_serial_contamination(self):
        plan = protocol('pld', 'suffix-backoff')
        primary, = [item for item in plan['contrasts'] if item['role'] == 'primary']
        a, b = primary['baseline'], primary['treatment']
        definitions = plan['arm_definitions']
        for block, order in enumerate(plan['orders']):
            logical = [definitions[arm]['comparison_arm'] for arm in order]
            self.assertEqual(logical[1:-1], [a, b, b, a] if block % 2 == 0 else [b, a, a, b])
            self.assertEqual((logical[0], logical[-1]), ('serial', 'serial'))
        rows, boots = observations(plan)
        self.assertEqual(c.schedule_errors(plan, rows, boots), [])
        for row in rows:
            logical = definitions[row['arm']]['comparison_arm']
            slot = plan['orders'][row['pair']].index(row['arm'])
            row['seconds'] = 10000 + slot if logical == 'serial' else (1 + .2 * slot) * (2 if logical == a else 1)
        effect = next(item for item in c.contrast_effects(plan, rows) if item['role'] == 'primary')
        for estimate in (effect['full_workload'], *effect['case_exposure'].values()):
            self.assertEqual(estimate['boot_blocks'], 4)
            for ratio in estimate['ratios']:
                self.assertAlmostEqual(ratio, 2)
        for arm in c.comparison_groups(plan)[a] + c.comparison_groups(plan)[b]:
            missing = [row for row in rows if not (row['pair'] == 0 and row['arm'] == arm and
                       row['case'] == c.CASES[0] and row['rep'] == -1)]
            self.assertTrue(c.schedule_errors(plan, missing, boots))
            incomplete = next(item for item in c.contrast_effects(plan, missing) if item['role'] == 'primary')
            self.assertEqual(incomplete['full_workload']['boot_blocks'], 3)
        with self.assertRaises(c.r.BenchError):
            c.component_orders('pld', 'suffix-backoff', ['serial', a, b], 4)

    def test_exact_saturating_windows_cumulative_emissions_and_reset(self):
        self.assertEqual(c.backoff_errors(self.row()), [])
        events = backoff_trace((1,), reset=True)
        state = dict(events[-1])
        state.update(event='no_match', window=1, left=1, misses=2)
        events.append(state)
        self.assertEqual(c.backoff_errors(self.row(events)), [])

    def test_stop_may_leave_unspent_budget_but_never_emit_extra_tokens(self):
        events = backoff_trace((1, 2, 4), reset=False)
        partial = events[:-2]  # Stop inside the final four-token window.
        self.assertGreater(partial[-1]['left'], 0)
        self.assertEqual(c.backoff_errors(self.row(partial)), [])
        overflow = copy.deepcopy(events)
        overflow.append(dict(overflow[-1], event='drain', left=0,
                             plain_tokens=overflow[-1]['plain_tokens'] + 1,
                             drain_tokens=overflow[-1]['drain_tokens'] + 1))
        self.assertTrue(c.backoff_errors(self.row(overflow)))

    def test_counter_sums_wrong_window_and_disabled_events_fail(self):
        self.assertTrue(c.backoff_errors(self.row(enabled=False)))
        self.assertEqual(c.backoff_errors(self.row([], enabled=False)), [])
        for key, value in (('window', 2), ('left', 0), ('plain_tokens', 1), ('misses', 0)):
            events = backoff_trace()
            events[0][key] = value
            with self.subTest(key=key):
                self.assertTrue(c.backoff_errors(self.row(events)))
        events = backoff_trace((1,), reset=False)
        events[1]['event'] = 'native'
        events[1]['native_tokens'], events[1]['cold_tokens'] = 1, 0
        self.assertTrue(c.backoff_errors(self.row(events)))

    def test_miss_or_success_during_skip_and_unpublished_tokens_fail(self):
        for event in ('no_match', 'accepted'):
            events = backoff_trace()
            events[1]['event'] = event
            self.assertTrue(c.backoff_errors(self.row(events)))
        row = self.row()
        row['commits'][0]['generated_ids'] = [1]
        self.assertIn('backoff_emissions_exceed_published_output', c.backoff_errors(row))

    def test_every_unrelated_prose_exposure_requires_real_backoff_execution(self):
        plan = protocol('pld', 'suffix-backoff')
        plan['reuse'] = {}
        rows, _ = observations(plan)
        for row in rows:
            row.update(rounds=[], commits=[], backoff_events=backoff_trace()
                       if plan['arm_definitions'][row['arm']].get('backoff') else [])
        self.assertTrue(c.mechanism_evidence(plan, rows)['pld-backoff-on']['backoff_requirement_pass'])
        for arm in c.comparison_groups(plan)['pld-backoff-on']:
            missing = copy.deepcopy(rows)
            next(row for row in missing if row['arm'] == arm and row['case'] == 'unrelated-prose')['backoff_events'] = []
            self.assertFalse(c.mechanism_evidence(plan, missing)['pld-backoff-on']['backoff_requirement_pass'])


class ReceiptControls(unittest.TestCase):
    def test_independent_copy_geometry_does_not_fake_lookup_offer_capacity(self):
        row = controlled_row(source='combined', selection='adaptive')
        receipt = row['rounds'][0]
        receipt.update(family='dflash', candidate_rows=4, legal_rows=4, assistant=False,
            selected_local=7, selected_neural=0, predicted_tokens=None, executed_geometry_hash=123,
            native_copy={'rows': 8, 'backing_length': 26, 'geometry_hash': 123,
                         'selected': True, 'ms_per_token': None, 'expected_tokens': None},
            route={'alternative_source': 'native_copy', 'decision': {
                'skip': True, 'reason': 'native_copy_unmeasured', 'native_copy': True,
                'native_copy_ms': None, 'native_copy_tokens': None}})
        self.assertEqual(c.component_receipt_errors(row), [])
        receipt['executed_geometry_hash'] = 456
        self.assertIn('round_0:native_copy_execution_mismatch', c.component_receipt_errors(row))

    def test_unsettled_native_entry_cannot_masquerade_as_verified_work(self):
        receipt = {'family': 'pld', 'kind': 'chain', 'selection': 'cut',
                   'timing_scope': 'native_entry_unsettled', 'lookup_candidates': True,
                   'candidate_rows': 1, 'legal_rows': 1, 'rows': 1, 'requested_rows': 0,
                   'accepted': 0, 'committed': 1, 'verify_commit_ms': None,
                   'predicted_verify_ms': None, 'priced': False,
                   'selected_local': 0, 'selected_corpus': 0,
                   'selected_persistent': 0, 'selected_neural': 0,
                   'lookup_order': 0, 'lookup_confidence': 0, 'selection_overhead_ms': None}
        row = {'arm': 'pld', 'family': 'pld', 'arm_definition': {'actual': 'cut'},
               'response': {'http_status': 200, 'text': 'answer', 'finish_reason': 'length',
                            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}},
               'commits': [{'prompt_ids': [1], 'generated_ids': [2], 'finish_reason': 'length'}],
               'rounds': [receipt], 'seconds': 1.0, 'body': {'max_tokens': 1}}
        self.assertEqual(c.d.check_sample(row), [])
        for key, value in (('rows', 2), ('accepted', 1), ('committed', 2),
                           ('verify_commit_ms', 1.0), ('priced', True), ('requested_rows', 8)):
            changed = copy.deepcopy(row)
            changed['rounds'][0][key] = value
            with self.subTest(key=key):
                self.assertIn('round_0:invalid_unsettled_native_entry', c.d.check_sample(changed))
        for selection in ('fixed', 'max'):
            changed = copy.deepcopy(row)
            changed['arm_definition']['actual'] = selection
            changed['rounds'][0]['selection'] = selection
            self.assertIn('round_0:invalid_unsettled_native_entry', c.d.check_sample(changed))
        row['arm_definition']['experiment'] = {'MLX_SERVE_LOOKUP_BACKOFF': '0'}
        self.assertIn('round_0:invalid_unsettled_native_entry', c.d.check_sample(row))

    def test_fixed_max_and_cut_receipts(self):
        for selection in ('fixed', 'max', 'cut'):
            with self.subTest(selection=selection):
                self.assertEqual(c.component_receipt_errors(controlled_row(selection=selection)), [])

    def test_cut_width_learning_is_charged_and_never_allowed_in_fixed_max_controls(self):
        probe = {'family': 'dflash', 'class': 0, 'rows': 16, 'variant': 'generic',
                 'reason': 'upper_width_learning', 'reservation_consumed': True,
                 'max_forced_executions': 2}
        row = controlled_row()
        row['width_probe_events'] = [probe]
        self.assertEqual(c.component_receipt_errors(row), [])
        self.assertIn('width trials', c.PROTOCOL['cost_accounting']['endpoint'])
        for selection in ('fixed', 'max'):
            control = controlled_row(selection=selection)
            control['width_probe_events'] = [probe]
            self.assertIn('width_probe_in_nonlearning_control', c.component_receipt_errors(control))
        row['width_probe_events'] = [dict(probe, max_forced_executions=3)]
        self.assertIn('invalid_width_learning_receipt', c.component_receipt_errors(row))

    def test_missing_or_lying_candidate_source_fails_closed(self):
        for value in (None, True, 0):
            row = controlled_row()
            row['rounds'][0]['lookup_candidates'] = value
            self.assertTrue(c.component_receipt_errors(row))
        row = controlled_row()
        row['rounds'][0]['selected_persistent'] = 1
        self.assertTrue(c.component_receipt_errors(row))

    def test_max_must_equal_legal_width_and_forced_dflash_uses_assistant(self):
        row = controlled_row(selection='max')
        row['rounds'][0]['rows'] -= 1
        self.assertTrue(c.component_receipt_errors(row))
        row = controlled_row()
        row['rounds'][0]['assistant'] = False
        self.assertTrue(c.component_receipt_errors(row))
        row = controlled_row()
        row['rounds'][0].pop('variant')
        self.assertTrue(c.component_receipt_errors(row))


class SharedBootExtensions(unittest.TestCase):
    def test_named_persistent_arm_clones_existing_frozen_seed_contract(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            seed = root / 'seed'
            seed.mkdir()
            (seed / 'tokens.bin').write_bytes(b'frozen training bytes')
            (seed / 'store.lock').touch()
            plan = {'arm_definitions': {'dflash-combined-cut': {'tier': 'persistent'}},
                    'reuse': {'paths': {'persistent': str(seed)}, 'hashes': {'persistent': c.d.tree_hashes(seed)}}}
            # Skip chmod only to keep the temporary fixture removable across platforms.
            with mock.patch.object(c.d, 'freeze'):
                store, corpus, before = c.d.prepare_boot_assets(root / 'boot', plan, 'dflash-combined-cut')
            self.assertIsNone(corpus)
            self.assertEqual(before['store'], plan['reuse']['hashes']['persistent'])
            self.assertEqual((store / 'tokens.bin').read_bytes(), b'frozen training bytes')


if __name__ == '__main__':
    unittest.main()
