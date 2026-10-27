"""CPU-only regressions for complete, predeclared confirmation schedules."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import lookup_confirmation_window as confirmation


def schedule():
    arms = ['legacy', 'dflash-legacy', 'persistent']
    cases = ['related-config', 'related-code', 'conflicting-config', 'unrelated-prose']
    plan = {'pairs': 1, 'arms': arms, 'orders': [arms],
            'cases': [{'id': case} for case in cases], 'total_requests': 24,
            'numerical_mode': 'row-exact',
            'cohort': {'root': '/private/cohort', 'hash': 'cohort', 'slot': 0, 'total': 4}}
    rows = [{'sample': f'0-{arm}-{case}-{rep}', 'pair': 0, 'arm': arm,
             'case': case, 'rep': rep, 'numerical_mode': 'row-exact'}
            for arm in arms for case in cases for rep in (-1, 0)]
    boots = [{'pair': 0, 'arm': arm, 'completed': True, 'assets_preserved': True,
              'numerical_mode': 'row-exact'} for arm in arms]
    return plan, rows, boots


class ConfirmationSchedule(unittest.TestCase):
    def test_full_case_sequence_survives_reversed_global_pair(self):
        plan, rows, boots = schedule()
        plan['cohort']['slot'] = 1
        plan['orders'] = [list(reversed(plan['arms']))]
        rows = [row for arm in plan['orders'][0] for row in rows if row['arm'] == arm]
        boots.reverse()
        self.assertEqual(confirmation.validate_schedule(plan, rows, boots), [])

    def test_missing_or_duplicated_request_cannot_complete_a_slot(self):
        plan, rows, boots = schedule()
        for changed in (rows[:-1], rows + [rows[-1]], rows[:-1] + [rows[0]]):
            with self.subTest(samples=[r['sample'] for r in changed]):
                self.assertTrue(confirmation.validate_schedule(plan, changed, boots))

    def test_reordering_cases_or_exposures_changes_the_workload(self):
        plan, rows, boots = schedule()
        for first, second in ((0, 1), (0, 2), (0, 8)):
            changed = copy.deepcopy(rows)
            changed[first], changed[second] = changed[second], changed[first]
            with self.subTest(swapped=(first, second)):
                self.assertTrue(confirmation.validate_schedule(plan, changed, boots))

    def test_local_pair_identity_cannot_be_reassigned_to_another_slot(self):
        plan, rows, boots = schedule()
        rows[0]['pair'] = 1
        self.assertTrue(confirmation.validate_schedule(plan, rows, boots))

    def test_one_arm_cannot_substitute_for_the_other_native_baseline(self):
        plan, rows, boots = schedule()
        for row in rows:
            if row['arm'] == 'dflash-legacy':
                row['arm'] = 'legacy'
        boots[1]['arm'] = 'legacy'
        self.assertTrue(confirmation.validate_schedule(plan, rows, boots))

    def test_duplicate_missing_or_reordered_boots_do_not_complete(self):
        plan, rows, boots = schedule()
        for changed in (boots[:-1], boots + [boots[-1]], [boots[1], boots[0], boots[2]],
                        [boots[0], boots[0], boots[2]]):
            with self.subTest(boots=changed):
                self.assertTrue(confirmation.validate_schedule(plan, rows, changed))


def cohort(root):
    plan, _, _ = schedule()
    plan.pop('cohort')
    plan.update(family='mtp', phase='evaluation', scenario='transfer', gate_schema=3,
                feature_arms=['persistent'], native_baseline_arms=plan['arms'][:2],
                baseline_arm='legacy', both_native_baselines=True, feature_selection='adaptive',
                serial_oracle=False, price_trace=False, cross_family_reference=None,
                warmup_per_case_per_arm_per_boot=1, timed_per_case_per_arm_per_boot=1,
                deadline_seconds=290, worker_stop_seconds=285, measurement_isolation='declared',
                evaluation_cases=copy.deepcopy(plan['cases']), native_defaults='row-exact',
                pass_rule='independent cold and repeat gates')
    config = {'version': 1, 'root': str(root), 'template': plan, 'wrapper_sha256': 'a' * 64,
              'orders': [plan['arms'][:], list(reversed(plan['arms']))] * 2}
    config['hash'] = confirmation.d.content_hash(config)
    return config


class ConfirmationDeclaration(unittest.TestCase):
    def test_global_pair_cannot_be_reassigned_or_reordered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = cohort(root)
            self.assertEqual(confirmation.pair_plan(config, 3)['orders'], [config['orders'][3]])
            for pair in (-1, 4, True, '1'):
                with self.subTest(pair=pair), self.assertRaises(confirmation.r.BenchError):
                    confirmation.pair_plan(config, pair)
            changed = copy.deepcopy(config)
            changed['orders'][1] = changed['orders'][0]
            changed['hash'] = confirmation.d.content_hash({k: v for k, v in changed.items() if k != 'hash'})
            with self.assertRaisesRegex(confirmation.r.BenchError, 'orders'):
                confirmation.validate_cohort(changed, root)

    def test_changed_protocol_or_relocated_cohort_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = cohort(root)
            changed = copy.deepcopy(config)
            changed['template']['worker_stop_seconds'] = 600
            with self.assertRaisesRegex(confirmation.r.BenchError, 'hash'):
                confirmation.validate_cohort(changed, root)
            changed['hash'] = confirmation.d.content_hash({k: v for k, v in changed.items() if k != 'hash'})
            with self.assertRaisesRegex(confirmation.r.BenchError, 'protocol'):
                confirmation.validate_cohort(changed, root)
            with self.assertRaisesRegex(confirmation.r.BenchError, 'root'):
                confirmation.validate_cohort(config, root / 'relocated')

    def test_mutated_final_artifact_is_rejected_before_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = cohort(root)
            slot = root / 'pair-0'
            slot.mkdir()
            plan = confirmation.pair_plan(config, 0)
            for name in confirmation.SEALED_FILES[:6]:
                confirmation.d.save(slot / name, {})
            confirmation.d.save(slot / 'plan.json', plan)
            confirmation.d.save(slot / 'seal.json', {
                'version': 1, 'root': str(slot), 'cohort': plan['cohort'],
                'artifacts': confirmation._artifact_hashes(slot)})
            confirmation.d.save(slot / 'boots.jsonl', {'completed': False})
            with self.assertRaisesRegex(confirmation.r.BenchError, 'seal/artifact'):
                confirmation.validate_slot(slot, config, 0)


class ConfirmationAggregate(unittest.TestCase):
    def reduce(self, *, missing=None, isolation='declared', family_failure=None,
               fastest_failure=None, drift=False, identity_mismatch=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = cohort(root)
            config['template']['measurement_isolation'] = isolation
            state = {'power_source': 'AC', 'power_settings': 'fixed', 'thermal': 'No thermal warning',
                     'pressure_level': 1, 'swap_used_bytes': 0, 'swapins': 0, 'swapouts': 0}
            slots = []
            for pair in range(4):
                summary = {'complete': True, 'invalid_samples': 0, 'speed_comparison_valid': True,
                           'arm_family_improvement_pass': {'persistent': pair != family_failure},
                           'fastest_native_baseline': {'arm_threshold_pass': {'persistent': pair != fastest_failure}}}
                rows = [{'case': case['id'], 'response': {'text': 'answer', 'usage': {'completion_tokens': 1},
                         'finish_reason': 'stop'}, 'commits': [{'prompt_ids': [1], 'generated_ids': [2]}]}
                        for case in config['template']['cases']]
                if identity_mismatch and pair == 3:
                    rows[0]['commits'][0]['generated_ids'] = [3]
                boot_state = dict(state, swapouts=1 if drift and pair == 3 else 0)
                boots = [{'machine_before': boot_state, 'machine_after': boot_state}] * 3
                slots.append((summary, rows, boots))
                if pair != missing:
                    slot = root / ('pair-' + str(pair))
                    slot.mkdir()
                    confirmation.d.save(slot / 'seal.json', {'slot': pair})
            with mock.patch.object(confirmation, 'load_cohort', return_value=config), \
                    mock.patch.object(confirmation, '_current_assets'), \
                    mock.patch.object(confirmation, 'validate_slot', side_effect=lambda out, conf, pair: slots[pair]), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = confirmation.aggregate(root, root / 'aggregate')
            return status, json.loads((root / 'aggregate/aggregate-summary.json').read_text())

    def test_all_four_pairs_can_confirm_but_never_promote_defaults(self):
        status, result = self.reduce()
        self.assertEqual(status, 0)
        self.assertTrue(result['family_confirmation_pass'])
        self.assertTrue(result['fastest_native_confirmation_pass'])
        self.assertFalse(result['promotion_pass'])
        self.assertFalse(result['canonical_speedup_claim'])

    def test_missing_pair_never_becomes_a_complete_subset(self):
        status, result = self.reduce(missing=2)
        self.assertEqual(status, 1)
        self.assertEqual(result['missing_slots'], [2])
        self.assertFalse(result['execution_complete'])
        self.assertFalse(result['confirmation_pass'])

    def test_one_failed_pair_blocks_confirmation_without_discarding_execution(self):
        for pair in range(4):
            with self.subTest(pair=pair):
                status, result = self.reduce(family_failure=pair)
                self.assertEqual(status, 0)
                self.assertTrue(result['execution_complete'])
                self.assertFalse(result['family_confirmation_pass'])
                self.assertFalse(result['fastest_native_confirmation_pass'])

    def test_family_win_is_not_a_fastest_decoder_win(self):
        _, result = self.reduce(fastest_failure=2)
        self.assertTrue(result['family_confirmation_pass'])
        self.assertFalse(result['fastest_native_confirmation_pass'])
        self.assertFalse(result['performance_pass'])

    def test_contended_execution_remains_complete_but_unqualified(self):
        _, result = self.reduce(isolation='contended')
        self.assertTrue(result['execution_complete'])
        self.assertFalse(result['measurement_qualified'])
        self.assertFalse(result['family_confirmation_pass'])
        self.assertIn('gpu_isolation_not_declared', result['measurement_exclusions'])

    def test_individually_stable_windows_must_also_be_stable_across_pairs(self):
        _, result = self.reduce(drift=True)
        self.assertTrue(result['execution_complete'])
        self.assertFalse(result['measurement_qualified'])
        self.assertIn('swapouts_changed', result['measurement_exclusions'])
        self.assertFalse(result['performance_pass'])

    def test_cross_pair_output_difference_invalidates_comparison(self):
        status, result = self.reduce(identity_mismatch=True)
        self.assertEqual(status, 1)
        self.assertFalse(result['complete'])
        self.assertIn('cross_window_exact_identity_mismatch:related-config', result['errors'])
        self.assertFalse(result['family_confirmation_pass'])


if __name__ == '__main__':
    unittest.main()
