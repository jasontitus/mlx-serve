#!/usr/bin/env python3
"""Deterministic reducer contracts; no model launches or copied host algorithms."""

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import bench_lookup_economics as economics


def observation(action, phase="heldout", repetition=0):
    width = int(action.split(":")[1]) if action.startswith("fixed:") else (2 if action in ("adaptive", "cut") else 3)
    receipt = {"version": 1, "family": "dflash", "kind": "tree", "variant": "generic", "class": 0,
               "context_bucket": economics.CONTEXT_BUCKET,
               "anchor_pos": 10, "prefix_hash": "prefix", "candidate_hash": "offer",
               "candidate_probability_hash": "scores", "candidate_rows": 3, "legal_rows": 3,
               "requested_rows": width if action.startswith("fixed:") else 0,
               "selection": "fixed" if action.startswith("fixed:") else action,
               "rows": width, "accepted": width - 1, "committed": width,
               "draft_ms": 1.0, "verify_commit_ms": {1: 4.0, 2: 5.0, 3: 9.0}[width],
               "round_ms": {1: 5.0, 2: 6.0, 3: 10.0}[width], "priced": True,
               "predicted_verify_ms": None, "predicted_tokens": None, "admissible": True}
    if action == "fixed:1":
        receipt.update(candidate_rows=1, legal_rows=1, candidate_hash="root", draft_ms=0.0,
                       assistant=False, selected_neural=0, root_no_assistant=True)
    return {"mode": "dflash-local", "profile": "copy-ctx0", "phase": phase,
            "repetition": repetition, "action": action, "errors": [], "round": receipt,
            "priming_commits": [], "commit": {"version": 1, "prompt_ids": [1, 2],
                "generated_ids": [10, 11, 12, 13], "finish_reason": "length"}}


def block(phase="heldout", repetition=0):
    return [observation(action, phase, repetition) for action in
            ("fixed:1", "fixed:2", "fixed:3", "adaptive", "cut", "max")]


def stable_block(phase="heldout", repetition=0, isolation="declared"):
    rows = block(phase, repetition)
    state = {"power_source": "ac", "power_settings": "lowpowermode 0",
             "thermal": "No thermal warning level has been recorded",
             "pressure_level": 1, "swap_used_bytes": 0, "swapins": 0, "swapouts": 0}
    for row in rows:
        row.update(isolation=isolation, machine_before=dict(state), machine_after=dict(state))
    return rows


WIDTHS = {"dflash-local/copy-ctx0": [1, 2, 3]}


class DiscoveryLifecycleTests(unittest.TestCase):
    def run_discovery(self, *, errors=(), legal_rows=3, unsupported=False,
                      discovery_only=False, preparation_error=False, training_transition=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workloads = root / "workloads.json"
            workloads.write_text(json.dumps({"cases": [
                {"id": name, "temperature": 0} for name in ("copy", "prose")]}))
            (root / "manifest.json").write_text(json.dumps({
                "version": 1, "binary": sys.executable, "workloads": str(workloads),
                "models": [{"id": "fake", "model_dir": str(root), "tokenizer_sha256": "sha"}]}))
            out = root / "output"
            profiles = [{"id": name + "-ctx0"} for name in ("copy", "prose")]
            calls = []

            def execute(args, binary, model, output, mode, profile, phase,
                        repetition, action, corpus, snapshot, *rest):
                calls.append((phase, profile["id"], action))
                if phase != "discovery":
                    if training_transition:
                        raise RuntimeError("training reached")
                    raise AssertionError("training must not start")
                row = observation(action, phase)
                row.update(profile=profile["id"], mode=mode)
                if profile["id"] == "copy-ctx0":
                    row["errors"] = list(errors)
                    row["round"]["legal_rows"] = legal_rows
                    if "missing_round_receipt" in errors:
                        row["round"] = None
                economics.runner.append(output / "observations.jsonl", row)
                return row

            with mock.patch.object(economics.runner, "digest", return_value="sha"), \
                    mock.patch.object(economics.runner, "inventory", return_value=[]), \
                    mock.patch.object(economics.runner, "unsupported",
                                      side_effect=lambda model, mode: "unavailable" if unsupported and mode == "mtp-local" else None), \
                    mock.patch.object(economics.runner, "machine_state", return_value={}), \
                    mock.patch.object(economics.runner, "Server") as server, \
                    mock.patch.object(economics, "prepare_profiles", return_value=profiles,
                                      side_effect=RuntimeError("tokenizer failed") if preparation_error else None), \
                    mock.patch.object(economics, "execute", side_effect=execute):
                argv = ["--kit", str(root), "--model", "fake", "--cases", "copy,prose",
                        "--modes", "dflash-local,mtp-local" if unsupported else "dflash-local",
                        "--output", str(out)]
                if discovery_only:
                    argv.append("--discovery-only")
                if preparation_error or training_transition:
                    with self.assertRaisesRegex(RuntimeError, "training reached" if training_transition else "tokenizer failed"):
                        economics.main(argv)
                    result = None
                else:
                    result = economics.main(argv)
                server.return_value.__exit__.assert_called_once()
            self.assertTrue((out / "machine-end.json").exists())
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary, json.loads((out / "discovery-summary.json").read_text()))
            self.assertNotIn("heldout_predictions", summary)
            self.assertFalse(summary["canonical_speedup_claim"])
            self.assertFalse(summary["causal_economics_valid"])
            self.assertEqual(summary["phase"], "discovery")
            self.assertEqual(json.loads((out / "run.json").read_text())["config"]["discovery_only"],
                             discovery_only)
            if calls:
                measured = [call for call in calls if call[0] == "discovery"]
                self.assertEqual(len((out / "observations.jsonl").read_text().splitlines()), len(measured))
            return result, summary, calls

    def test_discovery_only_stops_after_all_actual_max_offers(self):
        result, summary, calls = self.run_discovery(discovery_only=True)
        self.assertEqual(result, 0)
        self.assertTrue(summary["discovery_complete"])
        self.assertEqual(calls, [("discovery", name + "-ctx0", "max") for name in ("copy", "prose")])
        outcome = summary["profile_outcomes"]["dflash-local/copy-ctx0"]
        self.assertEqual(outcome["legal_widths"], [1, 2, 3])
        self.assertEqual(outcome["selected_origins"]["local"], 0)
        self.assertEqual(outcome["status"], "discovered")  # No declared origin: misses are legitimate.

    def test_failed_discovery_never_enters_training_and_retains_other_outcomes(self):
        for errors, width in ((["missing_round_receipt"], 3),
                              (["required_first_offer_origin_missing:local"], 3),
                              (["request_failed:timeout"], 3), ([], 1)):
            with self.subTest(errors=errors, width=width):
                result, summary, calls = self.run_discovery(errors=errors, legal_rows=width)
                self.assertEqual(result, 1)
                self.assertFalse(summary["discovery_complete"])
                outcomes = summary["profile_outcomes"]
                self.assertEqual(outcomes["dflash-local/copy-ctx0"]["status"], "failed")
                self.assertEqual(outcomes["dflash-local/prose-ctx0"]["status"], "discovered")
                self.assertEqual(outcomes["dflash-local/copy-ctx0"]["errors"],
                                 errors or ["no_nontrivial_first_offer"])
                self.assertEqual(len(calls), 2)

    def test_unsupported_requested_mode_blocks_training(self):
        result, summary, calls = self.run_discovery(unsupported=True)
        self.assertEqual(result, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(summary["profile_outcomes"]), 4)
        self.assertEqual(summary["profile_outcomes"]["mtp-local/prose-ctx0"]["status"], "unsupported")

    def test_preparation_failure_records_unrun_profiles_and_cleans_up(self):
        _, summary, calls = self.run_discovery(preparation_error=True)
        self.assertEqual(calls, [])
        self.assertFalse(summary["discovery_complete"])
        self.assertEqual(summary["failure"]["message"], "tokenizer failed")
        self.assertTrue(all(o["status"] == "not_run" for o in summary["profile_outcomes"].values()))

    def test_successful_normal_discovery_transitions_to_training(self):
        _, _, calls = self.run_discovery(training_transition=True)
        self.assertEqual([call[0] for call in calls], ["discovery", "discovery", "train"])


class ReceiptBoundaryTests(unittest.TestCase):
    def test_declared_lookup_witness_rejects_neural_only_full_offer_not_root(self):
        profile = {"economics_required_origin": "local"}
        receipt = observation("max")["round"]
        receipt.update(selected_neural=2, selected_local=0, selected_corpus=2)
        self.assertEqual(economics.first_offer_origin_errors(profile, receipt, "max"),
                         ["required_first_offer_origin_missing:local"])
        self.assertEqual(economics.first_offer_origin_errors(
            profile, observation("fixed:1")["round"], "fixed:1"), [])
        self.assertEqual(economics.first_offer_origin_errors({}, receipt, "max"), [])
        receipt["selected_local"] = 1
        self.assertEqual(economics.first_offer_origin_errors(profile, receipt, "max"), [])

    def test_neural_only_prices_do_not_claim_lookup_coverage(self):
        training, heldout = block("train"), block()
        curves = economics.fit_curves(training, WIDTHS)
        summary = economics.summarize(heldout, WIDTHS, curves)
        self.assertTrue(summary["frozen_prediction_evidence_valid"])
        self.assertFalse(summary["lookup_first_offer_exercised"])
        heldout[-1]["round"]["selected_local"] = 1
        summary = economics.summarize(heldout, WIDTHS, curves)
        self.assertTrue(summary["lookup_first_offer_exercised"])
        self.assertEqual(summary["first_offer_coverage"][0]["selected_origins"]["local"], 1)

    def test_exact_legal_widths_and_root(self):
        for action in ("fixed:1", "fixed:2", "fixed:3", "adaptive", "cut", "max"):
            self.assertEqual(economics.receipt_errors(observation(action)["round"], action), [])

    def test_executed_variant_is_required_not_inferred_from_width(self):
        receipt = observation("max")["round"]
        receipt.update(rows=16, candidate_rows=16, legal_rows=16)
        receipt.pop("variant")
        self.assertIn("missing_variant", economics.receipt_errors(receipt, "max"))
        for value in ("", "auto", "generic/linear_gdn", ["generic", "linear_gdn"], 1):
            receipt["variant"] = value
            self.assertIn("unsupported_executed_variant", economics.receipt_errors(receipt, "max"))
        for variant in economics.VERIFY_VARIANTS:
            receipt["variant"] = variant
            self.assertEqual(economics.receipt_errors(receipt, "max"), [])

    def test_cap_is_not_an_exact_measurement(self):
        receipt = observation("fixed:3")["round"]
        receipt.update(rows=2, accepted=1, committed=2)
        self.assertIn("requested_width_not_measured_exactly", economics.receipt_errors(receipt, "fixed:3"))

    def test_emission_censoring_is_invalid(self):
        receipt = observation("fixed:3")["round"]
        receipt["committed"] = 2
        self.assertIn("emission_censored", economics.receipt_errors(receipt, "fixed:3"))

    def test_missing_settled_cost_never_becomes_zero(self):
        receipt = observation("fixed:1")["round"]
        receipt["verify_commit_ms"] = None
        self.assertIn("missing_verify_commit_ms", economics.receipt_errors(receipt, "fixed:1"))
        self.assertEqual(economics.receipt_errors(None, "adaptive"), ["missing_round_receipt"])

    def test_unsettled_native_entry_never_becomes_a_price_observation(self):
        receipt = observation("adaptive")["round"]
        receipt.update(family="pld", kind="chain", rows=1, accepted=0, committed=1,
                       timing_scope="native_entry_unsettled", verify_commit_ms=None, priced=False)
        self.assertIn("unsettled_native_entry_noncomparable",
                      economics.receipt_errors(receipt, "adaptive"))
        # A populated number does not make an asynchronous interval settled.
        receipt.update(verify_commit_ms=1.0, round_ms=2.0, priced=True)
        self.assertIn("unsettled_native_entry_noncomparable",
                      economics.receipt_errors(receipt, "adaptive"))

    def test_invalid_numbers_and_contention(self):
        for value in (float("nan"), float("inf"), -1, True, "4"):
            receipt = observation("fixed:1")["round"]
            receipt["round_ms"] = value
            self.assertIn("invalid_round_ms", economics.receipt_errors(receipt, "fixed:1"))
        receipt = observation("fixed:1")["round"]
        receipt["priced"] = False
        self.assertIn("unpriced_observation", economics.receipt_errors(receipt, "fixed:1"))

    def test_settled_round_must_include_candidate_and_verify_costs(self):
        receipt = observation("fixed:2")["round"]
        receipt.update(draft_ms=100, verify_commit_ms=5, round_ms=5)
        self.assertIn("round_cost_excludes_candidate_construction_or_verify_commit",
                      economics.receipt_errors(receipt, "fixed:2"))
        receipt.update(draft_ms=1, verify_commit_ms=5, round_ms=5.96)
        self.assertEqual(economics.receipt_errors(receipt, "fixed:2"), [])
        receipt["round_ms"] = 5.9
        self.assertIn("round_cost_excludes_candidate_construction_or_verify_commit",
                      economics.receipt_errors(receipt, "fixed:2"))

    def test_max_must_exercise_full_offer(self):
        receipt = observation("max")["round"]
        receipt.update(rows=2, accepted=1, committed=2)
        self.assertIn("max_did_not_measure_full_offer", economics.receipt_errors(receipt, "max"))

    def test_commit_receipts_require_real_token_ids(self):
        commit = observation("fixed:1")["commit"]
        self.assertTrue(economics.complete_receipt(commit))
        for ids in ([], [True], [-1], ["1"]):
            changed = {**commit, "generated_ids": ids}
            self.assertFalse(economics.complete_receipt(changed))


class ControlledComparisonTests(unittest.TestCase):
    def test_regret_uses_measured_unchosen_widths(self):
        rows = block()
        result = economics.compare_block(rows, [1, 2, 3], {})
        self.assertTrue(result["comparable"])
        self.assertEqual(result["observed_winner"], "fixed:2")
        self.assertTrue(result["adaptive_winner_agreement"])
        self.assertEqual(result["adaptive_regret_ms_per_token"], 0)
        self.assertIsNone(result["predicted_winner"])
        rows[3]["round"].update(rows=1, accepted=0, committed=1)
        result = economics.compare_block(rows, [1, 2, 3], {})
        self.assertEqual(result["adaptive_regret_ms_per_token"], 3)
        self.assertAlmostEqual(result["adaptive_relative_regret"], 1)
        self.assertIsNone(result["adaptive_selected_width_regret_ms_per_token"])

    def test_every_offer_identity_component_is_required(self):
        for key, changed in (("prefix_hash", "other"), ("candidate_hash", "other"),
                             ("candidate_probability_hash", "other-scores"),
                             ("kind", "chain"), ("class", 4096), ("anchor_pos", 11)):
            rows = block()
            rows[1]["round"][key] = changed
            result = economics.compare_block(rows, [1, 2, 3], {})
            self.assertFalse(result["comparable"], key)
            self.assertNotIn("adaptive_regret_ms_per_token", result)

    def test_width_dependent_variants_are_allowed_but_selected_kernel_must_match(self):
        rows = block()
        # Recorded execution is authoritative, not a width-based guess.
        for row in rows:
            if row["round"]["rows"] == 3:
                row["round"]["variant"] = "linear_gdn"
        result = economics.compare_block(rows, [1, 2, 3], {})
        self.assertTrue(result["comparable"])
        self.assertEqual(result["executed_variants"]["max"], "linear_gdn")
        for action in ("adaptive", "cut", "max"):
            changed = copy.deepcopy(rows)
            receipt = next(row["round"] for row in changed if row["action"] == action)
            receipt["variant"] = "generic" if receipt["variant"] == "linear_gdn" else "linear_gdn"
            result = economics.compare_block(changed, [1, 2, 3], {})
            self.assertFalse(result["comparable"])
            self.assertIn("different_executed_variant_for_selected_width:" + action, result["errors"])

    def test_missing_or_duplicate_alternatives_never_shrink_oracle(self):
        self.assertFalse(economics.compare_block(block()[:-1], [1, 2, 3], {})["comparable"])
        self.assertFalse(economics.compare_block(block() + [observation("max")], [1, 2, 3], {})["comparable"])
        rows = [row for row in block() if row["action"] != "fixed:2"]
        result = economics.compare_block(rows, [1, 3], {})
        self.assertFalse(result["comparable"])
        self.assertIn("incomplete_legal_width_sweep", result["errors"])

    def test_committed_prompt_output_and_priming_identity_gate(self):
        for key in ("prompt_ids", "generated_ids", "finish_reason"):
            rows = block()
            rows[1]["commit"][key] = "stop" if key == "finish_reason" else [999]
            self.assertFalse(economics.compare_block(rows, [1, 2, 3], {})["comparable"])
        rows = block()
        rows[0]["priming_commits"] = [rows[0]["commit"]]
        self.assertFalse(economics.compare_block(rows, [1, 2, 3], {})["comparable"])

    def test_reducer_revalidates_even_if_caller_omitted_errors(self):
        rows = block()
        rows[0]["round"]["round_ms"] = None
        self.assertFalse(economics.compare_block(rows, [1, 2, 3], {})["comparable"])
        rows = block()
        rows[0]["commit"] = None
        self.assertFalse(economics.compare_block(rows, [1, 2, 3], {})["comparable"])

    def test_only_explicit_unassisted_root_may_omit_candidate_identity(self):
        rows = block()
        self.assertTrue(economics.compare_block(rows, [1, 2, 3], {})["comparable"])
        for field, value in (("draft_ms", 1), ("assistant", True), ("selected_neural", 1),
                             ("root_no_assistant", False)):
            changed = copy.deepcopy(rows)
            changed[0]["round"][field] = value
            self.assertFalse(economics.compare_block(changed, [1, 2, 3], {})["comparable"], field)
        rows[0]["round"]["prefix_hash"] = "different"
        self.assertFalse(economics.compare_block(rows, [1, 2, 3], {})["comparable"])

    def test_assistant_paid_root_has_no_unassisted_price_fallback(self):
        train = block("train")
        curves = economics.fit_curves(train, WIDTHS)
        paid = observation("adaptive")
        paid["round"].update(rows=1, accepted=0, committed=1)
        self.assertNotIn(economics.curve_key(paid), curves)

    def test_chain_families_use_same_strict_root_and_offer_contract(self):
        for family in ("mtp", "pld"):
            rows = block()
            for row in rows:
                row["mode"] = family + "-local"
                row["round"].update(family=family, kind="chain")
            self.assertTrue(economics.compare_block(rows, [1, 2, 3], {})["comparable"])
            rows[1]["round"]["candidate_hash"] = "other-chain"
            self.assertFalse(economics.compare_block(rows, [1, 2, 3], {})["comparable"])

    def test_full_offer_can_exceed_legal_verify_cap_without_silent_shrink(self):
        rows = block()
        for row in rows:
            if row["action"] != "fixed:1":
                row["round"]["candidate_rows"] = 8
        self.assertEqual(economics.receipt_errors(rows[-1]["round"], "max"), [])
        self.assertTrue(economics.compare_block(rows, [1, 2, 3], {})["comparable"])
        rows[-1]["round"]["legal_rows"] = 4
        self.assertIn("max_did_not_measure_full_offer", economics.receipt_errors(rows[-1]["round"], "max"))

    def test_plain_price_prior_is_only_measured_unassisted_verify_cost(self):
        curves = economics.fit_curves(block("train"), WIDTHS)
        with tempfile.TemporaryDirectory() as directory:
            paths = economics.price_files(Path(directory), curves)
            table = json.loads(paths["dflash-local/copy-ctx0"].read_text())
            self.assertEqual(table["schema"], 3)
            self.assertEqual(table["0"]["plain"], {"generic": {"1": 4.0}})
            self.assertEqual(table["0"]["tree"]["generic"]["2"], 5.0)
            paid_only = {key: cell for key, cell in curves.items()
                         if json.loads(key)[-1] != "no-assistant"}
            paths = economics.price_files(Path(directory), paid_only)
            table = json.loads(paths["dflash-local/copy-ctx0"].read_text())
            self.assertNotIn("plain", table["0"])

    def test_historical_context_semantics_cannot_be_relabelled_as_new_prices(self):
        historical = block("train")
        for row in historical:
            row["round"].pop("context_bucket")
        self.assertEqual(economics.fit_curves(historical, WIDTHS), {})
        current = economics.fit_curves(block("train"), WIDTHS)
        mixed = copy.deepcopy(current)
        next(iter(mixed.values())).pop("context_bucket")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(economics.runner.BenchError):
                economics.price_files(Path(directory), mixed)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_variant_prices_never_pool_or_backfill_at_identical_width(self):
        generic = block("train", 0)
        linear = block("train", 1)
        for row in linear:
            if row["round"]["rows"] == 3:
                row["round"].update(variant="linear_gdn", verify_commit_ms=18.0, round_ms=19.0)
        curves = economics.fit_curves(generic + linear, WIDTHS)
        self.assertEqual(curves[economics.curve_key(generic[2])]["verify_ms"], 9.0)
        self.assertEqual(curves[economics.curve_key(linear[2])]["verify_ms"], 18.0)
        self.assertEqual(curves[economics.curve_key(linear[2])]["n"], 1)
        with tempfile.TemporaryDirectory() as directory:
            paths = economics.price_files(Path(directory), curves)
            table = json.loads(paths["dflash-local/copy-ctx0"].read_text())
            self.assertEqual(table["0"]["tree"]["generic"]["3"], 9.0)
            self.assertEqual(table["0"]["tree"]["linear_gdn"], {"3": 18.0})
            self.assertEqual(table["0"]["plain"], {"generic": {"1": 4.0}})
        heldout = copy.deepcopy(linear)
        for row in heldout:
            row["phase"] = "heldout"
        summary = economics.summarize(heldout, WIDTHS, economics.fit_curves(generic, WIDTHS))
        self.assertFalse(summary["frozen_prediction_evidence_valid"])
        self.assertFalse(summary["same_state_counterfactual_valid"])
        missing = [row for row in summary["heldout_predictions"] if row["variant"] == "linear_gdn"]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(row["training_n"] == 0 for row in missing))

    def test_old_missing_ambiguous_or_conflicting_variant_evidence_is_rejected(self):
        historical = block("train")
        for row in historical:
            row["round"].pop("variant")
        self.assertEqual(economics.fit_curves(historical, WIDTHS), {})
        current = economics.fit_curves(block("train"), WIDTHS)
        key, cell = next(iter(current.items()))
        legacy_key = json.loads(key)
        del legacy_key[3]
        invalid = [
            {json.dumps(legacy_key): cell},
            {key: {k: v for k, v in cell.items() if k != "variant"}},
            {key: {**cell, "variant": "linear_gdn"}},
        ]
        for variant in (None, "auto", ["generic", "linear_gdn"]):
            parts = json.loads(key)
            parts[3] = variant
            invalid.append({json.dumps(parts): {**cell, "variant": variant}})
        with tempfile.TemporaryDirectory() as directory:
            for curves in invalid:
                with self.subTest(curves=curves):
                    with self.assertRaises(economics.runner.BenchError):
                        economics.price_files(Path(directory), {**current, **curves})
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prices_preserve_profile_binding_even_with_identical_context_and_variant(self):
        original = block("train")
        other = block("train")
        for row in other:
            row["profile"] = "prose-ctx0"
            row["round"]["verify_commit_ms"] *= 3
            row["round"]["round_ms"] *= 3
        widths = dict(WIDTHS, **{"dflash-local/prose-ctx0": [1, 2, 3]})
        curves = economics.fit_curves(original + other, widths)
        with tempfile.TemporaryDirectory() as directory:
            paths = economics.price_files(Path(directory), curves)
            self.assertEqual(set(paths), set(widths))
            copy_prices = json.loads(paths["dflash-local/copy-ctx0"].read_text())
            prose_prices = json.loads(paths["dflash-local/prose-ctx0"].read_text())
            self.assertEqual(copy_prices["0"]["tree"]["generic"]["2"], 5)
            self.assertEqual(prose_prices["0"]["tree"]["generic"]["2"], 15)

    def test_training_is_disjoint_and_complete(self):
        training = block("train")
        heldout = block("heldout")
        heldout[0]["round"]["verify_commit_ms"] = 100000
        curves = economics.fit_curves(training + heldout, WIDTHS)
        key = economics.curve_key(training[0])
        self.assertEqual(curves[key]["verify_ms"], 4)
        self.assertEqual(curves[key]["n"], 1)
        invalid = copy.deepcopy(training)
        invalid[1]["round"]["candidate_hash"] = "different"
        self.assertEqual(economics.fit_curves(invalid, WIDTHS), {})

    def test_no_cross_family_kind_variant_or_context_price_fallback(self):
        training = block("train")
        curves = economics.fit_curves(training, WIDTHS)
        for field, value in (("kind", "chain"), ("class", 4096), ("variant", "linear_gdn")):
            changed = observation("fixed:1")
            changed["round"][field] = value
            self.assertNotIn(economics.curve_key(changed), curves)
        changed = observation("fixed:1")
        changed["mode"] = "mtp-local"
        self.assertNotIn(economics.curve_key(changed), curves)

    def test_heldout_sample_counts_exclude_incomparables(self):
        train = block("train")
        good = block("heldout", 0)
        bad = block("heldout", 1)
        bad[1]["round"]["prefix_hash"] = "different"
        summary = economics.summarize(train + good + bad, WIDTHS, economics.fit_curves(train, WIDTHS))
        self.assertEqual(summary["n_heldout_blocks"], 2)
        self.assertEqual(summary["n_comparable_blocks"], 1)
        self.assertEqual(summary["prediction_errors"]["verify_ms"]["n"], 6)
        self.assertEqual(summary["prediction_errors"]["host_verify_ms"]["n"], 0)
        self.assertFalse(summary["canonical_speedup_claim"])
        self.assertFalse(summary["same_state_counterfactual_valid"])
        self.assertTrue(all(prediction["variant"] == "generic"
                            for prediction in summary["heldout_predictions"]))

    def test_contended_evidence_stays_diagnostic_not_hardware_calibration(self):
        train = stable_block("train", isolation="contended")
        heldout = stable_block(isolation="contended")
        diagnostic_curves = economics.fit_curves(train, WIDTHS)
        self.assertTrue(diagnostic_curves)
        self.assertEqual(economics.fit_curves(train, WIDTHS, causal=True), {})
        diagnostic = economics.summarize(heldout, WIDTHS, diagnostic_curves)
        causal = economics.summarize(heldout, WIDTHS, {}, causal=True)
        self.assertTrue(diagnostic["frozen_prediction_evidence_valid"])
        self.assertFalse(diagnostic["causal_economics_valid"])
        self.assertFalse(causal["causal_economics_valid"])
        self.assertEqual(causal["n_comparable_blocks"], 1)
        self.assertEqual(causal["n_evidence_blocks"], 0)
        self.assertEqual(causal["prediction_errors"]["verify_ms"]["n"], 0)
        self.assertIn("gpu_isolation_not_declared", causal["comparisons"][0]["causal_exclusions"])

    def test_one_unstable_action_excludes_whole_hardware_training_and_heldout_block(self):
        good_train = stable_block("train", 0)
        bad_train = stable_block("train", 1)
        bad_train[1]["machine_after"]["swapouts"] = 1
        curves = economics.fit_curves(good_train + bad_train, WIDTHS, causal=True)
        self.assertEqual(curves[economics.curve_key(good_train[0])]["n"], 1)
        good = stable_block("heldout", 0)
        bad = stable_block("heldout", 1)
        bad[1]["machine_after"]["pressure_level"] = 2
        summary = economics.summarize(good + bad, WIDTHS, curves, causal=True)
        self.assertEqual(summary["n_comparable_blocks"], 2)
        self.assertEqual(summary["n_evidence_blocks"], 1)
        self.assertEqual(summary["prediction_errors"]["verify_ms"]["n"], 6)
        self.assertFalse(summary["causal_economics_valid"])
        self.assertEqual(summary["boot_block_uncertainty"]["dflash-local/copy-ctx0"]
                         ["adaptive_regret_ms_per_token"]["n_blocks"], 1)

    def test_stability_must_cover_between_actions_not_only_each_request(self):
        rows = stable_block()
        rows[-1]["machine_before"]["power_source"] = "battery"
        rows[-1]["machine_after"]["power_source"] = "battery"
        self.assertIn("power_source_changed", economics.causal_block_errors(rows))
        missing = stable_block()
        missing[0]["machine_after"] = None
        self.assertTrue(economics.causal_block_errors(missing))

    def test_declared_stable_complete_blocks_can_have_separate_causal_evidence(self):
        train = stable_block("train")
        curves = economics.fit_curves(train, WIDTHS, causal=True)
        summary = economics.summarize(stable_block("heldout", 0) + stable_block("heldout", 1),
                                      WIDTHS, curves, causal=True)
        self.assertTrue(summary["causal_economics_valid"])
        self.assertEqual(summary["n_evidence_blocks"], 2)
        self.assertEqual(summary["prediction_scope"], "declared_isolation_observed_stable_blocks")

    def test_missing_training_withholds_prediction_validity(self):
        summary = economics.summarize(block(), WIDTHS, {})
        self.assertEqual(summary["n_comparable_blocks"], 1)
        self.assertFalse(summary["frozen_prediction_evidence_valid"])
        self.assertEqual(summary["n_completely_predicted_blocks"], 0)
        self.assertEqual(summary["prediction_errors"]["verify_ms"]["n"], 0)

    def test_bootstrap_unit_and_empty_prediction_boundaries(self):
        self.assertEqual(economics.interval([]), {"n_blocks": 0, "mean": None, "ci95": None})
        self.assertIsNone(economics.interval([1])["ci95"])
        self.assertEqual(economics.interval([1, 1])["ci95"], [1, 1])
        self.assertEqual(economics.prediction_summary([])["n"], 0)
        result = economics.prediction_summary([(2, 1), (4, 5)])
        self.assertEqual(result["mae"], 1)
        self.assertEqual(result["bias"], 0)
        self.assertEqual(sum(cell["n"] for cell in result["calibration"]), 2)

    def test_snapshot_fingerprint_tracks_content_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha = "11" * 32
            first_bytes = b"MLXLOOK1" + bytes.fromhex(sha) + struct.pack("<III", 10, 11, 1 << 30)
            second_bytes = b"MLXLOOK1" + bytes.fromhex(sha) + struct.pack("<III", 10, 12, 1 << 30)
            (root / "tokens.bin").write_bytes(first_bytes)
            first = economics.snapshot_files(root, sha)
            self.assertEqual(first[0]["documents"], 1)
            (root / "tokens.bin").write_bytes(second_bytes)
            self.assertNotEqual(first, economics.snapshot_files(root, sha))
            (root / "alias").symlink_to(root / "tokens.bin")
            with self.assertRaises(economics.runner.BenchError):
                economics.snapshot_files(root, sha)


    def test_snapshot_schema_rejects_corruption_and_wrong_tokenizer(self):
        sha = "11" * 32
        header = b"MLXLOOK1" + bytes.fromhex(sha)
        valid = header + struct.pack("<II", 10, 1 << 30)
        invalid = [b"", header, b"BROKEN!!" + valid[8:],
                   b"MLXLOOK1" + bytes.fromhex("22" * 32) + valid[40:],
                   valid[:-1], header + struct.pack("<II", 10, 11),
                   header + struct.pack("<II", 1 << 30, 1 << 30),
                   header + struct.pack("<III", 10, 1 << 30, 1 << 30),
                   header + struct.pack("<II", (1 << 30) + 1, 1 << 30)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for payload in invalid:
                (root / "tokens.bin").write_bytes(payload)
                with self.assertRaises(economics.runner.BenchError):
                    economics.snapshot_files(root, sha)
            (root / "tokens.bin").write_bytes(header + struct.pack("<IIIII", 10, 11, 1 << 30, 20, 1 << 30))
            self.assertEqual(economics.snapshot_files(root, sha)[0]["documents"], 2)

    def test_corpus_environment_is_limited_to_requested_tiers(self):
        for family in ("dflash", "mtp", "pld"):
            for tier in ("legacy", "local", "persistent"):
                self.assertIsNone(economics.corpus_for_mode(family + "-" + tier, "/prepared/corpus"))
            for tier in ("corpus", "all"):
                self.assertEqual(economics.corpus_for_mode(family + "-" + tier, "/prepared/corpus"),
                                 "/prepared/corpus")

    def test_persistence_requires_positive_matching_host_open_receipt(self):
        sha, path = "11" * 32, Path("/private/arm/store")
        opened = {"version": 1, "opened": True, "read_only": True, "path": str(path),
                  "tokenizer_sha256": sha, "token_count": 3}
        def log(receipt):
            return "[lookup-store] " + json.dumps(receipt) + "\n"
        self.assertEqual(economics.store_open_errors(log(opened), path, sha, 3), [])
        self.assertTrue(economics.store_open_errors("Model ready\n", path, sha, 3))
        for field, value in (("opened", False), ("read_only", False), ("path", "/other/store"),
                             ("tokenizer_sha256", "22" * 32), ("token_count", 0), ("version", 2)):
            self.assertTrue(economics.store_open_errors(log({**opened, field: value}), path, sha, 3), field)
        self.assertTrue(economics.store_open_errors(
            log(opened) + "[lookup] persistent store unavailable: TokenizerMismatch\n", path, sha, 3))


if __name__ == "__main__":
    unittest.main()
