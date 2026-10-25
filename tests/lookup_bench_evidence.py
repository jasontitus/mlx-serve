"""Pure evidence reducers for the owned offline benchmark (no inference or network)."""
import hashlib
import json
import math
import random
import statistics
import re


def stable_states(states):
    errors = []
    if not states:
        return ["machine_state_missing"]
    for key in ("power_source", "power_settings", "thermal", "pressure_level", "swap_used_bytes", "swapins", "swapouts"):
        values = [s.get(key) for s in states]
        if any(v is None or v == "" or v == "unknown" for v in values):
            errors.append(key + "_unknown")
        elif len(set(values)) != 1:
            errors.append(key + "_changed")
    if any(s.get("pressure_level") != 1 for s in states):
        errors.append("memory_pressure_not_normal")
    for state in states:
        thermal = (state.get("thermal") or "").lower()
        normal_notes = "no thermal warning" in thermal or "no performance warning" in thermal
        limits = re.findall(r"cpu_(?:speed|scheduler)_limit\s*=\s*(\d+)", thermal)
        if any(int(limit) < 100 for limit in limits) or (
                ("warning" in thermal and not normal_notes) or
                ("limit" in thermal and not normal_notes and not limits)):
            errors.append("thermal_warning")
    return sorted(set(errors))


def transfer_measurement_errors(boots, isolation):
    states = [boot.get(key, {}) for boot in boots for key in ("machine_before", "machine_after")]
    errors = stable_states(states)
    if isolation != "declared":
        errors.append("gpu_isolation_not_declared")
    return sorted(set(errors))


def selected_origin(rounds, tier):
    """Only settled selected-node provenance, never proposal/source unions."""
    keys = {"local": ("selected_local",), "corpus": ("selected_corpus",),
            "persistent": ("selected_persistent",),
            "all": ("selected_local", "selected_corpus", "selected_persistent")}.get(tier, ())
    return any(r.get("version") == 1 and any(isinstance(r.get(k), int) and r[k] > 0 for k in keys)
               and r.get("rows", 0) > 1 for r in rounds)


def response_equal(left, right):
    if not left or not right or not left.get("complete") or not right.get("complete"):
        return False
    if any(left.get(k) != right.get(k) for k in ("text", "finish_reason", "finish_details")):
        return False
    if (left.get("usage") or {}).get("completion_tokens") != (right.get("usage") or {}).get("completion_tokens"):
        return False
    a, b = left.get("commit"), right.get("commit")
    if a is not None or b is not None:
        return bool(a and b and all(a.get(k) == b.get(k) for k in ("prompt_ids", "generated_ids", "finish_reason")))
    return True


def request_identity(path, body):
    body = {k: v for k, v in body.items()
            if k not in ("enable_mtp", "enable_drafter", "enable_pld")}
    return hashlib.sha256(json.dumps([path, body], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def supported_correctness(requests):
    supported = [r for r in requests if r.get("exactness_contract_supported") and not r.get("error")]
    return bool(supported) and all(r.get("emission_exact") is True for r in supported)


def attach_canonical_receipts(captures, commits, rounds, decoded_prompts, rendered_prompt_ids=None):
    """Join actual commits; reference render IDs disambiguate tools/thinking."""
    rendered_prompt_ids = rendered_prompt_ids or {}
    def content_parts(body):
        if isinstance(body.get("prompt"), str):
            return [body["prompt"]]
        parts = []
        for message in body.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(p["text"] for p in content
                             if isinstance(p, dict) and isinstance(p.get("text"), str))
        return [part for part in parts if part]

    for capture in captures:
        body = capture["request"]
        parts = content_parts(body)
        ids = body.get("prompt")
        rendered = rendered_prompt_ids.get(request_identity(capture.get("path", ""), body))
        rendered_ambiguous = rendered is not None and len(rendered) != 1
        capture["receipt_match_basis"] = ("reference_rendered_prompt_ids" if rendered is not None
                                          else "decoded_content_candidates")
        capture["reference_prompt_ambiguous"] = rendered_ambiguous
        candidates = []
        for commit in commits:
            key = str(commit.get("prompt_hash"))
            text = decoded_prompts.get(key)
            if rendered is not None:
                matches = not rendered_ambiguous and commit.get("prompt_ids") == rendered[0]
            elif isinstance(ids, list):
                matches = commit.get("prompt_ids") == ids
            else:
                matches = bool(parts) and text is not None and all(p in text for p in parts)
            if matches:
                candidates.append(commit)
        identities = {json.dumps([c.get(k) for k in
                                  ("prompt_ids", "generated_ids", "finish_reason")],
                                 separators=(",", ":")) for c in candidates}
        capture["rounds"] = []
        capture["response"].pop("commit", None)
        capture["concurrent_or_receipt_ambiguous"] = rendered_ambiguous or len(identities) != 1
        if rendered is None:
            capture["content_receipt_ambiguous"] = capture["concurrent_or_receipt_ambiguous"]
        capture["round_attribution_ambiguous"] = len(candidates) != 1
        if len(identities) == 1:
            commit = candidates[0]
            capture["response"]["commit"] = commit
            if len(candidates) == 1:
                capture["rounds"] = [r for r in rounds
                                     if r.get("prompt_hash") == commit.get("prompt_hash")]
        capture["receipt_match_count"] = len(candidates)
    return captures


def canonical_reference_modes(mode):
    serial = "serial-dflash" if mode.startswith("dflash-") or mode == "serial-dflash" else "serial"
    legacy = mode.split("-")[0] + "-legacy" if not mode.startswith("serial") else mode
    return list(dict.fromkeys((serial, legacy)))


def token_receipt(response):
    commit = (response or {}).get("commit") or {}
    return bool(isinstance(commit.get("prompt_ids"), list) and commit["prompt_ids"]
                and isinstance(commit.get("generated_ids"), list)
                and commit.get("finish_reason") is not None)


def classify_canonical_replay(probe, replay_rows, reference_modes, mode, row_exact_tree=False):
    """Offline/live gate for defined metric requests, not blanket CLI correctness.

    Raw auxiliary responses and differences remain diagnostics. The reference
    commit teaches rendering only: generated IDs must still match an independent
    canonical commit, never be copied from the reference into the capture.
    """
    references = list(dict.fromkeys(reference_modes))
    captures = probe.get("responses", [])
    by_identity = {}
    rendered = {}
    for row in replay_rows:
        if row.get("reference") not in references or "request" not in row or "path" not in row:
            continue
        key = request_identity(row["path"], row["request"])
        by_identity.setdefault((key, row["reference"]), []).append(row)
        if token_receipt(row.get("response")):
            ids = row["response"]["commit"]["prompt_ids"]
            choices = rendered.setdefault(key, [])
            if ids not in choices:
                choices.append(ids)
    metrics = probe.get("metrics") or {}
    # Saved evidence already contains the whole-log commits and rounds. No
    # detokenization, prompt reconstruction, or timing-window inference is needed.
    resolvable = [c for c in captures if request_identity(c["path"], c["request"]) in rendered]
    if metrics.get("commits"):
        attach_canonical_receipts(resolvable, metrics["commits"], metrics.get("rounds", []),
                                  {}, rendered)
    defined = set(probe.get("metric_values") or {})
    expected = probe.get("metric_request_counts") or {}
    coverage_errors = []
    for metric in sorted(defined):
        count = sum(metric in c.get("metrics", []) for c in captures)
        if not count or (metric in expected and count != expected[metric]):
            coverage_errors.append({"metric": metric, "captures": count,
                                    "expected": expected.get(metric, "at_least_one")})
    measured = [c for c in captures if defined.intersection(c.get("metrics", []))]
    results = []
    for capture in captures:
        key = request_identity(capture["path"], capture["request"])
        in_scope = bool(defined.intersection(capture.get("metrics", [])))
        for reference in references:
            rows = by_identity.get((key, reference), [])
            rows = [r for r in rows if r.get("request_index") is None or
                    r["request_index"] == capture.get("request_index")]
            receipts_ok = bool(rows) and token_receipt(capture.get("response")) and all(
                token_receipt(r.get("response")) for r in rows)
            exact = bool(receipts_ok and all(
                response_equal(capture["response"], r["response"]) for r in rows))
            contract = bool(receipts_ok and not capture.get("concurrent_or_receipt_ambiguous") and
                            (capture["request"].get("temperature", 0) == 0 or
                             (mode.startswith("dflash-") and row_exact_tree)))
            results.append({"request_id": capture["request_id"],
                            "request_index": capture.get("request_index"), "reference": reference,
                            "metrics": capture.get("metrics", []), "in_metric_scope": in_scope,
                            "supported": in_scope and contract, "emission_exact": exact,
                            "token_ids_compared": bool(receipts_ok), "reference_matches": len(rows),
                            "scope_label": ("defined_metric_request" if in_scope else
                                            "unsupported_unmeasured_auxiliary_or_batching")})
    required = [r for r in results if r["in_metric_scope"]]
    auxiliary = [r for r in results if not r["in_metric_scope"]]
    covered = bool(defined and measured and references and required) and not coverage_errors
    return {"successful": covered and all(r["emission_exact"] for r in required),
            "supported": covered and all(r["supported"] for r in required),
            "scope": "defined_single_stream_metric_requests_only",
            "blanket_canonical_correctness": False,
            "requests": len(measured), "captured_requests": len(captures),
            "coverage_errors": coverage_errors, "results": results,
            "auxiliary_diagnostics": {
                "supported": False, "requests": len(captures) - len(measured),
                "comparisons": len(auxiliary),
                "observed_differences_or_missing_receipts": sum(not r["emission_exact"] for r in auxiliary),
                "label": "unmeasured_setup_and_batching_not_certified"},
            "reference_modes": references, "timing": "excluded diagnostic exact-request replay"}


def metric_cells(probe, tier):
    """Attribution is calculated independently for every metric's actual request set."""
    result = []
    for metric, value in (probe.get("metric_values") or {}).items():
        captures = [r for r in probe.get("responses", []) if metric in r.get("metrics", [])]
        execution = bool(captures) and all(r.get("response", {}).get("complete") and not r.get("concurrent_or_receipt_ambiguous") for r in captures)
        attributed = execution and all(selected_origin(r.get("rounds", []), tier) for r in captures)
        origin_counts = {name: sum(r.get("selected_" + name, 0) or 0
                                  for capture in captures for r in capture.get("rounds", []))
                         for name in ("local", "corpus", "persistent", "neural")}
        result.append({"metric": metric, "value": value, "execution_valid": execution,
                       "feature_attribution_valid": attributed,
                       "attribution_label": "selected_requested_origin" if attributed else "no_hit_or_missing_selected_origin",
                       "request_ids": [r["request_id"] for r in captures], "selected_origins": origin_counts})
    return result


def reduce_arms(events, requests, isolation, config):
    latest = {e["arm_id"]: dict(e) for e in events}
    active = [dict(r) for r in requests if r.get("attempt") == latest.get(r["arm_id"], {}).get("attempt")]
    # Old evidence has no explicit block IDs; still reduce it conservatively, never infer controls.
    blocks = {}
    for event in latest.values():
        blocks.setdefault((event.get("block", "legacy-evidence"), event["boot"]), []).append(event)
    diagnostics = []
    by_request = {(r["arm_id"], r["stage"], r["repetition"], r["case"]): r for r in active}
    for row in active:
        event = latest[row["arm_id"]]
        baseline_id = event.get("serial_arm_id", f"{row['boot']}:" + ("serial-dflash" if row["mode"].startswith("dflash-") else "serial"))
        baseline = by_request.get((baseline_id, row["stage"], row["repetition"], row["case"]))
        exact = None
        if baseline and not row.get("error") and not baseline.get("error"):
            exact = all(row.get(k) == baseline.get(k) for k in ("text", "finish_reason", "finish_details")) and (row.get("usage") or {}).get("completion_tokens") == (baseline.get("usage") or {}).get("completion_tokens")
            ca = row.get("metrics", {}).get("commits", [])
            cb = baseline.get("metrics", {}).get("commits", [])
            if ca or cb:
                exact = exact and len(ca) == len(cb) == 1 and all(ca[0].get(k) == cb[0].get(k) for k in ("prompt_ids", "generated_ids", "finish_reason"))
        row["emission_exact"] = exact
        labels = []
        if row.get("error"):
            labels.append("failed")
        if not row.get("exactness_contract_supported"):
            labels.append("unsupported_seeded_exactness_contract")
        elif exact is not True and not row["mode"].startswith("serial"):
            labels.append("mismatch" if exact is False else "missing_baseline")
        if not row.get("metrics", {}).get("family_engaged") and not row["mode"].startswith("serial"):
            labels.append("disengaged")
        if isolation != "declared":
            labels.append("contended")
        if event["status"] != "complete":
            labels.append("incomplete_arm")
        if row["stage"] in ("cold", "warm"):
            diagnostics.append({"arm_id": row["arm_id"], "mode": row["mode"], "boot": row["boot"],
                                "stage": row["stage"], "case": row["case"], "kind": row.get("kind"),
                                "repetition": row["repetition"], "emission_exact": exact,
                                "labels": ",".join(labels) or "emission_match",
                                "valid_comparison": exact is True and not labels,
                                "total_seconds": row.get("total_seconds"),
                                "feature_attribution_valid": selected_origin(row.get("metrics", {}).get("rounds", []), row["mode"].split("-", 1)[-1])})
    for block_events in blocks.values():
        states = []
        block_errors = []
        stability_errors = []
        if isolation != "declared":
            stability_errors.append("contended")
        if len({e.get("session_id") for e in block_events}) != 1:
            block_errors.append("cross_session_block")
        for event in block_events:
            probe = event.get("canonical_probe") or {}
            if event["status"] != "complete":
                block_errors.append("incomplete_block")
            if not probe.get("successful"):
                block_errors.append("canonical_execution_invalid")
            states.extend([probe.get("machine_before", {}), probe.get("machine_after", {})])
            states.extend(probe.get("machine_samples", []))
            own_requests = [r for r in active if r["arm_id"] == event["arm_id"]]
            for row in own_requests:
                states.extend([row.get("machine_before", {}), row.get("machine_after", {})])
            if not event["mode"].startswith("serial") and not supported_correctness(
                    [r for r in own_requests if r["stage"] in ("cold", "warm")]):
                block_errors.append("block_fixture_correctness_invalid")
            if probe.get("report_errors"):
                block_errors.append("canonical_report_invalid")
                if any("loadDrift" in error for error in probe["report_errors"]):
                    stability_errors.append("load_drift")
            stability_errors.extend(probe.get("stability_errors", []))
            replay = probe.get("matched_replay") or {}
            if not replay.get("successful") or not replay.get("supported"):
                block_errors.append("block_correctness_invalid")
        stability_errors += stable_states(states)
        block_errors += stability_errors
        for event in block_events:
            probe = event.get("canonical_probe") or {}
            own = [r for r in active if r["arm_id"] == event["arm_id"] and r["stage"] in ("cold", "warm")]
            serial = event["mode"].startswith("serial")
            fixture_ok = serial or supported_correctness(own)
            replay = probe.get("matched_replay") or {}
            replay_ok = bool(replay.get("successful") and replay.get("supported") and replay.get("requests", 0) > 0)
            execution = event["status"] == "complete" and probe.get("successful", False)
            correctness = fixture_ok and replay_ok
            controls = [latest.get(event.get(k)) for k in ("serial_arm_id", "legacy_arm_id", "aa_arm_id", "cut_arm_id", "max_arm_id")]
            controls_ok = all(c and c["status"] == "complete" and (c.get("canonical_probe") or {}).get("successful")
                              and ((c.get("canonical_probe") or {}).get("matched_replay") or {}).get("successful")
                              and ((c.get("canonical_probe") or {}).get("matched_replay") or {}).get("supported")
                              for c in controls)
            matching = bool(probe.get("request_sequence")) and controls_ok and all(
                (c.get("canonical_probe") or {}).get("request_sequence") == probe.get("request_sequence") for c in controls)
            frozen = event.get("persistent_state_valid", not event["mode"].endswith(("-persistent", "-all")))
            stable = not stability_errors
            cells = metric_cells(probe, event["mode"].split("-", 1)[-1])
            metric_execution = bool(cells) and all(c["execution_valid"] for c in cells)
            attributed = metric_execution and all(c["feature_attribution_valid"] for c in cells)
            engaged = serial or probe.get("metrics", {}).get("family_engaged", False)
            correctness = correctness and engaged
            if event["mode"].endswith(("-persistent", "-all")):
                peers = [c for c in controls if c and c["mode"].endswith(("-persistent", "-all"))]
                frozen = frozen and all(c.get("persistent_state_valid") and
                    (c.get("frozen_store") or {}).get("sha256") == (event.get("frozen_store") or {}).get("sha256") for c in peers)
            errors = list(block_errors)
            if not fixture_ok:
                errors.append("supported_correctness_missing_or_failed")
            if not replay_ok:
                errors.append("exact_request_replay_missing_or_failed")
            if not controls_ok:
                errors.append("missing_completed_legacy_aa_max_controls")
            if controls_ok and not matching:
                errors.append("canonical_request_sequence_mismatch")
            if not frozen:
                errors.append("persistent_state_invalid")
            if not engaged:
                errors.append("probe_disengaged")
            if not attributed:
                errors.append("metric_selected_origin_missing")
            if not metric_execution:
                errors.append("metric_request_execution_invalid")
            eligible = bool(not serial and execution and correctness and not block_errors and matching and frozen and metric_execution)
            for cell in cells:
                cell.update(correctness_valid=bool(correctness), stability_valid=stable,
                            incremental_comparison_valid=eligible and cell["execution_valid"])
            event.update(execution_valid=bool(execution), correctness_valid=bool(correctness),
                         stability_valid=stable, feature_attribution_valid=bool(attributed),
                         incremental_comparison_valid=eligible,
                         canonical_comparison_valid=eligible and attributed,
                         canonical_comparison_labels=sorted(set(errors)), metric_cells=cells)
    return list(latest.values()), diagnostics


def bootstrap_effect(values, margin, replications, minimum):
    if len(values) < minimum:
        return {"boot_blocks": len(values), "required_boot_blocks": minimum, "verdict": "inconclusive_insufficient_blocks", "ratios": values}
    logs = [math.log(v) for v in values]
    rng = random.Random(0)
    draws = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(replications))
    low, high = draws[int(.025 * replications)], draws[min(replications - 1, int(.975 * replications))]
    return {"boot_blocks": len(values), "ratio": math.exp(statistics.mean(logs)), "ci95": [low, high],
            "practical_margin": margin, "ratios": values,
            "verdict": "win" if low > 1 + margin else "regression" if high < 1 / (1 + margin) else "inconclusive"}


def candidate_match(left, right, metric):
    def index(probe):
        result = {}
        for capture in probe.get("responses", []):
            if metric not in capture.get("metrics", []):
                continue
            for row in capture.get("rounds", []):
                key = (capture["request_id"], row.get("family"), row.get("kind"),
                       row.get("class"), row.get("anchor_pos"), row.get("prefix_hash"))
                result[key] = (row.get("candidate_hash"), row.get("candidate_rows"),
                               row.get("candidate_probability_hash"), row.get("legal_rows"))
        return result
    a, b = index(left), index(right)
    common = set(a).intersection(b)
    valid = bool(common) and all(key[-1] is not None and
                                all(value is not None for value in a[key]) and
                                a[key] == b[key] for key in common)
    request_ids = {r["request_id"] for r in left.get("responses", []) if metric in r.get("metrics", [])}
    valid = valid and request_ids == {key[0] for key in common}
    return {"valid": valid, "matched_prefixes": len(common),
            "candidate_mismatches": sum(a[key] != b[key] for key in common),
            "left_only_prefixes": len(set(a) - set(b)), "right_only_prefixes": len(set(b) - set(a))}


def incremental_effects(arms, config):
    by_id = {a["arm_id"]: a for a in arms}
    groups = {}
    aa_groups = {}
    for arm in arms:
        if arm.get("slot") != "feature" or not arm.get("incremental_comparison_valid"):
            continue
        legacy = by_id[arm["legacy_arm_id"]]
        aa = by_id[arm["aa_arm_id"]]
        maximum = by_id[arm["max_arm_id"]]
        cut = by_id[arm["cut_arm_id"]]
        for cell in arm["metric_cells"]:
            metric, value = cell["metric"], cell["value"]
            lv = (legacy.get("canonical_probe", {}).get("metric_values") or {}).get(metric)
            av = (aa.get("canonical_probe", {}).get("metric_values") or {}).get(metric)
            mv = (maximum.get("canonical_probe", {}).get("metric_values") or {}).get(metric)
            cv = (cut.get("canonical_probe", {}).get("metric_values") or {}).get(metric)
            if not all(isinstance(v, (float, int)) and v > 0 for v in (value, lv, av, mv, cv)):
                continue
            ratio = lv / value if "ttft" in metric.lower() else value / lv
            aa_ratio = lv / av if "ttft" in metric.lower() else av / lv
            cut_ratio = mv / cv if "ttft" in metric.lower() else cv / mv
            key = (arm["mode"], metric)
            candidates = candidate_match(cut.get("canonical_probe") or {}, maximum.get("canonical_probe") or {}, metric)
            groups.setdefault(key, []).append((ratio, cut_ratio, cell["feature_attribution_valid"], candidates))
            aa_groups.setdefault(key, []).append(aa_ratio)
    results = []
    for key, data in groups.items():
        options = (config.get("practical_margin", .05), config.get("bootstrap_replications", 10000),
                   max(4, config.get("boot_repetitions", 4)))
        effect = bootstrap_effect([d[0] for d in data], *options)
        aa = bootstrap_effect(aa_groups[key], *options)
        aa_ok = "ci95" in aa and aa["ci95"][0] >= 1 / (1 + options[0]) and aa["ci95"][1] <= 1 + options[0]
        if not aa_ok:
            effect["verdict"] = "inconclusive_aa_noise"
        cut_effect = bootstrap_effect([d[1] for d in data], *options)
        if not all(d[3]["valid"] for d in data):
            cut_effect["verdict"] = "inconclusive_candidate_identity"
        elif not aa_ok:
            cut_effect["verdict"] = "inconclusive_aa_noise"
        results.append({"mode": key[0], "metric": key[1], "legacy_primary": effect, "aa_control": aa,
                        "cut_vs_max": cut_effect, "candidate_evidence": [d[3] for d in data],
                        "feature_attribution_valid": all(d[2] for d in data),
                        "performance_cell_retained_without_hits": not all(d[2] for d in data)})
    return results


def transfer_phase_pass(comparisons, cases, pairs, related):
    """Require every declared case/pair; never average away a bad cold control."""
    if len(comparisons) != len(cases) or {c["case"] for c in comparisons} != set(cases):
        return False
    for case in comparisons:
        rows = case.get("pairs", [])
        if len(rows) != pairs or {row.get("pair") for row in rows} != set(range(pairs)):
            return False
        threshold = 5 if case["case"] in related else -5
        for row in rows:
            value = row.get("latency_reduction_percent")
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
            if value < threshold and not math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12):
                return False
    return True


def transfer_gate_decisions(family_passes, fastest_passes, pairs, full_workload,
                            same_window, valid=True):
    """Keep family benefit distinct from beating every applicable native family.

    Input arm passes already include independent cold/timed thresholds, accepted
    tier activation, exactness, frozen history and numerical-mode checks.
    A separate-window native comparison is descriptive, never confirmation.
    """
    family = {arm: bool(valid and passed) for arm, passed in family_passes.items()}
    fastest = {arm: bool(family[arm] and same_window and fastest_passes.get(arm, False))
               for arm in family}
    external = ("corpus", "persistent")
    family_diagnostic = pairs in (2, 4) and any(family.get(arm, False) for arm in external)
    fastest_diagnostic = pairs in (2, 4) and any(fastest.get(arm, False) for arm in external)
    family_confirmation = bool(family_diagnostic and full_workload and pairs == 4)
    fastest_confirmation = bool(fastest_diagnostic and full_workload and pairs == 4)
    return {
        "gate_schema": 3,
        "evidence_level": "confirmation" if pairs == 4 else "diagnostic",
        "arm_family_improvement_pass": family,
        "arm_fastest_native_pass": fastest,
        "family_diagnostic_pass": bool(family_diagnostic),
        "family_confirmation_pass": family_confirmation,
        "fastest_native_diagnostic_pass": bool(fastest_diagnostic),
        "fastest_native_confirmation_pass": fastest_confirmation,
        # Unqualified historical fields remain conservative, never a family-only pass.
        "arm_performance_pass": fastest,
        "diagnostic_pass": bool(fastest_diagnostic),
        "full_workload_diagnostic_pass": bool(fastest_diagnostic and full_workload),
        "confirmation_pass": fastest_confirmation,
        "performance_pass": fastest_confirmation,
        "promotion_pass": False,
    }
