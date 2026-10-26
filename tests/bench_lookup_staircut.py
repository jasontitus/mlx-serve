#!/usr/bin/env python3
"""Offline, sequential StairCut comparisons using a prepared lookup-staircut kit.

Examples (run from the repository root after asset preparation):
  python3 tests/bench_lookup_staircut.py --model smoke --dry-run
  python3 tests/bench_lookup_staircut.py --model smoke --skip-probe --repetitions 1 --warmup 1 --max-tokens 64 --output /tmp/lookup-smoke
  python3 tests/bench_lookup_staircut.py --model flashnext --modes serial,mtp-legacy,mtp-local,mtp-corpus,mtp-persistent,mtp-all --output ~/lookup-flight

Each feature has an independent paired boot block: serial reference, adjacent
legacy A/A and legacy/feature controls, forced-proposal cut-only and max ablations.
The order reverses across predeclared --boot-repetitions (default 4).
Interrupted slots restart from fresh policy/store state; evidence is preserved.
Persistent seed documents are checked against exact commit IDs, reopened for a
designated seed-only retrieval, then frozen throughout prescribed training and
measurement. Cold means only the first request is policy-cold.
Canonical llmprobe changes only the recorded shared per-block cache-bust tag.
Every captured canonical body is replayed against unique serial/legacy references.
Exactness gates every defined decode/prefill/context and predictable/novel metric
request; unmeasured setup/batching responses remain unsupported diagnostics.
Raw token receipts are reversible private data (output directories/files 0700/0600).
Custom timings and replay are diagnostics; use --skip-probe for diagnostic-only work.
Chat diagnostic cases use messages instead of prompt; enable_thinking and
reasoning_effort are forwarded only when explicitly supplied by the case.
"""

import argparse
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import socket
import threading
import subprocess
import sys
import time
import struct
import urllib.error
import urllib.request

from lookup_bench_evidence import (incremental_effects, reduce_arms, request_identity,
                                   selected_origin, attach_canonical_receipts,
                                   canonical_reference_modes, classify_canonical_replay)


VERSION = 1
TIERS = ("legacy", "local", "corpus", "persistent", "all")
MODES = ("serial", "serial-dflash") + tuple(
    f"{family}-{tier}" for family in ("mtp", "dflash", "pld") for tier in TIERS
)
KV = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=(\"[^\"]*\"|\S+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class BenchError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BenchError(f"refusing HTTP redirect to {newurl}")


HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inventory(directory):
    """Cheap weight identity plus content hashes of small configuration files."""
    root = Path(directory)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns,
                          digest(path) if path.suffix == ".json" else None])
    return hashlib.sha256(canonical(files).encode()).hexdigest()


def absolute_asset(value, directory=False):
    path = Path(value)
    if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
        raise BenchError(f"missing absolute local {'directory' if directory else 'file'}: {path}")
    return str(path.resolve())



def runtime_identity(binary):
    """Freeze actual Darwin linkage and conservative Metal-library load candidates.

    Read installed Mach-O commands, never build sources. Resolve dyld run-path
    stacks in load order, retain logical/symlink paths and every resolution
    candidate, and hash all non-system transitive images. System shared-cache
    libraries are identified by their install names plus the OS build/kernel.
    Metal assets include installed resources and absolute paths embedded in the
    loaded images; absent candidates are recorded so a newly appearing override
    invalidates the declaration too. This does not launch the executable.
    """
    if sys.platform != "darwin":
        raise BenchError("runtime identity currently requires Darwin Mach-O linkage")
    executable = Path(os.path.abspath(binary))
    if not executable.is_file():
        raise BenchError("runtime executable is missing: " + str(executable))
    environment = clean_environment("legacy")
    architecture = os.uname().machine

    def command(arguments):
        try:
            run = subprocess.run(arguments, env=environment, capture_output=True,
                                 text=True, check=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BenchError("cannot inspect runtime linkage: " + str(exc)) from exc
        return run.stdout

    def system(path):
        return path.startswith(("/usr/lib/", "/System/Library/"))

    def expand(value, loader):
        for token, root in (("@loader_path", loader.parent), ("@executable_path", executable.parent)):
            if value == token or value.startswith(token + "/"):
                return os.path.normpath(str(root) + value[len(token):])
        if value.startswith("/"):
            return os.path.normpath(value)
        raise BenchError("unsupported relative runtime load path: " + value)

    images, edges, system_libraries, metal_candidates = {}, [], set(), set()
    metal_directories, mlx_images = set(), []

    def visit(path, inherited_rpaths):
        logical = str(path)
        resolved = str(path.resolve())
        if resolved in images:
            return
        if not path.is_file():
            raise BenchError("unresolved non-system runtime image: " + logical)
        listing = command(["/usr/bin/otool", "-arch", architecture, "-L", logical])
        commands = command(["/usr/bin/otool", "-arch", architecture, "-l", logical])
        blocks = re.split(r"(?m)^Load command \d+\s*$", commands)[1:]
        rpaths, dependencies, install_id = [], [], None
        for block in blocks:
            kind = re.search(r"(?m)^\s*cmd (LC_\w+)\s*$", block)
            if not kind:
                continue
            kind = kind.group(1)
            if kind == "LC_RPATH":
                value = re.search(r"(?m)^\s*path (.+) \(offset \d+\)\s*$", block)
                if not value:
                    raise BenchError("unparseable LC_RPATH in " + logical)
                rpaths.append(value.group(1))
            elif kind in ("LC_ID_DYLIB", "LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB",
                          "LC_REEXPORT_DYLIB", "LC_LAZY_LOAD_DYLIB", "LC_LOAD_UPWARD_DYLIB"):
                value = re.search(r"(?m)^\s*name (.+) \(offset \d+\)\s*$", block)
                if not value:
                    raise BenchError("unparseable dylib load command in " + logical)
                if kind == "LC_ID_DYLIB":
                    install_id = value.group(1)
                else:
                    dependencies.append({"name": value.group(1), "command": kind})
        listed = re.findall(r"(?m)^\s+(.+) \(compatibility version ", listing)
        if listed != ([install_id] if install_id else []) + [dep["name"] for dep in dependencies]:
            raise BenchError("otool dependency views disagree: " + logical)
        stack = list(dict.fromkeys([expand(value, path) for value in rpaths] + inherited_rpaths))
        images[resolved] = {"path": logical, "resolved": resolved, "sha256": digest(path),
                            "install_id": install_id, "rpaths": rpaths,
                            "effective_rpaths": stack, "dependencies": dependencies}
        # Extract literal absolute Metal paths from this actual image, not sources.
        for match in re.finditer(rb"/[ -~]{1,4096}\.metallib(?=\x00)", path.read_bytes()):
            metal_candidates.add(os.path.normpath(match.group().decode("ascii")))
        if path.name.startswith("libmlx"):
            mlx_images.append(logical)
            for image in (path, Path(resolved)):
                metal_directories.update((image.parent, image.parent / "Resources",
                                          image.parent.parent / "Resources"))
        for dep in dependencies:
            name = dep["name"]
            candidates = ([os.path.normpath(base + name[len("@rpath"):]) for base in stack]
                          if name.startswith("@rpath/") else [expand(name, path)])
            selected = next((value for value in candidates if system(value) or Path(value).is_file()), None)
            if selected is None:
                raise BenchError("unresolved non-system dependency " + name + " from " + logical)
            edges.append({"loader": logical, "name": name, "command": dep["command"],
                          "candidates": candidates, "selected": selected,
                          "resolved": selected if system(selected) else str(Path(selected).resolve())})
            if system(selected):
                system_libraries.add(selected)
            else:
                visit(Path(selected), stack)

    visit(executable, [])
    metal_directories.update((executable.parent, executable.parent / "Resources",
                              executable.parent.parent / "Resources"))
    for directory in metal_directories:
        metal_candidates.update(str(directory / name) for name in ("mlx.metallib", "default.metallib"))
        if directory.is_dir():
            metal_candidates.update(str(path) for path in directory.glob("*.metallib"))
    metal = {}
    for candidate in sorted(metal_candidates):
        path = Path(candidate)
        metal[candidate] = {"exists": path.is_file(), "resolved": str(path.resolve()),
                           "sha256": digest(path) if path.is_file() else None}
    if mlx_images and not any(asset["exists"] for asset in metal.values()):
        raise BenchError("linked MLX runtime has no identifiable Metal-library asset")
    return {"version": 1, "executable": str(executable), "architecture": architecture,
            "os_version": command(["/usr/bin/sw_vers"]).strip(),
            "kernel": dict(zip(("system", "node", "release", "version", "machine"), os.uname())),
            "images": images, "load_edges": edges, "system_libraries": sorted(system_libraries),
            "metal_assets": metal, "metal_search_directories": sorted(str(path) for path in metal_directories),
            "loading_policy": {"DYLD_*": "removed", "LD_LIBRARY_PATH": "removed", "LD_PRELOAD": "removed",
                               "MLX_*": "removed before explicit benchmark controls",
                               "resolution": "Mach-O loader-first inherited LC_RPATH stack; no unresolved non-system paths",
                               "metal": "Conservative installed/bundle and image-embedded candidates; all contents frozen"}}

def http_json(base, endpoint, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + endpoint, data,
                                     {"Content-Type": "application/json"})
    with HTTP.open(request, timeout=timeout) as response:
        result = json.load(response)
    if isinstance(result, dict) and result.get("error"):
        raise BenchError(canonical(result))
    return result


BENCHMARK_EXPERIMENT_KEYS = {
    "MLX_SERVE_LOOKUP_SELECTION", "MLX_SERVE_LOOKUP_STORE_WRITE",
    "MLX_SERVE_LOOKUP_RECEIPTS", "MLX_SERVE_LOOKUP_PRICES", "MLX_SERVE_ROW_EXACT",
    "MLX_SERVE_LOOKUP_PRICE_TRACE", "MLX_SERVE_LOOKUP_COUNTERFACTUAL",
    "MLX_SERVE_LOOKUP_COUNTERFACTUAL_ACTION", "MLX_SERVE_LOOKUP_COUNTERFACTUAL_PREPARATION",
    "MLX_SERVE_LOOKUP_CANDIDATES", "MLX_SERVE_LOOKUP_BACKOFF",
}


def evidence_environment(environment):
    """Record reproducibility controls, never arbitrary inherited credentials."""
    allowed = BENCHMARK_EXPERIMENT_KEYS | {
        "HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY", "NO_PROXY", "MLX_SERVE_LOOKUP_TRACE",
        "MLX_SERVE_ROUND_COST_PERSIST", "MLX_SERVE_LOOKUP",
        "MLX_SERVE_LOOKUP_CORPUS", "MLX_SERVE_LOOKUP_STORE",
    }
    return {key: value for key, value in environment.items() if key in allowed}


def clean_environment(tier, corpus=None, store=None, experiment=None):
    # Do not inherit GPU/kernel/acceptance experiments, price files or network proxies.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MLX_", "DSV", "QWEN", "ANE_", "LLMPROBE_", "NPM_", "DYLD_"))
           and k not in ("NODE_OPTIONS", "NODE_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD")
           and k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               NO_PROXY="127.0.0.1,localhost", MLX_SERVE_LOOKUP_TRACE="1",
               MLX_SERVE_ROUND_COST_PERSIST="0",
               MLX_SERVE_LOOKUP="0" if tier == "legacy" else "1")
    if corpus:
        env["MLX_SERVE_LOOKUP_CORPUS"] = str(corpus)
    if store:
        env["MLX_SERVE_LOOKUP_STORE"] = str(store)
    if experiment:
        if set(experiment) - BENCHMARK_EXPERIMENT_KEYS:
            raise BenchError("unsupported benchmark environment override")
        for key, value in experiment.items():
            value = str(value)
            if key == "MLX_SERVE_LOOKUP_SELECTION" and not re.fullmatch(r"adaptive|cut|max|fixed:[1-9][0-9]*", value):
                raise BenchError("invalid lookup row selection")
            if key in ("MLX_SERVE_LOOKUP_STORE_WRITE", "MLX_SERVE_LOOKUP_RECEIPTS",
                       "MLX_SERVE_LOOKUP_CANDIDATES", "MLX_SERVE_LOOKUP_BACKOFF") and value not in ("0", "1"):
                raise BenchError("invalid benchmark boolean")
            if key == "MLX_SERVE_ROW_EXACT" and value != "1":
                raise BenchError("row-exact mode requires an explicit value of 1")
            if key == "MLX_SERVE_LOOKUP_PRICE_TRACE" and value != "1":
                raise BenchError("diagnostic price tracing requires an explicit value of 1")
            if key == "MLX_SERVE_LOOKUP_COUNTERFACTUAL" and value not in ("0", "1"):
                raise BenchError("invalid counterfactual boolean")
            if key == "MLX_SERVE_LOOKUP_COUNTERFACTUAL_ACTION" and not re.fullmatch(r"(assistant|lookup)([1-9]|1[0-6])", value):
                raise BenchError("invalid counterfactual action")
            if key == "MLX_SERVE_LOOKUP_COUNTERFACTUAL_PREPARATION" and value not in ("merged16", "production-neural", "production-combined"):
                raise BenchError("invalid counterfactual preparation")
            if key == "MLX_SERVE_LOOKUP_PRICES":
                absolute_asset(value)
            env[key] = value
    return env


def numerical_mode(environment, startup_log):
    requested = environment.get("MLX_SERVE_ROW_EXACT") == "1"
    activated = "[row-exact] explicit numerical mode enabled" in ANSI.sub("", startup_log).splitlines()
    if requested != activated:
        raise BenchError("explicit row-exact activation does not match declared numerical mode")
    return "row-exact" if requested else "stock"


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # A just-stopped owned server can leave accepted sockets in TIME_WAIT.
        # Match the server's reusable bind; owners() still rejects live listeners.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise BenchError(f"port {port} is occupied; no existing process will be stopped") from exc


def owners(port):
    lsof = shutil.which("lsof") or ("/usr/sbin/lsof" if Path("/usr/sbin/lsof").exists() else None)
    if not lsof:
        raise BenchError("lsof is required to verify localhost listener ownership")
    result = subprocess.run([lsof, "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                            capture_output=True, text=True, timeout=10)
    if result.returncode not in (0, 1):
        raise BenchError(f"cannot inspect listener ownership: {result.stderr}")
    return {int(pid) for pid in result.stdout.split()}


def memory(pid):
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=5)
        return {"child_rss_kib": int(result.stdout.strip())}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"child_rss_kib": None}


def machine_state():
    result = {}
    for name, command in (("power", ["pmset", "-g", "batt"]),
                          ("power_settings", ["pmset", "-g", "custom"]),
                          ("thermal", ["pmset", "-g", "therm"]),
                          ("pressure", ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]),
                          ("swap", ["sysctl", "-n", "vm.swapusage"]),
                          ("vm_stat", ["vm_stat"])):
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = run.stdout.strip() if run.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    power = result["power"] or ""
    result["power_source"] = "ac" if "'AC Power'" in power else "battery" if "'Battery Power'" in power else "unknown"
    pressure = result.get("pressure")
    result["pressure_level"] = int(pressure) if pressure and pressure.isdigit() else None
    swap = re.search(r"used\s*=\s*([0-9.]+)([KMG])", result.get("swap") or "")
    result["swap_used_bytes"] = (int(float(swap[1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[swap[2]])
                                 if swap else None)
    for label, key in (("Swapins", "swapins"), ("Swapouts", "swapouts")):
        value = re.search(r"^" + label + r":\s*(\d+)", result.get("vm_stat") or "", re.M)
        result[key] = int(value[1]) if value else None
    return result


def probe_http_failed(row):
    """Only llmprobe's intentional POST {} surface discovery may return 400."""
    return bool(row.get("proxy_error") or row.get("inference_error") or
                (row.get("http_status", 200) >= 400 and not
                 (row.get("http_status") == 400 and row.get("request_kind") == "empty_discovery")))


def probe_report_errors(report, rungs, runs):
    """Validate the saved bench-only report, not merely the CLI exit status."""
    errors = []
    if not isinstance(report, dict) or report.get("incomplete"):
        return ["missing or incomplete canonical report"]
    bench = report.get("bench")
    if not isinstance(bench, dict):
        return ["missing canonical bench"]
    for metric in ("decodeTokPerSec", "ttftMs", "prefillTokPerSec"):
        stat = bench.get(metric)
        if (not isinstance(stat, dict) or not isinstance(stat.get("median"), (int, float))
                or stat["median"] <= 0 or len(stat.get("samples", [])) != runs):
            errors.append(f"missing successful {metric}")
    points = bench.get("contextScaling") or []
    for rung in rungs:
        point = next((p for p in points if p.get("targetTokens") == rung), None)
        if not point or point.get("runs", 0) != runs or not point.get("inputTokens") or not point.get("decodeTokPerSec"):
            errors.append(f"incomplete canonical context rung {rung}")
    speculative = bench.get("speculative") or {}
    for metric in ("predictableTokPerSec", "novelTokPerSec"):
        value = speculative.get(metric)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            errors.append("missing successful speculative:" + metric)
    if speculative.get("offScript") != 0:
        errors.append("canonical predictable workload has missing or off-script samples")
    for name, fields in (("prefixCache", ("coldTtftMs", "warmTtftMs")),
                         ("batching", ("singleTokPerSec", "aggregateTokPerSec"))):
        result = bench.get(name) or {}
        if any(type(result.get(field)) not in (int, float) or
               not math.isfinite(result[field]) or result[field] <= 0 for field in fields):
            errors.append("incomplete canonical " + name)
    if (bench.get("batching") or {}).get("streams") != 4:
        errors.append("canonical batching requires all four streams")
    drift = bench.get("loadDrift") or {}
    value = drift.get("driftPct")
    if drift.get("verdict") not in ("steady", "improved", "degraded") or type(value) not in (int, float) or not math.isfinite(value):
        errors.append("canonical loadDrift missing or malformed")
    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "error" and item:
                    errors.append(f"{path}.error: {canonical(item)}")
                elif key == "note" and isinstance(item, str) and re.search(r"HTTP \d|failed|no successful|error", item, re.I):
                    errors.append(f"{path}.note: {item}")
                else:
                    visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
    visit(bench, "bench")
    return errors


def probe_stability_errors(report):
    drift = (report.get("bench") or {}).get("loadDrift") or {}
    return [] if drift.get("verdict") == "steady" else ["load_drift"]


def canonical_metric_evidence(report, captures, builders, runs):
    """Bind pinned-client reported rates to their measured requests, not warmups."""
    bench = report["bench"]
    values = {key: bench[key]["median"] for key in ("decodeTokPerSec", "ttftMs", "prefillTokPerSec")}
    for key in ("predictableTokPerSec", "novelTokPerSec"):
        values["speculative:" + key] = bench["speculative"][key]
    for point in bench.get("contextScaling") or []:
        for key in ("decodeTokPerSec", "ttftMs", "prefillTokPerSec"):
            values[f"context:{point['targetTokens']}:{key}"] = point.get(key)
    prompts = {w["id"]: w.get("match_prompt", w["prompt"]) for w in builders}
    scenarios = [
        ("canonical-decode", 2, ["decodeTokPerSec", "ttftMs"]),
        ("canonical-prefill", 1, ["prefillTokPerSec"]),
        ("canonical-predictable", 1, ["speculative:predictableTokPerSec"]),
        ("canonical-novel", 1, ["speculative:novelTokPerSec"]),
    ]
    seen = {name: 0 for name, _, _ in scenarios}
    for capture in captures:
        content = canonical(capture["request"])
        capture["metrics"] = []
        for name, warmups, metrics in scenarios:
            prompt = prompts[name]
            if prompt in content or json.dumps(prompt)[1:-1] in content:
                seen[name] += 1
                if warmups < seen[name] <= runs + warmups:
                    capture["metrics"] = metrics[:]
    # Pinned 0.6.17: final drift runs follow rung code/count pairs.
    points = bench.get("contextScaling") or []
    start = len(captures) - runs - len(points) * runs * 2
    if start >= 0:
        for index, point in enumerate(points):
            for rep in range(runs):
                capture = captures[start + index * runs * 2 + rep * 2]
                capture["metrics"] = [f"context:{point['targetTokens']}:{key}"
                                      for key in ("decodeTokPerSec", "ttftMs", "prefillTokPerSec")]
    for capture in captures:
        capture["scenario"] = capture["metrics"] or ["auxiliary_or_batching_unattributed"]
    return {"metric_values": values, "metric_request_counts": {metric: runs for metric in values}}


def canonical_context_workloads(command, rungs):
    """Use the prepared canonical source's pure builders; never import its CLI."""
    sources = [Path(p) for p in command if str(p).endswith(".mjs")]
    if len(sources) != 1:
        raise BenchError("canonical context preflight requires the prepared local llmprobe .mjs command")
    node = command[0] if command[0] != str(sources[0]) else shutil.which("node")
    if not node:
        raise BenchError("local Node.js is required for canonical prompt preflight")
    script = r"""
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
function section(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  if (a < 0 || b < 0) throw Error('unsupported prepared llmprobe source contract: ' + start);
  return source.slice(a, b);
}
const code = section('function buildLongPrefix(', 'function buildHaystackWithNeedle(')
  + section('var DOMAINS =', 'function usedPlantedConstant(')
  + section('var PREDICTABLE_PASSAGE =', 'function echoedPassage(')
  + section('var DECODE_PROMPT =', 'var RUNG_FIXED_CHARS =');
const context = vm.createContext({});
vm.runInContext(code, context);
const rungs = JSON.parse(process.argv[2]);
const result = vm.runInContext(`[
  {id:'canonical-prefill', prompt:buildLongPrefix("The archive records the following entry.",40960)+"\\n\\nReply with just the word: OK.", max_tokens:8},
  {id:'canonical-prefix-cache', prompt:buildLongPrefix("Operations handbook, section on shift handover.",6144)+"\\n\\nReply with just the word: OK.", max_tokens:8},
  {id:'canonical-decode', prompt:DECODE_PROMPT, max_tokens:192},
  {id:'canonical-predictable', prompt:"You are a text normaliser. Output the user's text unchanged.\\n"+PREDICTABLE_PASSAGE, match_prompt:PREDICTABLE_PASSAGE, max_tokens:192},
  {id:'canonical-novel', prompt:"Invent an original, surprising paragraph of stream-of-consciousness prose. Avoid common phrases.", max_tokens:192},
  {id:'canonical-calibration', prompt:buildCodeContextWithConstant(2048)+"\\n\\n"+CODE_INSTRUCTION, max_tokens:1}
]`, context);
for (const rung of rungs) {
  context.bytes = rung * 8;
  for (const instruction of ['CODE_INSTRUCTION', 'COUNT_INSTRUCTION'])
    result.push({id:'canonical-rung-'+rung+'-'+instruction,
      prompt:vm.runInContext('buildCodeContextWithConstant(bytes)+"\\n\\n"+'+instruction, context),
      max_tokens:192, filler_byte_upper_bound:rung*8});
}
process.stdout.write(JSON.stringify(result));
"""
    run = subprocess.run([node, "-e", script, str(sources[0]), json.dumps(rungs)],
                         capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise BenchError(f"canonical prompt preflight failed: {run.stderr.strip()}")
    return json.loads(run.stdout)


def preflight_context(server, args, cases, command):
    workloads = [{**case, "max_tokens": min(case["max_tokens"], args.max_tokens)
                  if args.max_tokens else case["max_tokens"], "reserve": 0} for case in cases]
    if command:
        workloads += [{**case, "reserve": 512} for case in
                      canonical_context_workloads(command, [int(r) for r in args.probe_rungs.split(",")])]
    evidence = []
    for case in workloads:
        server.assert_owned()
        count = len(http_json(server.base, "/tokenize", {"content": case["prompt"]},
                              timeout=args.request_timeout)["tokens"])
        required = count + case["max_tokens"] + case["reserve"]
        evidence.append({"id": case["id"], "prompt_tokens": count,
                         "output_tokens": case["max_tokens"], "template_cache_reserve": case["reserve"],
                         "required_context": required,
                         "filler_byte_upper_bound": case.get("filler_byte_upper_bound")})
    insufficient = [e for e in evidence if e["required_context"] > args.ctx_size]
    if insufficient:
        worst = max(insufficient, key=lambda e: e["required_context"])
        raise BenchError(f"context preflight refused before timing: {worst['id']} needs "
                         f"{worst['prompt_tokens']} owned-tokenizer prompt + {worst['output_tokens']} output "
                         f"+ {worst['template_cache_reserve']} template/cache reserve = "
                         f"{worst['required_context']}; --ctx-size is {args.ctx_size}. "
                         "Canonical rung checks use the consumer's conservative 8-byte/token filler bound; "
                         "increase context or lower rungs. Canonical prompts are never truncated.")
    return evidence


def receipts(text, kind="round"):
    prefix = "[lookup-" + kind + "] "
    result = []
    for line in text.splitlines():
        if prefix in line:
            result.append(json.loads(line.split(prefix, 1)[1]))
    return result


def settled_log(server, offset, timeout=2):
    deadline = time.monotonic() + timeout
    while True:
        text = server.log_since(offset)
        if receipts(text, "commit") or server.proc.poll() is not None or time.monotonic() >= deadline:
            return text
        time.sleep(.02)


def response_receipt(raw, stream):
    result = {"events": [], "text": "", "usage": None, "finish_reason": None, "complete": False}
    done = not stream
    for line in (raw.splitlines() if stream else [raw]):
        if stream and not line.startswith("data:"):
            continue
        value = line[5:].strip() if stream else line
        if value == "[DONE]":
            done = True
            continue
        if not value:
            continue
        event = json.loads(value)
        result["events"].append(event)
        if event.get("error"):
            result["error"] = event["error"]
        for choice in event.get("choices", []):
            result["text"] += choice.get("text") or (choice.get("delta") or choice.get("message") or {}).get("content") or ""
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]
            if choice.get("finish_details") is not None:
                result["finish_details"] = choice["finish_details"]
        if event.get("usage") is not None:
            result["usage"] = event["usage"]
    result["complete"] = bool(done and result["usage"] is not None and
                              result["finish_reason"] in ("stop", "length") and
                              not result.get("finish_details") and not result.get("error"))
    return result


def controlled_probe_command(command, directory, block_tag):
    """Derive a pinned client copy; alter only its unique cache-bust tag initializer."""
    sources = [Path(p) for p in command if str(p).endswith(".mjs")]
    if len(sources) != 1:
        raise BenchError("controlled canonical run requires pinned local .mjs source")
    source = sources[0]
    text = source.read_text()
    marker = "var CACHE_BUST_TAG = Date.now().toString(36);"
    if text.count(marker) != 1:
        raise BenchError("pinned canonical cache-bust source contract mismatch")
    if 'version: "0.6.17"' not in text:
        raise BenchError("controlled canonical protocol requires pinned llmprobe 0.6.17")
    derived = directory / "llmprobe-controlled.mjs"
    # Node resolves dependencies relative to this module. The pinned bundle must
    # be self-contained; retain its original location via a symlinked dependency root.
    modules = source.parent
    while modules.parent != modules and modules.name != "node_modules":
        modules = modules.parent
    if modules.name != "node_modules":
        raise BenchError("pinned client is not inside a prepared node_modules tree")
    link = directory / "node_modules"
    if not link.exists():
        link.symlink_to(modules, target_is_directory=True)
    derived.write_text(text.replace(marker, "var CACHE_BUST_TAG = " + json.dumps(block_tag) + ";"))
    atomic_json(directory / "llmprobe-control.json", {
        "protocol": "controlled canonical llmprobe; only cache-bust tag initializer replaced",
        "version": "0.6.17",
        "original": str(source), "original_sha256": digest(source),
        "derived_sha256": digest(derived), "tag": block_tag, "replacement_count": 1,
        "timing_clock_unchanged": True})
    return [str(derived) if p == str(source) else p for p in command]


class ProbeProxy:
    """Loopback-only transparent relay; inject mode flags without changing probe prompts."""
    def __init__(self, server, mode, evidence):
        flags = {k: v for k, v in request_body(
            {"prompt": "", "max_tokens": 1, "temperature": 0, "seed": 0}, mode, None).items()
                 if k.startswith("enable_")}
        sequence = [0]
        sequence_lock = threading.Lock()
        pending = [0]
        condition = threading.Condition()
        self.pending, self.condition = pending, condition
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def relay_request(self):
                if self.path not in ("/health", "/v1/models", "/v1/completions", "/v1/chat/completions"):
                    self.send_error(404)
                    return
                payload = None
                offset = server.log_path.stat().st_size
                with sequence_lock:
                    request_index = sequence[0]
                    sequence[0] += 1
                if self.command == "POST":
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    request_kind = ("empty_discovery" if payload == {} and self.path in
                                    ("/v1/completions", "/v1/chat/completions") else "inference")
                    payload.update(flags)
                    append(evidence, {"path": self.path, "request": payload,
                                      "request_kind": request_kind, "timestamp": time.time()})
                try:
                    server.assert_owned()
                    req = urllib.request.Request(server.base + self.path,
                        None if payload is None else json.dumps(payload).encode(),
                        {"Content-Type": "application/json"})
                    try:
                        response = HTTP.open(req, timeout=server.args.request_timeout)
                    except urllib.error.HTTPError as exc:
                        response = exc
                    with response:
                        append(evidence, {"http_status": response.code, "path": self.path,
                                          "request_kind": request_kind if payload is not None else "status"})
                        self.send_response(response.code)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        body_chunks = []
                        while True:
                            chunk = response.read1(65536)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            body_chunks.append(chunk)
                            self.wfile.flush()
                        self.close_connection = True
                        if payload is not None and request_kind != "empty_discovery":
                            raw = b"".join(body_chunks).decode("utf-8", "replace")
                            normalized = response_receipt(raw, bool(payload.get("stream")))
                            log_text = settled_log(server, offset)
                            commits = receipts(log_text, "commit")
                            if len(commits) == 1:
                                normalized["commit"] = commits[0]
                            append(evidence, {"request_id": request_identity(self.path, payload),
                                "request_index": request_index,
                                "path": self.path, "request": payload, "response": normalized,
                                "raw_response": raw, "http_status": response.code,
                                "log_offset": offset, "log_end_offset": server.log_path.stat().st_size,
                                "rounds": receipts(log_text), "commits": commits})
                            for line in raw.splitlines():
                                candidate = line[5:].strip() if line.startswith("data:") else line.strip()
                                if not candidate or candidate == "[DONE]":
                                    continue
                                try:
                                    decoded = json.loads(candidate)
                                except ValueError:
                                    continue
                                if isinstance(decoded, dict) and decoded.get("error"):
                                    append(evidence, {"path": self.path, "inference_error": decoded["error"]})
                except (OSError, ValueError, BenchError, HTTPException) as exc:
                    append(evidence, {"proxy_error": str(exc), "timestamp": time.time()})
                    self.close_connection = True

            def relay(self):
                with condition:
                    pending[0] += 1
                try:
                    self.relay_request()
                except (OSError, ValueError, BenchError, HTTPException) as exc:
                    append(evidence, {"proxy_error": str(exc), "timestamp": time.time()})
                    self.close_connection = True
                finally:
                    with condition:
                        pending[0] -= 1
                        condition.notify_all()

            do_GET = relay
            do_POST = relay

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.http.server_port}"

    def __exit__(self, exc_type, *_):
        self.http.shutdown()
        with self.condition:
            settled = self.condition.wait_for(lambda: self.pending[0] == 0, timeout=5)
        self.http.server_close()
        self.thread.join()
        if not settled and exc_type is None:
            raise BenchError("canonical proxy handlers did not finish receipt capture")


class MachineMonitor:
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop_event.wait(5):
            self.samples.append({"timestamp": time.time(), **machine_state()})

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()


def canonical_probe(server, args, mode, command, arm):
    before = machine_state()
    offset = server.log_path.stat().st_size
    report = arm / "llmprobe.json"
    models = http_json(server.base, "/v1/models")["data"]
    if not models:
        raise BenchError("canonical probe requires an advertised model ID")
    model_id = models[0]["id"]
    server_settings = http_json(server.base, "/props").get("settings")
    with MachineMonitor() as monitor, ProbeProxy(server, mode, arm / "llmprobe-requests.jsonl") as base:
        launch = [*command, base, "-m", model_id, "--bench-only",
                  "--rungs", args.probe_rungs, "--runs", str(args.repetitions),
                  "--concurrency", "1", "--reasoning", "off",
                  "--save", str(report), "--no-color"]
        atomic_json(arm / "llmprobe-launch.json", {"command": launch, "machine_before": before})
        with (arm / "llmprobe.log").open("wb") as log:
            child = subprocess.Popen(launch, env=server.env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                code = child.wait(timeout=args.probe_timeout)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
    evidence = {"returncode": code, "report": str(report), "machine_before": before,
                "machine_after": machine_state(), "metrics": parse_stats(server.log_since(offset), mode.split("-")[0])}
    evidence["machine_samples"] = monitor.samples
    evidence["server_settings"] = server_settings
    evidence["submission_capacity_contract"] = {"configured_max_concurrent": 1, "queue_depth": 32,
                                               "configured_submit_capacity": 33,
                                               "batch_probe_unchanged": True,
                                               "batching_metric_attribution": "unsupported_when_offsets_overlap",
                                               "price_guard": "per-round priced receipt; concurrent rounds must not borrow single-slot prices"}
    proxy_rows = rows(arm / "llmprobe-requests.jsonl")
    proxy_failed = any(probe_http_failed(r) for r in proxy_rows)
    evidence["proxy_failed"] = proxy_failed
    captures = sorted((r for r in proxy_rows if "response" in r), key=lambda r: r["request_index"])
    whole_log = server.log_since(offset)
    all_commits = receipts(whole_log, "commit")
    decoded_prompts = {}
    for commit in all_commits:
        key = str(commit.get("prompt_hash"))
        if key not in decoded_prompts:
            decoded_prompts[key] = http_json(server.base, "/detokenize",
                {"tokens": commit["prompt_ids"]}, timeout=args.request_timeout)["content"]
    attach_canonical_receipts(captures, all_commits, receipts(whole_log), decoded_prompts)
    evidence["responses"] = captures
    # Tags preserve client issue order even in the fixed concurrent batch probe.
    def client_order(capture):
        tag = re.search(r"\[probe [a-z0-9]+-([a-z0-9]+)\]", canonical(capture["request"]))
        return (1, int(tag[1], 36)) if tag else (0, capture["request_index"])
    evidence["request_sequence"] = [r["request_id"] for r in sorted(captures, key=client_order)]
    evidence["metric_values"] = {}
    try:
        saved = json.loads(report.read_text())
        report_errors = probe_report_errors(saved, [int(r) for r in args.probe_rungs.split(",")], args.repetitions)
        evidence.update(canonical_metric_evidence(saved, captures, canonical_context_workloads(command, []), args.repetitions))
        evidence["stability_errors"] = probe_stability_errors(saved)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report_errors = [f"unreadable canonical report: {exc}"]
    evidence["report_errors"] = report_errors
    evidence["successful"] = code == 0 and not proxy_failed and not report_errors and bool(captures) and all(
        r["response"].get("complete") for r in captures)
    atomic_json(arm / "llmprobe-evidence.json", evidence)
    if not evidence["successful"]:
        raise BenchError(f"canonical llmprobe failed ({code}); {canonical(report_errors)}; inspect {arm / 'llmprobe.log'} and llmprobe-evidence.json")
    return evidence


class Server:
    def __init__(self, args, binary, model, mode, log_path, corpus=None, store=None, experiment=None):
        self.args, self.log_path = args, log_path
        self.base = f"http://127.0.0.1:{args.port}"
        self.proc = None
        self.log = None
        tier = mode.split("-", 1)[1] if "-" in mode else "legacy"
        self.env = clean_environment(tier, corpus, store, experiment)
        self.bench_mode = mode
        self.numerical_mode = None
        home = log_path.parent / (log_path.stem + "-home")
        home.mkdir(parents=True, exist_ok=True)
        self.env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"),
                        XDG_CACHE_HOME=str(home / ".cache"))
        self.command = [binary, "serve", "--model", model["model_dir"], "--host", "127.0.0.1",
                        "--port", str(args.port), "--ctx-size", str(args.ctx_size),
                        "--no-vision", "--prefix-cache-entries", "0", "--max-concurrent", "1"]
        self.auth_headers = {}
        if self.env.get("MLX_SERVE_LOOKUP_COUNTERFACTUAL") == "1":
            key = secrets.token_hex(32)
            self.command += ["--api-key", key, "--api-key-strict"]
            self.auth_headers["Authorization"] = "Bearer " + key
        if mode == "serial-dflash" or mode.startswith("dflash-"):
            self.command += ["--drafter", model["drafter_dir"]]
        else:
            self.command += ["--no-drafter"]
        if not mode.startswith("mtp-"):
            self.command += ["--no-mtp"]
        else:
            self.command += ["--mtp"]
        if not mode.startswith("pld-"):
            self.command += ["--no-pld"]

    def __enter__(self):
        port_free(self.args.port)
        if owners(self.args.port):
            raise BenchError("listener appeared before launch; refusing to use it")
        self.log = self.log_path.open("wb")
        try:
            self.proc = subprocess.Popen(self.command, env=self.env, stdout=self.log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            deadline = time.monotonic() + self.args.ready_timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise BenchError(f"child exited {self.proc.returncode}; inspect {self.log_path}")
                text = self.log_path.read_text(errors="replace")
                if "Model ready" in text:
                    self.numerical_mode = numerical_mode(self.env, text)
                    listener = owners(self.args.port)
                    if listener and listener != {self.proc.pid}:
                        raise BenchError(f"port owner {listener} is not child {self.proc.pid}")
                    if listener == {self.proc.pid}:
                        try:
                            http_json(self.base, "/health", timeout=2)
                            return self
                        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                            pass
                time.sleep(0.5)
            raise BenchError(f"model readiness timeout; inspect {self.log_path}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log is not None:
            self.log.close()

    def __exit__(self, *_):
        self.stop()

    def assert_owned(self):
        if self.proc.poll() is not None or owners(self.args.port) != {self.proc.pid}:
            raise BenchError("owned child is no longer the exclusive listener")

    def log_since(self, offset):
        with self.log_path.open("rb") as stream:
            stream.seek(offset)
            return ANSI.sub("", stream.read().decode("utf-8", "replace"))


def parse_stats(text, family):
    stats, traces = [], []
    for line in text.splitlines():
        if "[spec-stats]" in line:
            stats.append(dict(KV.findall(line)))
        if "[lookup]" in line:
            traces.append({"line": line, **dict(KV.findall(line))})
    primary = [s for s in stats if s.get("mode") == family]
    attempts = sum(int(s.get("attempts", "0")) for s in primary)
    lookup_rounds = None
    if family == "mtp":
        # generate.zig logSpecStats: rounds / drafted / accepted (NOT the reverse).
        lookup_rounds = sum(int(s["lookup"].split("/")[0]) for s in primary if "lookup" in s)
    elif family == "dflash":
        lookup_rounds = sum(int(t.get("look", 0)) + int(t.get("both", 0)) > 0
                            for t in traces if "[lookup] dflash " in t["line"])
    elif family == "pld":
        # PLD attempts are actual verify rounds; proposal traces alone are not engagement.
        lookup_rounds = attempts if any("[lookup] pld " in t["line"] for t in traces) else 0
    perf = re.search(r"prefill: ([0-9.]+) tok/s.*?decode: ([0-9.]+) tok/s", text)
    return {"spec_stats": stats, "lookup_traces": traces, "attempts": attempts,
            "accepts": sum(int(s.get("accepts", "0")) for s in primary),
            "family_engaged": attempts > 0 or (family == "mtp" and bool(lookup_rounds)),
            "runtime_disabled": any(s.get("runtime_disabled") == "true" for s in primary),
            "lookup_rounds": lookup_rounds,
            "proposed_sources": sorted({t["src"] for t in traces if "src" in t}),
            "rounds": receipts(text, "round"), "commits": receipts(text, "commit"),
            "server_prefill_tps": float(perf[1]) if perf else None,
            "server_decode_tps": float(perf[2]) if perf else None}


def request(server, body, timeout):
    started = time.perf_counter()
    result = {"events": [], "emitted_pieces": [], "text": "", "usage": None,
              "server_timings": None, "ttft_seconds": None, "finish_reason": None,
              "token_ids": None, "token_id_exactness": "unavailable_in_speculative_api"}
    chat = "messages" in body
    endpoint = "/v1/chat/completions" if chat else "/v1/completions"
    req = urllib.request.Request(server.base + endpoint, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Accept": "text/event-stream",
                                  **getattr(server, "auth_headers", {})})
    done = False
    try:
        with HTTP.open(req, timeout=timeout) as response:
            result["http_status"] = response.status
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                result["events"].append(event)
                if event.get("error"):
                    raise BenchError(canonical(event["error"]))
                for choice in event.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") if chat else choice.get("text")
                    if piece:
                        if result["ttft_seconds"] is None:
                            result["ttft_seconds"] = time.perf_counter() - started
                        result["emitted_pieces"].append(piece)
                        result["text"] += piece
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
                    if choice.get("finish_details") is not None:
                        result["finish_details"] = choice["finish_details"]
                if event.get("usage") is not None:
                    result["usage"] = event["usage"]
                if event.get("timings") is not None:
                    result["server_timings"] = event["timings"]
        if not done or result["usage"] is None or result["finish_reason"] is None:
            raise BenchError("incomplete SSE response (requires DONE, usage and finish reason)")
        if result["finish_reason"] not in ("stop", "length") or result.get("finish_details"):
            raise BenchError(f"abnormal generation finish: {result['finish_reason']} {result.get('finish_details')}")
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = exc.read().decode("utf-8", "replace")
    except (OSError, ValueError, BenchError, HTTPException) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["total_seconds"] = time.perf_counter() - started
    return result


def request_body(case, mode, cap):
    family = mode.split("-", 1)[0]
    body = {"model": "mlx-serve",
            "max_tokens": min(case["max_tokens"], cap) if cap else case["max_tokens"],
            "temperature": case["temperature"], "seed": case["seed"],
            "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "enable_mtp": family == "mtp", "enable_drafter": family == "dflash",
            "enable_pld": family == "pld", "stream": True,
            "stream_options": {"include_usage": True}}
    key = "messages" if "messages" in case else "prompt"
    body[key] = case[key]
    for key in ("enable_thinking", "reasoning_effort"):
        if key in case:
            body[key] = case[key]
    return body


def unsupported(model, mode):
    if mode.startswith("mtp-") and not model["native_mtp"]:
        return "profile has no native MTP head"
    if mode == "serial-dflash" or mode.startswith("dflash-"):
        if not model.get("drafter_dir"):
            return "profile has no local drafter"
        if mode.startswith("dflash-") and mode != "dflash-legacy" and not model["tree_candidate"]:
            return "architecture lacks row-exact specTreeSupported; no DFlash tier-tree coverage"
    return None


def recover_interrupted_tail(path):
    """Preserve a torn final JSONL record before replaying its unfinished arm."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    prefix, _, last = data.rpartition(b"\n")
    try:
        json.loads(last)
        recovered = data + b"\n"
    except (ValueError, UnicodeDecodeError):
        backup = path.with_name(path.name + f".interrupted-{time.time_ns()}")
        backup.write_bytes(data)
        recovered = prefix + b"\n" if prefix else b""
        print(f"Preserved interrupted JSONL at {backup}; replaying unfinished arm.", flush=True)
    temp = path.with_suffix(path.suffix + ".recover")
    temp.write_bytes(recovered)
    temp.replace(path)


def rows(path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BenchError(f"invalid JSONL at {path}:{number}; preserve/repair incomplete final line before resume") from exc
    return result

def validate_seed_store(store, tokenizer_sha, commits):
    data = (store / "tokens.bin").read_bytes()
    if len(data) < 40 or data[:8] != b"MLXLOOK1" or data[8:40] != bytes.fromhex(tokenizer_sha) or (len(data) - 40) % 4:
        raise BenchError("persistent seed header/tokenizer/alignment mismatch")
    expected = []
    for commit in commits:
        if commit.get("version") != 1 or not commit.get("prompt_ids") or commit.get("generated_ids") is None:
            raise BenchError("missing exact intended seed commit IDs")
        expected.extend(commit["prompt_ids"] + commit["generated_ids"] + [1 << 30])
    actual = list(struct.unpack("<" + "I" * ((len(data) - 40) // 4), data[40:]))
    if not commits or actual != expected:
        raise BenchError("persistent seed documents differ from exact intended IDs and separators")
    return {"sha256": digest(store / "tokens.bin"), "documents": len(commits),
            "tokenizer_sha256": tokenizer_sha, "exact_document_identity_valid": True,
            "commit_sha256": hashlib.sha256(canonical(commits).encode()).hexdigest()}


def seed_retrieval(server, commits, timeout):
    for commit in commits:
        document = commit["prompt_ids"] + commit["generated_ids"]
        for end in range(max(len(commit["prompt_ids"]), 8), len(document) - 1):
            prefix = document[end - 8:end]
            if len({tuple(prefix[i:i + 2]) for i in range(7)}) != 7:
                continue
            occurrences = sum(
                (other["prompt_ids"] + other["generated_ids"])[i:i + 8] == prefix
                for other in commits for i in range(max(0, len(other["prompt_ids"] + other["generated_ids"]) - 7)))
            if occurrences != 1:
                continue
            body = request_body({"prompt": document[:end], "max_tokens": min(8, len(document) - end),
                                 "temperature": 0, "seed": 0},
                                server.bench_mode, None)
            offset = server.log_path.stat().st_size
            response = request(server, body, timeout)
            text = settled_log(server, offset)
            found = receipts(text)
            evidence = {"synthetic_known_hit": True, "prompt_ids": document[:end],
                        "unique_seed_suffix": prefix,
                        "expected_seed_continuation": document[end:end + 8],
                        "response": response, "rounds": found, "commits": receipts(text, "commit")}
            emitted = evidence["commits"][0].get("generated_ids") if len(evidence["commits"]) == 1 else None
            evidence["successful"] = bool(not response.get("error") and selected_origin(found, "persistent") and
                                          emitted and emitted == document[end:end + len(emitted)])
            return evidence
    raise BenchError("no unique designated seed-only retrieval prefix")


def replay_canonical(args, binary, model, mode, probe, arm):
    """Replay all captures; certify only the complete defined metric request set."""
    references = canonical_reference_modes(mode)
    evidence = []
    captures = probe.get("responses", [])
    for reference_index, reference in enumerate(references):
        experiment = {"MLX_SERVE_LOOKUP_RECEIPTS": "1", "MLX_SERVE_LOOKUP_STORE_WRITE": "0"}
        if getattr(args, "row_exact", False):
            experiment["MLX_SERVE_ROW_EXACT"] = "1"
        with Server(args, binary, model, reference, arm / f"replay-{reference_index}.log",
                    experiment=experiment) as server:
            for capture in captures:
                body = dict(capture["request"])
                family = reference.split("-")[0]
                body.update(enable_mtp=family == "mtp", enable_drafter=family == "dflash", enable_pld=family == "pld")
                offset = server.log_path.stat().st_size
                server.assert_owned()
                req = urllib.request.Request(server.base + capture["path"], json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                with HTTP.open(req, timeout=args.request_timeout) as response:
                    raw = response.read().decode("utf-8")
                receipt = response_receipt(raw, bool(body.get("stream")))
                commits = receipts(settled_log(server, offset), "commit")
                if len(commits) == 1:
                    receipt["commit"] = commits[0]
                row = {"request_id": capture["request_id"], "reference": reference,
                       "request_index": capture["request_index"],
                       "path": capture["path"], "request": body, "response": receipt}
                append(arm / "canonical-replay.jsonl", row)
                evidence.append(row)
    return classify_canonical_replay(probe, evidence, references, mode, args.row_exact_tree)


def paired_schedule(modes, boots):
    features = [m for m in modes if not m.startswith("serial") and not m.endswith("-legacy")]
    if not features:
        features = [m for m in modes if not m.startswith("serial")] or list(modes)
    schedule = []
    for boot in range(boots):
        for feature in (features if boot % 2 == 0 else reversed(features)):
            family = feature.split("-")[0]
            serial = feature if feature.startswith("serial") else "serial-dflash" if family == "dflash" else "serial"
            legacy = family + "-legacy" if family != "serial" else serial
            slots = [("serial", serial, "adaptive"), ("aa", legacy, "adaptive"),
                     ("legacy", legacy, "adaptive"), ("feature", feature, "adaptive"),
                     ("cut", feature, "cut"), ("max", feature, "max")]
            if feature.startswith("serial"):
                slots = [("serial", feature, "adaptive")]
            for slot, mode, selection in (slots if boot % 2 == 0 else reversed(slots)):
                event = {"boot": boot, "block": feature, "slot": slot, "mode": mode,
                         "selection": selection, "arm_id": f"{boot}:{feature}:{slot}"}
                for key in ("serial", "legacy", "aa", "cut", "max"):
                    event[key + "_arm_id"] = f"{boot}:{feature}:{key}"
                schedule.append(event)
    return schedule



def summarize(out, isolation):
    config = json.loads((out / "run.json").read_text()).get("config", {}) if (out / "run.json").exists() else {}
    arms, comparisons = reduce_arms(rows(out / "arms.jsonl"), rows(out / "requests.jsonl"), isolation, config)
    atomic_json(out / "summary.json", {"version": VERSION, "arms": arms,
                "comparisons": comparisons, "incremental_effects": incremental_effects(arms, config),
                "notes": ["Canonical protocol: pinned llmprobe with recorded block-controlled cache-bust tag only.",
                          "Custom request and exact-request replay timings are diagnostics, not canonical measurements.",
                          "Independent boot blocks are the statistical unit; whole unstable blocks are excluded.",
                          "No-hit cells retain overhead/performance evidence but have no feature-benefit attribution.",
                          "Exact commit IDs are private reversible benchmark evidence; absent IDs cannot certify token identity.",
                          "Isolation is operator attestation, not detection of all GPU consumers."]})
    fields = list(comparisons[0]) if comparisons else ["mode", "labels", "valid_comparison"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    return comparisons


def run_sweep(server, args, mode, attempt, stage, repetition, cases, raw_path):
    stage_before = machine_state()
    for case in cases:
        server.assert_owned()
        body = request_body(case, mode, args.max_tokens)
        # The API otherwise silently caps generation to context headroom.
        tokenized = http_json(server.base, "/tokenize", {"content": case["prompt"]}, timeout=args.request_timeout)
        prompt_tokens = tokenized["tokens"]
        if len(prompt_tokens) + body["max_tokens"] > args.ctx_size:
            raise BenchError(f"{case['id']}: {len(prompt_tokens)} prompt + {body['max_tokens']} output exceeds --ctx-size {args.ctx_size}; increase context or explicitly cap --max-tokens")
        offset = server.log_path.stat().st_size
        before = memory(server.proc.pid)
        result = request(server, body, args.request_timeout)
        after = machine_state()
        log_text = settled_log(server, offset)
        commits = receipts(log_text, "commit")
        if len(commits) != 1:
            result["error"] = result.get("error") or "missing or ambiguous settled commit receipt"
        elif not result.get("error"):
            commit = commits[0]
            if (commit.get("prompt_ids") != prompt_tokens or
                    len(commit.get("generated_ids", [])) != (result.get("usage") or {}).get("completion_tokens") or
                    commit.get("finish_reason") != result.get("finish_reason")):
                result["error"] = "settled commit does not match intended prompt IDs / emitted count / finish"
            else:
                result["token_ids"] = commit["generated_ids"]
                result["token_id_exactness"] = "actual_generator_commit_receipt"
        family = mode.split("-", 1)[0]
        all_log = server.log_since(0)
        tier = mode.split("-", 1)[1] if "-" in mode else "legacy"
        armed = "[lookup] three-tier lookup armed:" in all_log
        if tier in ("corpus", "all"):
            armed = armed and "corpus=on" in all_log
        if tier in ("persistent", "all"):
            armed = armed and "store=on" in all_log
        result.update(mode=mode, arm_id=getattr(args, "arm_id", f"{args.boot}:{mode}"), boot=args.boot,
                      attempt=attempt, stage=stage, repetition=repetition,
                      machine_before=stage_before, machine_after=after,
                      session_id=args.session_id,
                      exactness_contract_supported=(body["temperature"] == 0 or
                          (mode.startswith("dflash-") and args.row_exact_tree)),
                      case=case["id"], kind=case["kind"], request=body,
                      timestamp=time.time(), context_size=args.ctx_size,
                      log_path=str(server.log_path), log_offset=offset, log=log_text,
                      memory_before=before, memory_after=memory(server.proc.pid),
                      metrics=parse_stats(log_text, family), tiers_armed=armed,
                      policy_cold=stage == "cold" and case is cases[0])
        append(raw_path, result)
        print(f"{mode} {stage} r{repetition} {case['id']}: "
              f"{result.get('error') or str((result.get('usage') or {}).get('completion_tokens')) + ' tokens'}", flush=True)
        if result.get("error"):
            raise BenchError(f"request failed without retry: {result['error']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", type=Path, default=Path("~/.mlx-serve/bench/lookup-staircut"))
    parser.add_argument("--model", default="smoke", metavar="PROFILE")
    parser.add_argument("--modes", help="comma-separated: " + ",".join(MODES))
    parser.add_argument("--cases", help="comma-separated workload IDs; default all (use copy,edit,code,prose,corpus,budget for greedy-only)")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--boot-repetitions", type=int, default=4, help="predeclared independent paired boot blocks; minimum 2")
    parser.add_argument("--practical-margin", type=float, default=.05)
    parser.add_argument("--bootstrap-replications", type=int, default=10000)
    parser.add_argument("--seeds", default="0,1,2", help="sampled-case seeds, comma-separated")
    parser.add_argument("--context-variants", default="0", help="neutral context paragraph counts, comma-separated")
    parser.add_argument("--seed-cases", help="seed-only workload IDs; default selected greedy cases")
    parser.add_argument("--probe", type=Path, help="local llmprobe executable override; otherwise manifest probe_cmd")
    parser.add_argument("--skip-probe", action="store_true", help="explicit diagnostics-only run; no canonical performance numbers")
    parser.add_argument("--probe-rungs", default="512", help="llmprobe context rungs, comma-separated (default 512)")
    parser.add_argument("--probe-timeout", type=float, default=3600, help="hard timeout for each owned llmprobe process")
    parser.add_argument("--warmup", type=int, default=1, help="full training sweeps, excluded from measurements")
    parser.add_argument("--max-tokens", type=int, help="explicit per-case output cap (never truncates prompts)")
    parser.add_argument("--ctx-size", type=int, default=16384,
                        help="owned context capacity (default 16384); canonical prompts are tokenized before timing")
    parser.add_argument("--port", type=int, default=11269)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--isolation", choices=("declared", "contended"), default="contended")
    parser.add_argument("--prepare-only", action="store_true", help="build/validate corpus with owned local server, no benchmark")
    parser.add_argument("--dry-run", action="store_true", help="validate local assets and show plan, never load a model")
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    args = parser.parse_args()
    args.boot = 0
    args.session_id = str(time.time_ns())
    if args.boot_repetitions < 2:
        parser.error("--boot-repetitions must be at least 2 for alternating laptop boot order")
    if not 0 < args.practical_margin < 1 or args.bootstrap_replications < 1000:
        parser.error("margin must be in (0,1); bootstrap replications must be >=1000")
    try:
        seeds = [int(v) for v in args.seeds.split(",")]
        context_variants = [int(v) for v in args.context_variants.split(",")]
    except ValueError:
        parser.error("seeds/context variants must be integer lists")
    if not seeds or not context_variants or min(seeds + context_variants) < 0 or len(set(seeds)) != len(seeds) or len(set(context_variants)) != len(context_variants):
        parser.error("seeds/context variants must be nonempty unique nonnegative integers")
    try:
        rungs = [int(r) for r in args.probe_rungs.split(",")]
    except ValueError:
        parser.error("--probe-rungs must be comma-separated integers")
    if not rungs or min(rungs) < 512:
        parser.error("--probe-rungs uses literal token integers >=512; llmprobe interprets smaller integers as thousands")
    if args.probe_timeout <= 0:
        parser.error("probe timeout must be positive")
    if args.repetitions < 1 or args.warmup < 0 or args.ctx_size < 1 or not 1 <= args.port <= 65535:
        parser.error("repetitions/context must be positive, warmup nonnegative and port in 1..65535")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.ready_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    args.kit = args.kit.expanduser().resolve()
    manifest_path = args.kit / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION:
        raise BenchError("unsupported manifest version")
    probe_cmd = [str(args.probe.expanduser().resolve())] if args.probe else manifest.get("probe_cmd")
    if args.skip_probe:
        probe_cmd = None
    if not args.skip_probe and not args.prepare_only:
        if not isinstance(probe_cmd, list) or not probe_cmd:
            raise BenchError("no prepared local llmprobe; prepare probe_cmd or explicitly --skip-probe for diagnostics")
        for item in probe_cmd:
            absolute_asset(item)
    binary = absolute_asset(manifest["binary"])
    if not os.access(binary, os.X_OK):
        raise BenchError(f"binary is not executable: {binary}")
    workloads_path = absolute_asset(manifest["workloads"])
    workloads = json.loads(Path(workloads_path).read_text())
    if workloads.get("version") != VERSION or not workloads.get("cases"):
        raise BenchError("workloads must be version 1 with nonempty cases")
    model = next((m for m in manifest["models"] if m["id"] == args.model), None)
    if model is None:
        raise BenchError("unknown profile; available: " + ", ".join(m["id"] for m in manifest["models"]))
    model = dict(model)
    model["model_dir"] = absolute_asset(model["model_dir"], directory=True)
    if model.get("drafter_dir"):
        model["drafter_dir"] = absolute_asset(model["drafter_dir"], directory=True)
    token_sha = digest(Path(model["model_dir"]) / "tokenizer.json")
    if model.get("tokenizer_sha256") not in (None, token_sha):
        raise BenchError("tokenizer changed since preparation; rebuild the kit manifest")
    family = "mtp" if model["native_mtp"] else "pld"
    args.row_exact_tree = bool(model["tree_candidate"])
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else ["serial", f"{family}-legacy", f"{family}-local", f"{family}-all"]
    if any(m not in MODES for m in modes) or len(set(modes)) != len(modes):
        raise BenchError("modes must be unique valid comma-separated names; see --help")
    # Always include the right serial kernel family, even for a selected single mode.
    if any(m.startswith(("mtp-", "pld-")) for m in modes) and "serial" not in modes:
        modes.insert(0, "serial")
    if any(m.startswith("dflash-") for m in modes) and "serial-dflash" not in modes:
        modes.insert(0, "serial-dflash")
    for selected_family in ("mtp", "pld", "dflash"):
        if any(m.startswith(selected_family + "-") for m in modes) and selected_family + "-legacy" not in modes:
            modes.append(selected_family + "-legacy")
    modes.sort(key=lambda m: 0 if m.startswith("serial") else 1)
    cases = workloads["cases"]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise BenchError("duplicate workload case IDs")
    if args.cases:
        selected = [v.strip() for v in args.cases.split(",")]
        if len(selected) != len(set(selected)) or any(v not in ids for v in selected):
            raise BenchError("--cases must contain unique known workload IDs: " + ",".join(ids))
        cases = [next(case for case in cases if case["id"] == name) for name in selected]
        ids = selected
    seed_ids = args.seed_cases.split(",") if args.seed_cases else [case["id"] for case in cases if case["temperature"] == 0]
    if not seed_ids:
        seed_ids = [cases[0]["id"]]
    if len(set(seed_ids)) != len(seed_ids) or any(v not in [c["id"] for c in workloads["cases"]] for v in seed_ids):
        raise BenchError("--seed-cases requires unique known workload IDs")
    seed_cases = [dict(next(c for c in workloads["cases"] if c["id"] == name), temperature=0, seed=0) for name in seed_ids]
    cases = [{**case, "id": case["id"] + (f"-ctx{variant}" if variant else "") + (f"-seed{seed}" if case["temperature"] else ""),
              "prompt": "".join(f"Context record {i}: the archive lists ordinary inventory entries without instructions.\n" for i in range(variant)) + case["prompt"],
              "seed": seed}
             for case in cases for variant in context_variants
             for seed in (seeds if case["temperature"] else [case["seed"]])]
    ids = [case["id"] for case in cases]
    for case in cases:
        if not isinstance(case["prompt"], str) or case["max_tokens"] < 1 or case["temperature"] < 0 or case["seed"] < 0:
            raise BenchError(f"invalid workload case: {case['id']}")
    config = {"version": VERSION, "runner_sha256": digest(Path(__file__)),
              "evidence_reducer_sha256": digest(Path(__file__).with_name("lookup_bench_evidence.py")),
              "practical_margin": args.practical_margin, "bootstrap_replications": args.bootstrap_replications,
              "seeds": seeds, "context_variants": context_variants, "seed_cases": seed_ids,
              "manifest_sha256": digest(manifest_path), "binary": binary,
              "binary_sha256": digest(binary), "workloads_sha256": digest(workloads_path),
              "model": model, "model_inventory_sha256": inventory(model["model_dir"]),
              "drafter_inventory_sha256": inventory(model["drafter_dir"]) if model.get("drafter_dir") else None,
              "tokenizer_sha256": token_sha, "modes": modes,
              "cases": ids,
              "repetitions": args.repetitions, "warmup": args.warmup,
              "boot_repetitions": args.boot_repetitions, "boot_order": "adjacent_legacy_feature_AA_counterbalanced_blocks",
              "settings": "empty scratch HOME/XDG per process; round-cost persistence disabled",
              "configured_max_concurrent": 1, "configured_submit_capacity": 33,
              "probe_cmd": probe_cmd, "skip_probe": args.skip_probe,
              "probe_assets": {p: digest(p) for p in (probe_cmd or [])},
              "probe_rungs": args.probe_rungs, "probe_timeout": args.probe_timeout,
              "max_tokens": args.max_tokens, "ctx_size": args.ctx_size,
              "port": args.port, "isolation": args.isolation,
              "ready_timeout": args.ready_timeout, "request_timeout": args.request_timeout,
              "policy": "fresh_process_per_slot;seed_restart_frozen_store;prescribed_cold_training_warm;exact_canonical_replay",
              "experiment_environment": evidence_environment(clean_environment("local"))}
    identity = hashlib.sha256(canonical(config).encode()).hexdigest()
    plan = {"run_identity": identity, "profile": args.model, "modes": modes,
            "unsupported": {m: unsupported(model, m) for m in modes if unsupported(model, m)},
            "cases": ids, "isolation": args.isolation,
            "boot_orders": paired_schedule(modes, args.boot_repetitions),
            "canonical_probe": None if args.skip_probe else probe_cmd,
            "ctx_size": args.ctx_size,
            "context_preflight": "owned /tokenize before timing; canonical fixed prompts and 8-byte/token rung filler upper bound, plus 512 template/cache-tag reserve; no truncation",
            "corpus_requires_local_tokenizer_server": True}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Run without --dry-run to execute. No model was loaded.")
        return 0
    os.umask(0o077)
    out = (args.output or args.kit / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + args.model)).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    run_path = out / "run.json"
    if run_path.exists():
        if not args.resume:
            raise BenchError(f"output already contains a run; use --resume: {out}")
        if json.loads(run_path.read_text()).get("identity") != identity:
            raise BenchError("resume identity/config/assets differ; use a new --output directory")
    else:
        if args.resume:
            raise BenchError("--resume requires an existing run.json")
        if any(out.iterdir()):
            raise BenchError("new --output must be empty")
        atomic_json(run_path, {"identity": identity, "config": config, "created": time.time()})
        atomic_json(out / "machine-start.json", machine_state())
    # fcntl is stdlib on the supported macOS/Linux hosts; no concurrent writers.
    import fcntl
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BenchError("another runner owns this output directory") from exc
        if args.resume:
            for name in ("requests.jsonl", "arms.jsonl"):
                recover_interrupted_tail(out / name)
        need_corpus = args.prepare_only or any(m.endswith(("-corpus", "-all")) and not unsupported(model, m) for m in modes)
        corpus = None
        if need_corpus or not args.prepare_only:
            from prepare_lookup_offline import ensure_corpus
            # Tokenization is a separate unmeasured owned process, never a second tokenizer.
            prep_log = out / ("corpus-" + str(time.time_ns()) + ".log")
            with Server(args, binary, model, "serial", prep_log) as server:
                server.assert_owned()
                context_evidence = preflight_context(server, args, cases + seed_cases, probe_cmd)
                atomic_json(out / "context-preflight.json", {"context_size": args.ctx_size,
                            "workloads": context_evidence, "before_timing": True})
                if need_corpus:
                    corpus = ensure_corpus(model, manifest, server.base, args.kit)
        if args.prepare_only:
            print(f"Corpus ready: {corpus}\nBenchmark: " + shlex.join([sys.executable, str(Path(__file__).resolve()), *[v for v in sys.argv[1:] if v != "--prepare-only"], "--resume"]) + ("" if args.output else " --output " + shlex.quote(str(out))))
            return 0
        events = rows(out / "arms.jsonl")
        complete = {e["arm_id"] for e in events if e["status"] in ("complete", "unsupported")}
        failures = False
        schedule = paired_schedule(modes, args.boot_repetitions)
        for slot in schedule:
            boot, mode, arm_id = slot["boot"], slot["mode"], slot["arm_id"]
            args.boot, args.arm_id = boot, arm_id
            if arm_id in complete:
                continue
            attempt = max([e["attempt"] for e in events if e["arm_id"] == arm_id] or [0]) + 1
            reason = unsupported(model, mode)
            event = {**slot, "attempt": attempt,
                     "session_id": args.session_id, "timestamp": time.time(),
                     "status": "unsupported" if reason else "running", "reason": reason}
            append(out / "arms.jsonl", event)
            events.append(event)
            if reason:
                print(f"{mode}: UNSUPPORTED: {reason}", flush=True)
                continue
            arm = out / "arms" / f"boot-{boot}" / slot["block"] / slot["slot"] / f"attempt-{attempt}"
            arm.mkdir(parents=True)
            tier = mode.split("-", 1)[1] if "-" in mode else "legacy"
            store = arm / "store" if tier in ("persistent", "all") else None
            corpus_for_arm = corpus if tier in ("corpus", "all") else None
            try:
                frozen = None
                seed_commits = []
                if store:
                    with Server(args, binary, model, mode, arm / "seed.log", None, store,
                                experiment={"MLX_SERVE_LOOKUP_RECEIPTS": "1", "MLX_SERVE_LOOKUP_STORE_WRITE": "1"}) as server:
                        run_sweep(server, args, mode, attempt, "store_seed", 0, seed_cases, arm / "seed-requests.jsonl")
                    seeded = rows(arm / "seed-requests.jsonl")
                    seed_commits = [r["metrics"]["commits"][0] for r in seeded]
                    frozen = validate_seed_store(store, token_sha, seed_commits)
                    seed_log = (arm / "seed.log").read_text(errors="replace")
                    if re.search(r"lookup.*(?:save|buffer|append|store).*(?:fail|refus|warn)", seed_log, re.I):
                        raise BenchError("persistent seed emitted storage failure/refusal warning")
                    atomic_json(arm / "store-seed.json", {**frozen, "commits": seed_commits})
                    with Server(args, binary, model, mode, arm / "retrieve.log", None, store,
                                experiment={"MLX_SERVE_LOOKUP_RECEIPTS": "1", "MLX_SERVE_LOOKUP_STORE_WRITE": "0",
                                            "MLX_SERVE_LOOKUP_SELECTION": "max"}) as server:
                        retrieval = seed_retrieval(server, seed_commits, args.request_timeout)
                    atomic_json(arm / "store-retrieval.json", retrieval)
                    if not retrieval["successful"] or digest(store / "tokens.bin") != frozen["sha256"]:
                        raise BenchError("designated seed-only retrieval/frozen-state gate failed")
                experiment = {"MLX_SERVE_LOOKUP_RECEIPTS": "1", "MLX_SERVE_LOOKUP_STORE_WRITE": "0",
                              "MLX_SERVE_LOOKUP_SELECTION": slot["selection"]}
                with Server(args, binary, model, mode, arm / "server.log", corpus_for_arm, store, experiment) as server:
                    atomic_json(arm / "launch.json", {"command": server.command,
                        "environment": evidence_environment(server.env),
                        "pid": server.proc.pid, "store_reopened": frozen is not None,
                        "server_settings": http_json(server.base, "/props").get("settings"),
                        "frozen_start_sha256": frozen["sha256"] if frozen else None})
                    phases = [("cold", 0)] + [("warmup", rep) for rep in range(args.warmup)] + [("warm", rep) for rep in range(args.repetitions)]
                    for stage, rep in phases:
                        before_sha = digest(store / "tokens.bin") if store else None
                        run_sweep(server, args, mode, attempt, stage, rep, cases, out / "requests.jsonl")
                        after_sha = digest(store / "tokens.bin") if store else None
                        append(arm / "frozen-phases.jsonl", {"stage": stage, "repetition": rep,
                               "before_sha256": before_sha, "after_sha256": after_sha})
                        if store and before_sha != after_sha:
                            raise BenchError("frozen persistent store changed during evaluation")
                    controlled = None if args.skip_probe else controlled_probe_command(
                        probe_cmd, arm, hashlib.sha256(f"{identity}:{boot}:{slot['block']}".encode()).hexdigest()[:16])
                    probe = None if args.skip_probe else canonical_probe(server, args, mode, controlled, arm)
                if store:
                    append(arm / "frozen-phases.jsonl", {"stage": "canonical_and_teardown",
                           "before_sha256": frozen["sha256"], "after_sha256": digest(store / "tokens.bin")})
                if store and digest(store / "tokens.bin") != frozen["sha256"]:
                    raise BenchError("frozen persistent store changed during canonical phase or teardown")
                if probe:
                    probe["matched_replay"] = replay_canonical(args, binary, model, mode, probe, arm)
                    atomic_json(arm / "llmprobe-evidence.json", probe)
                    if not probe["matched_replay"]["successful"] or not probe["matched_replay"]["supported"]:
                        raise BenchError("exact canonical token replay failed or lacks a supported contract")
                    if probe.get("stability_errors"):
                        failures = True
                        print(f"{mode}: performance excluded: {canonical(probe['stability_errors'])}; execution and replay retained", flush=True)
                event = {**event, "status": "complete", "timestamp": time.time(),
                         "persistent_state_valid": store is None or bool(frozen),
                         "frozen_store": frozen, "canonical_probe": probe, "diagnostics_only": args.skip_probe}
            except (OSError, ValueError, BenchError, subprocess.SubprocessError) as exc:
                failures = True
                event = {**event, "status": "failed", "reason": f"{type(exc).__name__}: {exc}", "timestamp": time.time()}
                print(f"{mode}: FAILED: {event['reason']}", file=sys.stderr, flush=True)
            append(out / "arms.jsonl", event)
            events.append(event)
            summarize(out, args.isolation)
        comparisons = summarize(out, args.isolation)
        bad = any(any(label in r["labels"].split(",") for label in ("mismatch", "failed", "incomplete_arm")) for r in comparisons)
        failures = failures or any((e.get("canonical_probe") or {}).get("stability_errors")
                                   for e in {e["arm_id"]: e for e in events}.values())
        print(f"Evidence: {out}\nSummary: {out / 'summary.json'}\nCSV: {out / 'summary.csv'}")
        print("Resume/replay unfinished arms: " + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + ("" if args.resume else " --resume") + ("" if args.output else " --output " + shlex.quote(str(out))))
        if any((e.get("canonical_probe") or {}).get("stability_errors") for e in events):
            print("Unstable measurements are retained, not retry-until-win samples; a new predeclared experiment is required.")
        return 1 if failures or bad else 0


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; owned child stopped. Preserve output and --resume to replay the unfinished arm.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, BenchError, subprocess.SubprocessError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        sys.exit(1)
