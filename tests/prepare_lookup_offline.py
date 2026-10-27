#!/usr/bin/env python3
"""Audit a local StairCut kit without loading models or accessing the network.

The corpus helpers are deliberately separate from main(): the runner calls them
against its own already-ready loopback baseline, never during asset preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import urllib.parse
import urllib.request

VERSION = 1
DOC_SEP = 1 << 30
MAX_ORDER = 8
HEADER_LIMIT = 64 * 1024 * 1024
DEFAULT_KIT = Path.home() / ".mlx-serve/bench/lookup-staircut"
ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "qwen27b": ("ddalcu/Qwen3.8-27B-MLX-Serve-4bit", "incoai/Qwen3.8-27B-DFlash2"),
    "flashnext": ("TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP", "incoai/Qwen3.8-Flash-Next-DFlash2"),
    "smoke": ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", None),
}
DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "U16": 16, "I16": 16,
    "U32": 32, "I32": 32, "U64": 64, "I64": 64,
    "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8, "F4": 4, "F4_E2M1": 4,
}
MTP_MARKERS = {
    "mtp.fc.weight", "language_model.mtp.fc.weight",
    "mtp.eh_proj.weight", "language_model.mtp.eh_proj.weight",
    "attn.mixer.q_proj.weight", "layers.0.eh_proj.weight",
}


class ReadinessError(ValueError):
    pass


def absolute(path):
    return Path(path).expanduser().resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReadinessError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path, limit=HEADER_LIMIT):
    path = Path(path)
    if not path.is_file():
        raise ReadinessError(f"Missing local asset: {path}; stage it before disconnecting")
    if path.stat().st_size > limit:
        raise ReadinessError(f"JSON exceeds {limit} byte safety bound: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as exc:
        raise ReadinessError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReadinessError(f"Expected JSON object: {path}")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def safetensors_header(path):
    """Validate all tensor regions using only the length prefix and JSON header."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ReadinessError(f"Truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= HEADER_LIMIT or length > size - 8:
            raise ReadinessError(f"Invalid/truncated safetensors header ({length} bytes): {path}")
        raw = stream.read(length)
    try:
        header = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as exc:
        raise ReadinessError(f"Invalid safetensors JSON: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ReadinessError(f"Safetensors header is not an object: {path}")
    intervals = []
    tensors = {}
    for key, entry in header.items():
        if key == "__metadata__":
            if entry is not None and not isinstance(entry, dict):
                raise ReadinessError(f"Invalid safetensors metadata: {path}")
            continue
        if not isinstance(entry, dict):
            raise ReadinessError(f"Invalid tensor header: {path}:{key}")
        shape, offsets, dtype = entry.get("shape"), entry.get("data_offsets"), entry.get("dtype")
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or not isinstance(dtype, str) or dtype not in DTYPE_BITS):
            raise ReadinessError(f"Invalid shape/dtype/offsets: {path}:{key}")
        start, end = offsets
        expected = (math.prod(shape) * DTYPE_BITS[dtype] + 7) // 8
        if end < start or end > size - 8 - length or end - start != expected:
            raise ReadinessError(f"Tensor payload bounds/size mismatch: {path}:{key}; re-stage complete shard")
        if end > start:
            intervals.append((start, end, key))
        tensors[key] = entry
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise ReadinessError(f"Overlapping tensor regions: {path}:{previous[2]} and {current[2]}")
    if not tensors:
        raise ReadinessError(f"No tensors in {path}")
    return tensors, {"path": str(path.resolve()), "size_bytes": size,
                     "header_sha256": hashlib.sha256(raw).hexdigest(), "tensors": len(tensors)}


def checkpoint_audit(directory):
    directory = absolute(directory)
    if not directory.is_dir():
        raise ReadinessError(f"Missing model/drafter directory: {directory}; stage local files first")
    index_path = directory / "model.safetensors.index.json"
    index = read_json(index_path) if index_path.exists() else None
    weight_map = index.get("weight_map") if index else None
    if index is not None and (not isinstance(weight_map, dict) or not weight_map):
        raise ReadinessError(f"Missing/non-object/empty weight_map: {index_path}")
    # Match loadWeightsFromOpenDir: indexed root shards, otherwise every
    # root-level safetensors file (including symlinked HF cache shards).
    paths = set(directory.glob("*.safetensors")) if index is None else set()
    for key, name in (weight_map or {}).items():
        if not isinstance(key, str) or not isinstance(name, str):
            raise ReadinessError(f"Invalid index mapping: {index_path}:{key}")
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ReadinessError(f"Indexed shard is not a root safetensors file: {index_path}: {name}")
        candidate = directory / name
        if not candidate.is_file():
            raise ReadinessError(f"Missing indexed shard: {index_path}: {name}")
        paths.add(candidate)
    if not paths:
        raise ReadinessError(f"Missing selected safetensors shards: {directory}")
    headers, audits, all_keys = {}, [], set()
    for path in sorted(paths):
        tensors, audit = safetensors_header(path)
        headers[path.resolve()] = tensors
        audits.append(audit)
        all_keys.update(tensors)
    for key, name in (weight_map or {}).items():
        if key not in headers[(directory / name).resolve()]:
            raise ReadinessError(f"Index key absent from named shard: {index_path}:{key} -> {name}")
    # Indexed ownership wins; absent an index the loader consumes all root files.
    served = {key: headers[(directory / name).resolve()][key] for key, name in weight_map.items()} if weight_map else {
        key: value for header in headers.values() for key, value in header.items()}
    return served, all_keys, {"shards": audits,
                             "index_sha256": sha256_file(index_path) if index else None,
                             "payload_verified": False}


def config_dimensions(config, path):
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ReadinessError(f"Invalid text_config in {path}")
    for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
        if type(text.get(key)) is not int or text[key] <= 0:
            raise ReadinessError(f"Missing/invalid {key} in {path}")
    return text


def audit_target_headers(tensors, config, text, architecture):
    """Check required embedding and layer anchors, not full execution parity."""
    if architecture not in ("qwen2", "qwen3_5", "qwen4_exp"):
        raise ReadinessError(f"Unsupported local-kit target architecture: {architecture}")
    guessed = "model" if architecture == "qwen2" else "language_model.model"
    prefixes = (guessed,) + tuple(p for p in ("language_model.model", "model.language_model", "model") if p != guessed)
    prefix = next((p for p in prefixes if any(k.startswith(p + ".") for k in tensors)), guessed)
    hidden = text["hidden_size"]

    def require(name, shape):
        entry = tensors.get(name)
        if entry is None or entry["shape"] != shape:
            raise ReadinessError(f"Missing/invalid required target tensor {name}; expected {shape}")
        return entry

    name = prefix + ".embed_tokens"
    weight = tensors.get(name + ".weight")
    if weight is None:
        raise ReadinessError(f"Missing required target tensor {name}.weight")
    if weight["dtype"] in ("BF16", "F16", "F32"):
        require(name + ".weight", [text["vocab_size"], hidden])
    else:
        shape = weight["shape"]
        scales = tensors.get(name + ".scales")
        if (weight["dtype"] != "U32" or len(shape) != 2 or shape[0] != text["vocab_size"]
                or shape[1] <= 0 or scales is None or len(scales["shape"]) != 2
                or scales["shape"][0] != shape[0] or scales["shape"][1] <= 0
                or hidden % scales["shape"][1] or (shape[1] * 32) % hidden
                or shape[1] * 32 // hidden not in (2, 3, 4, 5, 6, 8)):
            raise ReadinessError(f"Invalid packed target embedding geometry: {name}")
        quant = config.get("quantization", config.get("quantization_config", {}))
        if quant.get("mode", "affine") == "affine":
            require(name + ".biases", scales["shape"])
    if architecture == "qwen4_exp":
        key = prefix + ".hyper_connection_mixer.hc_norm.weight"
        entry = tensors.get(key)
        if entry is None or math.prod(entry["shape"]) != text.get("hc_count", 4) * hidden:
            raise ReadinessError(f"Missing/invalid grouped final target norm: {key}")
    else:
        require(prefix + ".norm.weight", [hidden])
    for layer in range(text["num_hidden_layers"]):
        base = f"{prefix}.layers.{layer}."
        if architecture == "qwen4_exp":
            hc = text.get("hc_count", 4)
            for block in ("attn_hyper_connection", "mlp_hyper_connection"):
                key = base + block + ".hc_norm.weight"
                entry = tensors.get(key)
                if entry is None or math.prod(entry["shape"]) != hc * hidden:
                    raise ReadinessError(f"Missing/invalid grouped target norm: {key}")
        else:
            require(base + "input_layernorm.weight", [hidden])
            require(base + "post_attention_layernorm.weight", [hidden])
        matrices = [v for k, v in tensors.items() if k.startswith(base)
                    and k.endswith(".weight") and len(v["shape"]) >= 2]
        if not matrices or any(not all(v["shape"]) for v in matrices):
            raise ReadinessError(f"Missing/empty target layer projections: {base}")
        interval = text.get("full_attention_interval", 4)
        linear = architecture != "qwen2" and interval > 0 and (layer + 1) % interval != 0
        candidates = ("linear_attn.in_proj_qkvz", "linear_attn.in_proj_qkv") if linear else ("self_attn.q_proj",)
        projection = next((base + p for p in candidates if base + p + ".weight" in tensors), None)
        if projection is None:
            raise ReadinessError(f"Missing required target attention projection: {base}")
        entry = tensors[projection + ".weight"]
        shape = entry["shape"]
        if len(shape) != 2 or not all(shape):
            raise ReadinessError(f"Invalid target attention matrix: {projection}")
        if entry["dtype"] in ("BF16", "F16", "F32"):
            if shape[1] != hidden:
                raise ReadinessError(f"Target attention input width mismatch: {projection}")
        elif entry["dtype"] == "U32":
            scales = tensors.get(projection + ".scales")
            if (shape[1] * 32 % hidden or shape[1] * 32 // hidden not in (2, 3, 4, 5, 6, 8)
                    or scales is None or len(scales["shape"]) != 2
                    or scales["shape"][0] != shape[0] or scales["shape"][1] <= 0
                    or hidden % scales["shape"][1]):
                raise ReadinessError(f"Invalid packed target attention geometry: {projection}")
    return prefix


def audit_dflash_contract(root, target_layers, path):
    nested = root.get("dflash_config")
    contract = dict(root)
    if isinstance(nested, dict):
        contract.update(nested)

    def u32(key, default=None):
        value = contract.get(key, default)
        if type(value) is not int or not 0 <= value <= 0xffffffff:
            raise ReadinessError(f"Missing/invalid DFlash {key}: {path}")
        return value

    block = u32("block_size")
    anchor = ("markov_rank" in root or contract.get("projector_type") == "dspark"
              or any("DSpark" in a for a in root.get("architectures", []) if isinstance(a, str)))
    if block + int(anchor) < 2 or (anchor and block == 0xffffffff):
        raise ReadinessError(f"Invalid DFlash block_size: {path}")
    u32("mask_token_id")
    if root.get("is_causal") is True:
        raise ReadinessError(f"Causal DFlash blocks unsupported: {path}")
    targets = contract.get("target_layer_ids")
    if (not isinstance(targets, list) or not targets
            or any(type(n) is not int or not 0 <= n < target_layers for n in targets)
            or any(b <= a for a, b in zip(targets, targets[1:]))):
        raise ReadinessError(f"Invalid DFlash capture layers: {path}")
    rank, top = u32("selector_rank", 0), u32("selector_top_k", 0)
    kernel, group = u32("conv_kernel_size", 0), u32("conv_group_size", 0)
    if ((rank > 0) != (top > 1) or (kernel > 0) != (group > 0)):
        raise ReadinessError(f"Incomplete DFlash selector/convolution pair: {path}")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                "head_dim", "intermediate_size"):
        if type(root.get(key)) is not int or not 0 < root[key] <= 0xffffffff:
            raise ReadinessError(f"Missing/invalid DFlash {key}: {path}")
    kv_heads = root.get("num_key_value_heads", root["num_attention_heads"])
    if type(kv_heads) is not int or not 0 < kv_heads <= 0xffffffff:
        raise ReadinessError(f"Invalid DFlash num_key_value_heads: {path}")
    if group and root["hidden_size"] % group:
        raise ReadinessError(f"DFlash convolution group does not divide hidden size: {path}")
    eps = root.get("rms_norm_eps")
    if type(eps) not in (int, float) or not math.isfinite(eps) or eps <= 0:
        raise ReadinessError(f"Missing/invalid DFlash rms_norm_eps: {path}")
    multiplier = contract.get("output_multiplier", 1.0)
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier == 0:
        raise ReadinessError(f"Invalid DFlash output_multiplier: {path}")
    return rank > 0


def audit_mtp_source(directory, tensors, audit, architecture):
    for relative in ("mtp/weights.safetensors", "mtp.safetensors", "model-mtp.safetensors",
                     "optiq/mtp.safetensors", "mtp_head.safetensors", "mtp-4bit.safetensors"):
        path = directory / relative
        if not path.is_file():
            continue
        try:
            with path.open("rb") as stream:
                prefix = stream.read(8)
                length = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else 0
                raw = stream.read(length) if 0 < length <= HEADER_LIMIT else b""
            selected = any(('"' + key + '"').encode() in raw for key in MTP_MARKERS)
        except OSError:
            selected = False
        if selected:
            _, sidecar = safetensors_header(path)
            audit["mtp_source"] = {"source": "sidecar", "file": sidecar}
            return True
    index_path = directory / "model.safetensors.index.json"
    keys = read_json(index_path)["weight_map"] if index_path.exists() else tensors
    generic = bool(MTP_MARKERS & keys.keys())
    # resolveMtpSource also probes model.safetensors after an index without
    # a generic marker; audit that source only when its header selects it.
    single = directory / "model.safetensors"
    if not generic and index_path.exists() and single.is_file():
        with single.open("rb") as stream:
            prefix_bytes = stream.read(8)
            length = struct.unpack("<Q", prefix_bytes)[0] if len(prefix_bytes) == 8 else 0
            raw = stream.read(length) if 0 < length <= HEADER_LIMIT else b""
        if any(('"' + key + '"').encode() in raw for key in MTP_MARKERS):
            _, single_audit = safetensors_header(single)
            audit["mtp_checkpoint_file"] = single_audit
            generic = True
    qwen4 = architecture == "qwen4_exp" and bool(
        {"mtp.fc_hidden.weight", "language_model.mtp.fc_hidden.weight"} & keys.keys())
    audit["mtp_source"] = {"source": "checkpoint" if generic or qwen4 else None}
    return generic or qwen4


def audit_ple(directory, tensors, config):
    parts = {}
    pattern = re.compile(r".*\.ple\.ple_embedding\.ngram_embedding\.(?:shards\.|shard_)(\d+)\.(weight|scales|biases|weight_scale)$")
    for key, value in tensors.items():
        match = pattern.fullmatch(key)
        if match:
            group = parts.setdefault(int(match[1]), {})
            if match[2] in group:
                raise ReadinessError(f"Duplicate embedded PLE shard part: {directory}:{key}")
            group[match[2]] = value
    if parts:
        expected_parts = config.get("split_ngram_parts", 128)
        if type(expected_parts) is not int or sorted(parts) != list(range(expected_parts)):
            raise ReadinessError(f"Incomplete embedded PLE shard numbering: {directory}; expected {expected_parts} parts")
        ngram_size = config.get("ngram_size", 3)
        heads = config.get("heads_per_ngram", 8)
        embed_dim = config.get("ple_embed_dim", 2560)
        if (any(type(n) is not int or n <= 0 for n in (ngram_size, heads, embed_dim))
                or ngram_size < 2 or embed_dim % ((ngram_size - 1) * heads)):
            raise ReadinessError(f"Invalid embedded PLE config geometry: {directory}")
        dimension = embed_dim // ((ngram_size - 1) * heads)
        geometry = None
        for number, group in parts.items():
            nvfp4 = "weight_scale" in group
            required = {"weight", "weight_scale"} if nvfp4 else {"weight", "scales", "biases"}
            if set(group) != required:
                raise ReadinessError(f"Incomplete embedded PLE shard {number}: {directory}")
            rows = None
            for name, value in group.items():
                shape = value["shape"]
                dtype = ("U8" if name == "weight" else "F8_E4M3") if nvfp4 else ("U32" if name == "weight" else "BF16")
                if (len(shape) != 2 or not all(shape) or value["dtype"] != dtype
                        or (rows is not None and shape[0] != rows)):
                    raise ReadinessError(f"Invalid embedded PLE matrix dimensions/dtype: {directory}:{number}")
                rows = shape[0]
            weight_cols = group["weight"]["shape"][1]
            scale_cols = group["weight_scale" if nvfp4 else "scales"]["shape"][1]
            packed = weight_cols * (8 if nvfp4 else 32)
            if (packed % dimension or packed // dimension not in ((4,) if nvfp4 else (2, 3, 4, 5, 6, 8))
                    or dimension % scale_cols or dimension // scale_cols > 1024
                    or (nvfp4 and dimension // scale_cols != 16)
                    or (not nvfp4 and group["biases"]["shape"][1] != scale_cols)):
                raise ReadinessError(f"Unsupported embedded PLE packing: {directory}:{number}")
            current = (nvfp4, weight_cols, scale_cols)
            if geometry is not None and current != geometry:
                raise ReadinessError(f"Inconsistent embedded PLE shard geometry: {directory}:{number}")
            geometry = current
        return {"source": "embedded", "shards": len(parts), "dimension": dimension}
    table = directory / "ngram_table.bin"
    if not table.is_file():
        raise ReadinessError(f"FlashNext needs embedded PLE tensors or local {table}; do not download at runtime")
    entries, audit = safetensors_header(table)
    with table.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        metadata = json.loads(stream.read(length)).get("__metadata__")
    if not isinstance(metadata, dict) or ("format" in metadata and metadata["format"] != "mlx-serve-ngram"):
        raise ReadinessError(f"Invalid external PLE metadata: {table}")
    values = []
    for key in ("bits", "group_size"):
        value = metadata.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\+?[0-9]+", value) or int(value) > 0xffffffff:
            raise ReadinessError(f"Invalid external PLE {key}: {table}")
        values.append(int(value))
    bits, group = values
    if bits not in (2, 3, 4, 5, 6, 8, 16) or (bits != 16 and not 0 < group <= 1024):
        raise ReadinessError(f"Unsupported external PLE packing: {table}")

    def matrix(key, dtype):
        value = entries.get(key)
        if (value is None or value["dtype"] != dtype or len(value["shape"]) != 2
                or any(not 0 < n <= 0xffffffff for n in value["shape"])):
            raise ReadinessError(f"Invalid external PLE matrix: {table}:{key}")
        return value["shape"]

    rows, cols = matrix("weight", "BF16" if bits == 16 else "U32")
    dimension = cols
    if bits != 16:
        sr, sc = matrix("scales", "BF16")
        br, bc = matrix("biases", "BF16")
        dimension = sc * group
        if sr != rows or br != rows or sc != bc or dimension > 0xffffffff or dimension * bits != cols * 32:
            raise ReadinessError(f"Inconsistent external PLE packing geometry: {table}")
    return {"source": "external", "file": audit, "dimension": dimension}


def audit_model(identifier, model_dir, drafter_dir):
    directory = absolute(model_dir)
    config_path = directory / "config.json"
    config = read_json(config_path)
    text = config_dimensions(config, config_path)
    architecture = config.get("model_type", text.get("model_type"))
    if not isinstance(architecture, str) or not architecture:
        raise ReadinessError(f"Missing model_type: {config_path}")
    tokenizer_path = directory / "tokenizer.json"
    tokenizer = read_json(tokenizer_path)
    tokenizer_model = tokenizer.get("model")
    if not isinstance(tokenizer_model, dict) or not tokenizer_model.get("vocab"):
        raise ReadinessError(f"Missing tokenizer model/vocabulary: {tokenizer_path}")
    for optional in ("tokenizer_config.json", "generation_config.json"):
        if (directory / optional).exists():
            read_json(directory / optional)
    tensors, all_keys, audit = checkpoint_audit(directory)
    audit["config_sha256"] = sha256_file(config_path)
    audit["weight_prefix"] = audit_target_headers(tensors, config, text, architecture)
    audit["limits"] = [
        "Header-only: payload values, quantization kernels and full model execution are not validated",
        "Layer anchors and nonempty projection geometry checked; complete architecture-specific projection binding requires runtime load",
        "MTP source markers establish discovery only, not complete head binding or engagement",
    ]
    if architecture == "qwen4_exp":
        audit["ple"] = audit_ple(directory, tensors, text)
    drafter_audit = None
    drafter = None
    selector = False
    if drafter_dir is not None:
        drafter = absolute(drafter_dir)
        dc_path = drafter / "config.json"
        dc = read_json(dc_path)
        dtext = config_dimensions(dc, dc_path)
        for key in ("hidden_size", "vocab_size"):
            if dtext[key] != text[key]:
                raise ReadinessError(f"Drafter/target {key} mismatch: {drafter} vs {directory}")
        selector = audit_dflash_contract(dc, text["num_hidden_layers"], dc_path)
        _, _, drafter_audit = checkpoint_audit(drafter)
        drafter_audit["config_sha256"] = sha256_file(dc_path)
    native_mtp = audit_mtp_source(directory, tensors, audit, architecture)
    # Architecture eligibility is only a candidate: loaded projection coverage,
    # Metal availability and actual engagement must still be checked by runner.
    candidate = bool(drafter and selector and architecture == "qwen3_5"
                     and not text.get("num_experts", 0) and not text.get("hadamard_block", 0))
    return {"id": identifier, "model_dir": str(directory),
            "drafter_dir": str(drafter) if drafter else None,
            "architecture": architecture, "native_mtp": native_mtp,
            "tree_candidate": candidate,
            "tree_candidate_note": "Runtime row-exact projection/Metal engagement required" if candidate else "Not eligible for the row-exact DFlash tier-tree path",
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "audit": audit, "drafter_audit": drafter_audit}


def audit_binary(binary):
    binary = absolute(binary)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ReadinessError(f"Missing/non-executable binary: {binary}; supply --binary or explicitly use --build-command with a local-only build")
    result = {"path": str(binary), "sha256": sha256_file(binary), "libraries": [], "metallibs": []}
    otool = shutil.which("otool")
    if not otool:
        result["linkage_audit"] = "otool unavailable; dependency closure not audited"
        if sys.platform == "darwin":
            raise ReadinessError("otool is unavailable; install/stage Apple command-line developer tools before disconnecting")
        result["warnings"] = ["Non-macOS binary: Metal readiness not established"]
        return result
    visited = set()

    def expand(value, owner):
        value = value.replace("@executable_path", str(binary.parent)).replace("@loader_path", str(owner.parent))
        return Path(value).resolve() if not value.startswith("@") else None

    def inspect(path, inherited):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        def run(flag):
            completed = subprocess.run([otool, flag, str(path)], text=True, capture_output=True, check=False)
            if completed.returncode:
                raise ReadinessError(f"otool {flag} failed for {path}: {completed.stderr.strip()}")
            return completed.stdout
        commands = run("-l")
        own = []
        lines = commands.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == "cmd LC_RPATH":
                for next_line in lines[index + 1:index + 5]:
                    match = re.match(r"\s*path (.+) \(offset \d+\)", next_line)
                    if match:
                        value = expand(match[1], path)
                        if value is None:
                            raise ReadinessError(f"Unresolvable rpath {match[1]} in {path}")
                        own.append(value)
                        break
        runpaths = own + inherited
        for line in run("-L").splitlines()[1:]:
            match = re.match(r"\s+(.+?) \(compatibility version", line)
            if not match:
                continue
            name = match[1]
            if name.startswith(("/System/Library/", "/usr/lib/")):
                continue  # These can exist only in Apple's dyld shared cache.
            if name.startswith("@rpath/"):
                candidates = [folder / name[len("@rpath/"):] for folder in runpaths]
            else:
                value = expand(name, path)
                candidates = [value] if value is not None else []
            resolved = next((p.resolve() for p in candidates if p.is_file()), None)
            if resolved is None:
                raise ReadinessError(f"Missing dylib {name} required by {path}; searched {[str(p) for p in candidates]}")
            result["libraries"].append({"loader": str(path), "install_name": name, "path": str(resolved)})
            if resolved.name == "libmlx.dylib":
                metallib = resolved.parent / "mlx.metallib"
                if not metallib.is_file() or metallib.stat().st_size == 0:
                    raise ReadinessError(f"Missing required MLX Metal library: {metallib}; stage alongside libmlx.dylib")
                result["metallibs"].append({"path": str(metallib), "sha256": sha256_file(metallib)})
            inspect(resolved, runpaths)
        result.setdefault("rpaths", {})[str(path)] = [str(p) for p in runpaths]
    inspect(binary, [])
    if not result["metallibs"]:
        raise ReadinessError("No local libmlx.dylib/mlx.metallib dependency found; this kit requires the MLX/Metal binary")
    result["linkage_audit"] = "non-system dependency closure resolved locally; Apple system paths accepted in dyld cache"
    return result


def audit_probe(probe):
    if probe is None:
        return [], {"ready": False, "reason": "No --probe supplied: diagnostics-only, not llmprobe performance-ready"}
    path = absolute(probe)
    if not path.is_file():
        raise ReadinessError(f"Missing local llmprobe CLI: {path}")
    node = shutil.which("node")
    if path.suffix in (".js", ".mjs", ".cjs"):
        if not node:
            raise ReadinessError("Local node executable is required for --probe JavaScript entry point")
        command = [str(absolute(node)), str(path)]
    elif os.access(path, os.X_OK):
        command = [str(path)]
        with path.open("rb") as stream:
            first = stream.readline(256).decode("utf-8", errors="replace").strip()
        if "node" in first and not node:
            raise ReadinessError(f"Node shebang in {path} but no local node executable")
    else:
        raise ReadinessError(f"Probe must be executable or a JavaScript entry point: {path}")
    package = next((p / "package.json" for p in path.parents if (p / "package.json").is_file()), None)
    if package is None:
        raise ReadinessError(f"Cannot audit llmprobe dependencies: no package.json above {path}")
    packages = []
    seen = set()

    def audit_package(package_json):
        package_json = package_json.resolve()
        if package_json in seen:
            return
        seen.add(package_json)
        metadata = read_json(package_json)
        packages.append({"path": str(package_json), "sha256": sha256_file(package_json), "name": metadata.get("name"), "version": metadata.get("version")})
        for dependency in metadata.get("dependencies", {}):
            dependency_package = next((parent / "node_modules" / dependency / "package.json"
                                       for parent in (package_json.parent, *package_json.parent.parents)
                                       if (parent / "node_modules" / dependency / "package.json").is_file()), None)
            if dependency_package is None:
                raise ReadinessError(f"Missing local probe dependency {dependency} required by {package_json}; stage complete node_modules before flight")
            audit_package(dependency_package)
    audit_package(package)
    return command, {"ready": True, "entry_sha256": sha256_file(path), "packages": packages,
                     "node": str(absolute(node)) if node else None,
                     "audit_scope": "local entry point and declared required dependency packages; no probe execution"}


def tokenize(base_url, text, timeout=60.0):
    parsed = urllib.parse.urlsplit(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ReadinessError("Corpus tokenizer must be an http:// loopback server owned by the runner")
    request = urllib.request.Request(base_url.rstrip("/") + "/tokenize",
                                     data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    # Ignore proxy environment and refuse redirects away from the owned server.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        data = json.load(response)
    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, list) or any(type(n) is not int or not 0 <= n < DOC_SEP for n in tokens):
        raise ReadinessError("/tokenize must return {tokens:[nonnegative integer IDs below DOC_SEP]}")
    return tokens


def write_npy_i32(path, values):
    header = f"{{'descr': '<i4', 'fortran_order': False, 'shape': ({len(values)},), }}".encode("ascii")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    if len(header) > 65535:
        raise ReadinessError("NPY v1 header too long")
    with Path(path).open("wb") as stream:
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
        for start in range(0, len(values), 4096):
            batch = values[start:start + 4096]
            stream.write(struct.pack("<" + "i" * len(batch), *batch))


def build_corpus(base_url, model_dir, corpus_texts, output_dir, timeout=60.0):
    """Tokenize documents via the live baseline and write loader-compatible assets."""
    if not isinstance(corpus_texts, list) or not corpus_texts or any(not isinstance(t, str) or not t for t in corpus_texts):
        raise ReadinessError("corpus_texts must be a nonempty list of nonempty strings")
    tokenizer_hash = sha256_file(absolute(model_dir) / "tokenizer.json")
    tokens = []
    for text in corpus_texts:
        tokens.extend(tokenize(base_url, text, timeout))
        tokens.append(DOC_SEP)
    if len(tokens) >= (1 << 31):
        raise ReadinessError("Corpus is too large for signed i32 suffix positions")
    # Bounded tuple order exactly matches the server's max_order-prefix check.
    suffix_array = sorted(range(len(tokens)), key=lambda position: tokens[position:position + MAX_ORDER])
    directory = absolute(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"path": str(directory), "tokenizer_sha256": tokenizer_hash,
                "max_order": MAX_ORDER, "doc_sep": DOC_SEP, "n_tokens": len(tokens),
                "documents": len(corpus_texts),
                "corpus_texts_sha256": hashlib.sha256(json.dumps(corpus_texts, ensure_ascii=False).encode()).hexdigest()}
    # Publish metadata last: an interrupted build cannot look complete.
    (directory / "meta.json").unlink(missing_ok=True)
    write_npy_i32(directory / "tokens.npy", tokens)
    write_npy_i32(directory / "sa.npy", suffix_array)
    metadata["tokens_sha256"] = sha256_file(directory / "tokens.npy")
    metadata["sa_sha256"] = sha256_file(directory / "sa.npy")
    write_json(directory / "meta.json", metadata)
    return metadata


def ensure_corpus(model, manifest, base_url, kit):
    tokenizer_hash = sha256_file(absolute(model["model_dir"]) / "tokenizer.json")
    directory = absolute(model.get("corpus_dir") or (Path(kit) / "corpora" / tokenizer_hash))
    workloads = read_json(manifest["workloads"])
    texts = workloads["corpus_texts"]
    text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    metadata_path = directory / "meta.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if (metadata.get("tokenizer_sha256") == tokenizer_hash and metadata.get("corpus_texts_sha256") == text_hash
                and metadata.get("max_order") == MAX_ORDER and metadata.get("doc_sep") == DOC_SEP
                and all((directory / name).is_file() and sha256_file(directory / name) == metadata.get(key)
                        for name, key in (("tokens.npy", "tokens_sha256"), ("sa.npy", "sa_sha256")))):
            return str(directory)
    build_corpus(base_url, model["model_dir"], texts, directory)
    return str(directory)


def overrides(values, option):
    result = {}
    for value in values or []:
        identifier, separator, path = value.partition("=")
        if not separator or not identifier or not path or identifier in result:
            raise ReadinessError(f"{option} requires unique id=path values: {value}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
            raise ReadinessError(f"Unsafe profile id: {identifier}")
        result[identifier] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", default=str(DEFAULT_KIT))
    parser.add_argument("--binary", default=str(ROOT / "zig-out/bin/mlx-serve"))
    parser.add_argument("--model", action="append", metavar="ID=PATH", help="Select only these profiles; absent selects all default profiles")
    parser.add_argument("--drafter", action="append", metavar="ID=PATH", help="Override selected profile drafter; PATH=none disables")
    parser.add_argument("--probe", help="Local llmprobe executable or JavaScript entry point (never npx)")
    parser.add_argument("--require-probe", action="store_true", help="Fail unless the llmprobe asset audit succeeds")
    parser.add_argument("--build-command", help="Explicit local-only command, run only if binary is missing; shell syntax is not supported")
    args = parser.parse_args(argv)
    try:
        models = overrides(args.model, "--model")
        drafters = overrides(args.drafter, "--drafter")
        if not models:
            models = {identifier: str(Path.home() / ".mlx-serve/models" / values[0]) for identifier, values in DEFAULTS.items()}
        if drafters.keys() - models.keys():
            raise ReadinessError(f"Drafter override has no selected model: {sorted(drafters.keys() - models.keys())}")
        binary = absolute(args.binary)
        if not binary.exists() and args.build_command:
            command = shlex.split(args.build_command)
            if not command:
                raise ReadinessError("--build-command must not be empty")
            completed = subprocess.run(command, cwd=ROOT, check=False,
                                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            if completed.returncode:
                raise ReadinessError(f"Explicit build command failed with exit {completed.returncode}")
        binary_audit = audit_binary(binary)
        probe_cmd, probe_audit = audit_probe(args.probe)
        if args.require_probe and not probe_cmd:
            raise ReadinessError("--require-probe needs --probe pointing to a completely local llmprobe installation")
        profiles, errors = [], []
        for identifier, path in models.items():
            default_drafter = DEFAULTS.get(identifier, (None, None))[1]
            drafter = drafters.get(identifier, str(Path.home() / ".mlx-serve/drafters" / default_drafter) if default_drafter else None)
            if drafter is not None and drafter.lower() == "none":
                drafter = None
            try:
                profiles.append(audit_model(identifier, path, drafter))
            except (ReadinessError, OSError) as exc:
                errors.append(f"{identifier}: {exc}")
        if errors:
            raise ReadinessError("Selected profiles are not flight-ready:\n  " + "\n  ".join(errors))
        source = ROOT / "tests/fixtures/lookup_staircut/workloads.json"
        workloads = read_json(source)
        required_cases = {"copy", "edit", "code", "prose", "corpus", "budget", "keyed", "lookup-hit", "lookup-edit"}
        cases = workloads.get("cases")
        if (workloads.get("version") != VERSION or not isinstance(cases, list)
                or {case.get("id") for case in cases} != required_cases
                or not workloads.get("corpus_texts")):
            raise ReadinessError(f"Invalid workload schema: {source}")
        kit = absolute(args.kit)
        destination = kit / "workloads.json"
        write_json(destination, workloads)
        manifest = {"version": VERSION, "binary": str(binary), "workloads": str(destination),
                    "models": profiles, "probe_cmd": probe_cmd,
                    "performance_probe_ready": bool(probe_cmd),
                    "readiness": "performance-probe-assets-ready" if probe_cmd else "diagnostics-only",
                    "binary_audit": binary_audit, "probe_audit": probe_audit,
                    "workloads_sha256": sha256_file(destination),
                    "offline_audit": {"network_used": False, "model_loaded": False,
                                      "tensor_payloads_read": False,
                                      "runtime_memory_admission": "not tested; server guard remains required",
                                      "corpus": "built later by runner against its owned ready baseline /tokenize endpoint"}}
        write_json(kit / "manifest.json", manifest)
        print(json.dumps({"manifest": str(kit / "manifest.json"), "readiness": manifest["readiness"],
                          "models": [model["id"] for model in profiles], "probe_cmd": probe_cmd}, indent=2))
        if not probe_cmd:
            print("NOTE: diagnostics-only; supply --probe and --require-probe for the llmprobe performance asset gate.", file=sys.stderr)
        return 0
    except (ReadinessError, OSError, subprocess.SubprocessError) as exc:
        print(f"OFFLINE READINESS FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
