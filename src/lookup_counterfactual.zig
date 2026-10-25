//! Opt-in readback diagnostic, not a throughput benchmark. The native verifier/commit
//! remains the only execution path. Construct Probe only for an exclusive diagnostic slot.
const std = @import("std");
const mlx = @import("mlx.zig");
const native = @import("transformer.zig");
const dflash = @import("dflash.zig");
const policy = @import("speculation_policy.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
pub const Digest = [32]u8;
pub const Action = enum {
    assistant8, assistant16, lookup8, lookup16,
    pub fn rows(self: Action) u32 { return switch (self) { .assistant8, .lookup8 => 8, .assistant16, .lookup16 => 16 }; }
    pub fn assistant(self: Action) bool { return self == .assistant8 or self == .assistant16; }
};
pub const Phase = enum { warmup, measured };

pub fn enabled() bool {
    const value = std.c.getenv("MLX_SERVE_LOOKUP_COUNTERFACTUAL") orelse return false;
    return std.mem.eql(u8, std.mem.sliceTo(value, 0), "1");
}

/// CPU-only setup evidence. The driver also checks assets before launch and after exit.
pub fn modelFingerprint(io: std.Io, allocator: std.mem.Allocator, paths: []const []const u8) !Digest {
    var identity: Hasher = .{};
    identity.bytes("mlx-serve-diagnostic-model-content-1");
    var buffer: [65536]u8 = undefined;
    for (paths) |path| {
        const dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
        defer dir.close(io);
        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |name| allocator.free(name);
            names.deinit(allocator);
        }
        var iterator = dir.iterate();
        var weight_count: usize = 0;
        while (try iterator.next(io)) |entry| {
            const weight = std.mem.endsWith(u8, entry.name, ".safetensors");
            if (!weight and !std.mem.endsWith(u8, entry.name, ".json") and !std.mem.endsWith(u8, entry.name, ".bin")) continue;
            const name = try allocator.dupe(u8, entry.name);
            errdefer allocator.free(name);
            try names.append(allocator, name);
            weight_count += @intFromBool(weight);
        }
        if (weight_count == 0) return error.DiagnosticUnsupportedModelFiles;
        std.mem.sort([]const u8, names.items, {}, struct {
            fn less(_: void, a: []const u8, b: []const u8) bool { return std.mem.lessThan(u8, a, b); }
        }.less);
        identity.integer(names.items.len);
        for (names.items) |name| {
            const file = try dir.openFile(io, name, .{});
            defer file.close(io);
            const before = try file.stat(io);
            if (before.kind != .file) return error.DiagnosticUnsupportedModelFiles;
            var hash = Sha256.init(.{});
            var offset: u64 = 0;
            while (offset < before.size) {
                const count: usize = @intCast(@min(buffer.len, before.size - offset));
                if (try file.readPositionalAll(io, buffer[0..count], offset) != count) return error.DiagnosticModelChanged;
                hash.update(buffer[0..count]);
                offset += count;
            }
            const after = try dir.statFile(io, name, .{});
            if (before.inode != after.inode or before.size != after.size or
                !std.meta.eql(before.mtime, after.mtime) or !std.meta.eql(before.ctime, after.ctime))
                return error.DiagnosticModelChanged;
            var digest: Digest = undefined;
            hash.final(&digest);
            identity.bytes(name);
            identity.integer(before.size);
            identity.bytes(&digest);
        }
    }
    return identity.finish();
}

/// Canonical fields, never struct padding, pointers, allocated capacity or map order.
pub const Hasher = struct {
    hash: Sha256 = Sha256.init(.{}),
    pub fn bytes(self: *Hasher, value: []const u8) void {
        self.integer(value.len);
        self.hash.update(value);
    }
    pub fn integer(self: *Hasher, value: anytype) void {
        const T = @TypeOf(value);
        const U = @Int(.unsigned, @bitSizeOf(T));
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, @as(U, @bitCast(value)), .little);
        self.hash.update(&buf);
    }
    pub fn flag(self: *Hasher, value: bool) void { self.integer(@as(u8, @intFromBool(value))); }
    pub fn finish(self: *Hasher) Digest { var out: Digest = undefined; self.hash.final(&out); return out; }

    /// Read every logical element after making a row-major view. No numeric conversion:
    /// bf16/f16 and float32 retain their actual bits. Unsupported dtypes fail closed.
    pub fn array(self: *Hasher, value: mlx.mlx_array, stream: mlx.mlx_stream) !void {
        self.flag(value.ctx != null);
        if (value.ctx == null) return;
        if (@import("builtin").target.cpu.arch.endian() != .little) return error.DiagnosticEndianUnsupported;
        var contiguous = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(contiguous);
        try mlx.check(mlx.mlx_contiguous(&contiguous, value, false, stream));
        try mlx.check(mlx.mlx_array_eval(contiguous));
        const shape = mlx.getShape(contiguous);
        self.integer(shape.len);
        for (shape) |dim| self.integer(dim);
        const dtype = mlx.mlx_array_dtype(contiguous);
        self.integer(@intFromEnum(dtype));
        const n = mlx.mlx_array_size(contiguous);
        if (n == 0) { self.bytes(&.{}); return; }
        const raw: []const u8 = switch (dtype) {
            .bool_ => std.mem.sliceAsBytes((mlx.mlx_array_data_bool(contiguous) orelse return error.DiagnosticReadbackFailed)[0..n]),
            .uint8 => (mlx.mlx_array_data_uint8(contiguous) orelse return error.DiagnosticReadbackFailed)[0..n],
            .uint32 => std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(contiguous) orelse return error.DiagnosticReadbackFailed)[0..n]),
            .int32 => std.mem.sliceAsBytes((mlx.mlx_array_data_int32(contiguous) orelse return error.DiagnosticReadbackFailed)[0..n]),
            .float16 => std.mem.sliceAsBytes((mlx.mlx_array_data_float16(contiguous) orelse return error.DiagnosticReadbackFailed)[0..n]),
            .bfloat16 => std.mem.sliceAsBytes((mlx.mlx_array_data_bfloat16(contiguous) orelse return error.DiagnosticReadbackFailed)[0..n]),
            .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(contiguous) orelse return error.DiagnosticReadbackFailed)[0..n]),
            else => return error.DiagnosticDtypeUnsupported,
        };
        self.bytes(raw);
    }

    fn slice(self: *Hasher, value: mlx.mlx_array, axis: usize, start_row: usize, rows: usize, stream: mlx.mlx_stream) !void {
        if (value.ctx == null) return self.array(value, stream);
        const shape = mlx.getShape(value);
        if (shape.len > 8 or axis >= shape.len or shape[axis] < 0 or start_row > @as(usize, @intCast(shape[axis])) or rows > @as(usize, @intCast(shape[axis])) - start_row) return error.DiagnosticInvalidState;
        var start: [8]c_int = @splat(0);
        var stop: [8]c_int = @splat(0);
        const stride: [8]c_int = @splat(1);
        @memcpy(stop[0..shape.len], shape);
        start[axis] = @intCast(start_row);
        stop[axis] = @intCast(start_row + rows);
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, value, &start, shape.len, &stop, shape.len, &stride, shape.len, stream));
        try self.array(view, stream);
    }

    pub fn kv(self: *Hasher, cache: *const native.KVCache, stream: mlx.mlx_stream) !void {
        self.integer(cache.step);
        self.integer(cache.entries.len);
        for (cache.entries) |entry| {
            self.flag(entry.initialized);
            if (!entry.initialized) continue;
            self.integer(entry.offset);
            self.integer(entry.base);
            self.integer(entry.keep);
            self.integer(entry.window);
            if (entry.base > entry.offset) return error.DiagnosticInvalidState;
            // All storage families share token axis 2. Hash only valid rows, never slack.
            inline for (.{ entry.keys, entry.values, entry.keys_scales, entry.keys_biases, entry.values_scales, entry.values_biases }) |arr| {
                try self.slice(arr, 2, 0, entry.offset - entry.base, stream);
            }
        }
    }
};

pub fn fingerprintKV(cache: *const native.KVCache, stream: mlx.mlx_stream) !Digest {
    var h: Hasher = .{};
    h.bytes("target-valid-kv-v1");
    try h.kv(cache, stream);
    return h.finish();
}

pub fn fingerprintSSM(entries: []const native.SSMCacheEntry, stream: mlx.mlx_stream) !Digest {
    var h: Hasher = .{};
    h.bytes("target-live-ssm-v1");
    h.integer(entries.len);
    for (entries) |entry| {
        h.flag(entry.initialized);
        if (!entry.initialized) continue;
        // Capture buffers belong to the previous round, not the committed state.
        // A live grouped slot must read its actual row rather than stale entry handles.
        if (entry.group) |group| {
            if (group.live) {
                if (entry.group_layer >= group.layers.len) return error.DiagnosticInvalidState;
                const live = group.layers[entry.group_layer];
                try h.slice(live.conv_state, 0, entry.group_row, 1, stream);
                try h.slice(live.ssm_state, 0, entry.group_row, 1, stream);
                if (live.aux_state.ctx != null) {
                    try h.slice(live.aux_state, 0, entry.group_row, 1, stream);
                } else try h.array(entry.aux_state, stream);
            } else {
                try h.array(entry.conv_state, stream);
                try h.array(entry.ssm_state, stream);
                try h.array(entry.aux_state, stream);
            }
        } else {
            try h.array(entry.conv_state, stream);
            try h.array(entry.ssm_state, stream);
            try h.array(entry.aux_state, stream);
        }
        try h.array(entry.qsa_pooled, stream);
        try h.array(entry.qsa_score_bank, stream);
        h.integer(entry.qsa_ratio);
        h.integer(entry.qsa_key_rows);
        h.integer(entry.qsa_pooled_blocks);
        h.integer(entry.qsa_score_blocks);
        h.integer(entry.qsa_hist_rows);
        h.flag(entry.ple_prev_valid);
        if (entry.ple_prev_valid) for (entry.ple_prev) |token| h.integer(token);
    }
    return h.finish();
}

pub fn fingerprintAssistant(context: ?*const dflash.DflashCtx, pending: []const mlx.mlx_array, pending_rounds: u32, stream: mlx.mlx_stream) !Digest {
    var h: Hasher = .{};
    h.bytes("assistant-live-context-v1");
    h.flag(context != null);
    if (context) |ctx| {
        h.integer(ctx.base_pos);
        try h.kv(&ctx.cache, stream);
        try h.array(ctx.anchor, stream);
    }
    h.integer(pending_rounds);
    h.integer(pending.len);
    for (pending) |arr| try h.array(arr, stream);
    return h.finish();
}

pub fn tokenHash(tokens: []const u32) Digest {
    return tokenHashParts(tokens, &.{});
}

pub fn tokenHashParts(first: []const u32, second: []const u32) Digest {
    var h: Hasher = .{};
    h.bytes("token-ids-v1");
    h.integer(first.len + second.len);
    for (first) |token| h.integer(token);
    for (second) |token| h.integer(token);
    return h.finish();
}

pub const OfferHash = struct { candidates: Digest, probabilities: Digest };
pub fn fingerprintOffer(tree: *const policy.Tree) !OfferHash {
    if (tree.len == 0 or tree.len > policy.MAX_TREE_ROWS) return error.DiagnosticInvalidOffer;
    var candidates: Hasher = .{};
    var probabilities: Hasher = .{};
    candidates.integer(tree.len);
    probabilities.integer(tree.len);
    probabilities.integer(@intFromBool(tree.scores_calibrated));
    for (0..tree.len) |i| {
        if ((i == 0 and tree.parents[i] != -1) or (i != 0 and (tree.parents[i] < 0 or tree.parents[i] >= @as(i32, @intCast(i))))) return error.DiagnosticInvalidOffer;
        candidates.integer(tree.tokens[i]);
        candidates.integer(tree.parents[i]);
        candidates.integer(tree.origins[i]);
        candidates.integer(@intFromEnum(tree.sources[i]));
        probabilities.integer(tree.look_order[i]);
        inline for (.{ tree.scores[i], tree.conditional_reach[i], tree.confidence[i], tree.draft_confidence[i], tree.look_confidence[i] }) |p| {
            if (!std.math.isFinite(p) or p < 0 or p > 1) return error.DiagnosticInvalidOffer;
            probabilities.integer(@as(u32, @bitCast(p)));
        }
    }
    return .{ .candidates = candidates.finish(), .probabilities = probabilities.finish() };
}

/// Capture BEFORE action-specific preparation. Controls includes actual next_token_id,
/// logical positions, sampler/stop controls and model/tokenizer IDs; prefix alone is not state.
pub const Identity = struct {
    prefix: Digest,
    target_kv: Digest,
    target_ssm: Digest,
    assistant: Digest,
    controls: Digest,
    training: Digest,
    learner: Digest,
    request_lookup: Digest,
};
pub const Preconditions = struct {
    row_exact: bool,
    greedy: bool,
    exclusive_owned_slot: bool,
    prefix_cache_off: bool,
    frozen_read_only_history: bool,
    seed: u64,
    pub fn validate(self: Preconditions) !void {
        if (!self.row_exact or !self.greedy or !self.exclusive_owned_slot or !self.prefix_cache_off or !self.frozen_read_only_history) return error.DiagnosticPreconditions;
    }
};

/// The dispatch digest records the kernels ACTUALLY dispatched (variant, dtype,
/// geometry and implementation identifier), not requested flags or `priced=true`.
pub const Kernel = struct { rows: u32, variant: policy.Variant, kind: policy.Kind, class: i64, dispatch: Digest };
pub const PreparedState = struct { target_kv: Digest, target_ssm: Digest, assistant: Digest };
pub const Timing = struct {
    verify_commit_ns: u64,
    /// Null unless all prep/selection/publish costs are settled and readback gaps excluded.
    full_round_ns: ?u64 = null,
    diagnostic_readback_ns: u64,
};
pub const Receipt = struct {
    phase: Phase,
    action: Action,
    first_offer: u32 = 0,
    identity: Identity,
    prepared: PreparedState,
    offer: OfferHash,
    selected: OfferHash,
    kernel: Kernel,
    timing: Timing,
    warmup_proven: bool,
    published: [33]u32 = @splat(0),
    published_len: usize,
    accepted_drafts: u32,
    pub fn tokens(self: *const Receipt) []const u32 { return self.published[0..self.published_len]; }
};

/// One fresh process/owned slot: excluded warmup then measured request, no third request.
/// Snapshot precedes BOTH requests. Kernel caches stay resident; every policy map/counter
/// and calibration population returns to the identical snapshot at request entry.
pub const Probe = struct {
    allocator: std.mem.Allocator,
    baseline: policy.Policy,
    action: Action,
    conditions: Preconditions,
    requests: u8 = 0,
    active: bool = false,
    captured: bool = false,
    common: ?Identity = null,
    prepared: ?PreparedState = null,
    offered: ?OfferHash = null,
    selected: ?OfferHash = null,
    warmup: ?Receipt = null,

    pub fn init(allocator: std.mem.Allocator, initial: *const policy.Policy, action: Action, conditions: Preconditions) !Probe {
        try conditions.validate();
        return .{ .allocator = allocator, .baseline = try initial.clone(allocator), .action = action, .conditions = conditions };
    }
    pub fn deinit(self: *Probe) void { self.baseline.deinit(); }

    pub fn beginRequest(self: *Probe, learner: *policy.Policy, trust: *policy.RequestTrust, conditions: Preconditions) !Phase {
        try conditions.validate();
        if (self.active or self.requests >= 2 or (self.requests == 1 and self.warmup == null) or !std.meta.eql(self.conditions, conditions)) return error.DiagnosticRequestOrder;
        const restored = try self.baseline.clone(self.allocator);
        learner.deinit();
        learner.* = restored;
        trust.* = .{};
        if (!std.mem.eql(u8, &learner.diagnosticFingerprint(), &self.baseline.diagnosticFingerprint())) return error.DiagnosticLearnerMismatch;
        self.active = true;
        self.captured = false;
        self.common = null;
        self.prepared = null;
        self.offered = null;
        self.selected = null;
        self.requests += 1;
        return if (self.requests == 1) .warmup else .measured;
    }

    /// Call at the first round BEFORE lookup or assistant preparation. No later-offer search.
    pub fn captureCommon(self: *Probe, identity: Identity) !void {
        if (!self.active or self.common != null) return error.DiagnosticRequestOrder;
        if (!std.mem.eql(u8, &identity.learner, &self.baseline.diagnosticFingerprint())) return error.DiagnosticLearnerMismatch;
        const cold_trust: policy.RequestTrust = .{};
        if (!std.mem.eql(u8, &identity.request_lookup, &cold_trust.diagnosticFingerprint())) return error.DiagnosticLearnerMismatch;
        if (self.warmup) |w| if (!std.meta.eql(w.identity, identity)) return error.DiagnosticStateMismatch;
        self.common = identity;
    }

    /// Hash the WHOLE pre-selection offer; the selected width must really be executable.
    pub fn captureOffer(self: *Probe, candidate: *const policy.Tree, selected_tree: *const policy.Tree, assistant_ran: bool, prepared: PreparedState) !void {
        if (!self.active or self.common == null or self.captured) return error.DiagnosticRequestOrder;
        if (selected_tree.len != self.action.rows() or candidate.len < self.action.rows() or assistant_ran != self.action.assistant()) return error.DiagnosticActionMismatch;
        const offered = try fingerprintOffer(candidate);
        const selected = try fingerprintOffer(selected_tree);
        if (self.warmup) |w| {
            if (!std.meta.eql(w.offer, offered) or !std.meta.eql(w.selected, selected)) return error.DiagnosticOfferMismatch;
            if (!std.meta.eql(w.prepared, prepared)) return error.DiagnosticStateMismatch;
        }
        self.offered = offered;
        self.selected = selected;
        self.prepared = prepared;
        self.captured = true;
    }

    /// Call ONLY after existing native verification, rollback/commit and publication settle.
    /// A truncated first round or failed kernel never establishes warmup. Never retry it.
    pub fn finishFirstRound(self: *Probe, kernel: Kernel, timing: Timing, published: []const u32, accepted_drafts: u32, complete: bool) !Receipt {
        if (!self.active or !self.captured) return error.DiagnosticRequestOrder;
        self.active = false;
        if (!complete or published.len == 0 or published.len > 33 or accepted_drafts >= kernel.rows or published.len != @as(usize, accepted_drafts) + 1 or kernel.rows != self.action.rows() or timing.verify_commit_ns == 0) return error.DiagnosticInvalidSample;
        if (timing.full_round_ns) |full| if (full < timing.verify_commit_ns) return error.DiagnosticInvalidSample;
        if (std.mem.allEqual(u8, &kernel.dispatch, 0)) return error.DiagnosticMissingDispatch;
        const measured = self.requests == 2;
        if (measured) {
            const warm = self.warmup orelse return error.DiagnosticMissingWarmup;
            if (!std.meta.eql(warm.kernel, kernel)) return error.DiagnosticKernelMismatch;
            if (!std.mem.eql(u32, warm.tokens(), published)) return error.DiagnosticOutputMismatch;
        }
        var receipt: Receipt = .{
            .phase = if (measured) .measured else .warmup,
            .action = self.action,
            .identity = self.common.?,
            .prepared = self.prepared.?,
            .offer = self.offered.?,
            .selected = self.selected.?,
            .kernel = kernel,
            .timing = timing,
            .warmup_proven = measured,
            .published_len = published.len,
            .accepted_drafts = accepted_drafts,
        };
        @memcpy(receipt.published[0..published.len], published);
        if (!measured) self.warmup = receipt;
        return receipt;
    }
};

/// Cross-process validation also requires common controls/model/history identities.
/// Assistant-vs-lookup deliberately differs in preparation and candidate set.
pub fn compare(a: *const Receipt, b: *const Receipt, serial_tokens: []const u32) !void {
    if (a.phase != .measured or b.phase != .measured or !a.warmup_proven or !b.warmup_proven or a.first_offer != 0 or b.first_offer != 0) return error.DiagnosticInvalidComparison;
    if (!std.meta.eql(a.identity, b.identity)) return error.DiagnosticStateMismatch;
    if (!std.mem.eql(u8, &a.prepared.target_kv, &b.prepared.target_kv) or !std.mem.eql(u8, &a.prepared.target_ssm, &b.prepared.target_ssm)) return error.DiagnosticStateMismatch;
    if (a.action.assistant() == b.action.assistant() and !std.meta.eql(a.prepared, b.prepared)) return error.DiagnosticStateMismatch;
    if (a.action.assistant() == b.action.assistant() and !std.meta.eql(a.offer, b.offer)) return error.DiagnosticOfferMismatch;
    for (.{ a, b }) |receipt| {
        if (serial_tokens.len < receipt.published_len or !std.mem.eql(u32, receipt.tokens(), serial_tokens[0..receipt.published_len])) return error.DiagnosticOutputMismatch;
    }
}

test "offer fingerprint ignores unused storage but distinguishes probability evidence" {
    const a = policy.Tree.chain(51, &.{ 73, 95 }, 0.7, 0.6);
    var b = a;
    b.tokens[31] = 999;
    try std.testing.expectEqualDeep(try fingerprintOffer(&a), try fingerprintOffer(&b));
    b.look_confidence[1] = 0.8;
    try std.testing.expect(!std.meta.eql(try fingerprintOffer(&a), try fingerprintOffer(&b)));
    b = a;
    b.look_order[1] = 2;
    try std.testing.expect(!std.meta.eql(try fingerprintOffer(&a), try fingerprintOffer(&b)));
    b = a;
    b.scores_calibrated = true;
    try std.testing.expect(!std.meta.eql(try fingerprintOffer(&a), try fingerprintOffer(&b)));
    b = a;
    b.conditional_reach[2] = 0.3;
    try std.testing.expect(!std.meta.eql(try fingerprintOffer(&a), try fingerprintOffer(&b)));
}

test "diagnostic preconditions fail closed" {
    const p: Preconditions = .{ .row_exact = true, .greedy = true, .exclusive_owned_slot = true, .prefix_cache_off = false, .frozen_read_only_history = true, .seed = 17 };
    try std.testing.expectError(error.DiagnosticPreconditions, p.validate());
}

test "excluded warmup restores every policy field before measured first offer" {
    const allocator = std.testing.allocator;
    var learner = policy.Policy.init(policy.PriceModel.init(allocator));
    defer learner.deinit();
    const conditions: Preconditions = .{ .row_exact = true, .greedy = true, .exclusive_owned_slot = true, .prefix_cache_off = true, .frozen_read_only_history = true, .seed = 43 };
    var probe = try Probe.init(allocator, &learner, .assistant8, conditions);
    defer probe.deinit();
    const baseline_hash = learner.diagnosticFingerprint();
    var trust: policy.RequestTrust = .{};
    try std.testing.expectEqual(Phase.warmup, try probe.beginRequest(&learner, &trust, conditions));
    const id: Identity = .{ .prefix = @splat(1), .target_kv = @splat(2), .target_ssm = @splat(3), .assistant = @splat(4), .controls = @splat(5), .training = @splat(6), .learner = baseline_hash, .request_lookup = trust.diagnosticFingerprint() };
    try probe.captureCommon(id);
    const tree = policy.Tree.chain(71, &.{ 73, 79, 83, 89, 97, 101, 103 }, 0.8, 0.7);
    const prepared: PreparedState = .{ .target_kv = id.target_kv, .target_ssm = id.target_ssm, .assistant = id.assistant };
    try probe.captureOffer(&tree, &tree, true, prepared);
    const kernel: Kernel = .{ .rows = 8, .variant = .generic, .kind = .chain, .class = 0, .dispatch = @splat(7) };
    const timing: Timing = .{ .verify_commit_ns = 19000, .diagnostic_readback_ns = 11000 };
    const warm = try probe.finishFirstRound(kernel, timing, &.{ 71, 73 }, 1, true);
    try std.testing.expect(!warm.warmup_proven);
    learner.observeChainAccepted(&tree, @intCast(tree.len - 1), 2);
    _ = learner.prices.observeKernel(8, 17, .chain, 0, .generic);
    try std.testing.expect(!std.mem.eql(u8, &baseline_hash, &learner.diagnosticFingerprint()));
    trust.learn(.look, 0.8, true);
    try std.testing.expect(!std.mem.eql(u8, &id.request_lookup, &trust.diagnosticFingerprint()));
    try std.testing.expectEqual(Phase.measured, try probe.beginRequest(&learner, &trust, conditions));
    try std.testing.expectEqualDeep(id.request_lookup, trust.diagnosticFingerprint());
    try std.testing.expectEqualDeep(baseline_hash, learner.diagnosticFingerprint());
    try probe.captureCommon(id);
    try probe.captureOffer(&tree, &tree, true, prepared);
    const measured = try probe.finishFirstRound(kernel, timing, &.{ 71, 73 }, 1, true);
    try std.testing.expect(measured.warmup_proven);
    try std.testing.expectError(error.DiagnosticRequestOrder, probe.beginRequest(&learner, &trust, conditions));
}
