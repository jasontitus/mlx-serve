//! DFlash draft-tree rounds.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const keyed_sample = @import("../sampling/keyed_sample.zig");
const transformer_mod = @import("../transformer.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const dflash_mod = @import("dflash/dflash.zig");
const DflashModel = dflash_mod.DflashModel;
const DrafterStepResult = @import("drafter.zig").DrafterStepResult;
const Generator = @import("../generate/generator.zig").Generator;
const capAcceptedForTokenBudget = @import("verify.zig").capAcceptedForTokenBudget;
const dflashTraceEnabled = @import("mtp/trace.zig").dflashTraceEnabled;
const verifyArgmax = @import("verify.zig").verifyArgmax;
const economics = @import("dflash/economics.zig");
const adapter = @import("lookup_adapter.zig");


var dflash_tree_logged = false;
var dflash_copy_logged = false;
/// A verify forward goes to the GPU every this many layers while the rest
/// of its graph is built (27B tree rounds: 52.8 -> 50.3 ms).
pub const VERIFY_PIPELINE_LAYERS: u32 = 4;
/// Rows of a draft-tree verify: the pending token and up to 15 nodes.
pub const DFLASH_TREE_W = 16;
/// Context tokens that must back a copied continuation (TensorFold's
/// "confident" bar: shorter backing lost more rounds than it won at 4k-16k).
pub const DFLASH_COPY_MATCH = 24;
/// Tokens the copy lookup scans back over each round (the copy and the scan are per round).
pub const DFLASH_COPY_WINDOW = 65536;

/// `[m, k]` Gumbel noise each lattice position's candidates get from the
/// keyed draw of its verify row (generated index G + 1 + position).
pub fn treeNoise(self: *const Generator, allocator: std.mem.Allocator, lat: *const dflash_mod.Lattice) ![]f32 {
    const out = try allocator.alloc(f32, lat.m * lat.k);
    const base = self.sampling.position_base + self.generated_ids.items.len + 1;
    for (0..lat.m) |d| for (0..lat.k) |j| {
        out[d * lat.k + j] = keyed_sample.gumbel(self.sampling.seed.?, base + d, @intCast(lat.cands[d * lat.k + j]));
    };
    return out;
}

/// One DFlash2 round over a best-first draft TREE instead of the selector's
/// single path: every node is verified in one forward (each row the serial
/// step along its ancestors), the target's token walked down the tree from
/// the root, and the accepted path committed into the KV cache, the GDN
/// state and the assistant context.
/// Nodes a draft tree takes: on the tensor units a 16-row window costs
/// about what an 8-row one does, so 15 within the lattice's depth;
/// elsewhere a node per position.
pub fn dflashTreeNodes(m: u32) usize {
    return if (transformer_mod.naxAvailable()) DFLASH_TREE_W - 1 else @min(m, DFLASH_TREE_W - 1);
}

pub fn dflashTreeRound(
    self: *Generator,
    allocator: std.mem.Allocator,
    model: *DflashModel,
    dctx: *dflash_mod.DflashCtx,
    blk_hidden: ?mlx.mlx_array,
    draft_logits: ?mlx.mlx_array,
    t1: u32,
    m: u32,
    anchor_pos: usize,
    kv_step_snap: usize,
    serial: bool,
    copy: []const u32,
    flight: ?*economics.Flight,
) !DrafterStepResult {
    const xfm = self.xfm;
    const s = xfm.s;
    const MAX_W = DFLASH_TREE_W;
    const tracing = dflashTraceEnabled();
    var ph = io_util.Stopwatch.init(self.timer.io);
    const max_nodes = if (flight != null) MAX_W - 1 else dflashTreeNodes(m);
    const depth_limit = economics.depthLimit(self.max_tokens -| self.completion_tokens);
    var candidate = economics.Tree.init(t1);
    var native = economics.Tree.init(t1);
    if (!dflash_tree_logged) {
        dflash_tree_logged = true;
        log.info("[dflash] draft trees engaged: up to {d} nodes a round\n", .{max_nodes});
    }

    // Rows: 0 = t1 (the root), then the lattice's best-first tree, or the
    // copied continuation as a chain.
    var w: usize = 1;
    var parents: [MAX_W]i32 = undefined;
    var depth: [MAX_W]i32 = undefined;
    var toks: [MAX_W]i32 = undefined;
    parents[0] = -1;
    depth[0] = 0;
    toks[0] = @intCast(t1);
    if (copy.len == 0 and draft_logits != null) {
        var lat = try dflash_mod.lattice(allocator, &model.selector.?, model.config.selector_top_k, blk_hidden.?, draft_logits.?, t1, s);
        defer lat.deinit(allocator);
        // A sampled target's scores carry its own noise at each position's candidates.
        const noise: ?[]f32 = if (serial) try self.treeNoise(allocator, &lat) else null;
        defer if (noise) |nz| allocator.free(nz);
        var admission: [MAX_W - 1]usize = undefined;
        var confidence: [MAX_W - 1]f32 = undefined;
        var tree = try dflash_mod.bestFirstTree(allocator, &lat, .{
            .max_nodes = max_nodes,
            .temperature = if (serial) self.sampling.temperature else 1.0,
            .noise = noise,
            .admission_order = if (flight != null) &admission else null,
            .confidence = if (flight != null) &confidence else null,
        });
        defer tree.deinit(allocator);
        if (flight != null) {
            candidate = adapter.neuralTree(t1, tree.tokens, tree.parents, tree.depth, confidence[0..tree.tokens.len], depth_limit);
            native = adapter.neuralAdmissionTree(&candidate, tree.tokens, tree.parents, admission[0..tree.tokens.len], dflashTreeNodes(m));
        } else for (tree.tokens, tree.parents, tree.depth) |tok, par, d| {
            parents[w] = if (par < 0) 0 else par + 1;
            depth[w] = @intCast(d + 1);
            toks[w] = @intCast(tok);
            w += 1;
        }
    }
    if (copy.len > 0 and !dflash_copy_logged) {
        dflash_copy_logged = true;
        log.info("[dflash] context copy verified as the round's chain ({d} rows)\n", .{copy.len + 1});
    }
    for (copy, 1..) |tok, d| {
        if (flight != null) {
            // The backed native copy remains a complete eligible incumbent;
            // mixed projection never silently replaces its chain geometry.
            const row = native.add(d - 1, tok, 1, .look, 1).?;
            native.look_order[row] = DFLASH_COPY_MATCH;
        } else {
            parents[w] = @intCast(w - 1);
            depth[w] = @intCast(d);
            toks[w] = @intCast(tok);
            w += 1;
        }
    }
    if (flight) |f| {
        if (copy.len > 0) candidate = native;
        f.select(&native, &candidate, copy.len > 0);
        w = f.chosen.len;
        std.debug.assert(w <= MAX_W);
        for (0..w) |i| {
            parents[i] = f.chosen.parents[i];
            depth[i] = if (i == 0) 0 else depth[@intCast(parents[i])] + 1;
            toks[i] = @intCast(f.chosen.tokens[i]);
        }
    }
    self.dflash_round_width = @intCast(w - 1);
    var max_depth: i32 = 0;
    for (depth[0..w]) |d| max_depth = @max(max_depth, d);
    if (tracing) {
        self.dflash_trace.add(.head, ph.read());
        ph.reset();
    }
    const spec_tree = Generator.specTreeFor(parents[0..w], depth[0..w], max_depth);
    defer spec_tree.deinit();
    const wc: c_int = @intCast(w);
    const verify_input = mlx.mlx_array_new_data(&toks, &[_]c_int{ 1, wc }, 2, .int32);
    defer _ = mlx.mlx_array_free(verify_input);

    const cap_out = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
    defer {
        for (cap_out) |a| _ = mlx.mlx_array_free(a);
        allocator.free(cap_out);
    }
    for (cap_out) |*a| a.* = mlx.mlx_array_new();
    if (flight) |f| try f.beginVerify(self);
    var cl = transformer_mod.CaptureLayers{ .ids = model.config.captureIds(), .out = cap_out };
    self.ctx.capture_layers = &cl;
    self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
    self.ctx.tree = &spec_tree;
    self.ctx.pipeline_build = VERIFY_PIPELINE_LAYERS;
    const verify_logits = xfm.forwardWith(&self.ctx, verify_input);
    self.ctx.capture_layers = null;
    self.ctx.capture_ssm_seq = false;
    self.ctx.tree = null;
    self.ctx.pipeline_build = 0;
    const logits = try verify_logits;
    defer if (self.ctx.ssm_entries) |entries| {
        for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
    };
    self.dflash_attempted += 1;

    var targets = blk: {
        defer _ = mlx.mlx_array_free(logits);
        break :blk if (serial)
            try self.verifySerialSamplesAt(logits, w, depth[0..w])
        else
            try verifyArgmax(logits, self.sampling.suppress_mask, s);
    };
    defer targets.deinit();
    try mlx.check(mlx.mlx_array_eval(targets.lazy()));
    const ids = try targets.ids(w);
    if (tracing) {
        self.dflash_trace.add(.verify, ph.read());
        ph.reset();
    }

    // Walk the target's tokens down the tree.
    var path_rows: [MAX_W]u32 = undefined;
    var accepted = walkTree(toks[0..w], parents[0..w], ids, &path_rows);
    const raw_accepted = accepted;
    if (flight) |f| {
        var reached: [MAX_W]u32 = undefined;
        for (path_rows[1 .. accepted + 1], 0..) |row, i| reached[i] = @intCast(toks[row]);
        reached[accepted] = @intCast(ids[path_rows[accepted]]);
        f.accept(reached[0 .. accepted + 1]);
    }
    accepted = capAcceptedForTokenBudget(accepted, self.completion_tokens, self.max_tokens);
    const eos_stop = clipAtEos(toks[0..w], path_rows[0 .. accepted + 1], self.eos_token_ids);
    if (eos_stop) |stop| accepted = stop.accepted;
    const next_pending: u32 = if (eos_stop) |stop| stop.token else @intCast(ids[path_rows[accepted]]);
    if (serial) self.sampling.draw = self.generated_ids.items.len + accepted + 2;
    // A copy that keeps nothing four rounds running pauses the copies.
    if (copy.len > 0 and (flight == null or flight.?.retained_native)) {
        self.dflash_copy_misses = if (accepted == 0) self.dflash_copy_misses + 1 else 0;
        if (self.dflash_copy_misses >= 4) {
            self.dflash_copy_misses = 0;
            self.dflash_copy_silent = 16;
        }
    }

    // Commit: the accepted path's KV rows move to consecutive positions,
    // the GDN state is the last kept node's, the conv window its path's.
    const n_commit: usize = 1 + @as(usize, accepted);
    var moved = false;
    for (1..n_commit) |j| {
        if (path_rows[j] != j) moved = true;
    }
    if (moved) try self.ctx.cache.compactRows(anchor_pos, path_rows[1..n_commit], s);
    try self.ctx.cache.truncate(anchor_pos + n_commit, s);
    self.ctx.cache.step = kv_step_snap;
    if (self.ctx.ssm_entries) |entries| {
        var conv_rows: [3]i32 = undefined;
        for (0..3) |i| {
            const dd: i32 = @as(i32, @intCast(accepted)) - 2 + @as(i32, @intCast(i));
            conv_rows[i] = if (dd < 0) dd + 3 else 3 + @as(i32, @intCast(path_rows[@intCast(dd)]));
        }
        var kept: [MAX_W]i32 = undefined;
        for (path_rows[0..n_commit], kept[0..n_commit]) |r, *k| k.* = @intCast(r);
        for (entries) |*entry| try transformer_mod.ssmCommitTreePath(entry, kept[0..n_commit], conv_rows, s);
    }
    self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
    if (accepted + 1 < w) self.partial_rounds += 1;
    if (tracing) {
        self.dflash_trace.add(.accept, ph.read());
        ph.reset();
    }

    // The assistant context grows by the kept rows' captures.
    {
        var idx_buf: [MAX_W]i32 = undefined;
        for (0..n_commit) |j| idx_buf[j] = @intCast(path_rows[j]);
        const idx = mlx.mlx_array_new_data(&idx_buf, &[_]c_int{@intCast(n_commit)}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        const kept = try allocator.alloc(mlx.mlx_array, cap_out.len);
        var kept_n: usize = 0;
        defer {
            for (kept[0..kept_n]) |a| _ = mlx.mlx_array_free(a);
            allocator.free(kept);
        }
        for (cap_out, kept) |full, *out| {
            out.* = mlx.mlx_array_new();
            kept_n += 1;
            try mlx.check(mlx.mlx_take_axis(out, full, idx, 1, s));
        }
        try dflash_mod.appendContext(model, dctx, kept, anchor_pos);
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        dctx.appendEvalArrays(eval_vec);
        if (tracing) {
            try mlx.check(mlx.mlx_eval(eval_vec));
            self.dflash_trace.add(.append, ph.read());
        } else {
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
    }

    const tokens = try allocator.alloc(u32, n_commit);
    errdefer allocator.free(tokens);
    try self.generated_ids.ensureUnusedCapacity(allocator, n_commit);
    for (0..n_commit) |j| tokens[j] = @intCast(toks[path_rows[j]]);
    // No fallible pricing work may follow publication of generated IDs.
    if (flight) |f| try f.finish(self, raw_accepted, @intCast(n_commit));
    self.generated_ids.appendSliceAssumeCapacity(tokens);
    self.dflash_accepted_tokens += accepted;
    self.next_token_id = next_pending;
    self.advanceStep(@intCast(n_commit));
    if (flight) |f| f.recordCommitted(self.external_history, path_rows[0..n_commit]);
    self.checkDflashRuntimeGate();
    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
    }
    if (tracing) {
        self.dflashTraceRoundEnd(accepted);
        self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
    }
    return DrafterStepResult{ .tokens = tokens, .accepted_tokens = accepted };
}

/// Host walk is the single source of rows used by KV compaction, recurrent
/// rollback and assistant capture gathering. A rejected sibling is never kept.
fn walkTree(tokens: []const i32, parents: []const i32, targets: []const i32, path: *[DFLASH_TREE_W]u32) u32 {
    std.debug.assert(tokens.len == parents.len and targets.len >= tokens.len and tokens.len <= DFLASH_TREE_W);
    path[0] = 0;
    var accepted: u32 = 0;
    walk: while (true) {
        const current = path[accepted];
        for (1..tokens.len) |row| {
            if (parents[row] == @as(i32, @intCast(current)) and tokens[row] == targets[current]) {
                accepted += 1;
                path[accepted] = @intCast(row);
                continue :walk;
            }
        }
        return accepted;
    }
}

const EosStop = struct { accepted: u32, token: u32 };
fn clipAtEos(tokens: []const i32, path: []const u32, eos: []const u32) ?EosStop {
    for (path[1..], 1..) |row, depth| {
        const token: u32 = @intCast(tokens[row]);
        for (eos) |stop| if (token == stop) return .{ .accepted = @intCast(depth - 1), .token = token };
    }
    return null;
}

test "DFlash tree walk supplies noncontiguous rollback rows and correction" {
    var path: [DFLASH_TREE_W]u32 = undefined;
    const accepted = walkTree(&.{ 10, 11, 12, 21, 22 }, &.{ -1, 0, 1, 0, 3 }, &.{ 21, 99, 99, 22, 30 }, &path);
    try std.testing.expectEqual(@as(u32, 2), accepted);
    try std.testing.expectEqualSlices(u32, &.{ 0, 3, 4 }, path[0 .. accepted + 1]);
    const clipped = capAcceptedForTokenBudget(accepted, 0, 2);
    try std.testing.expectEqual(@as(u32, 1), clipped);
    try std.testing.expectEqual(@as(u32, 3), path[clipped]);
}

test "DFlash EOS censors the kept path before rollback and publication" {
    const stop = clipAtEos(&.{ 10, 11, 99, 21, 22 }, &.{ 0, 3, 4 }, &.{22}).?;
    try std.testing.expectEqual(@as(u32, 1), stop.accepted);
    try std.testing.expectEqual(@as(u32, 22), stop.token);
    try std.testing.expect(clipAtEos(&.{ 10, 11, 99, 21, 22 }, &.{ 0, 3 }, &.{99}) == null);
}
