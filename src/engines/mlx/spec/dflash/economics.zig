//! Opt-in DFlash2 lookup economics. Native DFlash/tree/copy are upstream;
//! this module owns independent request-local three-tier selection and evidence.
//! Prices are operator-scoped schema-3 priors, not model/hardware attestation.
const std = @import("std");
const policy = @import("../speculation_policy.zig");
const adapter = @import("../lookup_adapter.zig");
const history = @import("../history.zig");
const settings = @import("../economics_settings.zig");
const io_util = @import("../../../../sdk/util/io_util.zig");
const log = @import("../../../../sdk/util/log.zig");
const mlx = @import("../../../../mlx/ffi.zig");
const Generator = @import("../../generate/generator.zig").Generator;

pub const Tree = policy.Tree;
pub const width = 16;
fn treeDispatch(gen: *const Generator) policy.TreeDispatch {
    if (gen.xfm.config.plugin) |binding| return if (binding.parsed.caps.spec_tree_linear_gdn) .linear_gdn else .generic;
    return if (gen.xfm.moe_layers != null) .linear_gdn else .generic;
}

pub fn enabled() bool {
    return settings.selection().mode != .native;
}

pub fn depthLimit(remaining: u32) u32 {
    return @min(width - 1, remaining -| 1);
}

fn contextIndex(class: i64) usize {
    if (class == 0) return 0;
    return @intCast(std.math.log2_int(u64, @intCast(class)) - 10);
}

pub const Runtime = struct {
    controller: policy.Policy,
    // Two actual executions per context/width: cold observation plus settled
    // price. Neither contention nor unexecuted proposals consumes this budget.
    probes: [2][9][width]u8 = @splat(@splat(@splat(0))),
    native_probes: [9]u8 = @splat(0),
    assistant_ms: [9]?f32 = @splat(null),
    rounds: u64 = 0,

    pub fn deinit(self: *Runtime) void {
        self.controller.deinit();
    }
};

/// Value-owned geometry survives history scratch reuse and every error path.
/// It owns no heap storage; abandoning a flight cannot publish proposals.
pub const Flight = struct {
    runtime: *Runtime,
    offered: Tree,
    chosen: Tree,
    native: Tree,
    class: i64,
    dispatch: policy.TreeDispatch = .generic,
    query_ms: f32,
    watch: io_util.Stopwatch,
    remaining_watch: ?io_util.Stopwatch = null,
    verify_watch: ?io_util.Stopwatch = null,
    routed: bool = false,
    copy: bool = false,
    assistant: bool = false,
    retained_native: bool = false,
    probe: bool = false,
    preparation_ms: f32 = 0,
    preparation_estimated: bool = false,
    preparation_known: bool = true,
    solo: bool = false,
    upper_trial: bool = false,
    proposed_rows: usize = 1,

    pub fn route(self: *Flight) void {
        self.query_ms = milliseconds(self.watch.read());
        self.proposed_rows = self.offered.len;
        if (settings.selection().mode == .fixed and settings.selection().rows == 1) {
            self.routed = true;
            return;
        }
        if (settings.selection().mode != .adaptive) return;
        var execution = self.offered;
        const decision = adapter.route(&self.runtime.controller, &execution, self.class, self.query_ms, width, self.dispatch, null, null);
        if (decision.decision.skip) {
            self.chosen = execution;
            self.routed = true;
        }
    }

    pub fn beginAssistant(self: *Flight, gen: *Generator) !void {
        self.assistant = true;
        const index = contextIndex(self.class);
        // Query/backed-copy scanning is CPU-only and already paid. Keep it
        // separate from drained assistant preparation; never price prior append.
        self.query_ms = milliseconds(self.watch.read());
        if (gen.spec_cost_solo and self.runtime.native_probes[index] < 2) {
            gen.dflash_unobserved = true;
            try mlx.check(mlx.mlx_synchronize(gen.xfm.s));
            self.remaining_watch = io_util.Stopwatch.init(gen.timer.io);
            self.runtime.native_probes[index] += 1;
        }
    }

    pub fn select(self: *Flight, native: *const Tree, neural: *const Tree, is_copy: bool) void {
        self.selectControlled(native, neural, is_copy, settings.selection());
    }

    fn selectControlled(self: *Flight, native: *const Tree, neural: *const Tree, is_copy: bool, selection: settings.Selection) void {
        self.native = native.*;
        self.copy = is_copy;
        if (is_copy) {
            // Backing is evidence, not a promise that every copy token survives.
            // Learn copy yield from reached decisions without changing its geometry.
            for (1..self.native.len) |row| {
                const q = self.runtime.controller.cal.q(if (row == 1) .look else .look_continuation, 1, self.native.look_order[row]);
                self.native.conditional_reach[row] = q;
                self.native.scores[row] = self.native.scores[@intCast(self.native.parents[row])] * q;
            }
            self.native.scores_calibrated = true;
        }
        const index = contextIndex(self.class);
        self.preparation_ms = if (self.assistant) self.query_ms else milliseconds(self.watch.read());
        if (self.remaining_watch) |watch| {
            const measured = milliseconds(watch.read());
            self.preparation_ms += measured;
            if (self.runtime.native_probes[index] >= 2) self.runtime.assistant_ms[index] = measured;
        } else if (self.assistant) {
            // No per-round barrier: reuse a settled request-local preparation
            // sample, explicitly labeled estimated in the receipt.
            self.preparation_ms += self.runtime.assistant_ms[index] orelse 0;
            self.preparation_estimated = true;
            self.preparation_known = self.runtime.assistant_ms[index] != null;
        }
        if (!self.routed) {
            var candidate = if (is_copy) self.native else neural.*;
            adapter.merge(&candidate, &self.offered);
            self.proposed_rows = candidate.len;
            const dynamic = selection.mode == .cut or selection.mode == .adaptive;
            // Two real native samples establish the assistant baseline before
            // a mixed upper-width trial can consume its second observation.
            const sample_native = dynamic and self.assistant and (self.remaining_watch != null or !self.preparation_known);
            self.chosen = if (sample_native) self.native else choose(&self.runtime.controller, &candidate, &self.native, self.class, self.preparation_ms, selection, self.dispatch);
            if (self.solo and !is_copy and dynamic and !sample_native) {
                const maximum = self.runtime.controller.chooseTreeIncumbentControlled(&candidate, native, self.class, self.preparation_ms, width, width, null, self.dispatch);
                const native_shape: policy.Shape = .{ .rows = @intCast(native.len), .variant = native.variant(self.dispatch) };
                const maximum_shape: policy.Shape = .{ .rows = @intCast(maximum.len), .variant = maximum.variant(self.dispatch) };
                if (self.runtime.controller.prices.upperTrialWidth(native_shape, maximum_shape, .tree, self.class)) |rows| {
                    const trial = self.runtime.controller.chooseTreeIncumbentControlled(&candidate, native, self.class, self.preparation_ms, width, rows, null, self.dispatch);
                    var trial_yield: f32 = 0;
                    var native_yield: f32 = 0;
                    for (trial.scores[0..trial.len]) |reach| trial_yield += reach;
                    for (native.scores[0..native.len]) |reach| native_yield += reach;
                    if (trial.len == rows and hasSelectedLookup(&trial) and trial_yield > native_yield and
                        self.runtime.controller.prices.reserveUpperTrial(native_shape, .{ .rows = rows, .variant = trial.variant(self.dispatch) }, .tree, self.class))
                    {
                        self.chosen = trial;
                        self.upper_trial = true;
                    }
                }
            }
        }
        self.retained_native = sameGeometry(&self.chosen, native);
        if (self.retained_native) retainLookupReceipts(&self.chosen, &self.offered);
    }

    pub fn beginVerify(self: *Flight, gen: *Generator) !void {
        const index = contextIndex(self.class);
        self.probe = gen.spec_cost_solo and self.runtime.probes[@intFromEnum(self.chosen.variant(self.dispatch))][index][self.chosen.len - 1] < 2;
        if (self.probe) {
            gen.dflash_unobserved = true;
            // Boundaries settle real work, not lazy graph-construction duration.
            // This runs only for bounded calibration, never every native round.
            try mlx.check(mlx.mlx_synchronize(gen.xfm.s));
            self.verify_watch = io_util.Stopwatch.init(gen.timer.io);
        }
    }

    /// Observe only reached target decisions. Never label the unexecuted offer
    /// from a different topology/row count as accepted or rejected.
    pub fn accept(self: *Flight, path: []const u32) void {
        self.runtime.controller.observeLookupTree(&self.chosen, path, null);
    }

    /// Attribute only published rows; raw acceptance can exceed this path at
    /// EOS or the token budget. A corroborated native edge is still lookup evidence.
    pub fn recordCommitted(self: *const Flight, request: ?*history.Request, path_rows: []const u32) void {
        const counters = request orelse return;
        for (self.chosen.origins[1..self.chosen.len]) |origin| if (origin & 6 != 0) {
            counters.rounds += 1;
            break;
        };
        for (path_rows[1..]) |row| if (self.chosen.origins[row] & 6 != 0) {
            counters.accepted += 1;
        };
    }

    pub fn finish(self: *Flight, gen: *Generator, raw_accepted: u32, committed: u32) !void {
        var verify_ms: ?f32 = null;
        var learned = false;
        const index = contextIndex(self.class);
        if (self.probe or self.remaining_watch != null) {
            try mlx.check(mlx.mlx_synchronize(gen.xfm.s));
            if (self.verify_watch) |watch| verify_ms = milliseconds(watch.read());
            if (gen.spec_cost_solo) {
                if (verify_ms) |ms| {
                    learned = self.runtime.controller.prices.observeKernel(@intCast(self.chosen.len), ms, .tree, self.class, self.chosen.variant(self.dispatch));
                    self.runtime.probes[@intFromEnum(self.chosen.variant(self.dispatch))][index][self.chosen.len - 1] += 1;
                    if (!self.assistant and self.chosen.len == 1)
                        _ = self.runtime.controller.observePlainProbe(ms, self.class, 1);
                }
                if (self.remaining_watch) |watch| {
                    // A mixed winner cannot teach the native assistant baseline.
                    if (self.assistant and self.retained_native and !self.routed) {
                        _ = self.runtime.controller.observeSpecRemainingProbe(milliseconds(watch.read()), committed, self.class, 1);
                    }
                }
            }
        }
        self.runtime.rounds += 1;
        if (settings.tracing()) log.info("[dflash-economics] class={d} mode={s} action={s} proposed_rows={d} native_rows={d} rows={d} variant={s} native_geometry={any} upper_trial={any} accepted_drafts={d} committed_tokens={d} preparation_ms={any} preparation_estimated={any} verify_ms={any} price_learned={any} solo={any}\n", .{
            self.class, @tagName(settings.selection().mode), if (self.routed) "lookup" else if (self.copy and self.retained_native) "native_copy" else if (self.retained_native) "native_neural" else "mixed",
            self.proposed_rows, self.native.len, self.chosen.len, @tagName(self.chosen.variant(self.dispatch)), self.retained_native, self.upper_trial,
            raw_accepted, committed, if (self.preparation_known) @as(?f32, self.preparation_ms) else null, self.preparation_estimated, verify_ms, learned, gen.spec_cost_solo,
        });
    }
};

/// Attach evidence by the complete parent/token path, without projecting mass
/// or adding rows. Native admission, ordering and neural scores stay untouched.
fn retainLookupReceipts(executed: *Tree, offered: *const Tree) void {
    if (executed.tokens[0] != offered.tokens[0]) return;
    const absent = policy.MAX_TREE_ROWS;
    var matched: [policy.MAX_TREE_ROWS]usize = @splat(absent);
    matched[0] = 0;
    for (1..executed.len) |row| {
        const parent = matched[@intCast(executed.parents[row])];
        if (parent == absent) continue;
        for (1..offered.len) |other| {
            if (offered.parents[other] != @as(i32, @intCast(parent)) or offered.tokens[other] != executed.tokens[row]) continue;
            matched[row] = other;
            if (offered.sources[other] != .draft) {
                if (executed.sources[row] == .draft or executed.look_confidence[row] <= offered.look_confidence[other])
                    executed.look_order[row] = offered.look_order[other];
                executed.look_confidence[row] = @max(executed.look_confidence[row], offered.look_confidence[other]);
                executed.origins[row] |= offered.origins[other] & 7;
                if (executed.sources[row] == .draft) executed.sources[row] = .both;
            }
            break;
        }
    }
}

pub fn prepare(gen: *Generator, allocator: std.mem.Allocator, anchor_pos: usize) !Flight {
    if (gen.dflash_economics == null) {
        const runtime = try allocator.create(Runtime);
        runtime.* = .{ .controller = policy.Policy.init(if (settings.pricesPath()) |path|
            policy.PriceModel.load(gen.timer.io, allocator, path)
        else
            policy.PriceModel.init(allocator)) };
        gen.dflash_economics = runtime;
    }
    const runtime = gen.dflash_economics.?;
    const watch = io_util.Stopwatch.init(gen.timer.io);
    const result = try history.queryTree(gen, allocator, depthLimit(gen.max_tokens -| gen.completion_tokens));
    const offered = adapter.lookupTree(gen.next_token_id, result.nodes, result.order, &runtime.controller.cal, depthLimit(gen.max_tokens -| gen.completion_tokens));
    return .{ .runtime = runtime, .offered = offered, .chosen = Tree.init(gen.next_token_id), .native = Tree.init(gen.next_token_id), .class = adapter.contextClass(anchor_pos + 1), .query_ms = milliseconds(watch.read()), .watch = watch, .solo = gen.spec_cost_solo, .dispatch = treeDispatch(gen) };
}

fn milliseconds(ns: u64) f32 {
    return @as(f32, @floatFromInt(ns)) / std.time.ns_per_ms;
}

pub fn sameGeometry(a: *const Tree, b: *const Tree) bool {
    return a.len == b.len and std.mem.eql(u32, a.tokens[0..a.len], b.tokens[0..b.len]) and std.mem.eql(i32, a.parents[0..a.len], b.parents[0..b.len]);
}

fn hasSelectedLookup(tree: *const Tree) bool {
    for (tree.origins[1..tree.len]) |origin| if (origin & 7 != 0 and origin & 8 == 0) return true;
    return false;
}

pub fn choose(controller: *const policy.Policy, candidate: *const Tree, native: *const Tree, class: i64, preparation_ms: f32, selection: settings.Selection, dispatch: policy.TreeDispatch) Tree {
    if (selection.mode == .native) return native.*;
    const requested: u32 = switch (selection.mode) {
        .fixed => selection.rows,
        .max => width,
        else => 0,
    };
    const chosen = controller.chooseTreeBootstrapControlled(candidate, class, preparation_ms, width, requested, native, null, dispatch);
    // No external selected path means the native exact geometry survives. Fixed
    // and maximum are explicit width experiments, so retain their requested cut.
    if (requested == 0 and !hasSelectedLookup(&chosen)) return native.*;
    return chosen;
}

test "DFlash flight owns its offer and only selected paths learn" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var nodes = [_]@import("../token_history.zig").VoteNode{
        .{ .token = 11, .parent = null, .depth = 1, .conditional_confidence = 0.8, .origins = 2 },
        .{ .token = 12, .parent = 0, .depth = 2, .conditional_confidence = 1, .origins = 2 },
    };
    const owned = adapter.lookupTree(10, &nodes, 8, &controller.cal, 15);
    nodes[0].token = 99;
    try std.testing.expectEqual(@as(u32, 11), owned.tokens[1]);
    var runtime = Runtime{ .controller = controller };
    var flight = Flight{ .runtime = &runtime, .offered = owned, .chosen = Tree.init(10), .native = Tree.init(10), .class = 0, .query_ms = 0, .watch = undefined };
    const before = runtime.controller.diagnosticFingerprint();
    flight.accept(&.{11});
    const after = runtime.controller.diagnosticFingerprint();
    try std.testing.expectEqualSlices(u8, &before, &after);
}

test "DFlash depth budget never becomes sibling width" {
    try std.testing.expectEqual(@as(u32, 1), depthLimit(2));
    const nodes = [_]@import("../token_history.zig").VoteNode{
        .{ .token = 11, .parent = null, .depth = 1, .conditional_confidence = 0.5, .origins = 1 },
        .{ .token = 12, .parent = null, .depth = 1, .conditional_confidence = 0.5, .origins = 4 },
        .{ .token = 13, .parent = 0, .depth = 2, .conditional_confidence = 1, .origins = 1 },
    };
    const cal: policy.Calibration = .{};
    const tree = adapter.lookupTree(10, &nodes, 8, &cal, depthLimit(2));
    try std.testing.expectEqual(@as(usize, 3), tree.len);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0 }, tree.parents[0..tree.len]);
}

test "DFlash unpriced mixed challenger retains native copy geometry" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var native = Tree.init(10);
    _ = native.add(0, 11, 0.9, .look, 0.9);
    _ = native.add(1, 12, 0.8, .look, 0.9);
    var mixed = native;
    _ = mixed.add(0, 99, 0.99, .look, 0.99);
    const selected = choose(&controller, &mixed, &native, 0, 0.1, .{ .mode = .adaptive, .rows = 0 }, .generic);
    try std.testing.expect(sameGeometry(&native, &selected));
    try std.testing.expectEqual(policy.Variant.generic, native.variant(.generic));
}

test "DFlash prices distinguish proven linear incumbent from unpriced branched challenger" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var native = Tree.init(10);
    for (1..9) |row| _ = native.add(row - 1, @intCast(10 + row), 0.9, .look, 0.99);
    var mixed = native;
    _ = mixed.add(0, 99, 0.99, .look, 0.99);
    controller.prices.observe(9, 1, .tree, 0, .linear_gdn);
    const selected = choose(&controller, &mixed, &native, 0, 0.1, .{ .mode = .adaptive, .rows = 0 }, .linear_gdn);
    try std.testing.expect(sameGeometry(&native, &selected));
    try std.testing.expectEqual(policy.Variant.linear_gdn, selected.variant(.linear_gdn));
    try std.testing.expectEqual(policy.Variant.generic, mixed.variant(.linear_gdn));
    try std.testing.expect(controller.prices.known(9, .tree, 0, .generic) == null);
}

test "DFlash adaptive empty lookup retains native neural geometry" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var native = Tree.init(10);
    _ = native.add(0, 11, 0.8, .draft, 0.8);
    _ = native.add(0, 12, 0.2, .draft, 0.2);
    controller.prices.observe(1, 0.01, .tree, 0, .generic);
    controller.prices.observe(3, 10, .tree, 0, .generic);
    const selected = choose(&controller, &native, &native, 0, 0, .{ .mode = .adaptive, .rows = 0 }, .generic);
    try std.testing.expect(sameGeometry(&native, &selected));
    const fixed = choose(&controller, &native, &native, 0, 0, .{ .mode = .fixed, .rows = 1 }, .generic);
    try std.testing.expectEqual(@as(usize, 1), fixed.len);
}

test "DFlash Flight retained native execution preserves overlapping lookup learning and committed attribution" {
    const tokens = @import("../token_history.zig");
    const cases = [_]struct {
        targets: []const u32,
        committed_rows: []const u32,
        continuation_samples: f32,
        accepted: u32,
    }{
        .{ .targets = &.{99}, .committed_rows = &.{0}, .continuation_samples = 0, .accepted = 0 },
        .{ .targets = &.{ 11, 98 }, .committed_rows = &.{ 0, 1 }, .continuation_samples = 1, .accepted = 1 },
        .{ .targets = &.{ 11, 12, 13 }, .committed_rows = &.{ 0, 1, 2 }, .continuation_samples = 1, .accepted = 2 },
        // Raw full acceptance, but EOS/budget only publishes the first draft.
        .{ .targets = &.{ 11, 12, 13 }, .committed_rows = &.{ 0, 1 }, .continuation_samples = 1, .accepted = 1 },
        // The same token 12 under another parent has no matching lookup receipt.
        .{ .targets = &.{ 77, 12, 100 }, .committed_rows = &.{ 0, 3, 4 }, .continuation_samples = 0, .accepted = 0 },
    };
    for ([_]bool{ false, true }) |settled| for (cases) |case| {
        var runtime = Runtime{ .controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator)) };
        defer runtime.deinit();
        const native = adapter.neuralTree(10, &.{ 11, 12, 77, 12 }, &.{ -1, 0, -1, 2 }, &.{ 0, 1, 0, 1 }, &.{ 0.8, 0.9, 0.2, 0.8 }, 15);
        if (settled) {
            runtime.assistant_ms[0] = 1;
            runtime.controller.prices.observe(@intCast(native.len), 10, .tree, 0, .generic);
        }
        const nodes = [_]tokens.VoteNode{
            .{ .token = 11, .parent = null, .depth = 1, .conditional_confidence = 0.7, .origins = 2 },
            .{ .token = 12, .parent = 0, .depth = 2, .conditional_confidence = 0.6, .origins = 4 },
            .{ .token = 13, .parent = 1, .depth = 3, .conditional_confidence = 0.01, .origins = 4 },
            .{ .token = 99, .parent = null, .depth = 1, .conditional_confidence = 0.1, .origins = 2 },
            .{ .token = 12, .parent = 3, .depth = 2, .conditional_confidence = 0.9, .origins = 2 },
        };
        const offered = adapter.lookupTree(10, &nodes, 24, &runtime.controller.cal, 15);
        var flight = Flight{
            .runtime = &runtime, .offered = offered, .chosen = Tree.init(10), .native = Tree.init(10),
            .class = 0, .query_ms = 1, .watch = undefined, .assistant = true,
        };
        flight.selectControlled(&native, &native, false, .{ .mode = .adaptive, .rows = 0 });
        try std.testing.expect(flight.retained_native);
        try std.testing.expect(sameGeometry(&native, &flight.chosen));
        try std.testing.expectEqualSlices(f32, native.scores[0..native.len], flight.chosen.scores[0..native.len]);
        try std.testing.expectEqualSlices(f32, native.conditional_reach[0..native.len], flight.chosen.conditional_reach[0..native.len]);
        try std.testing.expectEqualSlices(f32, native.confidence[0..native.len], flight.chosen.confidence[0..native.len]);
        try std.testing.expectEqualSlices(f32, native.draft_confidence[0..native.len], flight.chosen.draft_confidence[0..native.len]);
        try std.testing.expectEqual(policy.NodeSource.both, flight.chosen.sources[1]);
        try std.testing.expectEqual(policy.NodeSource.both, flight.chosen.sources[2]);
        try std.testing.expectEqual(policy.NodeSource.draft, flight.chosen.sources[4]);
        try std.testing.expectEqual(@as(u8, 10), flight.chosen.origins[1]);
        try std.testing.expectEqual(@as(u8, 12), flight.chosen.origins[2]);
        try std.testing.expectEqual(@as(u32, 24), flight.chosen.look_order[2]);
        try std.testing.expectEqual(@as(f32, 0), runtime.controller.cal.samples(.look));
        flight.accept(case.targets);
        try std.testing.expectEqual(@as(f32, 1), runtime.controller.cal.samples(.look));
        try std.testing.expectEqual(case.continuation_samples, runtime.controller.cal.samples(.look_continuation));

        var request = history.Request{ .allocator = std.testing.allocator, .drafter = tokens.Drafter.init(std.testing.allocator) };
        defer request.deinit();
        flight.recordCommitted(&request, case.committed_rows);
        try std.testing.expectEqual(@as(u32, 1), request.rounds);
        try std.testing.expectEqual(case.accepted, request.accepted);
    };
}

test "DFlash local-only receipts learn without claiming external history acceptance" {
    const tokens = @import("../token_history.zig");
    var runtime = Runtime{ .controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator)) };
    defer runtime.deinit();
    const nodes = [_]tokens.VoteNode{
        .{ .token = 11, .parent = null, .depth = 1, .conditional_confidence = 0.8, .origins = 1 },
    };
    const chosen = adapter.lookupTree(10, &nodes, 8, &runtime.controller.cal, 15);
    var flight = Flight{
        .runtime = &runtime, .offered = chosen, .chosen = chosen, .native = Tree.init(10),
        .class = 0, .query_ms = 0, .watch = undefined,
    };
    flight.accept(&.{ 11, 12 });
    try std.testing.expectEqual(@as(f32, 1), runtime.controller.cal.samples(.look));
    var request = history.Request{ .allocator = std.testing.allocator, .drafter = tokens.Drafter.init(std.testing.allocator) };
    defer request.deinit();
    flight.recordCommitted(&request, &.{ 0, 1 });
    try std.testing.expectEqual(@as(u32, 0), request.rounds);
    try std.testing.expectEqual(@as(u32, 0), request.accepted);
}
