//! Request-local lookup prices. Native probes never change head depth or numerics.
const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../../sdk/util/log.zig");
const Watch = @import("../../../../sdk/util/io_util.zig").Stopwatch;
const policy = @import("../speculation_policy.zig");
const adapter = @import("../lookup_adapter.zig");
const settings = @import("../economics_settings.zig");
const history = @import("../history.zig");
const lookup = @import("../mtp_lookup.zig");
const Generator = @import("../../generate/generator.zig").Generator;
const Chain = @import("chain.zig").MtpPreDraft;
const Plan = @import("ev_plan.zig").MtpRoundPlan;
const max_native_plans = @import("../../../../sdk/spec/mtp/mtp.zig").MAX_DEPTH;

// A single chunk never consults tau. An extending plan does: even a tiny tau
// change can change both its executed work and yield, so do not bucket thresholds.
fn sameNativePlan(a: Plan, b: Plan) bool {
    return a.m_lo == b.m_lo and a.m_hi == b.m_hi and
        (a.m_hi == a.m_lo or a.tau_ln == b.tau_ln);
}

// Keep old comparators and exhausted budgets when the planner revisits a plan.
// Use the native depth bound as the per-context plan exploration budget (not an
// enumeration of every depth/threshold combination). Exact-tau jitter can consume
// at most three probes per slot; unseen plans then conservatively remain native.
const NativeProbes = struct {
    const Cell = struct {
        plan: Plan,
        attempts: u32 = 0,
        warmed_widths: u2 = 0,
        sample: ?policy.RemainingRound = null,

        fn attemptLimit(self: *const Cell) u32 {
            // Eligible solo execution is either chunk A or A+B. Three attempts
            // repeat at least one width, whose first kernel observation is cold.
            return if (self.plan.m_hi > self.plan.m_lo) 3 else 2;
        }

        pub fn observe(self: *Cell, remaining_ms: f32, tokens: u32, rows: u32) void {
            if (self.sample != null) return;
            if (!std.math.isFinite(remaining_ms) or remaining_ms <= 0 or tokens == 0) return;
            const bit: u2 = if (rows == self.plan.m_lo + 1) 1 else if (rows == self.plan.m_hi + 1) 2 else return;
            if (self.warmed_widths & bit == 0) {
                self.warmed_widths |= bit;
                return;
            }
            self.sample = .{ .round_ms = remaining_ms, .expected_tokens = @floatFromInt(tokens) };
        }
    };
    cells: [max_native_plans]Cell = undefined,
    len: usize = 0,

    fn find(self: *const NativeProbes, plan: Plan) ?usize {
        for (self.cells[0..self.len], 0..) |cell, i| {
            if (sameNativePlan(cell.plan, plan)) return i;
        }
        return null;
    }

    fn pending(self: *const NativeProbes, plan: Plan, reference_priced: bool) bool {
        if (plan.width_trial) return false;
        const i = self.find(plan) orelse return self.len < self.cells.len;
        const cell = &self.cells[i];
        return (cell.sample == null or !reference_priced) and cell.attempts < cell.attemptLimit();
    }

    fn reserve(self: *NativeProbes, plan: Plan, reference_priced: bool) ?*Cell {
        if (!self.pending(plan, reference_priced)) return null;
        const i = self.find(plan) orelse blk: {
            const next = self.len;
            self.cells[next] = .{ .plan = plan };
            self.len += 1;
            break :blk next;
        };
        self.cells[i].attempts += 1;
        return &self.cells[i];
    }
};

pub const Runtime = struct {
    cut: policy.Policy,
    pick: std.ArrayList(u32) = .empty,
    selected: policy.Tree = .{},
    query_ns: u64 = 0,
    query_watch: ?Watch = null,
    probes: std.AutoHashMapUnmanaged(i64, NativeProbes) = .empty,
    demand: ?struct { class: i64, plan: Plan } = null,

    fn wantsNative(self: *const Runtime, class: i64, plan: Plan) bool {
        const demand = self.demand orelse return false;
        return demand.class == class and sameNativePlan(demand.plan, plan);
    }

    fn referencePriced(self: *const Runtime, class: i64, plan: Plan) bool {
        return self.cut.prices.known(plan.m_lo + 1, .chain, class, .generic) != null or
            self.cut.prices.known(plan.m_hi + 1, .chain, class, .generic) != null;
    }

    fn traceProbes(self: *const Runtime, class: i64, plan: Plan) void {
        if (!settings.tracing()) return;
        var attempts: u32 = 0;
        var measured: u32 = 0;
        const probes = self.probes.getPtr(class);
        if (probes) |p| for (p.cells[0..p.len]) |cell| {
            attempts += cell.attempts;
            measured += @intFromBool(cell.sample != null);
        };
        const reference_priced = self.referencePriced(class, plan);
        const pending = !plan.width_trial and self.wantsNative(class, plan) and
            (if (probes) |p| p.pending(plan, reference_priced) else true);
        const exhausted = !plan.width_trial and (self.nativeFor(class, plan) == null or !reference_priced) and
            (if (probes) |p| !p.pending(plan, reference_priced) else false);
        log.info("[lookup-mtp-probes] class={d} native_lo={d} native_hi={d} tau_ln={d:.6} plans={d} attempts={d} measurements={d} reference_priced={any} pending={any} exhausted={any}\n", .{
            class, plan.m_lo, plan.m_hi, plan.tau_ln, if (probes) |p| p.len else 0, attempts, measured, reference_priced, pending, exhausted,
        });
    }

    pub fn nativeFor(self: *const Runtime, class: i64, plan: Plan) ?policy.RemainingRound {
        if (plan.width_trial) return null;
        const probes = self.probes.getPtr(class) orelse return null;
        const i = probes.find(plan) orelse return null;
        return probes.cells[i].sample;
    }

    pub fn reserveNative(self: *Runtime, class: i64, plan: Plan) !?*NativeProbes.Cell {
        if (plan.width_trial) return null;
        const entry = try self.probes.getOrPut(self.cut.prices.allocator, class);
        if (!entry.found_existing) entry.value_ptr.* = .{};
        return entry.value_ptr.reserve(plan, self.referencePriced(class, plan));
    }

    pub fn deinit(self: *Runtime) void {
        self.pick.deinit(self.cut.prices.allocator);
        self.probes.deinit(self.cut.prices.allocator);
        self.cut.deinit();
    }
};

/// Owned by a chain, never by reusable query scratch. No allocation when disabled.
pub const Flight = struct {
    tree: policy.Tree,
    class: i64,
    rows: u32,
    anchor_position: usize = 0,
    query_ns: ?u64 = null,
    preparation_ns: u64 = 0,
    scope: policy.LookupPreparationScope = .cold,
    native: bool = false,
    solo: bool,
    remaining: ?Watch = null,
    target: ?Watch = null,
    handoff: ?Watch = null,
    head_settle_ns: u64 = 0,
};

pub fn enabled() bool {
    return settings.selection().mode != .native and history.enabled();
}

fn arm(gen: *Generator, allocator: std.mem.Allocator) !*Runtime {
    if (gen.mtp_economics) |rt| return rt;
    const rt = try allocator.create(Runtime);
    rt.* = .{ .cut = policy.Policy.init(if (settings.pricesPath()) |path|
        policy.PriceModel.load(gen.timer.io, allocator, path)
    else
        policy.PriceModel.init(allocator)) };
    gen.mtp_economics = rt;
    return rt;
}

fn solo(gen: *const Generator) bool {
    return gen.spec_cost_solo and !gen.mtp_batch_head and gen.mtp_group_cap == 0 and !gen.mtp_planner_owned;
}

fn supported(gen: *const Generator) bool {
    const head = gen.mtp orelse return false;
    const cache = gen.mtp_cache orelse return false;
    return (head == .plugin and cache == .plugin) or (head == .qwen and cache == .qwen);
}

fn ms(ns: u64) f32 {
    return @as(f32, @floatFromInt(ns)) / std.time.ns_per_ms;
}

/// Keep the stock suffix, token-budget and operator depth gates. Only width economics changes.
pub fn select(gen: *Generator, allocator: std.mem.Allocator, got: lookup.Match, cap: u32, floor: u32, t1: u32, watch: Watch, plan: Plan) !u32 {
    if (got.suffix < lookup.MIN_SUFFIX or cap < floor or cap == 0) return 0;
    const rt = try arm(gen, allocator);
    const request = gen.external_history.?;
    const suffix_cap: u32 = if (got.suffix >= lookup.STRONG_SUFFIX) lookup.MAX_DRAFT_STRONG else lookup.MAX_DRAFT;
    const offer = adapter.lookupChain(t1, got.draft, request.drafter.vote_nodes[0..request.drafter.node_count], got.suffix, &rt.cut.cal, @min(cap, suffix_cap) + 1);
    const class = adapter.contextClass(gen.mtpKvLen());
    const scope: policy.LookupPreparationScope = if (gen.mtp_economics_carried) .carried else .cold;
    // No native draft has been prepared at this query, even in an outgoing carry.
    const native = rt.nativeFor(class, plan);
    const fallback: policy.LookFallback = if (native) |cost| .{ .measured = cost } else .unknown;
    rt.query_ns = watch.read();
    rt.query_watch = watch;
    const control = settings.selection();
    // A historical candidate, not ordinary native decoding or a new context,
    // supplies demand for calibration. Preserve it across the bounded warmup.
    if ((control.mode == .adaptive or control.mode == .cut) and offer.len >= floor + 1)
        rt.demand = .{ .class = class, .plan = plan };
    rt.traceProbes(class, plan);
    const trial = if (control.mode == .adaptive and solo(gen) and native != null)
        rt.cut.reserveLookWidthTrial(&offer, class, floor + 1, ms(rt.query_ns), native, null, scope)
    else
        null;
    const choice = try rt.cut.chooseLookPreparedControlled(&offer, class, ms(rt.query_ns), &rt.pick, trial orelse control.rows, floor + 1, fallback, scope);
    const count = choice.rows;
    rt.selected = offer;
    rt.selected.len = 1 + count;
    if (settings.tracing()) log.info("[lookup-mtp-select] anchor_pos={d} anchor={d} class={d} mode={s} scope={s} native_lo={d} native_hi={d} offered_rows={d} selected_rows={d} native_measured={any} query_ms={d:.6} trial={any}\n", .{ gen.mtpKvLen(), t1, class, @tagName(control.mode), @tagName(scope), plan.m_lo, plan.m_hi, offer.len, if (count > 0) count + 1 else 0, native != null, ms(rt.query_ns), trial != null });
    return count;
}

pub fn probePending(gen: *const Generator, plan: Plan) bool {
    const mode = settings.selection().mode;
    if (plan.width_trial or !enabled() or (mode != .adaptive and mode != .cut) or !solo(gen) or !supported(gen) or
        gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_planner_pending or gen.sampling.temperature > 0.01) return false;
    const rt = gen.mtp_economics orelse return false;
    const class = adapter.contextClass(gen.mtpKvLen());
    if (!rt.wantsNative(class, plan)) return false;
    const probes = rt.probes.getPtr(class) orelse return true;
    return probes.pending(plan, rt.referencePriced(class, plan));
}

pub fn initChain(gen: *Generator, allocator: std.mem.Allocator, plan: Plan, t1: u32) !Chain {
    if (!probePending(gen, plan)) return gen.mtpChainInit(allocator, plan, t1);
    const rt = gen.mtp_economics.?;
    const class = adapter.contextClass(gen.mtpKvLen());
    const flight = try allocator.create(Flight);
    errdefer allocator.destroy(flight);
    try settle(gen);
    _ = (try rt.reserveNative(class, plan)).?;
    flight.* = .{ .tree = policy.Tree.init(t1), .anchor_position = gen.mtpKvLen(), .class = class, .rows = 0, .native = true, .solo = solo(gen), .remaining = Watch.init(gen.timer.io) };
    var chain = try gen.mtpChainInit(allocator, plan, t1);
    chain.economics = flight;
    return chain;
}

pub fn attachLookup(gen: *Generator, allocator: std.mem.Allocator, chain: *Chain) !void {
    if (!enabled() or !chain.external_history) return;
    const rt = gen.mtp_economics orelse return;
    const flight = try allocator.create(Flight);
    errdefer allocator.destroy(flight);
    // Separate existing head/stash debt from the target price, without forwarding it twice.
    try settle(gen);
    flight.* = .{
        .tree = rt.selected,
        .anchor_position = gen.mtpKvLen(),
        .class = adapter.contextClass(gen.mtpKvLen()),
        .rows = chain.m + 1,
        .query_ns = rt.query_ns,
        .preparation_ns = if (rt.query_watch) |watch| watch.read() -| rt.query_ns else 0,
        .scope = if (gen.mtp_economics_carried) .carried else .cold,
        .solo = solo(gen),
        .handoff = Watch.init(gen.timer.io),
    };
    chain.economics = flight;
}

/// Evaluate the actual served head state, including plugin recurrent state.
pub fn beginTarget(gen: *Generator, chain: *Chain, resume_watch: Watch) !void {
    const flight = chain.economics orelse return;
    if (flight.native) {
        const watch = Watch.init(gen.timer.io);
        const arrays = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(arrays);
        _ = mlx.mlx_vector_array_append_value(arrays, chain.t1_arr);
        for (chain.draft_arrs[0..chain.m]) |a| _ = mlx.mlx_vector_array_append_value(arrays, a);
        if (chain.conf_arrs) |slots| for (slots[0..chain.n_conf]) |a| { _ = mlx.mlx_vector_array_append_value(arrays, a); };
        if (chain.q_probs) |slots| for (slots[0..chain.n_qp]) |a| { _ = mlx.mlx_vector_array_append_value(arrays, a); };
        if (chain.h_chain) |a| _ = mlx.mlx_vector_array_append_value(arrays, a);
        if (chain.place) |place| _ = mlx.mlx_vector_array_append_value(arrays, place.live_end);
        if (gen.mtp_cache) |*cache| cache.appendEvalArrays(arrays);
        try mlx.check(mlx.mlx_eval(arrays));
        flight.head_settle_ns = watch.read();
        flight.rows = chain.m + 1;
    }
    if (!flight.native) {
        flight.preparation_ns += if (flight.scope == .carried) resume_watch.read() else if (flight.handoff) |watch| watch.read() else 0;
    }
    flight.target = Watch.init(gen.timer.io);
}

pub fn settle(gen: *Generator) !void {
    const arrays = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(arrays);
    gen.ctx.cache.appendEvalArrays(arrays);
    if (gen.mtp_cache) |*cache| cache.appendEvalArrays(arrays);
    if (gen.ctx.ssm_entries) |entries| for (entries) |entry| {
        if (!entry.initialized) continue;
        if (entry.conv_state.ctx != null) _ = mlx.mlx_vector_array_append_value(arrays, entry.conv_state);
        if (entry.ssm_state.ctx != null) _ = mlx.mlx_vector_array_append_value(arrays, entry.ssm_state);
    };
    if (gen.has_last_hidden) _ = mlx.mlx_vector_array_append_value(arrays, gen.last_hidden);
    if (gen.mtp_hist_stash) |stash| {
        _ = mlx.mlx_vector_array_append_value(arrays, stash.ids);
        _ = mlx.mlx_vector_array_append_value(arrays, stash.hidden);
    }
    if (mlx.mlx_vector_array_size(arrays) > 0) try mlx.check(mlx.mlx_eval(arrays));
}

pub fn priceEligible(flight: *const Flight, is_solo: bool, rows: u32, complete: bool) bool {
    return flight.solo and is_solo and complete and flight.rows == rows and flight.target != null;
}

/// Before successor preparation: counts are actually emitted, never merely proposed.
pub fn finish(gen: *Generator, chain: *Chain, accepted: u32, complete: bool, verify_len: u32) !void {
    const flight = chain.economics orelse return;
    const rt = gen.mtp_economics.?;
    if (!flight.native) rt.cut.observeChainAccepted(&flight.tree, if (complete) chain.m else accepted, accepted);
    try settle(gen);
    const target_ms = if (flight.target) |watch| ms(watch.read()) else 0;
    const eligible = priceEligible(flight, solo(gen), verify_len, complete) and supported(gen) and !chain.plan.width_trial;
    var learned = false;
    var prepare_learned = false;
    const remaining_ms = if (flight.remaining) |watch| ms(watch.read()) else 0;
    if (eligible) {
        learned = rt.cut.prices.observeKernel(chain.m + 1, target_ms, .chain, flight.class, .generic);
        if (flight.native) {
            if (rt.probes.getPtr(flight.class)) |probes| {
                if (probes.find(chain.plan)) |i| probes.cells[i].observe(remaining_ms, accepted + 1, chain.m + 1);
            }
        } else {
            prepare_learned = rt.cut.observeMtpPreparation(chain.m + 1, flight.class, flight.scope, ms(flight.preparation_ns));
        }
    }
    if (flight.native) rt.traceProbes(flight.class, chain.plan);
    if (settings.tracing()) log.info("[lookup-mtp-cost] anchor_pos={d} anchor={d} class={d} native={any} scope={s} variant=generic selected_rows={d} executed_rows={d} committed={d} complete={any} eligible={any} target_verify_commit_ms={d:.6} query_ms={?d} prepare_ms={d:.6} head_settle_ms={d:.6} native_remaining_ms={d:.6} target_learned={any} preparation_learned={any}\n", .{
        flight.anchor_position, flight.tree.tokens[0], flight.class, flight.native, @tagName(flight.scope), flight.rows, verify_len, accepted + 1, complete, eligible, target_ms, if (flight.query_ns) |ns| ms(ns) else @as(?f32, null), ms(flight.preparation_ns), ms(flight.head_settle_ns), remaining_ms, learned, prepare_learned,
    });
}

test "MTP economics: grouped clipped and incomplete samples cannot price" {
    var flight: Flight = .{ .tree = policy.Tree.init(1), .class = 0, .rows = 4, .solo = true, .target = Watch.init(std.testing.io) };
    try std.testing.expect(priceEligible(&flight, true, 4, true));
    try std.testing.expect(!priceEligible(&flight, false, 4, true));
    try std.testing.expect(!priceEligible(&flight, true, 3, true));
    try std.testing.expect(!priceEligible(&flight, true, 4, false));
    flight.solo = false;
    try std.testing.expect(!priceEligible(&flight, true, 4, true));
}

test "MTP economics: comparator keys retain extension thresholds and isolate plan warmups" {
    var rt: Runtime = .{ .cut = policy.Policy.init(policy.PriceModel.init(std.testing.allocator)) };
    defer rt.deinit();
    const first: Plan = .{ .m_lo = 2, .m_hi = 4, .tau_ln = -0.5 };
    var changed = first;
    changed.tau_ln = -0.5001;
    rt.cut.prices.observe(3, 1, .chain, 0, .generic);
    (try rt.reserveNative(0, first)).?.observe(100, 2, 3);
    try std.testing.expect(rt.nativeFor(0, first) == null);
    (try rt.reserveNative(0, first)).?.observe(10, 3, 3);
    try std.testing.expectEqual(@as(f32, 10), rt.nativeFor(0, first).?.round_ms);
    try std.testing.expect(rt.nativeFor(0, changed) == null);
    (try rt.reserveNative(0, changed)).?.observe(200, 1, 3);
    try std.testing.expect(rt.nativeFor(0, changed) == null);
    (try rt.reserveNative(0, changed)).?.observe(20, 2, 3);
    try std.testing.expectEqual(@as(f32, 20), rt.nativeFor(0, changed).?.round_ms);
    try std.testing.expectEqual(@as(f32, 2), rt.nativeFor(0, changed).?.expected_tokens);
    try std.testing.expectEqual(@as(f32, 10), rt.nativeFor(0, first).?.round_ms);
    try std.testing.expect((try rt.reserveNative(0, first)) == null);
    try std.testing.expect(rt.nativeFor(2048, first) == null);
    const flat: Plan = .{ .m_lo = 3, .m_hi = 3, .tau_ln = -0.1 };
    var flat_jitter = flat;
    flat_jitter.tau_ln = -0.2;
    (try rt.reserveNative(0, flat)).?.observe(30, 2, 4);
    (try rt.reserveNative(0, flat_jitter)).?.observe(15, 4, 4);
    try std.testing.expectEqual(@as(f32, 15), rt.nativeFor(0, flat).?.round_ms);
    try std.testing.expectEqual(@as(f32, 15), rt.nativeFor(0, flat_jitter).?.round_ms);
}

test "MTP economics: width trials spend nothing and threshold jitter has a finite budget" {
    var probes: NativeProbes = .{};
    var plan: Plan = .{ .m_lo = 2, .m_hi = 4, .tau_ln = -0.5, .width_trial = true };
    for (0..2) |_| {
        try std.testing.expect(!probes.pending(plan, false));
        try std.testing.expect(probes.reserve(plan, false) == null);
    }
    try std.testing.expectEqual(@as(usize, 0), probes.len);
    plan.width_trial = false;
    var attempts: u32 = 0;
    for (0..100) |n| {
        plan.tau_ln = -0.5 - @as(f32, @floatFromInt(n)) / 10000;
        for (0..3) |_| {
            if (probes.reserve(plan, false)) |cell| {
                attempts += 1;
                cell.observe(10, 3, 3);
            }
        }
    }
    try std.testing.expectEqual(@as(u32, 3 * max_native_plans), attempts);
    try std.testing.expectEqual(probes.cells.len, probes.len);
    plan.tau_ln = -0.5;
    try std.testing.expect(!probes.pending(plan, false));
    try std.testing.expectEqual(@as(f32, 10), probes.cells[probes.find(plan).?].sample.?.round_ms);
    // Incomplete/invalid rounds cannot replenish a consumed attempt.
    var incomplete: NativeProbes = .{};
    incomplete.reserve(plan, false).?.observe(0, 0, 3);
    incomplete.reserve(plan, false).?.observe(std.math.nan(f32), 1, 3);
    incomplete.reserve(plan, false).?.observe(0, 0, 3);
    try std.testing.expect(!incomplete.pending(plan, false));
    try std.testing.expect(incomplete.cells[0].sample == null);
}

test "MTP economics: native plan and context changes alone do not request probes" {
    var rt: Runtime = .{ .cut = policy.Policy.init(policy.PriceModel.init(std.testing.allocator)) };
    defer rt.deinit();
    const plan: Plan = .{ .m_lo = 3, .m_hi = 3, .tau_ln = 0 };
    try std.testing.expect(!rt.wantsNative(0, plan));
    rt.demand = .{ .class = 0, .plan = plan };
    try std.testing.expect(rt.wantsNative(0, plan));
    try std.testing.expect(!rt.wantsNative(2048, plan));
    var other = plan;
    other.m_hi += 1;
    try std.testing.expect(!rt.wantsNative(0, other));
    try std.testing.expectEqual(@as(usize, 0), rt.probes.count());
}

test "MTP economics: extension bootstrap needs a same-context executable verifier reference" {
    var rt: Runtime = .{ .cut = policy.Policy.init(policy.PriceModel.init(std.testing.allocator)) };
    defer rt.deinit();
    const plan: Plan = .{ .m_lo = 2, .m_hi = 4, .tau_ln = -0.5 };
    // Neither another shape nor another context can retire this calibration.
    rt.cut.prices.observe(2, 1, .chain, 0, .generic);
    rt.cut.prices.observe(3, 1, .chain, 2048, .generic);
    for ([_]u32{ 5, 3 }, 0..) |rows, i| {
        const cell = (try rt.reserveNative(0, plan)).?;
        try std.testing.expect(!rt.cut.prices.observeKernel(rows, 10, .chain, 0, .generic));
        cell.observe(20, @intCast(i + 1), rows);
    }
    try std.testing.expect(rt.nativeFor(0, plan) == null);
    try std.testing.expect(!rt.referencePriced(0, plan));
    try std.testing.expect(rt.probes.getPtr(0).?.pending(plan, rt.referencePriced(0, plan)));
    const third = (try rt.reserveNative(0, plan)).?;
    try std.testing.expect(rt.cut.prices.observeKernel(5, 10, .chain, 0, .generic));
    third.observe(99, 1, 5);
    const measured = rt.nativeFor(0, plan).?;
    try std.testing.expectEqual(@as(f32, 99), measured.round_ms);
    try std.testing.expect(rt.referencePriced(0, plan));
    try std.testing.expect((try rt.reserveNative(0, plan)) == null);
    var offer = policy.Tree.chain(0, &.{ 1, 2, 3, 4 }, 1, 1);
    offer.scores_calibrated = true;
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const choice = try rt.cut.chooseLookPreparedControlled(&offer, 0, 0, &out, 0, 3,
        .{ .measured = measured }, .cold);
    try std.testing.expectEqual(@as(u32, 4), choice.rows);
}

test "MTP economics: invalid third width probe exhausts without publishing a cold comparator" {
    var rt: Runtime = .{ .cut = policy.Policy.init(policy.PriceModel.init(std.testing.allocator)) };
    defer rt.deinit();
    const plan: Plan = .{ .m_lo = 2, .m_hi = 4, .tau_ln = -0.5 };
    for ([_]u32{ 3, 5 }) |rows| {
        const cell = (try rt.reserveNative(0, plan)).?;
        try std.testing.expect(!rt.cut.prices.observeKernel(rows, 10, .chain, 0, .generic));
        cell.observe(20, 2, rows);
    }
    try std.testing.expect(rt.nativeFor(0, plan) == null);
    _ = (try rt.reserveNative(0, plan)).?;
    // An incomplete/contended third execution contributes no observations.
    try std.testing.expect((try rt.reserveNative(0, plan)) == null);
    try std.testing.expect(!rt.referencePriced(0, plan));
    try std.testing.expect(rt.nativeFor(0, plan) == null);
}
