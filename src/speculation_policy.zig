//! Independently authored from the host behavioral contract; no source-port compatibility layer.
//! Prices are milliseconds of completed work, not enqueue latency. Stable marginal-reach
//! ordering preserves ancestors; a neural-supported native-width rerank retains its incumbent.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const MAX_TREE_ROWS = 32;
pub const Kind = enum { chain, tree, plain, spec };
pub const NodeSource = enum { draft, look, both };
pub const Variant = enum { generic, linear_gdn };
/// Capability of the tree verifier, not an inference from token-chain geometry.
pub const TreeDispatch = enum {
    generic,
    linear_gdn,

    pub fn variant(self: TreeDispatch, rows: usize, linear: bool) Variant {
        return if (self == .linear_gdn and rows > 8 and linear) .linear_gdn else .generic;
    }
};
pub const Shape = struct { rows: u32, variant: Variant };
pub const LookupPreparationScope = enum { cold, carried };

const mtp_contexts = [_]i64{ 0, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 };
const mtp_max_rows = 15;

fn mtpContextIndex(class: i64) ?usize {
    for (mtp_contexts, 0..) |label, index| if (class == label) return index;
    return null;
}

fn probability(p: f32) f32 {
    return if (std.math.isFinite(p)) std.math.clamp(p, 0, 1) else 0;
}
fn positive(p: f32) bool {
    return std.math.isFinite(p) and p > 0;
}
fn cost(p: f32) f32 {
    return if (positive(p)) std.math.clamp(p, 0.000001, 1.0e20) else 1.0e20;
}
fn preparation(p: f32) f32 {
    return if (std.math.isFinite(p) and p > 0) @min(p, 1.0e20) else 0;
}
fn origin(source: NodeSource) u8 {
    return switch (source) { .draft => 8, .look => 1, .both => 9 };
}

pub const Tree = struct {
    tokens: [MAX_TREE_ROWS]u32 = @splat(0),
    parents: [MAX_TREE_ROWS]i32 = @splat(-1),
    scores: [MAX_TREE_ROWS]f32 = @splat(0),
    /// Projected edge mass, retained even when its parent's marginal reach is zero.
    conditional_reach: [MAX_TREE_ROWS]f32 = @splat(0),
    sources: [MAX_TREE_ROWS]NodeSource = @splat(.draft),
    origins: [MAX_TREE_ROWS]u8 = @splat(0),
    confidence: [MAX_TREE_ROWS]f32 = @splat(0),
    draft_confidence: [MAX_TREE_ROWS]f32 = @splat(0),
    look_confidence: [MAX_TREE_ROWS]f32 = @splat(0),
    /// Matched query suffix length, not speculative depth or vote unanimity.
    look_order: [MAX_TREE_ROWS]u32 = @splat(0),
    len: usize = 0,
    /// True only when scores were projected with the current policy snapshot. This preserves
    /// excluded sibling mass when a host adapter prunes a larger offer to the transport cap.
    scores_calibrated: bool = false,

    pub fn init(anchor: u32) Tree {
        var tree: Tree = .{};
        tree.tokens[0] = anchor;
        tree.scores[0] = 1;
        tree.confidence[0] = 1;
        tree.len = 1;
        return tree;
    }

    pub fn variant(self: *const Tree, dispatch: TreeDispatch) Variant {
        var linear = true;
        for (1..self.len) |row| {
            if (self.parents[row] != @as(i32, @intCast(row - 1))) {
                linear = false;
                break;
            }
        }
        return dispatch.variant(self.len, linear);
    }

    /// Preserve projected mass and evidence while remapping an ancestor-closed subset.
    pub fn subset(self: *const Tree, selected: u32) Tree {
        var result = Tree.init(self.tokens[0]);
        result.scores_calibrated = self.scores_calibrated;
        var remap: [MAX_TREE_ROWS]usize = @splat(0);
        for (1..self.len) |i| {
            if (selected & (@as(u32, 1) << @as(u5, @intCast(i))) == 0) continue;
            const dst = result.len;
            remap[i] = dst;
            result.tokens[dst] = self.tokens[i];
            result.parents[dst] = @intCast(remap[@intCast(self.parents[i])]);
            result.scores[dst] = self.scores[i];
            result.conditional_reach[dst] = self.conditional_reach[i];
            result.sources[dst] = self.sources[i];
            result.origins[dst] = self.origins[i];
            result.confidence[dst] = self.confidence[i];
            result.draft_confidence[dst] = self.draft_confidence[i];
            result.look_confidence[dst] = self.look_confidence[i];
            result.look_order[dst] = self.look_order[i];
            result.len += 1;
        }
        return result;
    }

    /// Normalize siblings while retaining each descendant's previous conditional reach.
    fn normalize(self: *Tree) void {
        const before = self.scores;
        self.scores[0] = 1;
        for (0..self.len) |parent| {
            var sum: f32 = 0;
            for (parent + 1..self.len) |child| {
                if (self.parents[child] != @as(i32, @intCast(parent))) continue;
                sum += if (before[parent] > 0) probability(before[child] / before[parent]) else self.conditional_reach[child];
            }
            for (parent + 1..self.len) |child| {
                if (self.parents[child] != @as(i32, @intCast(parent))) continue;
                const conditional = if (before[parent] > 0) probability(before[child] / before[parent]) else self.conditional_reach[child];
                self.conditional_reach[child] = conditional / @max(1, sum);
                self.scores[child] = self.scores[parent] * self.conditional_reach[child];
            }
        }
    }

    pub fn add(self: *Tree, parent: usize, token: u32, reach: f32, source: NodeSource, conditional: f32) ?usize {
        if (self.len == 0 or self.len > MAX_TREE_ROWS or parent >= self.len) return null;
        self.scores_calibrated = false;
        const raw = probability(conditional);
        for (1..self.len) |i| {
            if (self.parents[i] != @as(i32, @intCast(parent)) or self.tokens[i] != token) continue;
            if (self.sources[i] != source) self.sources[i] = .both;
            self.origins[i] |= origin(source);
            self.confidence[i] = @max(self.confidence[i], raw);
            if (source != .look) self.draft_confidence[i] = @max(self.draft_confidence[i], raw);
            if (source != .draft) self.look_confidence[i] = @max(self.look_confidence[i], raw);
            self.conditional_reach[i] = @max(self.conditional_reach[i], raw);
            // Existing descendants preserve their conditionals if duplicate evidence raises reach.
            const old = self.scores[i];
            const raised = @max(old, @min(probability(reach), self.scores[parent]));
            self.scores[i] = raised;
            if (old > 0 and raised > old) {
                for (i + 1..self.len) |descendant| {
                    var ancestor = self.parents[descendant];
                    while (ancestor > @as(i32, @intCast(i))) ancestor = self.parents[@intCast(ancestor)];
                    if (ancestor == @as(i32, @intCast(i))) self.scores[descendant] *= raised / old;
                }
            }
            self.normalize();
            return i;
        }
        if (self.len == MAX_TREE_ROWS) return null;
        const i = self.len;
        self.len += 1;
        self.tokens[i] = token;
        self.parents[i] = @intCast(parent);
        self.scores[i] = @min(probability(reach), self.scores[parent]);
        self.conditional_reach[i] = raw;
        self.sources[i] = source;
        self.origins[i] = origin(source);
        self.confidence[i] = raw;
        self.draft_confidence[i] = if (source != .look) raw else 0;
        self.look_confidence[i] = if (source != .draft) raw else 0;
        self.normalize();
        return i;
    }

    pub fn chain(anchor: u32, tokens: []const u32, entry: f32, continuation: f32) Tree {
        var tree = Tree.init(anchor);
        var reach: f32 = 1;
        for (tokens[0..@min(tokens.len, MAX_TREE_ROWS - 1)], 0..) |token, i| {
            const conditional = probability(if (i == 0) entry else continuation);
            reach *= conditional;
            const row = i + 1;
            tree.tokens[row] = token;
            tree.parents[row] = @intCast(i);
            tree.scores[row] = reach;
            tree.conditional_reach[row] = conditional;
            tree.sources[row] = .look;
            tree.origins[row] = origin(.look);
            tree.confidence[row] = conditional;
            tree.look_confidence[row] = conditional;
            tree.len += 1;
        }
        return tree;
    }
};

/// A short rolling median suppresses isolated stalls without freezing after a regime change.
/// No platform timing constants are embedded in this estimator.
const Settled = struct {
    values: [7]f32 = @splat(0),
    len: u8 = 0,
    next: u8 = 0,

    fn observe(self: *Settled, value: f32) void {
        self.values[self.next] = cost(value);
        self.next = @intCast((self.next + 1) % self.values.len);
        self.len = @intCast(@min(self.len + 1, self.values.len));
    }
    fn estimate(self: Settled) ?f32 {
        if (self.len == 0) return null;
        var sorted = self.values;
        for (1..self.len) |i| {
            const v = sorted[i];
            var j = i;
            while (j > 0 and sorted[j - 1] > v) : (j -= 1) sorted[j] = sorted[j - 1];
            sorted[j] = v;
        }
        // Lower median is deliberately resistant even to one early, unlabelled cold stall.
        return sorted[(self.len - 1) / 2];
    }
};

const PriceKey = struct { class: i64, rows: u32, kind: Kind, variant: Variant };
const ColdState = enum { seen, retry_reserved };
const TrialKey = struct { class: i64, kind: Kind };
const WidthTrial = struct { rows: u32, variant: Variant, retry_reserved: bool = false };

const PreparationCell = struct {
    samples: Settled = .{},
    seen: bool = false,
};
const MtpPreparations = [mtp_contexts.len][2][mtp_max_rows]PreparationCell;
// Added model-owned storage is exactly 9 * 2 * 15 * @sizeOf(PreparationCell);
// no allocation or request/context reset accompanies observation or prediction.

pub const PriceModel = struct {
    allocator: Allocator,
    priors: std.AutoHashMap(PriceKey, f32),
    settled: std.AutoHashMap(PriceKey, Settled),
    warmed: std.AutoHashMap(PriceKey, ColdState),
    width_trials: std.AutoHashMap(TrialKey, WidthTrial),
    step_ms: ?f32 = null,

    pub fn init(allocator: Allocator) PriceModel {
        return .{
            .allocator = allocator,
            .priors = std.AutoHashMap(PriceKey, f32).init(allocator),
            .settled = std.AutoHashMap(PriceKey, Settled).init(allocator),
            .warmed = std.AutoHashMap(PriceKey, ColdState).init(allocator),
            .width_trials = std.AutoHashMap(TrialKey, WidthTrial).init(allocator),
        };
    }
    pub fn deinit(self: *PriceModel) void {
        self.priors.deinit();
        self.settled.deinit();
        self.warmed.deinit();
        self.width_trials.deinit();
    }
    pub fn count(self: *const PriceModel) usize {
        return self.settled.count();
    }
    pub fn clone(self: *const PriceModel, allocator: Allocator) !PriceModel {
        var copy = PriceModel.init(allocator);
        errdefer copy.deinit();
        copy.priors = try self.priors.cloneWithAllocator(allocator);
        copy.settled = try self.settled.cloneWithAllocator(allocator);
        copy.warmed = try self.warmed.cloneWithAllocator(allocator);
        copy.width_trials = try self.width_trials.cloneWithAllocator(allocator);
        copy.step_ms = self.step_ms;
        return copy;
    }

    fn number(value: std.json.Value) ?f32 {
        const n: f32 = switch (value) {
            .integer => |v| @floatFromInt(v),
            .float => |v| @floatCast(v),
            else => return null,
        };
        return if (positive(n)) cost(n) else null;
    }

    /// Input failures are optional-prior failures, not inference failures.
    pub fn load(io: std.Io, allocator: Allocator, path: []const u8) PriceModel {
        var result = PriceModel.init(allocator);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(4 * 1024 * 1024)) catch return result;
        defer allocator.free(bytes);
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return result;
        defer parsed.deinit();
        if (parsed.value != .object) return result;
        const schema = parsed.value.object.get("schema") orelse return result;
        if (schema != .integer or schema.integer != 3) return result;
        const buckets = parsed.value.object.get("context_bucket") orelse return result;
        if (buckets != .string or !std.mem.eql(u8, buckets.string, "floor-power-of-two-2048")) return result;
        if (parsed.value.object.get("step_ms")) |value| result.step_ms = number(value);
        var classes = parsed.value.object.iterator();
        while (classes.next()) |class_entry| {
            const class = std.fmt.parseInt(i64, class_entry.key_ptr.*, 10) catch continue;
            if (class_entry.value_ptr.* != .object) continue;
            var kinds = class_entry.value_ptr.object.iterator();
            while (kinds.next()) |kind_entry| {
                const kind = std.meta.stringToEnum(Kind, kind_entry.key_ptr.*) orelse continue;
                if (kind_entry.value_ptr.* != .object) continue;
                var variants = kind_entry.value_ptr.object.iterator();
                while (variants.next()) |variant_entry| {
                    const variant = std.meta.stringToEnum(Variant, variant_entry.key_ptr.*) orelse continue;
                    if (variant_entry.value_ptr.* != .object) continue;
                    var widths = variant_entry.value_ptr.object.iterator();
                    while (widths.next()) |width_entry| {
                        const rows = std.fmt.parseInt(u32, width_entry.key_ptr.*, 10) catch continue;
                        if (rows == 0) continue;
                        const ms = number(width_entry.value_ptr.*) orelse continue;
                        result.priors.put(.{ .class = class, .rows = rows, .kind = kind, .variant = variant }, ms) catch {};
                    }
                }
            }
        }
        return result;
    }

    pub fn prior(self: *const PriceModel, rows: u32, kind: Kind, class: i64, variant: Variant) ?f32 {
        return self.priors.get(.{ .class = class, .rows = rows, .kind = kind, .variant = variant });
    }
    /// Exact evidence only: unlike price(), this never invents a timing estimate.
    pub fn known(self: *const PriceModel, rows: u32, kind: Kind, class: i64, variant: Variant) ?f32 {
        if (self.settled.get(.{ .class = class, .rows = rows, .kind = kind, .variant = variant })) |sample| {
            if (sample.estimate()) |ms| return ms;
        }
        return self.prior(rows, kind, class, variant);
    }
    pub fn price(self: *const PriceModel, rows: u32, kind: Kind, class: i64, variant: Variant) f32 {
        const width = @max(rows, 1);
        const key: PriceKey = .{ .class = class, .rows = width, .kind = kind, .variant = variant };
        if (self.settled.get(key)) |sample| if (sample.estimate()) |ms| return ms;
        if (self.prior(width, kind, class, variant)) |ms| return ms;
        // Within-kind/context/variant extrapolation never assumes a wider action is cheaper.
        // Unknown widths carry a factor-two uncertainty penalty; this is not a timing prior.
        var nearest_distance: u32 = std.math.maxInt(u32);
        var nearest: ?f32 = null;
        // Probe only executable widths in this exact context/kind/variant. Prediction work is
        // bounded independently of the number of historical contexts in the maps.
        for (1..MAX_TREE_ROWS + 1) |candidate_width| {
            const old_width: u32 = @intCast(candidate_width);
            const distance = if (width > old_width) width - old_width else old_width - width;
            if (distance >= nearest_distance) continue;
            const ms = self.known(old_width, kind, class, variant) orelse continue;
            nearest_distance = distance;
            nearest = ms * @max(1, @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(old_width)));
        }
        return cost((nearest orelse ((self.step_ms orelse 1) * @as(f32, @floatFromInt(width)))) * 2);
    }
    pub fn observe(self: *PriceModel, rows: u32, ms: f32, kind: Kind, class: i64, variant: Variant) void {
        if (rows == 0 or !positive(ms)) return;
        const entry = self.settled.getOrPut(.{ .class = class, .rows = rows, .kind = kind, .variant = variant }) catch return;
        if (!entry.found_existing) entry.value_ptr.* = .{};
        entry.value_ptr.observe(ms);
    }
    pub fn observeKernel(self: *PriceModel, rows: u32, ms: f32, kind: Kind, class: i64, variant: Variant) bool {
        if (rows == 0 or !positive(ms)) return false;
        const key: PriceKey = .{ .class = class, .rows = rows, .kind = kind, .variant = variant };
        const entry = self.warmed.getOrPut(key) catch return false;
        if (!entry.found_existing) {
            entry.value_ptr.* = .seen;
            return false;
        }
        self.observe(rows, ms, kind, class, variant);
        return self.settled.contains(key);
    }


    /// One wider real shape per context/kind, with at most one matching cold retry.
    /// Both shapes must be classified from actual executable selected parents.
    /// A retry proposal retains its original variant; reserveUpperTrial checks the
    /// current offer's geometry before consuming the remaining attempt.
    pub fn upperTrialWidth(self: *const PriceModel, native: Shape, maximum: Shape, kind: Kind, class: i64) ?u32 {
        if (self.known(native.rows, kind, class, native.variant) == null) return null;
        if (self.width_trials.get(.{ .class = class, .kind = kind })) |trial| {
            if (trial.retry_reserved or trial.rows <= native.rows or trial.rows > maximum.rows or self.known(trial.rows, kind, class, trial.variant) != null) return null;
            return trial.rows;
        }
        if (maximum.rows > MAX_TREE_ROWS) return null;
        return if (maximum.rows > native.rows and self.known(maximum.rows, kind, class, maximum.variant) == null) maximum.rows else null;
    }

    pub fn reserveUpperTrial(self: *PriceModel, native: Shape, shape: Shape, kind: Kind, class: i64) bool {
        if (shape.rows <= native.rows or shape.rows > MAX_TREE_ROWS or self.known(native.rows, kind, class, native.variant) == null or self.known(shape.rows, kind, class, shape.variant) != null) return false;
        const key: TrialKey = .{ .class = class, .kind = kind };
        const price_key: PriceKey = .{ .class = class, .rows = shape.rows, .kind = kind, .variant = shape.variant };
        if (self.width_trials.getPtr(key)) |trial| {
            if (trial.rows != shape.rows or trial.variant != shape.variant or trial.retry_reserved) return false;
            const cold = self.warmed.getPtr(price_key) orelse return false;
            if (cold.* != .seen) return false;
            trial.retry_reserved = true;
            cold.* = .retry_reserved;
            return true;
        }
        const cold = self.warmed.getPtr(price_key);
        if (cold) |state| if (state.* == .retry_reserved) return false;
        // If this shape already ran cold naturally, reserve only its remaining retry.
        self.width_trials.put(key, .{ .rows = shape.rows, .variant = shape.variant, .retry_reserved = cold != null }) catch return false;
        if (cold) |state| state.* = .retry_reserved;
        return true;
    }
};

pub const Calibration = struct {
    pub const Kind = enum { look, look_continuation };
    const Bin = struct { hits: f32 = 0, total: f32 = 0 };
    const orders = 65; // token_history's bounded suffix lengths; zero is unspecified.
    bins: [2][orders][8]Bin = @splat(@splat(@splat(.{}))),

    fn bucket(p: f32) usize {
        return @min(7, @as(usize, @intFromFloat(probability(p) * 8)));
    }
    pub fn q(self: *const Calibration, kind: Calibration.Kind, p: f32, order: u32) f32 {
        const raw = probability(p);
        const index = bucket(raw);
        const bin = self.bins[@intFromEnum(kind)][@min(order, orders - 1)][index];
        return probability((4 * raw + bin.hits) / (4 + bin.total));
    }
    pub fn learn(self: *Calibration, kind: Calibration.Kind, p: f32, order: u32, hit: bool) void {
        const raw = probability(p);
        const index = bucket(raw);
        const bin = &self.bins[@intFromEnum(kind)][@min(order, orders - 1)][index];
        if (bin.total >= 128) {
            bin.hits *= 0.5;
            bin.total *= 0.5;
        }
        bin.total += 1;
        if (hit) bin.hits += 1;
    }
    pub fn samples(self: *const Calibration, kind: Calibration.Kind) f32 {
        var total: f32 = 0;
        for (self.bins[@intFromEnum(kind)]) |population| for (population) |bin| {
            total += bin.total;
        };
        return total;
    }
};

/// Request-local predictive support, pooled across suffix orders so a verified
/// copy can continue as its matching suffix grows. Model calibration still keys
/// exact orders; entry/continuation and confidence bins remain separate here.
/// This heuristic can transfer support to unrelated orders, not just copies.
pub const RequestTrust = struct {
    const Bin = struct { hits: f32 = 0, total: f32 = 0 };
    bins: [2][8]Bin = @splat(@splat(.{})),

    pub fn q(self: *const RequestTrust, kind: Calibration.Kind, raw: f32) f32 {
        const bin = self.bins[@intFromEnum(kind)][Calibration.bucket(raw)];
        // Miss-only evidence cannot create support that an unseen bin lacked.
        return if (bin.hits == 0) 0 else (bin.hits + 1) / (bin.total + 2);
    }
    pub fn learn(self: *RequestTrust, kind: Calibration.Kind, raw: f32, hit: bool) void {
        const bin = &self.bins[@intFromEnum(kind)][Calibration.bucket(raw)];
        if (bin.total >= 128) {
            bin.hits *= 0.5;
            bin.total *= 0.5;
        }
        bin.total += 1;
        if (hit) bin.hits += 1;
    }
    pub fn diagnosticFingerprint(self: *const RequestTrust) [32]u8 {
        var hash: Fingerprint = .{};
        hash.hash.update("mlx-serve-request-lookup-trust-3");
        for (self.bins) |kind| for (kind) |bin| {
            hash.float(bin.hits);
            hash.float(bin.total);
        };
        return hash.finish();
    }
};

pub const Choice = struct {
    pub const Source = enum { spine, look };
    tokens: []const u32,
    rows: u32,
    source: Source,
    expected: f32,
    value: f32,
};
pub const RemainingRound = struct { round_ms: f32, expected_tokens: f32 };
pub const LookFallback = union(enum) {
    plain,
    unknown,
    measured: RemainingRound,
};
pub const TreeEvaluation = struct {
    pub const Entry = struct {
        rows: u32,
        variant: Variant,
        verify_ms: f32,
        expected_tokens: f32,
        tokens_per_ms: f32,
        /// Geometry of this priced action, before any forced exploration override.
        /// A retained incumbent does not use the merged candidate's prefix mask.
        prefix_mask: u32,
        retained_native: bool,
    };
    entries: [MAX_TREE_ROWS]Entry = undefined,
    len: usize = 0,
};
pub const RouteDecision = struct {
    pub const Reason = enum { unmeasured, invalid_alternative, assistant, alternative, recovery, native_copy, native_copy_unmeasured };
    skip: bool,
    reason: Reason,
    assistant_ms: ?f32,
    alternative_ms: ?f32,
    skipped: u32,
    interval: u32,
    native_copy: bool = false,
    native_copy_ms: ?f32 = null,
    native_copy_tokens: ?f32 = null,
};
const RoundSamples = struct {
    values: [7]RemainingRound = @splat(.{ .round_ms = 0, .expected_tokens = 0 }),
    len: u8 = 0,
    next: u8 = 0,

    fn observe(self: *RoundSamples, ms: f32, tokens: u32) void {
        self.values[self.next] = .{ .round_ms = cost(ms), .expected_tokens = @floatFromInt(tokens) };
        self.next = @intCast((self.next + 1) % self.values.len);
        self.len = @intCast(@min(self.len + 1, self.values.len));
    }
    fn estimate(self: *const RoundSamples) ?RemainingRound {
        if (self.len == 0) return null;
        var ms: f64 = 0;
        var tokens: f64 = 0;
        for (self.values[0..self.len]) |sample| {
            ms += sample.round_ms;
            tokens += sample.expected_tokens;
        }
        const count: f64 = @floatFromInt(self.len);
        return .{ .round_ms = @floatCast(ms / count), .expected_tokens = @floatCast(tokens / count) };
    }
};

// Native backed-copy widths include the anchor. Each context owns bounded paired
// remaining-work/yield history; its first valid actual execution is cold.
const NativeCopyCell = struct {
    samples: RoundSamples = .{},
    seen: bool = false,
};
const native_copy_widths = 15; // rows 2...16

const RoundState = struct {
    plain: Settled = .{},
    spec: RoundSamples = .{},
    plain_seen: u32 = 0,
    spec_seen: u32 = 0,
    skipped: u32 = 0,
    native_copy: [native_copy_widths]NativeCopyCell = @splat(.{}),
};

pub const Policy = struct {
    prices: PriceModel,
    cal: Calibration = .{},
    rounds: std.AutoHashMap(i64, RoundState),
    mtp_preparations: MtpPreparations = @splat(@splat(@splat(.{}))),

    pub fn init(prices: PriceModel) Policy {
        return .{ .prices = prices, .rounds = std.AutoHashMap(i64, RoundState).init(prices.allocator) };
    }
    pub fn deinit(self: *Policy) void {
        self.rounds.deinit();
        self.prices.deinit();
    }
    pub fn clone(self: *const Policy, allocator: Allocator) !Policy {
        var copy = Policy.init(try self.prices.clone(allocator));
        errdefer copy.deinit();
        copy.rounds = try self.rounds.cloneWithAllocator(allocator);
        copy.cal = self.cal;
        copy.mtp_preparations = self.mtp_preparations;
        return copy;
    }

    /// Diagnostic-only state identity. Hash explicit values, never pointers or padding.
    /// Entry digests are XOR-folded with a map-specific domain and count, so rehashing or
    /// cloning the containers does not change identity. Keys are unique in each map.
    pub fn diagnosticFingerprint(self: *const Policy) [32]u8 {
        var all: Fingerprint = .{};
        all.hash.update("mlx-serve-independent-policy-state-8");
        all.word(@intFromBool(self.prices.step_ms != null));
        if (self.prices.step_ms) |step| all.float(step);
        for (self.cal.bins) |kind| for (kind) |population| for (population) |bin| {
            all.float(bin.hits);
            all.float(bin.total);
        };
        all.hash.update("mtp-lookup-preparation");
        for (self.mtp_preparations) |context| for (context) |scope| for (scope) |cell| {
            all.word(@intFromBool(cell.seen));
            all.window(cell.samples);
        };
        var fold: [32]u8 = @splat(0);
        var priors = self.prices.priors.iterator();
        while (priors.next()) |entry| {
            var one: Fingerprint = .{};
            one.priceKey(entry.key_ptr.*);
            one.float(entry.value_ptr.*);
            const digest = one.finish();
            for (&fold, digest) |*byte, value| byte.* ^= value;
        }
        all.hash.update("priors");
        all.word(self.prices.priors.count());
        all.hash.update(&fold);
        fold = @splat(0);
        var settled = self.prices.settled.iterator();
        while (settled.next()) |entry| {
            var one: Fingerprint = .{};
            one.priceKey(entry.key_ptr.*);
            one.window(entry.value_ptr.*);
            const digest = one.finish();
            for (&fold, digest) |*byte, value| byte.* ^= value;
        }
        all.hash.update("settled");
        all.word(self.prices.settled.count());
        all.hash.update(&fold);
        fold = @splat(0);
        var warmed = self.prices.warmed.iterator();
        while (warmed.next()) |entry| {
            var one: Fingerprint = .{};
            one.priceKey(entry.key_ptr.*);
            one.word(@intFromEnum(entry.value_ptr.*));
            const digest = one.finish();
            for (&fold, digest) |*byte, value| byte.* ^= value;
        }
        all.hash.update("variant-warmups");
        all.word(self.prices.warmed.count());
        all.hash.update(&fold);
        fold = @splat(0);
        var trials = self.prices.width_trials.iterator();
        while (trials.next()) |entry| {
            var one: Fingerprint = .{};
            one.word(@bitCast(entry.key_ptr.class));
            one.word(@intFromEnum(entry.key_ptr.kind));
            one.word(entry.value_ptr.rows);
            one.word(@intFromEnum(entry.value_ptr.variant));
            one.word(@intFromBool(entry.value_ptr.retry_reserved));
            const digest = one.finish();
            for (&fold, digest) |*byte, value| byte.* ^= value;
        }
        all.hash.update("bounded-width-trials");
        all.word(self.prices.width_trials.count());
        all.hash.update(&fold);
        fold = @splat(0);
        var rounds = self.rounds.iterator();
        while (rounds.next()) |entry| {
            var one: Fingerprint = .{};
            one.word(@bitCast(entry.key_ptr.*));
            one.window(entry.value_ptr.plain);
            one.word(entry.value_ptr.spec.len);
            one.word(entry.value_ptr.spec.next);
            for (entry.value_ptr.spec.values) |sample| {
                one.float(sample.round_ms);
                one.float(sample.expected_tokens);
            }
            one.word(entry.value_ptr.plain_seen);
            one.word(entry.value_ptr.spec_seen);
            one.word(entry.value_ptr.skipped);
            for (&entry.value_ptr.native_copy) |*cell| {
                one.word(@intFromBool(cell.seen));
                one.word(cell.samples.len);
                one.word(cell.samples.next);
                for (cell.samples.values) |sample| {
                    one.float(sample.round_ms);
                    one.float(sample.expected_tokens);
                }
            }
            const digest = one.finish();
            for (&fold, digest) |*byte, value| byte.* ^= value;
        }
        all.hash.update("rounds");
        all.word(self.rounds.count());
        all.hash.update(&fold);
        return all.finish();
    }
    pub fn predictVerify(self: *const Policy, rows: u32, kind: Kind, class: i64, variant: Variant) f32 {
        return self.prices.price(rows, kind, class, variant);
    }
    /// Only actual, aligned lookup executions may call this; invalid samples do not warm.
    pub fn observeMtpPreparation(self: *Policy, rows: u32, class: i64, scope: LookupPreparationScope, ms: f32) bool {
        const index = mtpContextIndex(class) orelse return false;
        if (rows == 0 or rows > mtp_max_rows or !positive(ms)) return false;
        const cell = &self.mtp_preparations[index][@intFromEnum(scope)][rows - 1];
        if (!cell.seen) {
            cell.seen = true;
            return false;
        }
        cell.samples.observe(ms);
        return true;
    }

    pub fn knownMtpPreparation(self: *const Policy, rows: u32, class: i64, scope: LookupPreparationScope) ?f32 {
        const index = mtpContextIndex(class) orelse return null;
        if (rows == 0 or rows > mtp_max_rows) return null;
        return self.mtp_preparations[index][@intFromEnum(scope)][rows - 1].samples.estimate();
    }

    fn lookKind(tree: *const Tree, row: usize) Calibration.Kind {
        const parent: usize = @intCast(tree.parents[row]);
        return if (parent == 0 or tree.sources[parent] == .draft) .look else .look_continuation;
    }
    fn calibrated(self: *const Policy, candidate: *const Tree) Tree {
        var tree = candidate.*;
        tree.len = @min(tree.len, MAX_TREE_ROWS);
        if (tree.len == 0) return Tree.init(0);
        tree.parents[0] = -1;
        tree.scores[0] = 1;
        for (1..tree.len) |i| {
            // Invalid tails are not executable. No synthetic ancestry is introduced.
            if (tree.parents[i] < 0 or tree.parents[i] >= @as(i32, @intCast(i))) {
                tree.len = i;
                break;
            }
        }
        if (tree.scores_calibrated) {
            tree.normalize();
            return tree;
        }
        for (0..tree.len) |parent| {
            var sum: f32 = 0;
            for (parent + 1..tree.len) |child| {
                if (tree.parents[child] != @as(i32, @intCast(parent))) continue;
                const draft_q = probability(tree.draft_confidence[child]);
                const look_q = self.cal.q(lookKind(&tree, child), tree.look_confidence[child], tree.look_order[child]);
                const q = switch (tree.sources[child]) { .draft => draft_q, .look => look_q, .both => @max(draft_q, look_q) };
                tree.scores[child] = q;
                sum += q;
            }
            for (parent + 1..tree.len) |child| {
                if (tree.parents[child] != @as(i32, @intCast(parent))) continue;
                tree.conditional_reach[child] = tree.scores[child] / @max(1, sum);
                tree.scores[child] = tree.scores[parent] * tree.conditional_reach[child];
            }
        }
        tree.scores_calibrated = true;
        return tree;
    }

    /// Restrict a pooled-ranked lookup offer without donating excluded sibling
    /// mass. Ranking/storage precedes this operation, so a cold request retains
    /// useful descendants for observations even when every execution reach is zero.
    pub fn restrictLookup(self: *const Policy, offer: *const Tree, trust: *const RequestTrust) Tree {
        var result = offer.*;
        for (1..offer.len) |row| {
            const parent: usize = @intCast(offer.parents[row]);
            const kind = lookKind(offer, row);
            const pooled = self.cal.q(kind, offer.look_confidence[row], offer.look_order[row]);
            const supported = @min(pooled, trust.q(kind, offer.look_confidence[row]));
            const conditional = if (offer.scores[parent] > 0) offer.scores[row] / offer.scores[parent] else offer.conditional_reach[row];
            result.conditional_reach[row] = conditional * (if (pooled > 0) supported / pooled else 0);
            result.scores[row] = result.scores[parent] * result.conditional_reach[row];
        }
        result.scores_calibrated = true;
        return result;
    }

    pub fn chooseTree(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, dispatch: TreeDispatch) Tree {
        return self.chooseTreeControlled(candidate, class, preparation_ms, max_rows, 0, null, dispatch);
    }
    pub fn chooseTreeControlled(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, requested: u32, evaluation: ?*TreeEvaluation, dispatch: TreeDispatch) Tree {
        return self.selectTree(candidate, class, preparation_ms, max_rows, requested, evaluation, .tree, dispatch, null, null);
    }

    /// At native width, preserve original neural geometry unless a genuinely new
    /// lookup path has a better eligible rate. Compare each action's actual variant.
    pub fn chooseTreeIncumbentControlled(self: *const Policy, candidate: *const Tree, native: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, requested: u32, evaluation: ?*TreeEvaluation, dispatch: TreeDispatch) Tree {
        return self.selectTree(candidate, class, preparation_ms, max_rows, requested, evaluation, .tree, dispatch, native, null);
    }

    /// The caller identifies its executable native action; no native width or hardware
    /// latency is guessed here. Bootstrap is a real action, not a root-only calibration probe.
    pub fn chooseTreeBootstrapControlled(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, requested: u32, incumbent: *const Tree, evaluation: ?*TreeEvaluation, dispatch: TreeDispatch) Tree {
        if (requested == 0) {
            const native = self.chooseTreeIncumbentControlled(incumbent, incumbent, class, preparation_ms, max_rows, @intCast(incumbent.len), evaluation, dispatch);
            if (self.prices.known(@intCast(native.len), .tree, class, native.variant(dispatch)) == null) {
                return native;
            }
        }
        return self.chooseTreeIncumbentControlled(candidate, incumbent, class, preparation_ms, max_rows, requested, evaluation, dispatch);
    }

    fn selectTree(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, requested: u32, evaluation: ?*TreeEvaluation, kind: Kind, dispatch: TreeDispatch, incumbent: ?*const Tree, mtp_scope: ?LookupPreparationScope) Tree {
        const tree = self.calibrated(candidate);
        const cap = @min(tree.len, @min(MAX_TREE_ROWS, @max(1, max_rows)));
        const wanted = @min(cap, @max(1, requested));
        if (evaluation) |log| log.len = 0;
        // A child's marginal reach never exceeds its parent's. Stable descending order
        // therefore has ancestor-closed prefixes, including ties. Except for retained
        // native geometry, each prefix maximizes summed reach at that width. Decide
        // the actual action before evaluating its yield and kernel price.
        var order: [MAX_TREE_ROWS]usize = undefined;
        for (0..tree.len) |i| {
            var at = i;
            while (at > 0 and tree.scores[order[at - 1]] < tree.scores[i]) : (at -= 1) {
                order[at] = order[at - 1];
            }
            order[at] = i;
        }
        var selected: u32 = 1;
        var prefix: u32 = 0;
        var reward: f32 = 0;
        var best_value: f32 = -1;
        var linear = true;
        var lookup_only = false;
        var selected_native = false;
        for (1..cap + 1) |width| {
            const row = order[width - 1];
            prefix |= @as(u32, 1) << @as(u5, @intCast(row));
            reward += tree.scores[row];
            // In an ancestor-closed linear prefix, the next node must extend its
            // sole leaf. Once a sibling enters, every wider prefix remains branched.
            if (width > 1 and tree.parents[row] != @as(i32, @intCast(order[width - 2]))) linear = false;
            if (width > 1 and tree.origins[row] & 8 == 0) lookup_only = true;
            if (requested > 0 and width != wanted) continue;
            var keep_native = if (incumbent) |native| width == native.len and !lookup_only else false;
            var variant = if (keep_native) incumbent.?.variant(dispatch) else dispatch.variant(width, linear);
            var expected = reward;
            if (keep_native) {
                expected = 0;
                for (incumbent.?.scores[0..width]) |reach| expected += reach;
            }
            // Unknown widths are exploration actions, not measured staircase cells.
            // Native bootstrap and bounded explicit trials establish their prices;
            // an extrapolated latency must never win adaptive width selection.
            var known_ms = self.prices.known(@intCast(width), kind, class, variant) orelse
                if (requested > 0) @as(?f32, self.predictVerify(@intCast(width), kind, class, variant)) else null;
            // A mixed prefix is a challenger, not a replacement for the native
            // action's eligibility. Its projection may lower neural reach or
            // change the dispatch to an unpriced kernel at the very same width.
            if (!keep_native) if (incumbent) |native| {
                if (width == native.len) {
                    const native_variant = native.variant(dispatch);
                    const native_ms = self.prices.known(@intCast(width), kind, class, native_variant) orelse
                        if (requested > 0) @as(?f32, self.predictVerify(@intCast(width), kind, class, native_variant)) else null;
                    if (native_ms) |ms| {
                        var native_expected: f32 = 0;
                        for (native.scores[0..width]) |reach| native_expected += reach;
                        if (known_ms == null or native_expected / (preparation(preparation_ms) + ms) >=
                            expected / (preparation(preparation_ms) + known_ms.?))
                        {
                            keep_native = true;
                            variant = native_variant;
                            expected = native_expected;
                            known_ms = ms;
                        }
                    }
                }
            };
            const verify_ms = known_ms orelse continue;
            // An unknown P preserves the previous zero-additional-preparation
            // hypothesis, not a measured zero. Known P corrects each width here.
            const lookup_preparation = if (mtp_scope) |scope|
                self.knownMtpPreparation(@intCast(width), class, scope) orelse 0
            else
                0;
            const value = expected / (preparation(preparation_ms) + lookup_preparation + verify_ms);
            if (evaluation) |log| {
                log.entries[log.len] = .{ .rows = @intCast(width), .variant = variant, .verify_ms = verify_ms, .expected_tokens = expected, .tokens_per_ms = value, .prefix_mask = prefix, .retained_native = keep_native };
                log.len += 1;
            }
            if (value > best_value) {
                best_value = value;
                selected = prefix;
                selected_native = keep_native;
            }
        }
        if (selected_native) return incumbent.?.*;
        return tree.subset(selected);
    }

    /// Standalone lookup otherwise cannot learn any wider chain from root-only
    /// execution: conservative unknown-width prices keep selecting the root.
    /// The caller admits only unconstrained, solo PLD rounds with a capped offer.
    pub fn reservePlainWidthTrial(self: *Policy, candidate: *const Tree, class: i64) ?u32 {
        if (!candidate.scores_calibrated or candidate.len < 2 or candidate.len > MAX_TREE_ROWS) return null;
        const root: Shape = .{ .rows = 1, .variant = .generic };
        const trial = self.prices.upperTrialWidth(root, .{ .rows = @intCast(candidate.len), .variant = .generic }, .chain, class) orelse return null;
        var gain: f32 = 0;
        for (candidate.scores[1..trial], 1..) |reach, row| {
            if (!std.math.isFinite(reach) or reach < 0 or reach > 1 or
                candidate.parents[row] != @as(i32, @intCast(row - 1))) return null;
            gain += reach;
        }
        if (gain <= 0 or !self.prices.reserveUpperTrial(root, .{ .rows = trial, .variant = .generic }, .chain, class)) return null;
        return trial;
    }

    pub const LookWidthTrialEvidence = struct {
        rows: u32,
        reference_rows: u32,
        reference_ms: f32,
        preparation_ms: f32,
        lookup_preparation_ms: ?f32,
        mtp_scope: ?LookupPreparationScope,
        expected_tokens: f32,
        optimistic_rate: f32,
        native_rate: f32,
    };

    /// Screen bounded exploration assuming zero added verification latency over
    /// the first priced narrow reference. This is not an upper-width price or a
    /// mathematical lower bound: wider kernels can actually become cheaper.
    /// The caller supplies the same paid preparation as ordinary admission.
    /// P never opens or renews this verifier-only budget. Its optimistic screen
    /// charges known P, retaining the zero-P hypothesis only while it is unknown.
    pub fn reserveLookWidthTrial(self: *Policy, candidate: *const Tree, class: i64, minimum_rows: u32, preparation_ms: f32, native_cost: ?RemainingRound, evidence: ?*?LookWidthTrialEvidence, mtp_scope: ?LookupPreparationScope) ?u32 {
        if (evidence) |out| out.* = null;
        const floor = @max(2, minimum_rows);
        if (!candidate.scores_calibrated or candidate.len <= floor or candidate.len > MAX_TREE_ROWS) return null;
        const native = native_cost orelse return null;
        if (!(native.round_ms > 0) or !std.math.isFinite(native.round_ms) or
            !(native.expected_tokens > 0) or !std.math.isFinite(native.expected_tokens)) return null;
        var reference_rows: ?u32 = null;
        var reference_ms: f32 = undefined;
        for (floor..candidate.len) |rows| {
            if (self.prices.known(@intCast(rows), .chain, class, .generic)) |ms| {
                reference_rows = @intCast(rows);
                reference_ms = ms;
                break;
            }
        }
        const reference = reference_rows orelse return null;
        const trial = self.prices.upperTrialWidth(.{ .rows = reference, .variant = .generic }, .{ .rows = @intCast(candidate.len), .variant = .generic }, .chain, class) orelse return null;
        var expected: f32 = 0;
        var reference_expected: f32 = 0;
        for (candidate.scores[0..trial], 0..) |reach, row| {
            if (!std.math.isFinite(reach) or reach < 0 or reach > 1) return null;
            if (row > 0 and candidate.parents[row] != @as(i32, @intCast(row - 1))) return null;
            expected += reach;
            if (row < reference) reference_expected += reach;
        }
        if (expected <= reference_expected) return null;
        const paid = preparation(preparation_ms);
        const lookup_preparation = if (mtp_scope) |scope| self.knownMtpPreparation(trial, class, scope) else null;
        const optimistic_rate = expected / (paid + (lookup_preparation orelse 0) + reference_ms);
        const native_rate = native.expected_tokens / (paid + native.round_ms);
        if (evidence) |out| out.* = .{
            .rows = trial,
            .reference_rows = reference,
            .reference_ms = reference_ms,
            .preparation_ms = paid,
            .lookup_preparation_ms = lookup_preparation,
            .mtp_scope = mtp_scope,
            .expected_tokens = expected,
            .optimistic_rate = optimistic_rate,
            .native_rate = native_rate,
        };
        if (!(optimistic_rate > native_rate)) return null;
        if (!self.prices.reserveUpperTrial(.{ .rows = reference, .variant = .generic }, .{ .rows = trial, .variant = .generic }, .chain, class)) return null;
        return trial;
    }

    /// Q is shared paid query time; known P is a lookup-only omitted-cost correction.
    /// Unknown P retains the prior hypothesis, not a measured zero or exact total price.
    /// Native fallback pricing and explicit fixed/maximum widths remain unchanged.
    pub fn chooseLookPreparedControlled(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, out: *std.ArrayList(u32), requested: u32, fallback: LookFallback, mtp_scope: ?LookupPreparationScope) !Choice {
        out.clearRetainingCapacity();
        const paid = preparation(preparation_ms);
        const baseline: ?f32 = switch (fallback) {
            .unknown => null,
            .plain => 1 / (paid + (self.plainFor(class) orelse self.predictVerify(1, .plain, class, .generic))),
            .measured => |value| if (positive(value.round_ms) and positive(value.expected_tokens))
                value.expected_tokens / (paid + cost(value.round_ms))
            else
                null,
        };
        if (requested == 0 and baseline == null) return .{ .tokens = out.items, .rows = 0, .source = .spine, .expected = 0, .value = 0 };
        const selected = self.selectTree(candidate, class, paid, MAX_TREE_ROWS, requested, null, .chain, .generic, null, mtp_scope);
        if (requested == 0 and self.prices.known(@intCast(selected.len), .chain, class, .generic) == null)
            return .{ .tokens = out.items, .rows = 0, .source = .spine, .expected = 0, .value = baseline orelse 0 };
        var expected: f32 = 0;
        for (selected.scores[1..selected.len]) |reach| expected += reach;
        const lookup_preparation = if (mtp_scope) |scope|
            self.knownMtpPreparation(@intCast(selected.len), class, scope) orelse 0
        else
            0;
        const value = (1 + expected) / (paid + lookup_preparation + self.predictVerify(@intCast(selected.len), .chain, class, .generic));
        if (requested == 0 and baseline != null and value <= baseline.?) return .{ .tokens = out.items, .rows = 0, .source = .spine, .expected = 0, .value = baseline.? };
        try out.appendSlice(self.prices.allocator, selected.tokens[1..selected.len]);
        return .{ .tokens = out.items, .rows = @intCast(out.items.len), .source = .look, .expected = expected, .value = value };
    }

    /// A verified linear prefix observes successes and, unless fully accepted, one
    /// rejection. Scheduling-clipped and unreachable descendants remain censored.
    pub fn observeChainAccepted(self: *Policy, tree: *const Tree, executed_drafts: u32, accepted: u32) void {
        if (tree.len <= 1 or tree.len > MAX_TREE_ROWS) return;
        const drafted = @min(tree.len - 1, executed_drafts);
        const hits = @min(drafted, accepted);
        for (1..@min(drafted + 1, hits + 2)) |row| {
            self.cal.learn(if (row == 1) .look else .look_continuation, tree.look_confidence[row], tree.look_order[row], row <= hits);
        }
    }


    /// path_tokens excludes the anchor and contains only actually reached target decisions.
    /// A missing next decision is censored, not a rejection. A supplied mismatch rejects
    /// every candidate child at that parent, but never any of their descendants.
    pub fn observeLookupTree(self: *Policy, tree: *const Tree, path_tokens: []const u32, trust: ?*RequestTrust) void {
        if (tree.len == 0 or tree.len > MAX_TREE_ROWS) return;
        var parent: usize = 0;
        for (path_tokens) |actual| {
            var matching: ?usize = null;
            for (parent + 1..tree.len) |child| {
                if (tree.parents[child] != @as(i32, @intCast(parent))) continue;
                const hit = tree.tokens[child] == actual;
                if (tree.sources[child] != .draft) {
                    const kind = lookKind(tree, child);
                    self.cal.learn(kind, tree.look_confidence[child], tree.look_order[child], hit);
                    if (trust) |request| request.learn(kind, tree.look_confidence[child], hit);
                }
                if (hit) matching = child;
            }
            parent = matching orelse break;
        }
    }

    /// A bounded branching offer may omit part of a separately selected chain.
    /// After observing that offer, retain executed-chain coverage without learning
    /// the shared prefix twice. Membership is by parent-prefix, never token alone.
    pub fn observeLookupChainRemainder(self: *Policy, chain: *const Tree, offered: *const Tree, path_tokens: []const u32, trust: *RequestTrust) void {
        if (chain.len == 0 or chain.len > MAX_TREE_ROWS or offered.len == 0 or offered.len > MAX_TREE_ROWS or chain.tokens[0] != offered.tokens[0]) return;
        var offered_parent: ?usize = 0;
        for (path_tokens, 1..) |actual, row| {
            if (row >= chain.len or chain.parents[row] != @as(i32, @intCast(row - 1))) break;
            var represented: ?usize = null;
            if (offered_parent) |parent| {
                for (parent + 1..offered.len) |child| {
                    if (offered.parents[child] == @as(i32, @intCast(parent)) and offered.tokens[child] == chain.tokens[row]) {
                        represented = child;
                        break;
                    }
                }
            }
            const hit = actual == chain.tokens[row];
            if (represented == null) {
                const kind: Calibration.Kind = if (row == 1) .look else .look_continuation;
                self.cal.learn(kind, chain.look_confidence[row], chain.look_order[row], hit);
                trust.learn(kind, chain.look_confidence[row], hit);
            }
            if (!hit) break;
            offered_parent = represented;
        }
    }

    fn state(self: *Policy, class: i64) ?*RoundState {
        const entry = self.rounds.getOrPut(class) catch return null;
        if (!entry.found_existing) entry.value_ptr.* = .{};
        return entry.value_ptr;
    }
    pub fn plainFor(self: *const Policy, class: i64) ?f32 {
        const context = self.rounds.getPtr(class) orelse return null;
        return context.plain.estimate();
    }
    pub fn specFor(self: *const Policy, class: i64) ?f32 {
        const sample = self.specRemainingFor(class) orelse return null;
        return sample.round_ms / sample.expected_tokens;
    }
    pub fn specRemainingFor(self: *const Policy, class: i64) ?RemainingRound {
        const context = self.rounds.getPtr(class) orelse return null;
        return context.spec.estimate();
    }
    pub fn nativeCopyRemainingFor(self: *const Policy, rows: u32, class: i64) ?RemainingRound {
        if (rows < 2 or rows > 16) return null;
        const context = self.rounds.getPtr(class) orelse return null;
        return context.native_copy[rows - 2].samples.estimate();
    }
    /// Settled post-query work and actual yield, never a shadow offer or assistant cost.
    pub fn observeNativeCopyRemaining(self: *Policy, ms: f32, tokens: u32, rows: u32, class: i64) bool {
        if (rows < 2 or rows > 16 or !positive(ms) or tokens == 0 or tokens > rows) return false;
        const context = self.state(class) orelse return false;
        const cell = &context.native_copy[rows - 2];
        if (!cell.seen) {
            cell.seen = true;
            return false;
        }
        cell.samples.observe(ms, tokens);
        return true;
    }
    pub fn observePlain(self: *Policy, ms: f32, class: i64) void {
        _ = self.observePlainProbe(ms, class, 0);
    }
    pub fn observePlainProbe(self: *Policy, ms: f32, class: i64, warmup: u32) bool {
        if (!positive(ms)) return false;
        const context = self.state(class) orelse return false;
        const cold = context.plain_seen < warmup;
        context.plain_seen +|= 1;
        if (cold) return false;
        context.plain.observe(ms);
        return true;
    }
    /// Only native work after the lookup query; never a mixed scheduler/predraft cycle.
    pub fn observeSpecRemainingProbe(self: *Policy, ms: f32, tokens: u32, class: i64, warmup: u32) bool {
        if (!positive(ms) or tokens == 0) return false;
        const context = self.state(class) orelse return false;
        const cold = context.spec_seen < warmup;
        context.spec_seen +|= 1;
        if (cold) return false;
        context.spec.observe(ms, tokens);
        return true;
    }
    pub fn routeAgainstPrepared(self: *Policy, class: i64, alternative_ms: ?f32, preparation_ms: f32, native_copy_rows: ?u32) RouteDecision {
        const remaining = self.specRemainingFor(class);
        const assistant: ?f32 = if (remaining) |sample|
            (preparation(preparation_ms) + sample.round_ms) / sample.expected_tokens
        else
            null;
        var result: RouteDecision = .{ .skip = false, .reason = .unmeasured, .assistant_ms = assistant, .alternative_ms = alternative_ms, .skipped = 0, .interval = 12 };
        if (native_copy_rows) |rows| {
            const copy = self.nativeCopyRemainingFor(rows, class) orelse {
                // Baseline preservation, not an observation or a recovery trial.
                result.skip = true;
                result.native_copy = true;
                result.reason = .native_copy_unmeasured;
                result.alternative_ms = null;
                if (self.rounds.getPtr(class)) |context| result.skipped = context.skipped;
                return result;
            };
            const copy_rate = (preparation(preparation_ms) + copy.round_ms) / copy.expected_tokens;
            result.native_copy_ms = copy_rate;
            result.native_copy_tokens = copy.expected_tokens;
            var copy_wins = true;
            var best_rate = copy_rate;
            if (alternative_ms) |ms| {
                if (positive(ms) and ms < best_rate) {
                    copy_wins = false;
                    best_rate = ms;
                }
            }
            const context = self.rounds.getPtr(class).?;
            const assistant_wins = if (assistant) |ms|
                ms < best_rate or (!copy_wins and ms == best_rate)
            else
                false;
            if (assistant_wins) {
                context.skipped = 0;
                result.reason = .assistant;
            } else if (context.skipped >= result.interval) {
                context.skipped = 0;
                result.reason = .recovery;
            } else {
                context.skipped += 1;
                result.skip = true;
                result.native_copy = copy_wins;
                result.reason = if (copy_wins) .native_copy else .alternative;
            }
            result.skipped = context.skipped;
            if (result.native_copy) result.alternative_ms = copy_rate;
            return result;
        }
        if (alternative_ms) |ms| {
            if (!positive(ms)) {
                result.reason = .invalid_alternative;
                if (self.rounds.getPtr(class)) |context| context.skipped = 0;
                return result;
            }
        } else {
            if (self.rounds.getPtr(class)) |context| context.skipped = 0;
            return result;
        }
        const incumbent = assistant orelse return result;
        const context = self.rounds.getPtr(class) orelse return result;
        if (incumbent <= alternative_ms.?) {
            context.skipped = 0;
            result.reason = .assistant;
        } else if (context.skipped >= result.interval) {
            context.skipped = 0;
            result.reason = .recovery;
        } else {
            context.skipped += 1;
            result.skip = true;
            result.reason = .alternative;
        }
        result.skipped = context.skipped;
        return result;
    }
};

test "tree boundaries, evidence union, and sibling mass preserve descendant conditionals" {
    var tree = Tree.init(70);
    const a = tree.add(0, 71, 0.8, .draft, 0.8).?;
    const b = tree.add(a, 72, 0.4, .draft, 0.5).?;
    _ = tree.add(0, 73, 0.8, .look, 0.8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), tree.scores[b] / tree.scores[a], 0.00001);
    try std.testing.expect(tree.scores[a] + tree.scores[3] <= 1);
    try std.testing.expectEqual(a, tree.add(0, 71, 0.2, .look, 0.3).?);
    try std.testing.expectEqual(NodeSource.both, tree.sources[a]);
    try std.testing.expectEqual(@as(u8, 9), tree.origins[a]);
    try std.testing.expect(tree.add(99, 1, 1, .look, 1) == null);
    _ = tree.add(0, 74, std.math.nan(f32), .look, std.math.inf(f32));
    for (tree.scores[0..tree.len]) |p| try std.testing.expect(std.math.isFinite(p) and p >= 0 and p <= 1);
    while (tree.len < MAX_TREE_ROWS) _ = tree.add(tree.len - 1, @intCast(tree.len + 80), 0, .look, 0);
    try std.testing.expect(tree.add(0, 999, 1, .look, 1) == null);
    try std.testing.expect(tree.add(0, 71, 0, .look, 0) != null);
}

test "cold exclusions are variant kind width and context specific" {
    var prices = PriceModel.init(std.testing.allocator);
    defer prices.deinit();
    try std.testing.expect(!prices.observeKernel(3, std.math.nan(f32), .tree, 101, .generic));
    try std.testing.expect(!prices.observeKernel(3, 900, .tree, 101, .generic));
    try std.testing.expect(prices.observeKernel(3, 6, .tree, 101, .generic));
    try std.testing.expect(!prices.observeKernel(3, 800, .tree, 101, .linear_gdn));
    try std.testing.expectEqual(@as(f32, 6), prices.price(3, .tree, 101, .generic));
    try std.testing.expect(prices.known(3, .tree, 101, .linear_gdn) == null);
    try std.testing.expect(!prices.observeKernel(3, 700, .chain, 101, .generic));
    try std.testing.expect(!prices.observeKernel(4, 700, .tree, 101, .generic));
    try std.testing.expect(!prices.observeKernel(3, 700, .tree, 102, .generic));
    prices.observe(3, 60000, .tree, 101, .generic);
    prices.observe(3, 7, .tree, 101, .generic);
    try std.testing.expectEqual(@as(f32, 7), prices.price(3, .tree, 101, .generic));
    try std.testing.expect(prices.prior(3, .tree, 101, .generic) == null);
    try std.testing.expect(positive(prices.price(0, .chain, -10, .generic)));
    try std.testing.expectEqual(@as(usize, 1), prices.count());
}

test "calibration populations censor unreachable descendants" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const tree = Tree.chain(50, &.{ 51, 52, 53 }, 0.6, 0.7);
    policy.observeLookupTree(&tree, &.{99}, null);
    try std.testing.expectEqual(@as(f32, 1), policy.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 0), policy.cal.samples(.look_continuation));
    policy.observeLookupTree(&tree, &.{51}, null);
    try std.testing.expectEqual(@as(f32, 0), policy.cal.samples(.look_continuation));
    policy.observeLookupTree(&tree, &.{ 51, 99 }, null);
    try std.testing.expectEqual(@as(f32, 1), policy.cal.samples(.look_continuation));
    policy.observeChainAccepted(&tree, @intCast(tree.len - 1), 0);
    try std.testing.expectEqual(@as(f32, 1), policy.cal.samples(.look_continuation));
    for ([_]f32{ -1, 0, 0.4, 1, 2, std.math.nan(f32), std.math.inf(f32) }) |raw| {
        const p = policy.cal.q(.look, raw, 0);
        try std.testing.expect(std.math.isFinite(p) and p >= 0 and p <= 1);
    }
}

test "root preparation participates and forced widths evaluate only their action" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    policy.prices.observe(1, 1, .tree, 5, .generic);
    policy.prices.observe(2, 3, .tree, 5, .generic);
    const tree = Tree.chain(8, &.{9}, 0.9, 0.9);
    var log: TreeEvaluation = .{};
    const free = policy.chooseTreeControlled(&tree, 5, 0, 32, 0, &log, .generic);
    try std.testing.expectEqual(@as(usize, 1), free.len);
    const paid = policy.chooseTreeControlled(&tree, 5, 20, 32, 0, &log, .generic);
    try std.testing.expectEqual(@as(usize, 2), paid.len);
    try std.testing.expectEqual(@as(usize, 2), log.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 21.0), log.entries[0].tokens_per_ms, 0.00001);
    const forced = policy.chooseTreeControlled(&tree, 5, 20, 32, 100, &log, .generic);
    try std.testing.expectEqual(@as(usize, 2), forced.len);
    try std.testing.expectEqual(@as(usize, 1), log.len);
    try std.testing.expectEqual(@as(u32, 2), log.entries[0].rows);
    const root = policy.chooseTreeControlled(&tree, 5, 20, 0, 100, &log, .generic);
    try std.testing.expectEqual(@as(usize, 1), root.len);
}

test "width selection preserves ancestors and stable ties across branches" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var tree = Tree.init(20);
    _ = tree.add(0, 21, 0.25, .draft, 0.25);
    _ = tree.add(0, 22, 0.65, .look, 0.65);
    _ = tree.add(2, 23, 0.65, .look, 1);
    _ = tree.add(3, 24, 0.65, .look, 1);
    const chosen = policy.chooseTreeControlled(&tree, 0, 0, 32, 4, null, .generic);
    try std.testing.expectEqualSlices(u32, &.{ 20, 22, 23, 24 }, chosen.tokens[0..chosen.len]);
    for (chosen.parents[1..chosen.len], 1..) |parent, row| try std.testing.expect(parent >= 0 and @as(usize, @intCast(parent)) < row);
    try std.testing.expectEqual(@as(f32, 1), chosen.look_confidence[3]);
}

test "unknown incumbent retains neural path except explicit force" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const candidate = Tree.chain(0, &.{ 4, 5, 6 }, 0.99, 0.99);
    const adaptive = try policy.chooseLookPreparedControlled(&candidate, 0, 0, &out, 0, .unknown, null);
    try std.testing.expectEqual(@as(u32, 0), adaptive.rows);
    try std.testing.expectEqual(Choice.Source.spine, adaptive.source);
    const forced = try policy.chooseLookPreparedControlled(&candidate, 0, 0, &out, 3, .unknown, null);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, forced.tokens);
    try std.testing.expectEqual(@as(u32, 2), forced.rows);
    const anchor_only = try policy.chooseLookPreparedControlled(&candidate, 0, 0, &out, 1, .unknown, null);
    try std.testing.expectEqual(@as(u32, 0), anchor_only.rows);
}

test "round warmup isolation bounded recovery and changed assistant costs" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    try std.testing.expect(!policy.observePlainProbe(-1, 12, 2));
    try std.testing.expect(!policy.observePlainProbe(100, 12, 2));
    try std.testing.expect(!policy.observePlainProbe(90, 12, 2));
    try std.testing.expect(policy.observePlainProbe(3, 12, 2));
    try std.testing.expect(!policy.observeSpecRemainingProbe(100, 1, 12, 2));
    try std.testing.expect(!policy.observeSpecRemainingProbe(90, 1, 12, 2));
    try std.testing.expect(policy.observeSpecRemainingProbe(18, 3, 12, 2));
    try std.testing.expect(policy.plainFor(13) == null);
    try std.testing.expect(!policy.observeSpecRemainingProbe(100, 1, 13, 2));
    try std.testing.expect(!policy.routeAgainstPrepared(13, 1, 0, null).skip);
    try std.testing.expect(!policy.routeAgainstPrepared(12, std.math.nan(f32), 0, null).skip);
    var skipped: u32 = 0;
    while (true) {
        const route = policy.routeAgainstPrepared(12, 3, 0, null);
        if (!route.skip) {
            try std.testing.expectEqual(RouteDecision.Reason.recovery, route.reason);
            try std.testing.expectEqual(route.interval, skipped);
            break;
        }
        skipped += 1;
        try std.testing.expect(skipped <= route.interval);
    }
    for (0..7) |_| _ = policy.observeSpecRemainingProbe(2, 2, 12, 0);
    const restored = policy.routeAgainstPrepared(12, 3, 0, null);
    try std.testing.expect(!restored.skip);
    try std.testing.expectEqual(RouteDecision.Reason.assistant, restored.reason);
}

test "clone owns prices warmups routing and calibration independently" {
    var original = Policy.init(PriceModel.init(std.testing.allocator));
    defer original.deinit();
    original.prices.observe(2, 7, .tree, 18, .generic);
    _ = original.prices.observeKernel(2, 100, .tree, 18, .linear_gdn);
    _ = original.observeSpecRemainingProbe(15, 3, 18, 0);
    _ = original.routeAgainstPrepared(18, 2, 0, null);
    const candidate = Tree.chain(0, &.{ 1, 2 }, 0.8, 0.8);
    original.observeChainAccepted(&candidate, 2, 1);
    var copy = try original.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(original.prices.price(2, .tree, 18, .generic), copy.prices.price(2, .tree, 18, .generic));
    try std.testing.expectEqual(original.cal.samples(.look), copy.cal.samples(.look));
    try std.testing.expectEqual(@as(u32, 2), copy.routeAgainstPrepared(18, 2, 0, null).skipped);
    try std.testing.expectEqual(@as(u32, 2), original.routeAgainstPrepared(18, 2, 0, null).skipped);
    try std.testing.expect(copy.prices.observeKernel(2, 2, .tree, 18, .linear_gdn));
    for (0..7) |_| copy.prices.observe(2, 2, .tree, 18, .generic);
    copy.observeChainAccepted(&candidate, 2, 0);
    try std.testing.expectEqual(@as(f32, 7), original.prices.price(2, .tree, 18, .generic));
    try std.testing.expectEqual(@as(f32, 2), copy.prices.price(2, .tree, 18, .generic));
    try std.testing.expectEqual(@as(f32, 1), original.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 2), copy.cal.samples(.look));
}

test "optional prior files missing malformed and partially valid remain safe" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}/prices.json", .{tmp.sub_path});
    defer std.testing.allocator.free(path);
    var missing = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer missing.deinit();
    try std.testing.expect(missing.known(4, .tree, 77, .generic) == null);
    try std.testing.expect(positive(missing.price(4, .tree, 77, .generic)));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "prices.json", .data = "{not-json" });
    var malformed = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer malformed.deinit();
    try std.testing.expect(malformed.known(4, .tree, 77, .generic) == null);
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "prices.json",
        .data = "{\"step_ms\":2.5,\"77\":{\"tree\":{\"4\":6.25}}}",
    });
    var legacy = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer legacy.deinit();
    try std.testing.expect(legacy.known(4, .tree, 77, .generic) == null);
    try std.testing.expect(legacy.step_ms == null);
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "prices.json",
        .data = "{\"schema\":2,\"context_bucket\":\"floor-power-of-two-2048\",\"step_ms\":2.5,\"77\":{\"tree\":{\"4\":6.25,\"2\":-7,\"8\":\"bad\"},\"chain\":{\"4\":9.5}},\"78\":false}",
    });
    var old_schema = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer old_schema.deinit();
    try std.testing.expect(old_schema.known(4, .tree, 77, .generic) == null);
    try std.testing.expect(old_schema.known(4, .tree, 77, .linear_gdn) == null);
    try std.testing.expect(old_schema.step_ms == null);
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "prices.json",
        .data = "{\"schema\":3,\"context_bucket\":\"floor-power-of-two-2048\",\"step_ms\":2.5,\"77\":{\"tree\":{\"generic\":{\"4\":6.25,\"2\":-7,\"8\":\"bad\"},\"linear_gdn\":{\"4\":3.25},\"unlabelled\":{\"4\":1},\"5\":1},\"chain\":{\"generic\":{\"4\":9.5}}},\"78\":false}",
    });
    var loaded = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer loaded.deinit();
    try std.testing.expectEqual(@as(?f32, 6.25), loaded.known(4, .tree, 77, .generic));
    try std.testing.expectEqual(@as(?f32, 3.25), loaded.known(4, .tree, 77, .linear_gdn));
    try std.testing.expectEqual(@as(?f32, 9.5), loaded.prior(4, .chain, 77, .generic));
    try std.testing.expect(loaded.known(2, .tree, 77, .generic) == null);
    try std.testing.expect(loaded.known(4, .tree, 78, .generic) == null);
    try std.testing.expect(loaded.known(5, .tree, 77, .generic) == null);
    try std.testing.expect(loaded.known(4, .chain, 77, .linear_gdn) == null);
    // Serialize the loaded variant-labelled values and read them back as priors.
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .schema = @as(u32, 3),
        .context_bucket = "floor-power-of-two-2048",
        .step_ms = loaded.step_ms,
        .@"77" = .{
            .tree = .{
                .generic = .{ .@"4" = loaded.prior(4, .tree, 77, .generic).? },
                .linear_gdn = .{ .@"4" = loaded.prior(4, .tree, 77, .linear_gdn).? },
            },
            .chain = .{ .generic = .{ .@"4" = loaded.prior(4, .chain, 77, .generic).? } },
        },
    }, .{});
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "prices.json", .data = encoded });
    var roundtrip = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer roundtrip.deinit();
    try std.testing.expectEqual(loaded.step_ms, roundtrip.step_ms);
    for ([_]Variant{ .generic, .linear_gdn }) |variant| {
        try std.testing.expectEqual(loaded.prior(4, .tree, 77, variant), roundtrip.prior(4, .tree, 77, variant));
    }
    try std.testing.expectEqual(loaded.prior(4, .chain, 77, .generic), roundtrip.prior(4, .chain, 77, .generic));
}

test "cold adaptive selection retains caller native width until its cost is known" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const tree = Tree.chain(30, &.{ 31, 32, 33, 34, 35, 36 }, 0.9, 0.9);
    const native = tree.subset(0b11111);
    var log: TreeEvaluation = .{};
    const cold = policy.chooseTreeBootstrapControlled(&tree, 81, 0, 16, 0, &native, &log, .generic);
    try std.testing.expectEqual(@as(usize, 5), cold.len);
    try std.testing.expectEqual(@as(usize, 1), log.len);
    try std.testing.expect(policy.prices.known(5, .tree, 81, .generic) == null);
    _ = policy.prices.observeKernel(5, 400, .tree, 81, .generic);
    const still_cold = policy.chooseTreeBootstrapControlled(&tree, 81, 0, 16, 0, &native, null, .generic);
    try std.testing.expectEqual(@as(usize, 5), still_cold.len);
    _ = policy.prices.observeKernel(5, 10, .tree, 81, .generic);
    _ = policy.chooseTreeBootstrapControlled(&tree, 81, 0, 16, 0, &native, &log, .generic);
    try std.testing.expectEqual(@as(usize, 1), log.len);
    try std.testing.expectEqual(@as(u32, 5), log.entries[0].rows);
    const forced = policy.chooseTreeBootstrapControlled(&tree, 82, 0, 16, 2, &native, null, .generic);
    try std.testing.expectEqual(@as(usize, 2), forced.len);
}

const Fingerprint = struct {
    hash: std.crypto.hash.sha2.Sha256 = std.crypto.hash.sha2.Sha256.init(.{}),

    fn word(self: *Fingerprint, value: u64) void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        self.hash.update(&bytes);
    }
    fn float(self: *Fingerprint, value: f32) void {
        self.word(@as(u32, @bitCast(value)));
    }
    fn priceKey(self: *Fingerprint, key: PriceKey) void {
        self.word(@bitCast(key.class));
        self.word(key.rows);
        self.word(@intFromEnum(key.kind));
        self.word(@intFromEnum(key.variant));
    }
    fn window(self: *Fingerprint, sample: Settled) void {
        self.word(sample.len);
        self.word(sample.next);
        for (sample.values) |value| self.float(value);
    }
    fn finish(self: *Fingerprint) [32]u8 {
        var digest: [32]u8 = undefined;
        self.hash.final(&digest);
        return digest;
    }
};

test "diagnostic identity includes cold markers and is independent of insertion order" {
    var left = Policy.init(PriceModel.init(std.testing.allocator));
    defer left.deinit();
    var right = Policy.init(PriceModel.init(std.testing.allocator));
    defer right.deinit();
    left.prices.observe(2, 8, .tree, 65, .generic);
    left.prices.observe(4, 11, .chain, 66, .generic);
    right.prices.observe(4, 11, .chain, 66, .generic);
    right.prices.observe(2, 8, .tree, 65, .generic);
    try std.testing.expectEqual(left.diagnosticFingerprint(), right.diagnosticFingerprint());
    var copy = try left.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(left.diagnosticFingerprint(), copy.diagnosticFingerprint());
    _ = copy.prices.observeKernel(7, 90, .tree, 65, .linear_gdn);
    try std.testing.expect(!std.mem.eql(u8, &left.diagnosticFingerprint(), &copy.diagnosticFingerprint()));
}

test "already projected pruning retains excluded sibling mass and raw evidence" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var tree = Tree.init(300);
    _ = tree.add(0, 301, 0.2, .look, 0.8);
    _ = tree.add(1, 302, 0.1, .look, 0.7);
    tree.scores_calibrated = true;
    const selected = policy.chooseTreeControlled(&tree, 0, 0, 32, 3, null, .generic);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), selected.scores[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), selected.scores[2], 0.00001);
    try std.testing.expectEqual(@as(f32, 0.8), selected.look_confidence[1]);
    try std.testing.expectEqual(@as(f32, 0.7), selected.look_confidence[2]);
    try std.testing.expect(selected.scores_calibrated);
}

test "paid preparation is amortized over the native round yield" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    controller.prices.observe(1, 4, .chain, 0, .generic);
    controller.prices.observe(2, 4, .chain, 0, .generic);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const candidate = Tree.chain(0, &.{7}, 0.5, 0.5);
    const selected = try controller.chooseLookPreparedControlled(&candidate, 0, 4, &out, 0,
        .{ .measured = .{ .round_ms = 4, .expected_tokens = 2 } }, null);
    try std.testing.expectEqual(Choice.Source.spine, selected.source);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), selected.value, 0.00001);
}

test "native remaining cost preserves paired round yield when preparation changes" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    _ = controller.observeSpecRemainingProbe(4, 2, 0, 0);
    _ = controller.observeSpecRemainingProbe(12, 4, 0, 0);
    const remaining = controller.specRemainingFor(0).?;
    try std.testing.expectApproxEqAbs(@as(f32, 8), remaining.round_ms, 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 3), remaining.expected_tokens, 0.00001);
    const route = controller.routeAgainstPrepared(0, 3.5, 4, null);
    try std.testing.expectApproxEqAbs(@as(f32, 4), route.assistant_ms.?, 0.00001);
    try std.testing.expect(route.skip);
}

test "lookup width learning escapes narrow-price starvation without synthetic timing" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var selected: std.ArrayList(u32) = .empty;
    defer selected.deinit(std.testing.allocator);
    const raw = Tree.chain(0, &.{ 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84 }, 1, 0.9);
    const candidate = policy.calibrated(&raw);
    const fallback: LookFallback = .{ .measured = .{ .round_ms = 90, .expected_tokens = 2 } };
    try std.testing.expect(policy.observeSpecRemainingProbe(90, 2, 0, 0));
    policy.prices.observe(3, 84, .chain, 0, .generic);
    const locked = try policy.chooseLookPreparedControlled(&candidate, 0, 1, &selected, 0, fallback, null);
    try std.testing.expectEqual(@as(u32, 2), locked.rows);
    const first = policy.reserveLookWidthTrial(&candidate, 0, 2, 1, policy.specRemainingFor(0), null, null).?;
    try std.testing.expectEqual(@as(u32, 15), first);
    try std.testing.expect(policy.prices.known(first, .chain, 0, .generic) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 1, policy.specRemainingFor(0), null, null) == null);
    try std.testing.expect(!policy.prices.observeKernel(first, 400, .chain, 0, .generic));
    try std.testing.expectEqual(@as(?u32, 15), policy.reserveLookWidthTrial(&candidate, 0, 2, 1, policy.specRemainingFor(0), null, null));
    try std.testing.expect(policy.prices.known(first, .chain, 0, .generic) == null);
    try std.testing.expect(policy.prices.observeKernel(first, 140, .chain, 0, .generic));
    const learned = try policy.chooseLookPreparedControlled(&candidate, 0, 1, &selected, 0, fallback, null);
    try std.testing.expectEqual(@as(u32, 14), learned.rows);
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 1, policy.specRemainingFor(0), null, null) == null);
}

test "lookup width trials respect floors caps declining yield and failed retries" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const raw = Tree.chain(0, &.{ 11, 12, 13, 14, 15, 16, 17, 18 }, 1, 1);
    const strong = policy.calibrated(&raw);
    const weak_raw = Tree.chain(0, &.{ 11, 12, 13, 14, 15, 16, 17, 18 }, 0.05, 1);
    const weak = policy.calibrated(&weak_raw);
    policy.prices.observe(3, 84, .chain, 0, .generic);
    try std.testing.expect(policy.reserveLookWidthTrial(&strong, 0, 2, 0, policy.specRemainingFor(0), null, null) == null);
    try std.testing.expect(policy.observeSpecRemainingProbe(90, 2, 0, 0));
    try std.testing.expect(policy.reserveLookWidthTrial(&strong, 0, 4, 0, policy.specRemainingFor(0), null, null) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&strong, 2048, 2, 0, policy.specRemainingFor(2048), null, null) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&weak, 0, 2, 0, policy.specRemainingFor(0), null, null) == null);
    try std.testing.expectEqual(@as(?u32, 9), policy.reserveLookWidthTrial(&strong, 0, 2, 0, policy.specRemainingFor(0), null, null));
    try std.testing.expect(!policy.prices.observeKernel(9, 400, .chain, 0, .generic));
    var shortened = strong;
    shortened.len = 8;
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(policy.reserveLookWidthTrial(&shortened, 0, 2, 0, policy.specRemainingFor(0), null, null) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&weak, 0, 2, 0, policy.specRemainingFor(0), null, null) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    var copy = try policy.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(@as(?u32, 9), copy.reserveLookWidthTrial(&strong, 0, 2, 0, copy.specRemainingFor(0), null, null));
    try std.testing.expect(!copy.prices.observeKernel(9, std.math.nan(f32), .chain, 0, .generic));
    try std.testing.expect(copy.reserveLookWidthTrial(&strong, 0, 2, 0, copy.specRemainingFor(0), null, null) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    try std.testing.expectEqual(@as(?u32, 9), policy.reserveLookWidthTrial(&strong, 0, 2, 0, policy.specRemainingFor(0), null, null));
}

test "upper width exploration is one shape and one real cold retry" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const native: Shape = .{ .rows = 4, .variant = .generic };
    const upper: Shape = .{ .rows = 12, .variant = .generic };
    try std.testing.expect(policy.prices.upperTrialWidth(native, upper, .tree, 0) == null);
    policy.prices.observe(4, 50, .tree, 0, .generic);
    try std.testing.expectEqual(@as(?u32, 12), policy.prices.upperTrialWidth(native, upper, .tree, 0));
    try std.testing.expect(policy.prices.reserveUpperTrial(native, upper, .tree, 0));
    try std.testing.expect(!policy.prices.reserveUpperTrial(native, upper, .tree, 0));
    try std.testing.expect(!policy.prices.reserveUpperTrial(native, .{ .rows = 10, .variant = .generic }, .tree, 0));
    try std.testing.expect(policy.prices.known(12, .tree, 0, .generic) == null);
    try std.testing.expect(!policy.prices.observeKernel(12, 500, .tree, 0, .generic));
    try std.testing.expect(policy.prices.upperTrialWidth(native, .{ .rows = 11, .variant = .generic }, .tree, 0) == null);
    try std.testing.expect(!policy.prices.reserveUpperTrial(native, .{ .rows = 12, .variant = .linear_gdn }, .tree, 0));
    var copy = try policy.clone(std.testing.allocator);
    defer copy.deinit();
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(copy.prices.reserveUpperTrial(native, upper, .tree, 0));
    try std.testing.expect(!std.mem.eql(u8, &before, &copy.diagnosticFingerprint()));
    try std.testing.expectEqualDeep(before, policy.diagnosticFingerprint());
    try std.testing.expect(copy.prices.upperTrialWidth(native, .{ .rows = 16, .variant = .generic }, .tree, 0) == null);
    try std.testing.expect(!copy.prices.observeKernel(12, std.math.inf(f32), .tree, 0, .generic));
    try std.testing.expect(!copy.prices.reserveUpperTrial(native, upper, .tree, 0));
    try std.testing.expect(policy.prices.reserveUpperTrial(native, upper, .tree, 0));
    try std.testing.expect(policy.prices.observeKernel(12, 95, .tree, 0, .generic));
    try std.testing.expectEqual(@as(?f32, 95), policy.prices.known(12, .tree, 0, .generic));
    try std.testing.expect(policy.prices.upperTrialWidth(native, .{ .rows = 16, .variant = .generic }, .tree, 0) == null);
}

test "a naturally cold width has only one remaining trial attempt" {
    var prices = PriceModel.init(std.testing.allocator);
    defer prices.deinit();
    const native: Shape = .{ .rows = 3, .variant = .generic };
    const upper: Shape = .{ .rows = 15, .variant = .generic };
    prices.observe(3, 84, .chain, 0, .generic);
    try std.testing.expect(!prices.observeKernel(15, 300, .chain, 0, .generic));
    try std.testing.expect(prices.reserveUpperTrial(native, upper, .chain, 0));
    try std.testing.expect(!prices.observeKernel(15, std.math.nan(f32), .chain, 0, .generic));
    try std.testing.expect(!prices.reserveUpperTrial(native, upper, .chain, 0));
    try std.testing.expect(prices.upperTrialWidth(native, upper, .chain, 0) == null);
}

test "strong suffix history cannot wash out weak suffix mismatches" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var strong = Tree.chain(0, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 }, 1, 1);
    strong.look_order = @splat(32);
    var weak = strong;
    weak.look_order = @splat(1);
    // Successful templates continue arriving between unrelated low-order misses.
    for (0..8) |_| {
        for (0..32) |_| controller.observeChainAccepted(&strong, 14, 14);
        controller.observeChainAccepted(&weak, 14, 0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), controller.cal.q(.look, 1, 1), 0.00001);
    try std.testing.expectEqual(@as(f32, 1), controller.cal.q(.look, 1, 32));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.q(.look_continuation, 1, 1));
    // The same measured action remains useful for the strong context only.
    controller.prices.observe(1, 50, .chain, 0, .generic);
    for (2..16) |width| controller.prices.observe(@intCast(width), 150, .chain, 0, .generic);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const fallback: LookFallback = .{ .measured = .{ .round_ms = 100, .expected_tokens = 4 } };
    const weak_choice = try controller.chooseLookPreparedControlled(&weak, 0, 0, &out, 0, fallback, null);
    try std.testing.expectEqual(Choice.Source.spine, weak_choice.source);
    const strong_choice = try controller.chooseLookPreparedControlled(&strong, 0, 0, &out, 0, fallback, null);
    try std.testing.expectEqual(Choice.Source.look, strong_choice.source);
    try std.testing.expectEqual(@as(u32, 14), strong_choice.rows);
    var copy = try controller.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqualSlices(u8, &controller.diagnosticFingerprint(), &copy.diagnosticFingerprint());
    copy.cal.learn(.look, 1, 2, false);
    try std.testing.expect(!std.mem.eql(u8, &controller.diagnosticFingerprint(), &copy.diagnosticFingerprint()));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.q(.look, 1, 2));
}

test "execution width censors the tail of a fully accepted carried chain" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var tree = Tree.chain(10, &.{ 11, 12, 13, 14 }, 0.5, 0.5);
    tree.look_order = @splat(3);
    tree.look_confidence[3] = 0.8;
    controller.observeChainAccepted(&tree, 2, 2);
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look_continuation));
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), controller.cal.q(.look_continuation, 0.5, 3), 0.00001);
    try std.testing.expectEqual(@as(f32, 0.8), controller.cal.q(.look_continuation, 0.8, 3));
    const learned = controller.diagnosticFingerprint();
    controller.observeChainAccepted(&tree, 0, 4);
    try std.testing.expectEqual(learned, controller.diagnosticFingerprint());
    controller.observeChainAccepted(&tree, 1, 0);
    try std.testing.expectEqual(@as(f32, 2), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look_continuation));
}

test "request lookup trust needs reached success and resets independently of pooled history" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var tree = Tree.chain(1, &.{ 2, 3, 4 }, 1, 1);
    tree.look_order = @splat(32);
    for (0..200) |_| controller.observeLookupTree(&tree, &.{ 2, 3, 4 }, null);
    var request: RequestTrust = .{};
    try std.testing.expectEqual(@as(f32, 0), request.q(.look, 1));
    controller.observeLookupTree(&tree, &.{99}, &request);
    try std.testing.expectEqual(@as(f32, 0), request.q(.look, 1));
    try std.testing.expectEqual(@as(f32, 0), request.q(.look_continuation, 1));
    // Continuation support cannot silently establish entry support.
    request.learn(.look_continuation, 1, true);
    try std.testing.expectEqual(@as(f32, 0), request.q(.look, 1));
    controller.observeLookupTree(&tree, &.{2}, &request);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), request.q(.look, 1), 0.00001);
    const next_request: RequestTrust = .{};
    try std.testing.expectEqual(@as(f32, 0), next_request.q(.look, 1));
    try std.testing.expect(!std.mem.eql(u8, &request.diagnosticFingerprint(), &next_request.diagnosticFingerprint()));
}

test "lookup request support enables bounded reuse only after actual native width pricing" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    var raw = Tree.chain(0, &tokens, 1, 1);
    raw.look_order = @splat(8);
    const offer = controller.chooseTreeControlled(&raw, 0, 0, 15, 15, null, .generic);
    var request: RequestTrust = .{};
    _ = controller.observeSpecRemainingProbe(100, 2, 0, 1);
    _ = controller.observeSpecRemainingProbe(90, 2, 0, 1);
    const cold = controller.restrictLookup(&offer, &request);
    try std.testing.expectEqual(@as(usize, 15), cold.len);
    for (cold.scores[1..cold.len]) |reach| try std.testing.expectEqual(@as(f32, 0), reach);
    try std.testing.expect(controller.reserveLookWidthTrial(&cold, 0, 2, 0, controller.specRemainingFor(0), null, null) == null);
    // One genuine shape sample remains cold. Different second widths do not
    // manufacture a warm exact reference or reopen the bounded opportunities.
    try std.testing.expect(!controller.prices.observeKernel(3, 70, .chain, 0, .generic));
    try std.testing.expect(!controller.prices.observeKernel(4, 80, .chain, 0, .generic));
    for (0..8) |_| controller.observeLookupTree(&offer, &tokens, &request);
    const supported = controller.restrictLookup(&offer, &request);
    try std.testing.expect(controller.reserveLookWidthTrial(&supported, 0, 2, 0, controller.specRemainingFor(0), null, null) == null);
    try std.testing.expect(controller.prices.observeKernel(3, 60, .chain, 0, .generic));
    const fresh: RequestTrust = .{};
    const unsupported = controller.restrictLookup(&offer, &fresh);
    try std.testing.expect(controller.reserveLookWidthTrial(&unsupported, 0, 2, 0, controller.specRemainingFor(0), null, null) == null);
    try std.testing.expectEqual(@as(u32, 15), controller.reserveLookWidthTrial(&supported, 0, 2, 0, controller.specRemainingFor(0), null, null).?);
    var expected: f32 = 0;
    for (supported.scores[0..supported.len]) |reach| expected += reach;
    try std.testing.expect(expected > 10);
}

test "lookup bounded shadow offer preserves nonoverlapping executed chain evidence" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var trust: RequestTrust = .{};
    var selected = Tree.chain(10, &.{ 11, 12, 13, 14 }, 0.8, 0.6);
    @memset(selected.look_order[1..selected.len], 6);
    var offered = Tree.chain(10, &.{11}, 0.8, 0.6);
    // Equal token 12 under another prefix is not evidence for 11 -> 12.
    _ = offered.add(0, 99, 0.2, .look, 0.2).?;
    _ = offered.add(2, 12, 0.1, .look, 0.3).?;
    @memset(offered.look_order[1..offered.len], 64);
    const reached = [_]u32{ 11, 12, 88 };
    controller.observeLookupTree(&offered, &reached, &trust);
    controller.observeLookupChainRemainder(&selected, &offered, &reached, &trust);
    try std.testing.expectEqual(@as(f32, 2), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 2), controller.cal.samples(.look_continuation));
    try std.testing.expectApproxEqAbs(@as(f32, 0.84), controller.cal.q(.look, 0.8, 64), 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), trust.q(.look, 0.8), 0.00001);
    // One continuation hit and its first rejection, never the unreached 14.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), trust.q(.look_continuation, 0.6), 0.00001);
    try std.testing.expectEqual(@as(f32, 0), trust.q(.look_continuation, 0.3));
}

test "settled prices and nearest width extrapolation never cross kernel variants" {
    var prices = PriceModel.init(std.testing.allocator);
    defer prices.deinit();
    try std.testing.expect(!prices.observeKernel(9, 900, .tree, 0, .generic));
    try std.testing.expect(prices.observeKernel(9, 90, .tree, 0, .generic));
    try std.testing.expectEqual(@as(?f32, 90), prices.known(9, .tree, 0, .generic));
    try std.testing.expect(prices.known(9, .tree, 0, .linear_gdn) == null);
    // No linear timing exists: retain the unmeasured default, not generic 90ms.
    try std.testing.expectEqual(@as(f32, 20), prices.price(10, .tree, 0, .linear_gdn));
    try std.testing.expectEqual(@as(f32, 180), prices.price(8, .tree, 0, .generic));
    try std.testing.expect(!prices.observeKernel(9, 600, .tree, 0, .linear_gdn));
    try std.testing.expect(prices.known(9, .tree, 0, .linear_gdn) == null);
    try std.testing.expect(prices.observeKernel(9, 9, .tree, 0, .linear_gdn));
    try std.testing.expectEqual(@as(?f32, 9), prices.known(9, .tree, 0, .linear_gdn));
    try std.testing.expectEqual(@as(?f32, 90), prices.known(9, .tree, 0, .generic));
    try std.testing.expectEqual(@as(f32, 18), prices.price(8, .tree, 0, .linear_gdn));
    try std.testing.expectEqual(@as(f32, 180), prices.price(8, .tree, 0, .generic));
    try prices.priors.put(.{ .class = 1, .rows = 9, .kind = .tree, .variant = .generic }, 90);
    try std.testing.expect(prices.prior(9, .tree, 1, .linear_gdn) == null);
    try std.testing.expectEqual(@as(f32, 20), prices.price(10, .tree, 1, .linear_gdn));
}

test "ranked prefixes price their remapped shape and preserve forced geometry" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var tree = Tree.init(100);
    _ = tree.add(0, 101, 0.1, .draft, 0.1);
    _ = tree.add(0, 102, 0.9, .draft, 0.9);
    for (3..10) |row| _ = tree.add(row - 1, @intCast(100 + row), 0.9, .draft, 1);
    try std.testing.expectEqual(Variant.generic, tree.variant(.linear_gdn));
    for (1..11) |width| policy.prices.observe(@intCast(width), if (width == 9) 0.01 else 100, .tree, 0, .generic);
    policy.prices.observe(9, 1, .tree, 0, .linear_gdn);
    var log: TreeEvaluation = .{};
    const chosen = policy.chooseTreeControlled(&tree, 0, 0, 32, 0, &log, .linear_gdn);
    try std.testing.expectEqual(@as(usize, 9), chosen.len);
    try std.testing.expectEqual(Variant.linear_gdn, chosen.variant(.linear_gdn));
    try std.testing.expectEqual(Variant.generic, log.entries[7].variant);
    try std.testing.expectEqual(Variant.linear_gdn, log.entries[8].variant);
    try std.testing.expectEqual(@as(f32, 1), log.entries[8].verify_ms);
    try std.testing.expectEqual(Variant.generic, log.entries[9].variant);
    try std.testing.expectEqual(@as(f32, 100), log.entries[9].verify_ms);
    try std.testing.expectEqualSlices(u32, &.{ 100, 102, 103, 104, 105, 106, 107, 108, 109 }, chosen.tokens[0..chosen.len]);
    for (chosen.parents[1..chosen.len], 1..) |parent, row| try std.testing.expectEqual(@as(i32, @intCast(row - 1)), parent);
    const before = policy.chooseTreeControlled(&tree, 0, 0, 32, 9, null, .linear_gdn);
    for (0..7) |_| policy.prices.observe(9, 1000, .tree, 0, .linear_gdn);
    const after = policy.chooseTreeControlled(&tree, 0, 0, 32, 9, &log, .linear_gdn);
    try std.testing.expectEqualDeep(before, after);
    try std.testing.expectEqual(@as(usize, 1), log.len);
    try std.testing.expectEqual(@as(f32, 1000), log.entries[0].verify_ms);
    _ = policy.chooseTreeControlled(&tree, 0, 0, 32, 9, &log, .generic);
    try std.testing.expectEqual(Variant.generic, log.entries[0].variant);
    try std.testing.expectEqual(@as(f32, 0.01), log.entries[0].verify_ms);
}

test "native bootstrap requires the actual selected prefix variant" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var tree = Tree.chain(0, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, 0.9, 1);
    _ = tree.add(0, 9, 0.1, .draft, 0.1);
    const native = tree.subset(0x1ff);
    policy.prices.observe(1, 0.01, .tree, 0, .generic);
    policy.prices.observe(9, 90, .tree, 0, .generic);
    var log: TreeEvaluation = .{};
    const cold = policy.chooseTreeBootstrapControlled(&tree, 0, 0, 32, 0, &native, &log, .linear_gdn);
    try std.testing.expectEqual(@as(usize, 9), cold.len);
    try std.testing.expectEqual(Variant.linear_gdn, cold.variant(.linear_gdn));
    try std.testing.expectEqual(@as(usize, 1), log.len);
    try std.testing.expect(!policy.prices.observeKernel(9, 900, .tree, 0, .linear_gdn));
    const still_cold = policy.chooseTreeBootstrapControlled(&tree, 0, 0, 32, 0, &native, null, .linear_gdn);
    try std.testing.expectEqual(@as(usize, 9), still_cold.len);
    try std.testing.expect(policy.prices.observeKernel(9, 9, .tree, 0, .linear_gdn));
    _ = policy.chooseTreeBootstrapControlled(&tree, 0, 0, 32, 0, &native, &log, .linear_gdn);
}

test "upper width eligibility uses both actual variants without renewing the trial budget" {
    var prices = PriceModel.init(std.testing.allocator);
    defer prices.deinit();
    const native: Shape = .{ .rows = 9, .variant = .linear_gdn };
    const upper: Shape = .{ .rows = 12, .variant = .linear_gdn };
    prices.observe(9, 90, .tree, 0, .generic);
    prices.observe(12, 120, .tree, 0, .generic);
    try std.testing.expect(prices.upperTrialWidth(native, upper, .tree, 0) == null);
    try std.testing.expect(!prices.reserveUpperTrial(native, upper, .tree, 0));
    prices.observe(9, 9, .tree, 0, .linear_gdn);
    try std.testing.expectEqual(@as(?u32, 12), prices.upperTrialWidth(native, upper, .tree, 0));
    try std.testing.expect(prices.reserveUpperTrial(native, upper, .tree, 0));
    try std.testing.expect(!prices.observeKernel(12, 1200, .tree, 0, .linear_gdn));
    try std.testing.expectEqual(@as(?u32, 12), prices.upperTrialWidth(native, upper, .tree, 0));
    try std.testing.expect(prices.reserveUpperTrial(native, upper, .tree, 0));
    try std.testing.expect(prices.upperTrialWidth(native, .{ .rows = 16, .variant = .generic }, .tree, 0) == null);
    try std.testing.expect(!prices.reserveUpperTrial(native, .{ .rows = 16, .variant = .generic }, .tree, 0));
    // A reference and wider action need not use the same kernel.
    prices.observe(8, 80, .tree, 1, .generic);
    const generic_native: Shape = .{ .rows = 8, .variant = .generic };
    try std.testing.expectEqual(@as(?u32, 12), prices.upperTrialWidth(generic_native, upper, .tree, 1));
    try std.testing.expect(prices.reserveUpperTrial(generic_native, upper, .tree, 1));
}

test "policy identity separates otherwise identical price and cold variants" {
    var generic = Policy.init(PriceModel.init(std.testing.allocator));
    defer generic.deinit();
    var linear = Policy.init(PriceModel.init(std.testing.allocator));
    defer linear.deinit();
    generic.prices.observe(9, 9, .tree, 0, .generic);
    linear.prices.observe(9, 9, .tree, 0, .linear_gdn);
    try std.testing.expect(!std.mem.eql(u8, &generic.diagnosticFingerprint(), &linear.diagnosticFingerprint()));
    var copy = try linear.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(linear.diagnosticFingerprint(), copy.diagnosticFingerprint());
    // Equalize settled state, then isolate the prior-key contribution.
    generic.prices.observe(9, 9, .tree, 0, .linear_gdn);
    linear.prices.observe(9, 9, .tree, 0, .generic);
    try std.testing.expectEqual(generic.diagnosticFingerprint(), linear.diagnosticFingerprint());
    try generic.prices.priors.put(.{ .class = 0, .rows = 10, .kind = .tree, .variant = .generic }, 10);
    try linear.prices.priors.put(.{ .class = 0, .rows = 10, .kind = .tree, .variant = .linear_gdn }, 10);
    try std.testing.expect(!std.mem.eql(u8, &generic.diagnosticFingerprint(), &linear.diagnosticFingerprint()));
    generic.prices.priors.clearRetainingCapacity();
    linear.prices.priors.clearRetainingCapacity();
    try std.testing.expect(!generic.prices.observeKernel(11, 100, .tree, 0, .generic));
    try std.testing.expect(!linear.prices.observeKernel(11, 100, .tree, 0, .linear_gdn));
    try std.testing.expect(!std.mem.eql(u8, &generic.diagnosticFingerprint(), &linear.diagnosticFingerprint()));
}

test "ordinary lookup chain pricing stays generic above the tree dispatch threshold" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const tree = Tree.chain(0, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, 1, 1);
    policy.prices.observe(9, 90, .chain, 0, .generic);
    policy.prices.observe(9, 1, .chain, 0, .linear_gdn);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const chosen = try policy.chooseLookPreparedControlled(&tree, 0, 0, &out, 9, .unknown, null);
    try std.testing.expectEqual(@as(u32, 8), chosen.rows);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), chosen.value, 0.00001);
}

test "standalone lookup escapes root-only prices with one shape and one cold retry" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var candidate = Tree.chain(10, &.{ 11, 12, 13, 14, 15 }, 1, 1);
    candidate.scores_calibrated = true;
    var selected: std.ArrayList(u32) = .empty;
    defer selected.deinit(std.testing.allocator);
    try std.testing.expect(policy.reservePlainWidthTrial(&candidate, 0) == null);
    // Tree-verifier prices must not unlock a different chain verifier.
    policy.prices.observe(1, 50, .tree, 0, .generic);
    try std.testing.expect(policy.reservePlainWidthTrial(&candidate, 0) == null);
    try std.testing.expect(!policy.prices.observeKernel(1, 50, .chain, 0, .generic));
    try std.testing.expect(policy.reservePlainWidthTrial(&candidate, 0) == null);
    try std.testing.expect(policy.prices.observeKernel(1, 50, .chain, 0, .generic));
    policy.observePlain(50, 0);
    const locked = try policy.chooseLookPreparedControlled(&candidate, 0, 0.2, &selected, 0, .plain, null);
    try std.testing.expectEqual(@as(u32, 0), locked.rows);
    const first = policy.reservePlainWidthTrial(&candidate, 0).?;
    try std.testing.expectEqual(@as(u32, 6), first);
    const forced = try policy.chooseLookPreparedControlled(&candidate, 0, 0.2, &selected, first, .plain, null);
    try std.testing.expectEqual(@as(u32, 5), forced.rows);
    try std.testing.expect(policy.reservePlainWidthTrial(&candidate, 0) == null);
    try std.testing.expect(!policy.prices.observeKernel(first, 100, .chain, 0, .generic));
    var shorter = candidate;
    shorter.len = 5;
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(policy.reservePlainWidthTrial(&shorter, 0) == null);
    var empty = Tree.init(10);
    empty.scores_calibrated = true;
    try std.testing.expect(policy.reservePlainWidthTrial(&empty, 0) == null);
    var no_gain = candidate;
    @memset(no_gain.scores[1..no_gain.len], 0);
    try std.testing.expect(policy.reservePlainWidthTrial(&no_gain, 0) == null);
    try std.testing.expectEqualDeep(before, policy.diagnosticFingerprint());
    try std.testing.expectEqual(@as(?u32, 6), policy.reservePlainWidthTrial(&candidate, 0));
    try std.testing.expect(policy.prices.observeKernel(first, 60, .chain, 0, .generic));
    const learned = try policy.chooseLookPreparedControlled(&candidate, 0, 0.2, &selected, 0, .plain, null);
    try std.testing.expectEqual(@as(u32, 5), learned.rows);
    try std.testing.expect(policy.reservePlainWidthTrial(&candidate, 0) == null);
    var longer = Tree.chain(10, &.{ 11, 12, 13, 14, 15, 16, 17 }, 1, 1);
    longer.scores_calibrated = true;
    try std.testing.expect(policy.reservePlainWidthTrial(&longer, 0) == null);
}

test "adaptive widths never win using invented prices even with large paid preparation" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const tree = Tree.chain(0, &.{ 1, 2, 3, 4 }, 1, 1);
    policy.prices.observe(2, 10, .tree, 0, .generic);
    // Huge sunk preparation previously made unmeasured wider widths attractive
    // despite the extrapolation penalty. Only this actual measured action competes.
    var log: TreeEvaluation = .{};
    const chosen = policy.chooseTreeControlled(&tree, 0, 1000, 5, 0, &log, .generic);
    try std.testing.expectEqual(@as(usize, 2), chosen.len);
    try std.testing.expectEqual(@as(usize, 1), log.len);
    policy.prices.observe(4, 11, .tree, 0, .generic);
    const plateau = policy.chooseTreeControlled(&tree, 0, 1000, 5, 0, null, .generic);
    try std.testing.expectEqual(@as(usize, 4), plateau.len);
    const bounded = policy.chooseTreeControlled(&tree, 0, 1000, 3, 0, null, .generic);
    try std.testing.expectEqual(@as(usize, 2), bounded.len);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const lookup = try policy.chooseLookPreparedControlled(&tree, 0, 1000, &out, 0,
        .{ .measured = .{ .round_ms = 100, .expected_tokens = 1 } }, null);
    try std.testing.expectEqual(Choice.Source.spine, lookup.source);
}

test "lookup feedback crosses pure neural ancestors without learning them" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var tree = Tree.init(10);
    _ = tree.add(0, 11, 0.8, .draft, 0.8).?;
    _ = tree.add(1, 12, 0.64, .look, 0.8).?;
    _ = tree.add(2, 13, 0.32, .look, 0.5).?;
    var trust: RequestTrust = .{};
    controller.observeLookupTree(&tree, &.{ 11, 12, 99 }, &trust);
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look_continuation));
    try std.testing.expectApproxEqAbs(@as(f32, 0.84), controller.cal.q(.look, 0.8, 0), 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), controller.cal.q(.look_continuation, 0.5, 0), 0.00001);
    try std.testing.expect(trust.q(.look, 0.8) > 0);
    try std.testing.expectEqual(@as(f32, 0), trust.q(.look_continuation, 0.5));
}

test "lookup growing copy retains support while exact-order calibration bounds admission" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var offer = Tree.chain(10, &.{ 11, 12, 13 }, 1, 1);
    @memset(offer.look_order[1..offer.len], 8);
    var request: RequestTrust = .{};
    var selected: std.ArrayList(u32) = .empty;
    defer selected.deinit(std.testing.allocator);
    controller.prices.observe(4, 40, .chain, 0, .generic);
    const native: LookFallback = .{ .measured = .{ .round_ms = 100, .expected_tokens = 3 } };
    var scored = controller.calibrated(&offer);
    const cold = controller.restrictLookup(&scored, &request);
    const declined = try controller.chooseLookPreparedControlled(&cold, 0, 1, &selected, 0, native, null);
    try std.testing.expectEqual(@as(u32, 0), declined.rows);
    controller.observeLookupTree(&offer, &.{ 11, 12, 13 }, &request);
    for ([_]u32{ 11, 14, 17 }) |order| {
        @memset(offer.look_order[1..offer.len], order);
        scored = controller.calibrated(&offer);
        const supported = controller.restrictLookup(&scored, &request);
        const choice = try controller.chooseLookPreparedControlled(&supported, 0, 1, &selected, 0, native, null);
        try std.testing.expectEqualSlices(u32, &.{ 11, 12, 13 }, choice.tokens);
        controller.observeLookupTree(&offer, &.{ 11, 12, 13 }, &request);
    }
    // Pooling is a support heuristic, not permission to ignore an order's misses.
    @memset(offer.look_order[1..offer.len], 6);
    for (0..100) |_| controller.observeLookupTree(&offer, &.{99}, null);
    scored = controller.calibrated(&offer);
    const weak = controller.restrictLookup(&scored, &request);
    const rejected = try controller.chooseLookPreparedControlled(&weak, 0, 1, &selected, 0, native, null);
    try std.testing.expectEqual(@as(u32, 0), rejected.rows);
    const fresh: RequestTrust = .{};
    const reset = controller.restrictLookup(&scored, &fresh);
    try std.testing.expectEqual(@as(f32, 0), reset.scores[1]);
}

test "lookup request support keeps confidence populations and decay independent" {
    var request: RequestTrust = .{};
    const cold = request.diagnosticFingerprint();
    request.learn(.look, 0.8, true);
    try std.testing.expect(!std.mem.eql(u8, &cold, &request.diagnosticFingerprint()));
    try std.testing.expectEqual(@as(f32, 0), request.q(.look, 0.6));
    try std.testing.expectEqual(@as(f32, 0), request.q(.look_continuation, 0.8));
    for (0..127) |_| request.learn(.look, 0.8, false);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 130.0), request.q(.look, 0.8), 0.00001);
    request.learn(.look, 0.8, true);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5 / 67.0), request.q(.look, 0.8), 0.00001);
    request = .{};
    try std.testing.expectEqual(cold, request.diagnosticFingerprint());
}

test "lookup exploration rejects a slower optimistic rate before reserving" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var candidate = Tree.chain(0, &.{ 1, 2, 3 }, 1, 1);
    candidate.scores_calibrated = true;
    policy.prices.observe(2, 150, .chain, 0, .generic);
    // A stale shared probe favors lookup; only caller-current economics count.
    try std.testing.expect(policy.observeSpecRemainingProbe(96, 1, 0, 0));
    var evidence: ?Policy.LookWidthTrialEvidence = null;
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 1, null, &evidence, null) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 1, .{ .round_ms = 50, .expected_tokens = 3 }, &evidence, null) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    try std.testing.expectEqual(@as(u32, 2), evidence.?.reference_rows);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 151.0), evidence.?.optimistic_rate, 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / 51.0), evidence.?.native_rate, 0.000001);
    try std.testing.expectEqual(@as(?u32, 4), policy.reserveLookWidthTrial(&candidate, 0, 2, 1, .{ .round_ms = 200, .expected_tokens = 3 }, &evidence, null));
    try std.testing.expect(policy.prices.known(4, .chain, 0, .generic) == null);
    try std.testing.expect(!policy.prices.observeKernel(4, 400, .chain, 0, .generic));
    const cold = policy.diagnosticFingerprint();
    // A real cold execution permits a retry, but not against a faster incumbent.
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 1, .{ .round_ms = 50, .expected_tokens = 3 }, null, null) == null);
    try std.testing.expectEqual(cold, policy.diagnosticFingerprint());
    try std.testing.expectEqual(@as(?u32, 4), policy.reserveLookWidthTrial(&candidate, 0, 2, 1, .{ .round_ms = 200, .expected_tokens = 3 }, null, null));
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 1, .{ .round_ms = 200, .expected_tokens = 3 }, null, null) == null);
    try std.testing.expect(policy.prices.known(4, .chain, 0, .generic) == null);
    try std.testing.expect(policy.prices.observeKernel(4, 170, .chain, 0, .generic));
    try std.testing.expectApproxEqAbs(@as(f32, 170), policy.prices.known(4, .chain, 0, .generic).?, 0.00001);
}

test "lookup exploration responds to the exact narrow reference cost" {
    for ([_]f32{ 150, 40 }) |reference_ms| {
        var policy = Policy.init(PriceModel.init(std.testing.allocator));
        defer policy.deinit();
        var candidate = Tree.chain(0, &.{ 1, 2, 3 }, 1, 1);
        candidate.scores_calibrated = true;
        policy.prices.observe(2, reference_ms, .chain, 0, .generic);
        const before = policy.diagnosticFingerprint();
        const trial = policy.reserveLookWidthTrial(&candidate, 0, 2, 1, .{ .round_ms = 100, .expected_tokens = 3 }, null, null);
        if (reference_ms == 150) {
            try std.testing.expect(trial == null);
            try std.testing.expectEqual(before, policy.diagnosticFingerprint());
        } else {
            try std.testing.expectEqual(@as(?u32, 4), trial);
        }
        try std.testing.expect(policy.prices.known(4, .chain, 0, .generic) == null);
    }
}

test "lookup exploration admits fewer tokens when rate wins and charges paid preparation" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var candidate = Tree.chain(0, &.{ 1, 2, 3 }, 1, 1);
    candidate.scores_calibrated = true;
    @memcpy(candidate.scores[0..4], &[_]f32{ 1, 0.5, 0.3, 0.2 });
    policy.prices.observe(2, 40, .chain, 0, .generic);
    const native: RemainingRound = .{ .round_ms = 200, .expected_tokens = 3 };
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 1000, native, null, null) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    try std.testing.expectEqual(@as(?u32, 4), policy.reserveLookWidthTrial(&candidate, 0, 2, 1, native, null, null));
    try std.testing.expect(policy.prices.known(4, .chain, 0, .generic) == null);
}

test "MTP preparation excludes first actual sample independently in every bounded cell" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const initial = policy.diagnosticFingerprint();
    for ([_]f32{ 0, -1, std.math.nan(f32), std.math.inf(f32) }) |invalid| {
        try std.testing.expect(!policy.observeMtpPreparation(2, 0, .cold, invalid));
    }
    for ([_]u32{ 0, 16, std.math.maxInt(u32) }) |rows|
        try std.testing.expect(!policy.observeMtpPreparation(rows, 0, .cold, 1));
    for ([_]i64{ -1, 1, 2047, 3072, 524288 }) |class|
        try std.testing.expect(!policy.observeMtpPreparation(2, class, .cold, 1));
    try std.testing.expectEqual(initial, policy.diagnosticFingerprint());
    for (mtp_contexts) |class| {
        for ([_]LookupPreparationScope{ .cold, .carried }) |scope| {
            for (1..mtp_max_rows + 1) |width| {
                const rows: u32 = @intCast(width);
                const ms: f32 = @floatFromInt(width);
                try std.testing.expect(policy.knownMtpPreparation(rows, class, scope) == null);
                try std.testing.expect(!policy.observeMtpPreparation(rows, class, scope, 9999));
                try std.testing.expect(policy.knownMtpPreparation(rows, class, scope) == null);
                try std.testing.expect(policy.observeMtpPreparation(rows, class, scope, ms));
                try std.testing.expectEqual(@as(?f32, ms), policy.knownMtpPreparation(rows, class, scope));
            }
        }
    }
    // Context changes never evict earlier model-owned populations.
    try std.testing.expectEqual(@as(?f32, 2), policy.knownMtpPreparation(2, 0, .cold));
}

test "MTP preparation median rolls and clone preserves samples warmup and identity" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const initial = policy.diagnosticFingerprint();
    try std.testing.expect(!policy.observeMtpPreparation(4, 2048, .carried, 9999));
    const cold = policy.diagnosticFingerprint();
    try std.testing.expect(!std.mem.eql(u8, &initial, &cold));
    var copy = try policy.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(cold, copy.diagnosticFingerprint());
    for ([_]f32{ 7, 1, 6, 2, 5, 3, 4 }) |ms| {
        try std.testing.expect(policy.observeMtpPreparation(4, 2048, .carried, ms));
        try std.testing.expect(copy.observeMtpPreparation(4, 2048, .carried, ms));
    }
    try std.testing.expectEqual(@as(?f32, 4), policy.knownMtpPreparation(4, 2048, .carried));
    try std.testing.expectEqual(policy.diagnosticFingerprint(), copy.diagnosticFingerprint());
    for (0..7) |_| try std.testing.expect(copy.observeMtpPreparation(4, 2048, .carried, 8));
    try std.testing.expectEqual(@as(?f32, 8), copy.knownMtpPreparation(4, 2048, .carried));
    try std.testing.expectEqual(@as(?f32, 4), policy.knownMtpPreparation(4, 2048, .carried));
    try std.testing.expect(!std.mem.eql(u8, &policy.diagnosticFingerprint(), &copy.diagnosticFingerprint()));
    try std.testing.expect(copy.knownMtpPreparation(3, 2048, .carried) == null);
    try std.testing.expect(copy.knownMtpPreparation(4, 2048, .cold) == null);
    try std.testing.expect(copy.knownMtpPreparation(4, 4096, .carried) == null);
}

test "MTP known preparation reranks every width and unknown retains existing hypothesis" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var candidate = Tree.chain(0, &.{ 1, 2, 3 }, 1, 1);
    candidate.scores_calibrated = true;
    policy.prices.observe(2, 10, .chain, 0, .generic);
    policy.prices.observe(4, 10, .chain, 0, .generic);
    const fallback: LookFallback = .{ .measured = .{ .round_ms = 20, .expected_tokens = 2 } };
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const initial = policy.diagnosticFingerprint();
    const unmeasured = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, 0, fallback, .cold);
    try std.testing.expectEqual(@as(u32, 3), unmeasured.rows);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 12.0), unmeasured.value, 0.000001);
    try std.testing.expectEqual(initial, policy.diagnosticFingerprint());
    try std.testing.expect(policy.knownMtpPreparation(4, 0, .cold) == null);
    try std.testing.expect(!policy.observeMtpPreparation(4, 0, .cold, 100));
    try std.testing.expect(policy.observeMtpPreparation(4, 0, .cold, 100));
    // Winner-only surcharge would reject width four and miss profitable width two.
    const reranked = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, 0, fallback, .cold);
    try std.testing.expectEqualSlices(u32, &.{1}, reranked.tokens);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 12.0), reranked.value, 0.000001);
    const carried = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, 0, fallback, .carried);
    try std.testing.expectEqual(@as(u32, 3), carried.rows);
    const unchanged = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, 0, fallback, null);
    try std.testing.expectEqual(@as(u32, 3), unchanged.rows);
    try std.testing.expect(!policy.observeMtpPreparation(2, 0, .cold, 100));
    try std.testing.expect(policy.observeMtpPreparation(2, 0, .cold, 100));
    const native = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, 0, fallback, .cold);
    try std.testing.expectEqual(Choice.Source.spine, native.source);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 22.0), native.value, 0.000001);
    for ([_]u32{ 2, 4, MAX_TREE_ROWS }) |requested| {
        const forced = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, requested, fallback, .cold);
        try std.testing.expectEqual(@min(requested, 4) - 1, forced.rows);
    }
    const forced = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, 4, fallback, .cold);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 112.0), forced.value, 0.000001);
}

test "MTP preparation prices never substitute for unknown verifier prices" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var candidate = Tree.chain(0, &.{ 1, 2, 3 }, 1, 1);
    candidate.scores_calibrated = true;
    policy.prices.observe(2, 10, .chain, 0, .generic);
    try std.testing.expect(!policy.observeMtpPreparation(4, 0, .cold, 1));
    try std.testing.expect(policy.observeMtpPreparation(4, 0, .cold, 1));
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const choice = try policy.chooseLookPreparedControlled(&candidate, 0, 2, &out, 0,
        .{ .measured = .{ .round_ms = 20, .expected_tokens = 2 } }, .cold);
    try std.testing.expectEqual(@as(u32, 1), choice.rows);
}

test "MTP preparation changes trial screen but not verifier eligibility or retry budget" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var candidate = Tree.chain(0, &.{ 1, 2, 3 }, 1, 1);
    candidate.scores_calibrated = true;
    policy.prices.observe(2, 10, .chain, 0, .generic);
    const native: RemainingRound = .{ .round_ms = 20, .expected_tokens = 2 };
    try std.testing.expect(!policy.observeMtpPreparation(4, 0, .cold, 100));
    try std.testing.expect(policy.observeMtpPreparation(4, 0, .cold, 100));
    var evidence: ?Policy.LookWidthTrialEvidence = null;
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 2, native, &evidence, .cold) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    try std.testing.expectEqual(@as(?f32, 100), evidence.?.lookup_preparation_ms);
    try std.testing.expectEqual(@as(?LookupPreparationScope, .cold), evidence.?.mtp_scope);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 112.0), evidence.?.optimistic_rate, 0.000001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 22.0), evidence.?.native_rate, 0.000001);
    // Unknown carried P preserves the existing eligible verifier trial.
    try std.testing.expectEqual(@as(?u32, 4), policy.reserveLookWidthTrial(&candidate, 0, 2, 2, native, &evidence, .carried));
    try std.testing.expect(evidence.?.lookup_preparation_ms == null);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 12.0), evidence.?.optimistic_rate, 0.000001);
    // P observations do not manufacture the actual V execution needed for retry.
    try std.testing.expect(!policy.observeMtpPreparation(4, 0, .carried, 1));
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 2, native, null, .carried) == null);
    try std.testing.expect(!policy.prices.observeKernel(4, 100, .chain, 0, .generic));
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 2, native, null, .cold) == null);
    try std.testing.expectEqual(@as(?u32, 4), policy.reserveLookWidthTrial(&candidate, 0, 2, 2, native, null, .carried));
    // Neither scope switching nor cloning can reopen the consumed second attempt.
    var copy = try policy.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(policy.diagnosticFingerprint(), copy.diagnosticFingerprint());
    for ([_]?LookupPreparationScope{ null, .cold, .carried }) |scope|
        try std.testing.expect(copy.reserveLookWidthTrial(&candidate, 0, 2, 2, native, null, scope) == null);
    try std.testing.expect(policy.knownMtpPreparation(4, 0, .carried) == null);
}

test "known verifier and missing MTP preparation never authorize a new trial" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    var candidate = Tree.chain(0, &.{ 1, 2, 3 }, 1, 1);
    candidate.scores_calibrated = true;
    policy.prices.observe(2, 10, .chain, 0, .generic);
    policy.prices.observe(4, 10, .chain, 0, .generic);
    const before = policy.diagnosticFingerprint();
    for ([_]?LookupPreparationScope{ null, .cold, .carried }) |scope|
        try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2, 2,
            .{ .round_ms = 20, .expected_tokens = 2 }, null, scope) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
}

test "unknown native copy preserves nullable economics without observations or recovery budget" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    _ = controller.observeSpecRemainingProbe(1, 1, 0, 0);
    const before = controller.diagnosticFingerprint();
    for (0..30) |_| {
        const choice = controller.routeAgainstPrepared(0, 0.1, 4, 9);
        try std.testing.expect(choice.skip and choice.native_copy);
        try std.testing.expectEqual(RouteDecision.Reason.native_copy_unmeasured, choice.reason);
        try std.testing.expect(choice.native_copy_ms == null and choice.native_copy_tokens == null);
        try std.testing.expect(choice.alternative_ms == null);
        try std.testing.expectEqual(@as(u32, 0), choice.skipped);
    }
    try std.testing.expectEqual(before, controller.diagnosticFingerprint());
    try std.testing.expect(controller.nativeCopyRemainingFor(9, 0) == null);
    try std.testing.expect(!controller.observeNativeCopyRemaining(100, 9, 9, 0));
    const cold = controller.routeAgainstPrepared(0, 0.1, 4, 9);
    try std.testing.expect(cold.native_copy and cold.native_copy_tokens == null);
}

test "native copy compares its own paired remaining cost and yield against each action" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    _ = controller.observeSpecRemainingProbe(15, 3, 0, 0);
    _ = controller.observeNativeCopyRemaining(100, 9, 9, 0);
    _ = controller.observeNativeCopyRemaining(4, 2, 9, 0);
    _ = controller.observeNativeCopyRemaining(12, 4, 9, 0);
    const copy = controller.nativeCopyRemainingFor(9, 0).?;
    try std.testing.expectEqual(@as(f32, 8), copy.round_ms);
    try std.testing.expectEqual(@as(f32, 3), copy.expected_tokens);
    const wins = controller.routeAgainstPrepared(0, 5, 4, 9);
    try std.testing.expect(wins.skip and wins.native_copy);
    try std.testing.expectEqual(RouteDecision.Reason.native_copy, wins.reason);
    try std.testing.expectEqual(@as(f32, 4), wins.native_copy_ms.?);
    try std.testing.expectEqual(wins.native_copy_ms, wins.alternative_ms);
    try std.testing.expectEqual(@as(f32, 3), wins.native_copy_tokens.?);
    try std.testing.expectApproxEqAbs(@as(f32, 19.0 / 3.0), wins.assistant_ms.?, 0.00001);
    const lookup = controller.routeAgainstPrepared(0, 3.5, 4, 9);
    try std.testing.expect(lookup.skip and !lookup.native_copy);
    try std.testing.expectEqual(RouteDecision.Reason.alternative, lookup.reason);
    try std.testing.expectEqual(wins.native_copy_ms, lookup.native_copy_ms);
    try std.testing.expectEqual(wins.native_copy_tokens, lookup.native_copy_tokens);
    for (0..7) |_| _ = controller.observeSpecRemainingProbe(2, 3, 0, 0);
    const assistant = controller.routeAgainstPrepared(0, 3.5, 4, 9);
    try std.testing.expect(!assistant.skip and !assistant.native_copy);
    try std.testing.expectEqual(RouteDecision.Reason.assistant, assistant.reason);
    try std.testing.expectEqual(@as(f32, 2), assistant.assistant_ms.?);
    try std.testing.expectEqual(wins.native_copy_ms, assistant.native_copy_ms);
    // No eligible copy leaves all newly added metadata absent, even with history.
    const absent = controller.routeAgainstPrepared(0, 3.5, 4, null);
    try std.testing.expect(!absent.native_copy);
    try std.testing.expect(absent.native_copy_ms == null and absent.native_copy_tokens == null);
}

test "native copy retains exact ties against assistant and lookup" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    _ = controller.observeNativeCopyRemaining(8, 3, 9, 0);
    _ = controller.observeNativeCopyRemaining(8, 3, 9, 0);
    _ = controller.observeSpecRemainingProbe(12, 4, 0, 0);
    const tie = controller.routeAgainstPrepared(0, 4, 4, 9);
    try std.testing.expect(tie.skip and tie.native_copy);
    try std.testing.expectEqual(tie.assistant_ms, tie.native_copy_ms);
    try std.testing.expectEqual(tie.alternative_ms, tie.native_copy_ms);
    const cheaper = controller.routeAgainstPrepared(0, 3.99, 4, 9);
    try std.testing.expect(cheaper.skip and !cheaper.native_copy);
    try std.testing.expectEqual(RouteDecision.Reason.alternative, cheaper.reason);
}

test "known native copy excludes unknown assistant and shares bounded route recovery" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    _ = controller.observeNativeCopyRemaining(12, 2, 4, 0);
    _ = controller.observeNativeCopyRemaining(12, 2, 4, 0);
    for (0..12) |index| {
        const choice = controller.routeAgainstPrepared(0, if (index % 2 == 0) 8 else 5, 0, 4);
        try std.testing.expect(choice.skip);
        try std.testing.expectEqual(index % 2 == 0, choice.native_copy);
        try std.testing.expect(choice.assistant_ms == null);
        try std.testing.expectEqual(@as(u32, @intCast(index + 1)), choice.skipped);
    }
    // An unmeasured other width neither spends nor resets the existing counter.
    const unknown = controller.routeAgainstPrepared(0, 1, 0, 5);
    try std.testing.expectEqual(@as(u32, 12), unknown.skipped);
    try std.testing.expectEqual(RouteDecision.Reason.native_copy_unmeasured, unknown.reason);
    const recovery = controller.routeAgainstPrepared(0, null, 0, 4);
    try std.testing.expect(!recovery.skip and !recovery.native_copy);
    try std.testing.expectEqual(RouteDecision.Reason.recovery, recovery.reason);
    try std.testing.expectEqual(@as(u32, 0), recovery.skipped);
    try std.testing.expectEqual(@as(f32, 6), recovery.native_copy_ms.?);
    const again = controller.routeAgainstPrepared(0, std.math.nan(f32), 0, 4);
    try std.testing.expect(again.skip and again.native_copy);
    try std.testing.expectEqual(@as(u32, 1), again.skipped);
}

test "native copy rejects malformed timing before cold width and context state" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const before = controller.diagnosticFingerprint();
    for ([_]f32{ 0, -1, std.math.nan(f32), std.math.inf(f32) }) |ms| {
        try std.testing.expect(!controller.observeNativeCopyRemaining(ms, 2, 4, 0));
    }
    try std.testing.expect(!controller.observeNativeCopyRemaining(4, 0, 4, 0));
    try std.testing.expect(!controller.observeNativeCopyRemaining(4, 5, 4, 0));
    try std.testing.expect(!controller.observeNativeCopyRemaining(4, 1, 1, 0));
    try std.testing.expect(!controller.observeNativeCopyRemaining(4, 1, 17, 0));
    try std.testing.expectEqual(before, controller.diagnosticFingerprint());
    try std.testing.expectEqual(@as(u32, 0), controller.rounds.count());
    try std.testing.expect(!controller.observeNativeCopyRemaining(100, 4, 4, 0));
    try std.testing.expect(controller.nativeCopyRemainingFor(4, 0) == null);
    try std.testing.expect(controller.observeNativeCopyRemaining(6, 2, 4, 0));
    try std.testing.expect(!controller.observeNativeCopyRemaining(100, 4, 4, 2048));
    try std.testing.expect(!controller.observeNativeCopyRemaining(100, 5, 5, 0));
    try std.testing.expect(controller.nativeCopyRemainingFor(4, 2048) == null);
    try std.testing.expect(controller.nativeCopyRemainingFor(5, 0) == null);
    try std.testing.expectEqual(@as(f32, 6), controller.nativeCopyRemainingFor(4, 0).?.round_ms);
    try std.testing.expect(controller.specRemainingFor(0) == null);
    try std.testing.expect(controller.specRemainingFor(2048) == null);
    // Copy observations cannot warm assistant history.
    try std.testing.expect(!controller.observeSpecRemainingProbe(100, 1, 0, 1));
    try std.testing.expect(controller.specRemainingFor(0) == null);
    try std.testing.expect(controller.observeSpecRemainingProbe(9, 3, 0, 1));
    try std.testing.expectEqual(@as(f32, 9), controller.specRemainingFor(0).?.round_ms);
}

test "native copy clone fingerprints cold markers and bounded paired ring independently" {
    var original = Policy.init(PriceModel.init(std.testing.allocator));
    defer original.deinit();
    // Precreate the context so the first fingerprint difference is the cold marker.
    original.observePlain(2, 0);
    const before = original.diagnosticFingerprint();
    _ = original.observeNativeCopyRemaining(100, 9, 9, 0);
    try std.testing.expect(!std.mem.eql(u8, &before, &original.diagnosticFingerprint()));
    var clone = try original.clone(std.testing.allocator);
    defer clone.deinit();
    try std.testing.expectEqual(original.diagnosticFingerprint(), clone.diagnosticFingerprint());
    try std.testing.expect(clone.observeNativeCopyRemaining(99, 9, 9, 0));
    for (0..7) |index| {
        try std.testing.expect(clone.observeNativeCopyRemaining(@floatFromInt(index + 1), @intCast(index + 1), 9, 0));
    }
    const estimate = clone.nativeCopyRemainingFor(9, 0).?;
    try std.testing.expectEqual(@as(f32, 4), estimate.round_ms);
    try std.testing.expectEqual(@as(f32, 4), estimate.expected_tokens);
    try std.testing.expectEqual(@as(u8, 7), clone.rounds.getPtr(0).?.native_copy[7].samples.len);
    try std.testing.expect(original.nativeCopyRemainingFor(9, 0) == null);
    try std.testing.expect(!std.mem.eql(u8, &original.diagnosticFingerprint(), &clone.diagnosticFingerprint()));
    var snapshot = try clone.clone(std.testing.allocator);
    defer snapshot.deinit();
    try std.testing.expectEqual(clone.diagnosticFingerprint(), snapshot.diagnosticFingerprint());
    try std.testing.expect(!clone.observeNativeCopyRemaining(100, 2, 2, 2048));
    try std.testing.expect(snapshot.nativeCopyRemainingFor(2, 2048) == null);
    try std.testing.expect(snapshot.specRemainingFor(0) == null);
}
