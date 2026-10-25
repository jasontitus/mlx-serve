//! Independently authored from the host behavioral contract; no source-port compatibility layer.
//! Prices are milliseconds of completed work, not enqueue latency. Stable marginal-reach
//! ordering gives the greatest additive reward at each width while preserving ancestors.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const MAX_TREE_ROWS = 32;
pub const Kind = enum { chain, tree, plain, spec };
pub const NodeSource = enum { draft, look, both };
pub const Variant = enum { generic, linear_gdn };

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

const PriceKey = struct { class: i64, rows: u32, kind: Kind };
const ColdKey = struct { price: PriceKey, variant: Variant };
const ColdState = enum { seen, retry_reserved };
const TrialKey = struct { class: i64, kind: Kind };
const WidthTrial = struct { rows: u32, variant: Variant, retry_reserved: bool = false };

pub const PriceModel = struct {
    allocator: Allocator,
    priors: std.AutoHashMap(PriceKey, f32),
    settled: std.AutoHashMap(PriceKey, Settled),
    warmed: std.AutoHashMap(ColdKey, ColdState),
    width_trials: std.AutoHashMap(TrialKey, WidthTrial),
    step_ms: ?f32 = null,

    pub fn init(allocator: Allocator) PriceModel {
        return .{
            .allocator = allocator,
            .priors = std.AutoHashMap(PriceKey, f32).init(allocator),
            .settled = std.AutoHashMap(PriceKey, Settled).init(allocator),
            .warmed = std.AutoHashMap(ColdKey, ColdState).init(allocator),
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
        if (schema != .integer or schema.integer != 2) return result;
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
                var widths = kind_entry.value_ptr.object.iterator();
                while (widths.next()) |width_entry| {
                    const rows = std.fmt.parseInt(u32, width_entry.key_ptr.*, 10) catch continue;
                    if (rows == 0) continue;
                    const ms = number(width_entry.value_ptr.*) orelse continue;
                    result.priors.put(.{ .class = class, .rows = rows, .kind = kind }, ms) catch {};
                }
            }
        }
        return result;
    }

    pub fn prior(self: *const PriceModel, rows: u32, kind: Kind, class: i64) ?f32 {
        return self.priors.get(.{ .class = class, .rows = rows, .kind = kind });
    }
    /// Exact evidence only: unlike price(), this never invents a timing estimate.
    pub fn known(self: *const PriceModel, rows: u32, kind: Kind, class: i64) ?f32 {
        if (self.settled.get(.{ .class = class, .rows = rows, .kind = kind })) |sample| {
            if (sample.estimate()) |ms| return ms;
        }
        return self.prior(rows, kind, class);
    }
    pub fn price(self: *const PriceModel, rows: u32, kind: Kind, class: i64) f32 {
        const width = @max(rows, 1);
        const key: PriceKey = .{ .class = class, .rows = width, .kind = kind };
        if (self.settled.get(key)) |sample| if (sample.estimate()) |ms| return ms;
        if (self.prior(width, kind, class)) |ms| return ms;
        // Within-kind/context extrapolation never assumes a wider action is cheaper.
        // Unknown widths carry a factor-two uncertainty penalty; this is not a timing prior.
        var nearest_distance: u32 = std.math.maxInt(u32);
        var nearest: ?f32 = null;
        // Probe only executable widths in this exact context/kind. Prediction work is
        // bounded independently of the number of historical contexts in the maps.
        for (1..MAX_TREE_ROWS + 1) |candidate_width| {
            const old_width: u32 = @intCast(candidate_width);
            const distance = if (width > old_width) width - old_width else old_width - width;
            if (distance >= nearest_distance) continue;
            const ms = self.known(old_width, kind, class) orelse continue;
            nearest_distance = distance;
            nearest = ms * @max(1, @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(old_width)));
        }
        return cost((nearest orelse ((self.step_ms orelse 1) * @as(f32, @floatFromInt(width)))) * 2);
    }
    pub fn observe(self: *PriceModel, rows: u32, ms: f32, kind: Kind, class: i64) void {
        if (rows == 0 or !positive(ms)) return;
        const entry = self.settled.getOrPut(.{ .class = class, .rows = rows, .kind = kind }) catch return;
        if (!entry.found_existing) entry.value_ptr.* = .{};
        entry.value_ptr.observe(ms);
    }
    pub fn observeKernel(self: *PriceModel, rows: u32, ms: f32, kind: Kind, class: i64, variant: Variant) bool {
        if (rows == 0 or !positive(ms)) return false;
        const key: ColdKey = .{ .price = .{ .class = class, .rows = rows, .kind = kind }, .variant = variant };
        const entry = self.warmed.getOrPut(key) catch return false;
        if (!entry.found_existing) {
            entry.value_ptr.* = .seen;
            return false;
        }
        self.observe(rows, ms, kind, class);
        return self.settled.contains(key.price);
    }


    /// One wider real shape per context/kind, with at most one matching cold retry.
    /// The caller must compare calibrated yields and check execution eligibility.
    pub fn upperTrialWidth(self: *const PriceModel, native_rows: u32, maximum: u32, kind: Kind, class: i64) ?u32 {
        if (self.known(native_rows, kind, class) == null) return null;
        if (self.width_trials.get(.{ .class = class, .kind = kind })) |trial| {
            if (trial.retry_reserved or trial.rows > maximum or self.known(trial.rows, kind, class) != null) return null;
            return trial.rows;
        }
        const width = @min(MAX_TREE_ROWS, maximum);
        return if (width > native_rows and self.known(width, kind, class) == null) width else null;
    }

    pub fn reserveUpperTrial(self: *PriceModel, native_rows: u32, rows: u32, kind: Kind, class: i64, variant: Variant) bool {
        if (rows <= native_rows or rows > MAX_TREE_ROWS or self.known(native_rows, kind, class) == null or self.known(rows, kind, class) != null) return false;
        const key: TrialKey = .{ .class = class, .kind = kind };
        if (self.width_trials.getPtr(key)) |trial| {
            if (trial.rows != rows or trial.variant != variant or trial.retry_reserved) return false;
            const cold = self.warmed.getPtr(.{ .price = .{ .class = class, .rows = rows, .kind = kind }, .variant = variant }) orelse return false;
            if (cold.* != .seen) return false;
            trial.retry_reserved = true;
            cold.* = .retry_reserved;
            return true;
        }
        const cold = self.warmed.getPtr(.{ .price = .{ .class = class, .rows = rows, .kind = kind }, .variant = variant });
        if (cold) |state| if (state.* == .retry_reserved) return false;
        // If this shape already ran cold naturally, reserve only its remaining retry.
        self.width_trials.put(key, .{ .rows = rows, .variant = variant, .retry_reserved = cold != null }) catch return false;
        if (cold) |state| state.* = .retry_reserved;
        return true;
    }
};

pub const Calibration = struct {
    pub const Kind = enum { neural, look, look_continuation };
    const Bin = struct { hits: f32 = 0, total: f32 = 0 };
    const orders = 65; // token_history's bounded suffix lengths; zero is unspecified.
    neural: [8]Bin = @splat(.{}),
    bins: [2][orders][8]Bin = @splat(@splat(@splat(.{}))),

    fn bucket(p: f32) usize {
        return @min(7, @as(usize, @intFromFloat(probability(p) * 8)));
    }
    pub fn q(self: *const Calibration, kind: Calibration.Kind, p: f32, order: u32) f32 {
        const raw = probability(p);
        const bin = if (kind == .neural) self.neural[bucket(raw)] else self.bins[@intFromEnum(kind) - 1][@min(order, orders - 1)][bucket(raw)];
        return probability((4 * raw + bin.hits) / (4 + bin.total));
    }
    pub fn learn(self: *Calibration, kind: Calibration.Kind, p: f32, order: u32, hit: bool) void {
        const bin = if (kind == .neural) &self.neural[bucket(p)] else &self.bins[@intFromEnum(kind) - 1][@min(order, orders - 1)][bucket(p)];
        if (bin.total >= 128) {
            bin.hits *= 0.5;
            bin.total *= 0.5;
        }
        bin.total += 1;
        if (hit) bin.hits += 1;
    }
    pub fn samples(self: *const Calibration, kind: Calibration.Kind) f32 {
        var total: f32 = 0;
        if (kind == .neural) {
            for (self.neural) |bin| total += bin.total;
        } else {
            for (self.bins[@intFromEnum(kind) - 1]) |population| for (population) |bin| {
                total += bin.total;
            };
        }
        return total;
    }
};

/// Request-local predictive evidence. Vote unanimity and successes in another
/// request do not establish support here. Orders share these small populations;
/// pooled calibration remains exact-order-specific. This is a predictive mean,
/// not a confidence bound or a statistical admission guarantee.
pub const RequestTrust = struct {
    const Bin = struct { hits: f32 = 0, total: f32 = 0 };
    bins: [2][8]Bin = @splat(@splat(.{})),

    pub fn q(self: *const RequestTrust, kind: Calibration.Kind, raw: f32) f32 {
        std.debug.assert(kind != .neural);
        const bin = self.bins[@intFromEnum(kind) - 1][Calibration.bucket(raw)];
        // Miss-only evidence cannot create support that an unseen bin lacked.
        return if (bin.hits == 0) 0 else (bin.hits + 1) / (bin.total + 2);
    }
    pub fn learn(self: *RequestTrust, kind: Calibration.Kind, raw: f32, hit: bool) void {
        std.debug.assert(kind != .neural);
        const bin = &self.bins[@intFromEnum(kind) - 1][Calibration.bucket(raw)];
        if (bin.total >= 128) {
            bin.hits *= 0.5;
            bin.total *= 0.5;
        }
        bin.total += 1;
        if (hit) bin.hits += 1;
    }
    pub fn diagnosticFingerprint(self: *const RequestTrust) [32]u8 {
        var hash: Fingerprint = .{};
        hash.hash.update("mlx-serve-request-lookup-trust-1");
        for (self.bins) |population| for (population) |bin| {
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
    pub const Entry = struct { rows: u32, verify_ms: f32, expected_tokens: f32, tokens_per_ms: f32 };
    entries: [MAX_TREE_ROWS]Entry = undefined,
    len: usize = 0,
};
pub const RouteDecision = struct {
    pub const Reason = enum { unmeasured, invalid_alternative, assistant, alternative, recovery };
    skip: bool,
    reason: Reason,
    assistant_ms: ?f32,
    alternative_ms: ?f32,
    skipped: u32,
    interval: u32,
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

const RoundState = struct {
    plain: Settled = .{},
    spec: RoundSamples = .{},
    plain_seen: u32 = 0,
    spec_seen: u32 = 0,
    skipped: u32 = 0,
};

pub const Policy = struct {
    prices: PriceModel,
    cal: Calibration = .{},
    rounds: std.AutoHashMap(i64, RoundState),

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
        return copy;
    }

    /// Diagnostic-only state identity. Hash explicit values, never pointers or padding.
    /// Entry digests are XOR-folded with a map-specific domain and count, so rehashing or
    /// cloning the containers does not change identity. Keys are unique in each map.
    pub fn diagnosticFingerprint(self: *const Policy) [32]u8 {
        var all: Fingerprint = .{};
        all.hash.update("mlx-serve-independent-policy-state-2");
        all.word(@intFromBool(self.prices.step_ms != null));
        if (self.prices.step_ms) |step| all.float(step);
        for (self.cal.neural) |bin| {
            all.float(bin.hits);
            all.float(bin.total);
        }
        for (self.cal.bins) |kind| for (kind) |population| for (population) |bin| {
            all.float(bin.hits);
            all.float(bin.total);
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
            one.priceKey(entry.key_ptr.price);
            one.word(@intFromEnum(entry.key_ptr.variant));
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
            const digest = one.finish();
            for (&fold, digest) |*byte, value| byte.* ^= value;
        }
        all.hash.update("rounds");
        all.word(self.rounds.count());
        all.hash.update(&fold);
        return all.finish();
    }
    pub fn predictVerify(self: *const Policy, rows: u32, kind: Kind, class: i64) f32 {
        return self.prices.price(rows, kind, class);
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
                const draft_q = self.cal.q(.neural, tree.draft_confidence[child], 0);
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

    pub fn chooseTree(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32) Tree {
        return self.chooseTreeControlled(candidate, class, preparation_ms, max_rows, 0, null);
    }
    pub fn chooseTreeControlled(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, requested: u32, evaluation: ?*TreeEvaluation) Tree {
        return self.selectTree(candidate, class, preparation_ms, max_rows, requested, evaluation, .tree);
    }

    /// The caller identifies its executable native action; no native width or hardware
    /// latency is guessed here. Bootstrap is a real action, not a root-only calibration probe.
    pub fn chooseTreeBootstrapControlled(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, requested: u32, native_rows: u32, evaluation: ?*TreeEvaluation) Tree {
        const native = @min(@max(1, native_rows), @min(@max(1, max_rows), @max(1, @min(MAX_TREE_ROWS, candidate.len))));
        const effective: u32 = if (requested == 0 and self.prices.known(@intCast(native), .tree, class) == null) @intCast(native) else requested;
        return self.chooseTreeControlled(candidate, class, preparation_ms, max_rows, effective, evaluation);
    }

    fn selectTree(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, max_rows: u32, requested: u32, evaluation: ?*TreeEvaluation, kind: Kind) Tree {
        const tree = self.calibrated(candidate);
        const cap = @min(tree.len, @min(MAX_TREE_ROWS, @max(1, max_rows)));
        const wanted = @min(cap, @max(1, requested));
        if (evaluation) |log| log.len = 0;
        // A child's marginal reach never exceeds its parent's. Stable descending order
        // therefore has ancestor-closed prefixes, including ties. Each prefix maximizes
        // summed reach among equal-width subsets; verification cost depends only on width.
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
        for (1..cap + 1) |width| {
            const row = order[width - 1];
            prefix |= @as(u32, 1) << @as(u5, @intCast(row));
            reward += tree.scores[row];
            if (requested > 0 and width != wanted) continue;
            const verify_ms = self.predictVerify(@intCast(width), kind, class);
            const value = reward / (preparation(preparation_ms) + verify_ms);
            if (evaluation) |log| {
                log.entries[log.len] = .{ .rows = @intCast(width), .verify_ms = verify_ms, .expected_tokens = reward, .tokens_per_ms = value };
                log.len += 1;
            }
            if (value > best_value) {
                best_value = value;
                selected = prefix;
            }
        }
        var result = Tree.init(tree.tokens[0]);
        result.scores_calibrated = true;
        var remap: [MAX_TREE_ROWS]usize = @splat(0);
        for (1..tree.len) |i| {
            if (selected & (@as(u32, 1) << @as(u5, @intCast(i))) == 0) continue;
            const dst = result.len;
            remap[i] = dst;
            result.tokens[dst] = tree.tokens[i];
            result.parents[dst] = @intCast(remap[@intCast(tree.parents[i])]);
            result.scores[dst] = tree.scores[i];
            result.conditional_reach[dst] = tree.conditional_reach[i];
            result.sources[dst] = tree.sources[i];
            result.origins[dst] = tree.origins[i];
            result.confidence[dst] = tree.confidence[i];
            result.draft_confidence[dst] = tree.draft_confidence[i];
            result.look_confidence[dst] = tree.look_confidence[i];
            result.look_order[dst] = tree.look_order[i];
            result.len += 1;
        }
        return result;
    }

    /// Bounded chain-width learning uses actual known prices and the current offer's
    /// calibrated yields, even when ordinary selection has fallen back to native.
    /// The caller enforces solo/adaptive/fresh-native execution eligibility.
    pub fn reserveLookWidthTrial(self: *Policy, candidate: *const Tree, class: i64, minimum_rows: u32) ?u32 {
        const floor = @max(2, minimum_rows);
        if (!candidate.scores_calibrated or candidate.len <= floor or candidate.len > MAX_TREE_ROWS) return null;
        const native = self.specRemainingFor(class) orelse return null;
        var reference_rows: ?u32 = null;
        for (floor..candidate.len) |rows| {
            if (self.prices.known(@intCast(rows), .chain, class) != null) {
                reference_rows = @intCast(rows);
                break;
            }
        }
        const reference = reference_rows orelse return null;
        const trial = self.prices.upperTrialWidth(reference, @intCast(candidate.len), .chain, class) orelse return null;
        var expected: f32 = 0;
        var reference_expected: f32 = 0;
        for (candidate.scores[0..trial], 0..) |reach, row| {
            if (!std.math.isFinite(reach) or reach < 0 or reach > 1) return null;
            if (row > 0 and candidate.parents[row] != @as(i32, @intCast(row - 1))) return null;
            expected += reach;
            if (row < reference) reference_expected += reach;
        }
        if (expected <= reference_expected or expected <= native.expected_tokens) return null;
        if (!self.prices.reserveUpperTrial(reference, trial, .chain, class, .generic)) return null;
        return trial;
    }

    /// Use this entry point when lookup preparation has already consumed measured time.
    pub fn chooseLookPreparedControlled(self: *const Policy, candidate: *const Tree, class: i64, preparation_ms: f32, out: *std.ArrayList(u32), requested: u32, fallback: LookFallback) !Choice {
        out.clearRetainingCapacity();
        const paid = preparation(preparation_ms);
        const baseline: ?f32 = switch (fallback) {
            .unknown => null,
            .plain => 1 / (paid + (self.plainFor(class) orelse self.predictVerify(1, .plain, class))),
            .measured => |value| if (positive(value.round_ms) and positive(value.expected_tokens))
                value.expected_tokens / (paid + cost(value.round_ms))
            else
                null,
        };
        if (requested == 0 and baseline == null) return .{ .tokens = out.items, .rows = 0, .source = .spine, .expected = 0, .value = 0 };
        const selected = self.selectTree(candidate, class, paid, MAX_TREE_ROWS, requested, null, .chain);
        var expected: f32 = 0;
        for (selected.scores[1..selected.len]) |reach| expected += reach;
        const value = (1 + expected) / (paid + self.predictVerify(@intCast(selected.len), .chain, class));
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
    pub const CalibrationPopulations = enum { all, neural, lookup };


    /// path_tokens excludes the anchor and contains only actually reached target decisions.
    /// A missing next decision is censored, not a rejection. A supplied mismatch rejects
    /// every candidate child at that parent, but never any of their descendants.
    pub fn observeTree(self: *Policy, tree: *const Tree, path_tokens: []const u32, populations: CalibrationPopulations, trust: ?*RequestTrust) void {
        if (tree.len == 0 or tree.len > MAX_TREE_ROWS) return;
        var parent: usize = 0;
        for (path_tokens) |actual| {
            var matching: ?usize = null;
            for (parent + 1..tree.len) |child| {
                if (tree.parents[child] != @as(i32, @intCast(parent))) continue;
                const hit = tree.tokens[child] == actual;
                if (populations != .lookup and tree.sources[child] != .look) self.cal.learn(.neural, tree.draft_confidence[child], 0, hit);
                if (populations != .neural and tree.sources[child] != .draft) {
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
        const context = self.rounds.get(class) orelse return null;
        return context.plain.estimate();
    }
    pub fn specFor(self: *const Policy, class: i64) ?f32 {
        const sample = self.specRemainingFor(class) orelse return null;
        return sample.round_ms / sample.expected_tokens;
    }
    pub fn specRemainingFor(self: *const Policy, class: i64) ?RemainingRound {
        const context = self.rounds.get(class) orelse return null;
        return context.spec.estimate();
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
    pub fn routeAgainstPrepared(self: *Policy, class: i64, alternative_ms: ?f32, preparation_ms: f32) RouteDecision {
        const remaining = self.specRemainingFor(class);
        const assistant: ?f32 = if (remaining) |sample|
            (preparation(preparation_ms) + sample.round_ms) / sample.expected_tokens
        else
            null;
        var result: RouteDecision = .{ .skip = false, .reason = .unmeasured, .assistant_ms = assistant, .alternative_ms = alternative_ms, .skipped = 0, .interval = 12 };
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
    try std.testing.expectEqual(@as(f32, 6), prices.price(3, .tree, 101));
    try std.testing.expect(!prices.observeKernel(3, 700, .chain, 101, .generic));
    try std.testing.expect(!prices.observeKernel(4, 700, .tree, 101, .generic));
    try std.testing.expect(!prices.observeKernel(3, 700, .tree, 102, .generic));
    prices.observe(3, 60000, .tree, 101);
    prices.observe(3, 7, .tree, 101);
    try std.testing.expectEqual(@as(f32, 7), prices.price(3, .tree, 101));
    try std.testing.expect(prices.prior(3, .tree, 101) == null);
    try std.testing.expect(positive(prices.price(0, .chain, -10)));
    try std.testing.expectEqual(@as(usize, 1), prices.count());
}

test "calibration populations censor unreachable descendants" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const tree = Tree.chain(50, &.{ 51, 52, 53 }, 0.6, 0.7);
    policy.observeTree(&tree, &.{99}, .all, null);
    try std.testing.expectEqual(@as(f32, 1), policy.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 0), policy.cal.samples(.look_continuation));
    policy.observeTree(&tree, &.{51}, .all, null);
    try std.testing.expectEqual(@as(f32, 0), policy.cal.samples(.look_continuation));
    policy.observeTree(&tree, &.{ 51, 99 }, .all, null);
    try std.testing.expectEqual(@as(f32, 1), policy.cal.samples(.look_continuation));
    try std.testing.expectEqual(@as(f32, 0), policy.cal.samples(.neural));
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
    policy.prices.observe(1, 1, .tree, 5);
    policy.prices.observe(2, 3, .tree, 5);
    const tree = Tree.chain(8, &.{9}, 0.9, 0.9);
    var log: TreeEvaluation = .{};
    const free = policy.chooseTreeControlled(&tree, 5, 0, 32, 0, &log);
    try std.testing.expectEqual(@as(usize, 1), free.len);
    const paid = policy.chooseTreeControlled(&tree, 5, 20, 32, 0, &log);
    try std.testing.expectEqual(@as(usize, 2), paid.len);
    try std.testing.expectEqual(@as(usize, 2), log.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 21.0), log.entries[0].tokens_per_ms, 0.00001);
    const forced = policy.chooseTreeControlled(&tree, 5, 20, 32, 100, &log);
    try std.testing.expectEqual(@as(usize, 2), forced.len);
    try std.testing.expectEqual(@as(usize, 1), log.len);
    try std.testing.expectEqual(@as(u32, 2), log.entries[0].rows);
    const root = policy.chooseTreeControlled(&tree, 5, 20, 0, 100, &log);
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
    const chosen = policy.chooseTreeControlled(&tree, 0, 0, 32, 4, null);
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
    const adaptive = try policy.chooseLookPreparedControlled(&candidate, 0, 0, &out, 0, .unknown);
    try std.testing.expectEqual(@as(u32, 0), adaptive.rows);
    try std.testing.expectEqual(Choice.Source.spine, adaptive.source);
    const forced = try policy.chooseLookPreparedControlled(&candidate, 0, 0, &out, 3, .unknown);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, forced.tokens);
    try std.testing.expectEqual(@as(u32, 2), forced.rows);
    const anchor_only = try policy.chooseLookPreparedControlled(&candidate, 0, 0, &out, 1, .unknown);
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
    try std.testing.expect(!policy.routeAgainstPrepared(13, 1, 0).skip);
    try std.testing.expect(!policy.routeAgainstPrepared(12, std.math.nan(f32), 0).skip);
    var skipped: u32 = 0;
    while (true) {
        const route = policy.routeAgainstPrepared(12, 3, 0);
        if (!route.skip) {
            try std.testing.expectEqual(RouteDecision.Reason.recovery, route.reason);
            try std.testing.expectEqual(route.interval, skipped);
            break;
        }
        skipped += 1;
        try std.testing.expect(skipped <= route.interval);
    }
    for (0..7) |_| _ = policy.observeSpecRemainingProbe(2, 2, 12, 0);
    const restored = policy.routeAgainstPrepared(12, 3, 0);
    try std.testing.expect(!restored.skip);
    try std.testing.expectEqual(RouteDecision.Reason.assistant, restored.reason);
}

test "clone owns prices warmups routing and calibration independently" {
    var original = Policy.init(PriceModel.init(std.testing.allocator));
    defer original.deinit();
    original.prices.observe(2, 7, .tree, 18);
    _ = original.prices.observeKernel(2, 100, .tree, 18, .linear_gdn);
    _ = original.observeSpecRemainingProbe(15, 3, 18, 0);
    _ = original.routeAgainstPrepared(18, 2, 0);
    const candidate = Tree.chain(0, &.{ 1, 2 }, 0.8, 0.8);
    original.observeChainAccepted(&candidate, 2, 1);
    var copy = try original.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(original.prices.price(2, .tree, 18), copy.prices.price(2, .tree, 18));
    try std.testing.expectEqual(original.cal.samples(.look), copy.cal.samples(.look));
    try std.testing.expectEqual(@as(u32, 2), copy.routeAgainstPrepared(18, 2, 0).skipped);
    try std.testing.expectEqual(@as(u32, 2), original.routeAgainstPrepared(18, 2, 0).skipped);
    try std.testing.expect(copy.prices.observeKernel(2, 2, .tree, 18, .linear_gdn));
    for (0..7) |_| copy.prices.observe(2, 2, .tree, 18);
    copy.observeChainAccepted(&candidate, 2, 0);
    try std.testing.expectEqual(@as(f32, 7), original.prices.price(2, .tree, 18));
    try std.testing.expectEqual(@as(f32, 2), copy.prices.price(2, .tree, 18));
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
    try std.testing.expect(missing.known(4, .tree, 77) == null);
    try std.testing.expect(positive(missing.price(4, .tree, 77)));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "prices.json", .data = "{not-json" });
    var malformed = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer malformed.deinit();
    try std.testing.expect(malformed.known(4, .tree, 77) == null);
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "prices.json",
        .data = "{\"step_ms\":2.5,\"77\":{\"tree\":{\"4\":6.25}}}",
    });
    var legacy = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer legacy.deinit();
    try std.testing.expect(legacy.known(4, .tree, 77) == null);
    try std.testing.expect(legacy.step_ms == null);
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "prices.json",
        .data = "{\"schema\":2,\"context_bucket\":\"floor-power-of-two-2048\",\"step_ms\":2.5,\"77\":{\"tree\":{\"4\":6.25,\"2\":-7,\"8\":\"bad\"},\"chain\":{\"4\":9.5}},\"78\":false}",
    });
    var loaded = PriceModel.load(std.testing.io, std.testing.allocator, path);
    defer loaded.deinit();
    try std.testing.expectEqual(@as(?f32, 6.25), loaded.known(4, .tree, 77));
    try std.testing.expectEqual(@as(?f32, 9.5), loaded.prior(4, .chain, 77));
    try std.testing.expect(loaded.known(2, .tree, 77) == null);
    try std.testing.expect(loaded.known(4, .tree, 78) == null);
}

test "cold adaptive selection retains caller native width until its cost is known" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const tree = Tree.chain(30, &.{ 31, 32, 33, 34, 35, 36 }, 0.9, 0.9);
    var log: TreeEvaluation = .{};
    const cold = policy.chooseTreeBootstrapControlled(&tree, 81, 0, 16, 0, 5, &log);
    try std.testing.expectEqual(@as(usize, 5), cold.len);
    try std.testing.expectEqual(@as(usize, 1), log.len);
    try std.testing.expect(policy.prices.known(5, .tree, 81) == null);
    _ = policy.prices.observeKernel(5, 400, .tree, 81, .generic);
    const still_cold = policy.chooseTreeBootstrapControlled(&tree, 81, 0, 16, 0, 5, null);
    try std.testing.expectEqual(@as(usize, 5), still_cold.len);
    _ = policy.prices.observeKernel(5, 10, .tree, 81, .generic);
    _ = policy.chooseTreeBootstrapControlled(&tree, 81, 0, 16, 0, 5, &log);
    try std.testing.expectEqual(tree.len, log.len);
    const forced = policy.chooseTreeBootstrapControlled(&tree, 82, 0, 16, 2, 5, null);
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
    left.prices.observe(2, 8, .tree, 65);
    left.prices.observe(4, 11, .chain, 66);
    right.prices.observe(4, 11, .chain, 66);
    right.prices.observe(2, 8, .tree, 65);
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
    const selected = policy.chooseTreeControlled(&tree, 0, 0, 32, 3, null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), selected.scores[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), selected.scores[2], 0.00001);
    try std.testing.expectEqual(@as(f32, 0.8), selected.look_confidence[1]);
    try std.testing.expectEqual(@as(f32, 0.7), selected.look_confidence[2]);
    try std.testing.expect(selected.scores_calibrated);
}

test "paid preparation is amortized over the native round yield" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    controller.prices.observe(1, 4, .chain, 0);
    controller.prices.observe(2, 4, .chain, 0);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const candidate = Tree.chain(0, &.{7}, 0.5, 0.5);
    const selected = try controller.chooseLookPreparedControlled(&candidate, 0, 4, &out, 0,
        .{ .measured = .{ .round_ms = 4, .expected_tokens = 2 } });
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
    const route = controller.routeAgainstPrepared(0, 3.5, 4);
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
    policy.prices.observe(3, 84, .chain, 0);
    const locked = try policy.chooseLookPreparedControlled(&candidate, 0, 1, &selected, 0, fallback);
    try std.testing.expectEqual(@as(u32, 2), locked.rows);
    const first = policy.reserveLookWidthTrial(&candidate, 0, 2).?;
    try std.testing.expectEqual(@as(u32, 15), first);
    try std.testing.expect(policy.prices.known(first, .chain, 0) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2) == null);
    try std.testing.expect(!policy.prices.observeKernel(first, 400, .chain, 0, .generic));
    try std.testing.expectEqual(@as(?u32, 15), policy.reserveLookWidthTrial(&candidate, 0, 2));
    try std.testing.expect(policy.prices.known(first, .chain, 0) == null);
    try std.testing.expect(policy.prices.observeKernel(first, 140, .chain, 0, .generic));
    const learned = try policy.chooseLookPreparedControlled(&candidate, 0, 1, &selected, 0, fallback);
    try std.testing.expectEqual(@as(u32, 14), learned.rows);
    try std.testing.expect(policy.reserveLookWidthTrial(&candidate, 0, 2) == null);
}

test "lookup width trials respect floors caps declining yield and failed retries" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    const raw = Tree.chain(0, &.{ 11, 12, 13, 14, 15, 16, 17, 18 }, 1, 1);
    const strong = policy.calibrated(&raw);
    const weak_raw = Tree.chain(0, &.{ 11, 12, 13, 14, 15, 16, 17, 18 }, 0.05, 1);
    const weak = policy.calibrated(&weak_raw);
    policy.prices.observe(3, 84, .chain, 0);
    try std.testing.expect(policy.reserveLookWidthTrial(&strong, 0, 2) == null);
    try std.testing.expect(policy.observeSpecRemainingProbe(90, 2, 0, 0));
    try std.testing.expect(policy.reserveLookWidthTrial(&strong, 0, 4) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&strong, 2048, 2) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&weak, 0, 2) == null);
    try std.testing.expectEqual(@as(?u32, 9), policy.reserveLookWidthTrial(&strong, 0, 2));
    try std.testing.expect(!policy.prices.observeKernel(9, 400, .chain, 0, .generic));
    var shortened = strong;
    shortened.len = 8;
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(policy.reserveLookWidthTrial(&shortened, 0, 2) == null);
    try std.testing.expect(policy.reserveLookWidthTrial(&weak, 0, 2) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    var copy = try policy.clone(std.testing.allocator);
    defer copy.deinit();
    try std.testing.expectEqual(@as(?u32, 9), copy.reserveLookWidthTrial(&strong, 0, 2));
    try std.testing.expect(!copy.prices.observeKernel(9, std.math.nan(f32), .chain, 0, .generic));
    try std.testing.expect(copy.reserveLookWidthTrial(&strong, 0, 2) == null);
    try std.testing.expectEqual(before, policy.diagnosticFingerprint());
    try std.testing.expectEqual(@as(?u32, 9), policy.reserveLookWidthTrial(&strong, 0, 2));
}

test "upper width exploration is one shape and one real cold retry" {
    var policy = Policy.init(PriceModel.init(std.testing.allocator));
    defer policy.deinit();
    try std.testing.expect(policy.prices.upperTrialWidth(4, 12, .tree, 0) == null);
    policy.prices.observe(4, 50, .tree, 0);
    try std.testing.expectEqual(@as(?u32, 12), policy.prices.upperTrialWidth(4, 12, .tree, 0));
    try std.testing.expect(policy.prices.reserveUpperTrial(4, 12, .tree, 0, .generic));
    try std.testing.expect(!policy.prices.reserveUpperTrial(4, 12, .tree, 0, .generic));
    try std.testing.expect(!policy.prices.reserveUpperTrial(4, 10, .tree, 0, .generic));
    try std.testing.expect(policy.prices.known(12, .tree, 0) == null);
    try std.testing.expect(!policy.prices.observeKernel(12, 500, .tree, 0, .generic));
    try std.testing.expect(policy.prices.upperTrialWidth(4, 11, .tree, 0) == null);
    try std.testing.expect(!policy.prices.reserveUpperTrial(4, 12, .tree, 0, .linear_gdn));
    var copy = try policy.clone(std.testing.allocator);
    defer copy.deinit();
    const before = policy.diagnosticFingerprint();
    try std.testing.expect(copy.prices.reserveUpperTrial(4, 12, .tree, 0, .generic));
    try std.testing.expect(!std.mem.eql(u8, &before, &copy.diagnosticFingerprint()));
    try std.testing.expectEqualDeep(before, policy.diagnosticFingerprint());
    try std.testing.expect(copy.prices.upperTrialWidth(4, 16, .tree, 0) == null);
    try std.testing.expect(!copy.prices.observeKernel(12, std.math.inf(f32), .tree, 0, .generic));
    try std.testing.expect(!copy.prices.reserveUpperTrial(4, 12, .tree, 0, .generic));
    try std.testing.expect(policy.prices.reserveUpperTrial(4, 12, .tree, 0, .generic));
    try std.testing.expect(policy.prices.observeKernel(12, 95, .tree, 0, .generic));
    try std.testing.expectEqual(@as(?f32, 95), policy.prices.known(12, .tree, 0));
    try std.testing.expect(policy.prices.upperTrialWidth(4, 16, .tree, 0) == null);
}

test "a naturally cold width has only one remaining trial attempt" {
    var prices = PriceModel.init(std.testing.allocator);
    defer prices.deinit();
    prices.observe(3, 84, .chain, 0);
    try std.testing.expect(!prices.observeKernel(15, 300, .chain, 0, .generic));
    try std.testing.expect(prices.reserveUpperTrial(3, 15, .chain, 0, .generic));
    try std.testing.expect(!prices.observeKernel(15, std.math.nan(f32), .chain, 0, .generic));
    try std.testing.expect(!prices.reserveUpperTrial(3, 15, .chain, 0, .generic));
    try std.testing.expect(prices.upperTrialWidth(3, 15, .chain, 0) == null);
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
    try std.testing.expectEqual(@as(f32, 1), controller.cal.q(.neural, 1, 0));
    // The same measured action remains useful for the strong context only.
    controller.prices.observe(1, 50, .chain, 0);
    for (2..16) |width| controller.prices.observe(@intCast(width), 150, .chain, 0);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const fallback: LookFallback = .{ .measured = .{ .round_ms = 100, .expected_tokens = 4 } };
    const weak_choice = try controller.chooseLookPreparedControlled(&weak, 0, 0, &out, 0, fallback);
    try std.testing.expectEqual(Choice.Source.spine, weak_choice.source);
    const strong_choice = try controller.chooseLookPreparedControlled(&strong, 0, 0, &out, 0, fallback);
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
    for (0..200) |_| controller.observeTree(&tree, &.{ 2, 3, 4 }, .lookup, null);
    var request: RequestTrust = .{};
    try std.testing.expectEqual(@as(f32, 0), request.q(.look, 1));
    controller.observeTree(&tree, &.{99}, .lookup, &request);
    try std.testing.expectEqual(@as(f32, 0), request.q(.look, 1));
    try std.testing.expectEqual(@as(f32, 0), request.q(.look_continuation, 1));
    // Continuation support cannot silently establish entry support.
    request.learn(.look_continuation, 1, true);
    try std.testing.expectEqual(@as(f32, 0), request.q(.look, 1));
    controller.observeTree(&tree, &.{2}, .lookup, &request);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), request.q(.look, 1), 0.00001);
    const next_request: RequestTrust = .{};
    try std.testing.expectEqual(@as(f32, 0), next_request.q(.look, 1));
    try std.testing.expect(!std.mem.eql(u8, &request.diagnosticFingerprint(), &next_request.diagnosticFingerprint()));
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(RequestTrust));
    try std.testing.expectEqual(@as(f32, 0), controller.cal.samples(.neural));
}

test "lookup request support enables bounded reuse only after actual native width pricing" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    var raw = Tree.chain(0, &tokens, 1, 1);
    raw.look_order = @splat(8);
    const offer = controller.chooseTreeControlled(&raw, 0, 0, 15, 15, null);
    var request: RequestTrust = .{};
    _ = controller.observeSpecRemainingProbe(100, 2, 0, 1);
    _ = controller.observeSpecRemainingProbe(90, 2, 0, 1);
    const cold = controller.restrictLookup(&offer, &request);
    try std.testing.expectEqual(@as(usize, 15), cold.len);
    for (cold.scores[1..cold.len]) |reach| try std.testing.expectEqual(@as(f32, 0), reach);
    try std.testing.expect(controller.reserveLookWidthTrial(&cold, 0, 2) == null);
    // One genuine shape sample remains cold. Different second widths do not
    // manufacture a warm exact reference or reopen the bounded opportunities.
    try std.testing.expect(!controller.prices.observeKernel(3, 70, .chain, 0, .generic));
    try std.testing.expect(!controller.prices.observeKernel(4, 80, .chain, 0, .generic));
    for (0..8) |_| controller.observeTree(&offer, &tokens, .lookup, &request);
    const supported = controller.restrictLookup(&offer, &request);
    try std.testing.expect(controller.reserveLookWidthTrial(&supported, 0, 2) == null);
    try std.testing.expect(controller.prices.observeKernel(3, 60, .chain, 0, .generic));
    const fresh: RequestTrust = .{};
    const unsupported = controller.restrictLookup(&offer, &fresh);
    try std.testing.expect(controller.reserveLookWidthTrial(&unsupported, 0, 2) == null);
    try std.testing.expectEqual(@as(u32, 15), controller.reserveLookWidthTrial(&supported, 0, 2).?);
    var expected: f32 = 0;
    for (supported.scores[0..supported.len]) |reach| expected += reach;
    try std.testing.expect(expected > 10);
}

test "lookup bounded shadow offer preserves nonoverlapping executed chain evidence" {
    var controller = Policy.init(PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var trust: RequestTrust = .{};
    const selected = Tree.chain(10, &.{ 11, 12, 13, 14 }, 0.8, 0.6);
    var offered = Tree.chain(10, &.{11}, 0.8, 0.6);
    // Equal token 12 under another prefix is not evidence for 11 -> 12.
    _ = offered.add(0, 99, 0.2, .look, 0.2).?;
    _ = offered.add(2, 12, 0.1, .look, 0.3).?;
    const reached = [_]u32{ 11, 12, 88 };
    controller.observeTree(&offered, &reached, .lookup, &trust);
    controller.observeLookupChainRemainder(&selected, &offered, &reached, &trust);
    try std.testing.expectEqual(@as(f32, 2), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 2), controller.cal.samples(.look_continuation));
    try std.testing.expectApproxEqAbs(@as(f32, 0.84), controller.cal.q(.look, 0.8, 0), 0.00001);
    // One continuation hit and its first rejection, never the unreached 14.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), trust.q(.look_continuation, 0.6), 0.00001);
    try std.testing.expectEqual(@as(f32, 0), trust.q(.look_continuation, 0.3));
}
