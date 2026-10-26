//! Pure host adapters authored from the independent behavioral contracts.
//! Inputs are borrowed; all proposal selection and routing scratch is stack bounded.
const std = @import("std");
const policy = @import("speculation_policy.zig");
const history = @import("token_history.zig");

pub const Tree = policy.Tree;
pub const Calibration = policy.Calibration;
pub const VoteNode = history.VoteNode;
pub const max_proposals = 1024;
const absent = std.math.maxInt(u16);

fn probability(value: f32) f32 {
    return if (std.math.isFinite(value)) std.math.clamp(value, 0, 1) else 0;
}
fn positive(value: f32) bool {
    return std.math.isFinite(value) and value > 0;
}

/// Labels are lower bounds, matching native round-cost context classes.
pub fn contextClass(kv_length: usize) i64 {
    if (kv_length < 2048) return 0;
    var label: usize = 2048;
    while (label < 262144 and kv_length >= label * 2) label *= 2;
    return @intCast(label);
}

const Proposal = struct {
    token: u32 = 0,
    parent: u16 = 0,
    depth: u32 = 0,
    first_child: u16 = absent,
    next_sibling: u16 = absent,
    raw: f32 = 0,
    conditional: f32 = 0,
    reach: f32 = 0,
    origins: u8 = 0,
};

/// Canonical parents refer to this arena, never to truncated external row numbers.
const Proposals = struct {
    rows: [max_proposals + 1]Proposal = undefined,
    input_map: [max_proposals]u16 = @splat(absent),
    len: usize = 1,

    fn init() Proposals {
        var result: Proposals = .{};
        result.rows[0] = .{ .reach = 1 };
        return result;
    }

    fn insert(self: *Proposals, input: usize, parent: u16, token: u32, depth: u32, raw: f32, calibrated: f32, origins: u8) void {
        var child = self.rows[parent].first_child;
        while (child != absent) : (child = self.rows[child].next_sibling) {
            if (self.rows[child].token != token) continue;
            self.rows[child].raw = @max(self.rows[child].raw, probability(raw));
            self.rows[child].conditional = @max(self.rows[child].conditional, probability(calibrated));
            self.rows[child].origins |= origins;
            self.input_map[input] = child;
            return;
        }
        const row: u16 = @intCast(self.len);
        self.len += 1;
        self.rows[row] = .{
            .token = token,
            .parent = parent,
            .depth = depth,
            .raw = probability(raw),
            .conditional = probability(calibrated),
            .origins = origins,
            .next_sibling = self.rows[parent].first_child,
        };
        self.rows[parent].first_child = row;
        self.input_map[input] = row;
    }

    fn finish(self: *Proposals, anchor: u32, source: policy.NodeSource, order: u32) Tree {
        // One sibling projection, before pruning. Removed alternatives do not donate
        // probability to surviving alternatives; duplicate evidence is not extra mass.
        for (0..self.len) |parent| {
            var sum: f32 = 0;
            var child = self.rows[parent].first_child;
            while (child != absent) : (child = self.rows[child].next_sibling) sum += self.rows[child].conditional;
            child = self.rows[parent].first_child;
            while (child != absent) : (child = self.rows[child].next_sibling) {
                self.rows[child].conditional /= @max(1, sum);
                self.rows[child].reach = self.rows[parent].reach * self.rows[child].conditional;
            }
        }
        var selected: [max_proposals + 1]bool = @splat(false);
        selected[0] = true;
        var count: usize = 1;
        while (count < policy.MAX_TREE_ROWS and count < self.len) : (count += 1) {
            var best: ?usize = null;
            for (1..self.len) |row| {
                if (selected[row] or !selected[self.rows[row].parent]) continue;
                if (best == null or self.rows[row].reach > self.rows[best.?].reach) best = row;
            }
            const row = best orelse break;
            selected[row] = true;
        }
        var tree = Tree.init(anchor);
        var remap: [max_proposals + 1]u16 = @splat(absent);
        remap[0] = 0;
        for (1..self.len) |row| {
            if (!selected[row]) continue;
            const proposal = self.rows[row];
            const dst = tree.len;
            remap[row] = @intCast(dst);
            tree.tokens[dst] = proposal.token;
            tree.parents[dst] = @intCast(remap[proposal.parent]);
            tree.scores[dst] = proposal.reach;
            tree.conditional_reach[dst] = proposal.conditional;
            tree.sources[dst] = source;
            tree.origins[dst] = proposal.origins;
            tree.confidence[dst] = proposal.raw;
            tree.look_confidence[dst] = if (source == .look) proposal.raw else 0;
            tree.look_order[dst] = if (source == .look) order else 0;
            tree.draft_confidence[dst] = if (source == .draft) proposal.raw else 0;
            tree.len += 1;
        }
        tree.scores_calibrated = true;
        return tree;
    }
};

/// depth_limit counts drafts along a path, not rows across different branches.
/// At most the first max_proposals input records are examined.
pub fn lookupTree(anchor: u32, nodes: []const VoteNode, order: u32, calibration: *const Calibration, depth_limit: u32) Tree {
    var proposals = Proposals.init();
    const depth_cap = @min(depth_limit, policy.MAX_TREE_ROWS - 1);
    for (nodes[0..@min(nodes.len, max_proposals)], 0..) |node, input| {
        if (node.depth == 0 or node.depth > depth_cap) continue;
        const parent: u16 = if (node.parent) |index| blk: {
            if (index >= input) continue;
            break :blk proposals.input_map[index];
        } else 0;
        if (parent == absent or node.depth != proposals.rows[parent].depth + 1) continue;
        const kind: Calibration.Kind = if (parent == 0) .look else .look_continuation;
        proposals.insert(input, parent, node.token, node.depth, node.conditional_confidence, calibration.q(kind, node.conditional_confidence, order), node.origins);
    }
    return proposals.finish(anchor, .look, order);
}

/// Native lattice: -1 means the anchor and depth zero means the first draft.
/// Misaligned input tails, forward references and inconsistent depths are ignored.
pub fn neuralTree(anchor: u32, tokens: []const u32, parents: []const i32, depths: []const u32, confidence: []const f32, calibration: *const Calibration, depth_limit: u32) Tree {
    var proposals = Proposals.init();
    const len = @min(max_proposals, @min(@min(tokens.len, parents.len), @min(depths.len, confidence.len)));
    const depth_cap = @min(depth_limit, policy.MAX_TREE_ROWS - 1);
    for (0..len) |input| {
        if (depths[input] >= depth_cap) continue;
        const parent: u16 = if (parents[input] == -1) 0 else blk: {
            if (parents[input] < 0 or parents[input] >= @as(i32, @intCast(input))) continue;
            break :blk proposals.input_map[@intCast(parents[input])];
        };
        if (parent == absent or depths[input] != proposals.rows[parent].depth) continue;
        proposals.insert(input, parent, tokens[input], depths[input] + 1, confidence[input], calibration.q(.neural, confidence[input], 0), 8);
    }
    return proposals.finish(anchor, .draft, 0);
}

fn conditionals(tree: *const Tree) [policy.MAX_TREE_ROWS]f32 {
    var result: [policy.MAX_TREE_ROWS]f32 = @splat(0);
    for (1..tree.len) |row| {
        const parent: usize = @intCast(tree.parents[row]);
        result[row] = if (tree.scores[parent] > 0) probability(tree.scores[row] / tree.scores[parent]) else probability(tree.conditional_reach[row]);
    }
    return result;
}

fn project(tree: *Tree, conditional: *const [policy.MAX_TREE_ROWS]f32) void {
    tree.scores[0] = 1;
    for (0..tree.len) |parent| {
        var sum: f32 = 0;
        for (parent + 1..tree.len) |child| {
            if (tree.parents[child] == @as(i32, @intCast(parent))) sum += conditional[child];
        }
        for (parent + 1..tree.len) |child| {
            if (tree.parents[child] != @as(i32, @intCast(parent))) continue;
            tree.conditional_reach[child] = conditional[child] / @max(1, sum);
            tree.scores[child] = tree.scores[parent] * tree.conditional_reach[child];
        }
    }
}

fn validTree(tree: *const Tree) bool {
    if (tree.len == 0 or tree.len > policy.MAX_TREE_ROWS or tree.parents[0] != -1) return false;
    for (1..tree.len) |row| {
        if (tree.parents[row] < 0 or tree.parents[row] >= @as(i32, @intCast(row))) return false;
    }
    return true;
}

/// Merge only trees describing the same anchor. Capacity failure leaves descendants
/// unmapped, but later duplicate rows can still contribute evidence to retained rows.
pub fn merge(target: *Tree, other: *const Tree) void {
    if (!validTree(target) or !validTree(other) or target.tokens[0] != other.tokens[0]) return;
    const target_projected = target.scores_calibrated or target.len == 1;
    var conditional = conditionals(target);
    const incoming = conditionals(other);
    var remap: [policy.MAX_TREE_ROWS]u16 = @splat(absent);
    remap[0] = 0;
    for (1..other.len) |row| {
        const parent = remap[@intCast(other.parents[row])];
        if (parent == absent) continue;
        var match: ?usize = null;
        for (1..target.len) |dst| {
            if (target.parents[dst] == @as(i32, @intCast(parent)) and target.tokens[dst] == other.tokens[row]) {
                match = dst;
                break;
            }
        }
        const dst = match orelse blk: {
            if (target.len == policy.MAX_TREE_ROWS) continue;
            const added = target.len;
            target.len += 1;
            target.tokens[added] = other.tokens[row];
            target.parents[added] = @intCast(parent);
            target.sources[added] = other.sources[row];
            target.origins[added] = 0;
            target.confidence[added] = 0;
            target.draft_confidence[added] = 0;
            target.look_confidence[added] = 0;
            target.look_order[added] = 0;
            break :blk added;
        };
        remap[row] = @intCast(dst);
        if (other.sources[row] != .draft and
            (target.sources[dst] == .draft or target.look_confidence[dst] <= other.look_confidence[row]))
        {
            target.look_order[dst] = other.look_order[row];
        }
        if (target.sources[dst] != other.sources[row]) target.sources[dst] = .both;
        target.origins[dst] |= other.origins[row];
        target.confidence[dst] = @max(target.confidence[dst], probability(other.confidence[row]));
        target.draft_confidence[dst] = @max(target.draft_confidence[dst], probability(other.draft_confidence[row]));
        target.look_confidence[dst] = @max(target.look_confidence[dst], probability(other.look_confidence[row]));
        conditional[dst] = @max(conditional[dst], incoming[row]);
    }
    project(target, &conditional);
    target.scores_calibrated = target_projected and (other.scores_calibrated or other.len == 1);
}

pub const Alternative = enum { plain, lookup };
pub const Route = struct {
    decision: policy.RouteDecision,
    source: Alternative,
    plain_ms: ?f32,
    lookup_ms: ?f32,
    lookup_rows: u32,
    lookup_expected_tokens: f32,
    lookup_variant: ?policy.Variant,
    lookup_verify_ms: ?f32,
};

/// Opt-in diagnostics retain losing and unpriced ranked-prefix alternatives.
pub const RouteEvaluation = struct {
    pub const Entry = struct {
        rows: u32,
        variant: policy.Variant,
        verify_ms: ?f32,
        expected_tokens: f32,
        ms_per_token: ?f32,
    };
    entries: [policy.MAX_TREE_ROWS]Entry = undefined,
    len: usize = 0,
};

/// All diagnostic costs are milliseconds per expected committed token. Query time
/// is already paid and appears once in each alternative's complete-round numerator.
/// Assistant history retains native remaining work and yield, not a mixed cycle.
/// Unknown exact widths do not become actionable merely from extrapolated prices.
/// Prune only when bypassing the assistant; otherwise retain proposals for merging.
pub fn route(controller: *policy.Policy, lookup: *Tree, class: i64, query_ms: f32, row_limit: u32, dispatch: policy.TreeDispatch, evaluation: ?*RouteEvaluation) Route {
    if (evaluation) |log| log.len = 0;
    var plain: ?f32 = null;
    var lookup_cost: ?f32 = null;
    var best = lookup.*;
    var expected: f32 = 0;
    var best_verify_ms: ?f32 = null;
    const valid_preparation = std.math.isFinite(query_ms) and query_ms >= 0;
    if (valid_preparation) {
        if (controller.plainFor(class)) |ms| {
            const total = query_ms + ms;
            if (positive(total)) plain = total;
        }
        if (validTree(lookup)) {
            const cap = @min(lookup.len, @min(policy.MAX_TREE_ROWS, row_limit));
            if (cap >= 2) for (2..cap + 1) |width| {
                const candidate = controller.chooseTreeControlled(lookup, class, query_ms, @intCast(cap), @intCast(width), null, dispatch);
                const variant = candidate.variant(dispatch);
                const verify_ms = controller.prices.known(@intCast(candidate.len), .tree, class, variant);
                if (verify_ms == null and evaluation == null) continue;
                var reward: f32 = 0;
                for (candidate.scores[0..candidate.len]) |reach| reward += reach;
                const total: ?f32 = if (verify_ms) |ms| blk: {
                    const value = (query_ms + ms) / reward;
                    break :blk if (positive(value)) value else null;
                } else null;
                if (evaluation) |log| {
                    log.entries[log.len] = .{ .rows = @intCast(candidate.len), .variant = variant,
                        .verify_ms = verify_ms, .expected_tokens = reward, .ms_per_token = total };
                    log.len += 1;
                }
                const measured = total orelse continue;
                if (lookup_cost == null or measured < lookup_cost.?) {
                    lookup_cost = measured;
                    best = candidate;
                    expected = reward;
                    best_verify_ms = verify_ms;
                }
            };
        }
    }
    const source: Alternative = if (lookup_cost != null and (plain == null or lookup_cost.? < plain.?)) .lookup else .plain;
    const alternative = if (source == .lookup) lookup_cost else plain;
    const decision = controller.routeAgainstPrepared(class, alternative, query_ms);
    if (decision.skip) lookup.* = if (source == .lookup) best else Tree.init(lookup.tokens[0]);
    return .{
        .decision = decision,
        .source = source,
        .plain_ms = plain,
        .lookup_ms = lookup_cost,
        .lookup_rows = if (lookup_cost != null) @intCast(best.len) else 0,
        .lookup_expected_tokens = expected,
        .lookup_variant = if (lookup_cost != null) best.variant(dispatch) else null,
        .lookup_verify_ms = best_verify_ms,
    };
}

/// Build executable chain evidence from the exact token-prefix vote path.
/// Missing evidence truncates the offer; no continuation confidence is invented.
pub fn lookupChain(anchor: u32, tokens: []const u32, nodes: []const VoteNode, order: u32, calibration: *const Calibration, cap: u32) Tree {
    var tree = Tree.init(anchor);
    const count = @min(tokens.len, @min(policy.MAX_TREE_ROWS, @max(1, cap)) - 1);
    var matched: [max_proposals]bool = @splat(false);
    var found: [policy.MAX_TREE_ROWS]bool = @splat(false);
    var raw: [policy.MAX_TREE_ROWS]f32 = @splat(0);
    var origins: [policy.MAX_TREE_ROWS]u8 = @splat(0);
    for (nodes[0..@min(nodes.len, max_proposals)], 0..) |node, input| {
        if (node.depth == 0 or node.depth > count or node.token != tokens[node.depth - 1]) continue;
        if (node.parent) |parent| {
            if (parent >= input or !matched[parent] or nodes[parent].depth + 1 != node.depth) continue;
        } else if (node.depth != 1) continue;
        matched[input] = true;
        found[node.depth] = true;
        raw[node.depth] = @max(raw[node.depth], probability(node.conditional_confidence));
        origins[node.depth] |= node.origins;
    }
    for (tokens[0..count], 0..) |token, offset| {
        const row = offset + 1;
        if (!found[row]) break;
        const confidence = raw[row];
        tree.tokens[row] = token;
        tree.parents[row] = @intCast(offset);
        tree.sources[row] = .look;
        tree.origins[row] = origins[row];
        tree.confidence[row] = confidence;
        tree.look_confidence[row] = confidence;
        tree.look_order[row] = order;
        tree.conditional_reach[row] = calibration.q(if (offset == 0) .look else .look_continuation, confidence, order);
        tree.scores[row] = tree.scores[offset] * tree.conditional_reach[row];
        tree.len += 1;
    }
    tree.scores_calibrated = true;
    return tree;
}

fn vote(token: u32, parent: ?u32, depth: u32, confidence: f32, origins: u8) VoteNode {
    return .{ .token = token, .parent = parent, .depth = depth, .conditional_confidence = confidence, .origins = origins };
}

fn expectClosed(tree: *const Tree) !void {
    try std.testing.expect(validTree(tree));
    for (1..tree.len) |row| {
        const parent: usize = @intCast(tree.parents[row]);
        try std.testing.expect(std.math.isFinite(tree.scores[row]));
        try std.testing.expect(tree.scores[row] >= 0 and tree.scores[row] <= tree.scores[parent]);
    }
}

test "context labels are native lower bounds with saturated upper bucket" {
    try std.testing.expectEqual(@as(i64, 0), contextClass(2047));
    try std.testing.expectEqual(@as(i64, 2048), contextClass(3073));
    try std.testing.expectEqual(@as(i64, 4096), contextClass(4096));
    try std.testing.expectEqual(@as(i64, 131072), contextClass(250000));
    try std.testing.expectEqual(@as(i64, 262144), contextClass(std.math.maxInt(usize)));
}

test "assistant retention preserves proposals and plain routing selects only root" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    controller.prices.observe(2, 8, .tree, 0, .generic);
    _ = controller.observeSpecRemainingProbe(3, 3, 0, 0);
    var offer = Tree.chain(83, &.{ 41, 59, 26 }, 0.7, 0.8);
    const before = offer;
    const assistant = route(&controller, &offer, 0, 0.2, 4, .generic, null);
    try std.testing.expect(!assistant.decision.skip);
    try std.testing.expectEqual(before.len, offer.len);
    try std.testing.expectEqualSlices(u32, before.tokens[0..before.len], offer.tokens[0..offer.len]);

    var plain_controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer plain_controller.deinit();
    plain_controller.prices.observe(2, 50, .tree, 0, .generic);
    plain_controller.observePlain(2, 0);
    _ = plain_controller.observeSpecRemainingProbe(80, 2, 0, 0);
    const plain_route = route(&plain_controller, &offer, 0, 0.2, 4, .generic, null);
    try std.testing.expect(plain_route.decision.skip);
    try std.testing.expectEqual(Alternative.plain, plain_route.source);
    try std.testing.expectEqual(@as(usize, 1), offer.len);
    try std.testing.expectEqual(@as(u32, 83), offer.tokens[0]);
}

test "lookup rejects bad ancestry and keeps duplicate source evidence" {
    const cal: Calibration = .{};
    const nodes = [_]VoteNode{
        vote(17, null, 1, 0.4, 2),
        vote(17, null, 1, 0.6, 4),
        vote(23, 1, 2, 0.7, 4),
        vote(91, 8, 2, 1, 1),
        vote(92, 3, 3, 1, 1),
        vote(93, null, 2, 1, 1),
        vote(94, 0, 3, 1, 1),
    };
    const tree = lookupTree(5, &nodes, 7, &cal, 7);
    try std.testing.expectEqual(@as(usize, 3), tree.len);
    try std.testing.expectEqual(@as(u8, 6), tree.origins[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), tree.look_confidence[1], 0.00001);
    try std.testing.expectEqual(@as(u32, 7), tree.look_order[2]);
    try std.testing.expectEqual(@as(i32, 1), tree.parents[2]);
    try expectClosed(&tree);
}

test "calibration separates lookup entry from conditional continuation" {
    var cal: Calibration = .{};
    for (0..37) |_| {
        cal.learn(.look, 0.6, 2, false);
        cal.learn(.look_continuation, 0.6, 2, true);
    }
    const nodes = [_]VoteNode{ vote(31, null, 1, 0.6, 1), vote(32, 0, 2, 0.6, 1) };
    const tree = lookupTree(3, &nodes, 2, &cal, 2);
    try std.testing.expect(tree.scores[1] < 0.1);
    try std.testing.expect(tree.scores[2] / tree.scores[1] > 0.9);
    try std.testing.expectEqual(@as(f32, 0.6), tree.look_confidence[2]);
}

test "bounded pruning considers late alternatives and depth does not cap breadth" {
    const cal: Calibration = .{};
    var nodes: [max_proposals + 9]VoteNode = undefined;
    for (&nodes, 0..) |*node, index| node.* = vote(@intCast(index + 100), null, 1, if (index == max_proposals - 1) 0.9 else 0.001, 2);
    const tree = lookupTree(4, &nodes, 1, &cal, 1);
    try std.testing.expectEqual(@as(usize, policy.MAX_TREE_ROWS), tree.len);
    try std.testing.expectEqual(@as(u32, max_proposals - 1 + 100), tree.tokens[tree.len - 1]);
    for (tree.parents[1..tree.len]) |parent| try std.testing.expectEqual(@as(i32, 0), parent);
    try expectClosed(&tree);
}

test "neural lattice retains siblings and rejects nonnative parent markers" {
    const cal: Calibration = .{};
    const tree = neuralTree(7, &.{ 41, 42, 43, 44, 45 }, &.{ -1, -1, 0, -2, 3 }, &.{ 0, 0, 1, 0, 1 }, &.{ 0.55, 0.35, 0.8, 1, 1 }, &cal, 1);
    try std.testing.expectEqual(@as(usize, 3), tree.len);
    try std.testing.expectEqual(@as(u8, 8), tree.origins[2]);
    try std.testing.expectEqual(@as(i32, 0), tree.parents[2]);
    try expectClosed(&tree);
}

test "merge shares evidence without multiplying agreement and rescales descendants" {
    const cal: Calibration = .{};
    var target = neuralTree(9, &.{ 51, 52 }, &.{ -1, 0 }, &.{ 0, 1 }, &.{ 0.7, 0.4 }, &cal, 3);
    const nodes = [_]VoteNode{ vote(51, null, 1, 0.8, 4), vote(53, null, 1, 0.8, 2) };
    const other = lookupTree(9, &nodes, 17, &cal, 3);
    merge(&target, &other);
    try std.testing.expectEqual(@as(usize, 4), target.len);
    try std.testing.expectEqual(policy.NodeSource.both, target.sources[1]);
    try std.testing.expectEqual(@as(u8, 12), target.origins[1]);
    try std.testing.expectEqual(@as(f32, 0.7), target.draft_confidence[1]);
    try std.testing.expectEqual(@as(f32, 0.8), target.look_confidence[1]);
    try std.testing.expectEqual(@as(u32, 17), target.look_order[1]);
    try std.testing.expectEqual(@as(u32, 17), target.look_order[3]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), target.scores[2] / target.scores[1], 0.00001);
    try expectClosed(&target);
}

test "full merge never attaches dropped descendants to anchor" {
    const cal: Calibration = .{};
    var nodes: [31]VoteNode = undefined;
    for (&nodes, 0..) |*node, index| node.* = vote(@intCast(index + 60), null, 1, 0.02, 1);
    var target = lookupTree(2, &nodes, 1, &cal, 1);
    const incoming = [_]VoteNode{ vote(400, null, 1, 0.8, 2), vote(401, 0, 2, 0.9, 2), vote(60, null, 1, 0.3, 4) };
    const other = lookupTree(2, &incoming, 2, &cal, 3);
    merge(&target, &other);
    try std.testing.expectEqual(@as(usize, 32), target.len);
    try std.testing.expectEqual(@as(u8, 5), target.origins[1]);
    try std.testing.expectEqual(@as(u32, 2), target.look_order[1]);
    for (target.tokens[0..target.len]) |token| try std.testing.expect(token != 400 and token != 401);
    try expectClosed(&target);
}

test "chain evidence follows exact ancestry and truncates missing continuation" {
    const cal: Calibration = .{};
    const nodes = [_]VoteNode{ vote(71, null, 1, 0.6, 2), vote(72, 0, 2, 0.7, 2), vote(81, null, 1, 0.4, 4), vote(72, 2, 2, 0.9, 4) };
    const tree = lookupChain(1, &.{ 71, 72, 73 }, &nodes, 4, &cal, 4);
    try std.testing.expectEqual(@as(usize, 3), tree.len);
    try std.testing.expectEqual(@as(u8, 2), tree.origins[2]);
    try std.testing.expectEqual(@as(f32, 0.7), tree.look_confidence[2]);
    try expectClosed(&tree);
}

test "routing retains unknown assistant and accounts for paid query in both alternatives" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const nodes = [_]VoteNode{vote(101, null, 1, 1, 1)};
    var tree = lookupTree(10, &nodes, 1, &controller.cal, 1);
    var choice = route(&controller, &tree, 4096, 0, 2, .generic, null);
    try std.testing.expect(!choice.decision.skip);
    try std.testing.expect(choice.plain_ms == null and choice.lookup_ms == null);
    _ = controller.observeSpecRemainingProbe(8, 4, 4096, 0);
    controller.observePlain(2, 4096);
    controller.prices.observe(2, 3, .tree, 4096, .generic);
    choice = route(&controller, &tree, 4096, 5, 2, .generic, null);
    try std.testing.expectApproxEqAbs(@as(f32, 7), choice.plain_ms.?, 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 4), choice.lookup_ms.?, 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.25), choice.decision.assistant_ms.?, 0.00001);
    try std.testing.expect(!choice.decision.skip);
    choice = route(&controller, &tree, 4096, 0, 2, .generic, null);
    try std.testing.expectEqual(Alternative.lookup, choice.source);
    try std.testing.expect(choice.decision.skip);
    choice = route(&controller, &tree, 8192, 0, 2, .generic, null);
    try std.testing.expect(!choice.decision.skip);
    try std.testing.expect(choice.lookup_ms == null);
}

test "routing rejects invalid preparation instead of silently making it free" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    controller.observePlain(1, 0);
    _ = controller.observeSpecRemainingProbe(9, 1, 0, 0);
    var tree = Tree.init(3);
    const choice = route(&controller, &tree, 0, std.math.nan(f32), 8, .generic, null);
    try std.testing.expect(!choice.decision.skip);
    try std.testing.expect(choice.plain_ms == null);
}

test "pruned alternatives do not donate probability during policy selection" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var nodes: [47]VoteNode = undefined;
    for (&nodes, 0..) |*node, index| node.* = vote(@intCast(index + 200), null, 1, 0.5, 1);
    const tree = lookupTree(6, &nodes, 3, &controller.cal, 1);
    const selected = controller.chooseTreeControlled(&tree, 0, 0, 16, 16, null, .generic);
    try std.testing.expectEqual(@as(usize, 16), selected.len);
    for (selected.look_order[1..selected.len]) |order| try std.testing.expectEqual(@as(u32, 3), order);
    for (selected.scores[1..selected.len]) |reach| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 47.0), reach, 0.00001);
    }
}

test "nonfinite raw proposals remain bounded without losing parent closure" {
    const cal: Calibration = .{};
    const nodes = [_]VoteNode{
        vote(301, null, 1, std.math.nan(f32), 1),
        vote(302, 0, 2, std.math.inf(f32), 1),
        vote(303, null, 1, -4, 2),
        vote(304, null, 1, 5, 4),
    };
    const tree = lookupTree(8, &nodes, 2, &cal, 2);
    try expectClosed(&tree);
    try std.testing.expectEqual(@as(f32, 0), tree.look_confidence[1]);
    try std.testing.expectEqual(@as(f32, 1), tree.look_confidence[4]);
}

test "chain pricing and learning use each offered edge rather than root unanimity" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const nodes = [_]VoteNode{
        vote(31, null, 1, 1, 4),
        vote(32, 0, 2, 0.25, 4),
        vote(33, 1, 3, 0.8, 4),
    };
    const candidate = lookupChain(3, &.{ 31, 32, 33 }, &nodes, 2, &controller.cal, 4);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(std.testing.allocator);
    const choice = try controller.chooseLookPreparedControlled(&candidate, 0, 0, &out, 4, .unknown);
    try std.testing.expectApproxEqAbs(@as(f32, 1.45), choice.expected, 0.00001);
    var receipt_expected: f32 = 0;
    for (candidate.scores[1..candidate.len]) |reach| receipt_expected += reach;
    try std.testing.expectApproxEqAbs(receipt_expected, choice.expected, 0.00001);
    controller.observeChainAccepted(&candidate, @intCast(candidate.len - 1), 1);
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look_continuation));
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), controller.cal.q(.look_continuation, 0.25, 2), 0.00001);
    try std.testing.expectEqual(@as(f32, 0.8), controller.cal.q(.look_continuation, 0.8, 2));
    try std.testing.expectEqual(@as(f32, 0.25), controller.cal.q(.look_continuation, 0.25, 1));
}

test "reached neural path calibrates unselected lookup without rejected-branch evidence" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const selected = neuralTree(10, &.{ 20, 21, 99, 100 }, &.{ -1, 0, -1, 2 }, &.{ 0, 1, 0, 1 }, &.{ 0.8, 0.7, 0.2, 0.9 }, &controller.cal, 3);
    const nodes = [_]VoteNode{
        vote(20, null, 1, 0.9, 4),
        vote(22, 0, 2, 0.6, 4),
        vote(23, 1, 3, 0.7, 4),
        vote(30, null, 1, 0.4, 2),
        vote(31, 3, 2, 0.3, 2),
    };
    const offered = lookupTree(10, &nodes, 2, &controller.cal, 3);
    // No target decision survives a cancellation: neither offer gains observations.
    controller.observeTree(&selected, &.{}, .neural, null);
    controller.observeTree(&offered, &.{}, .lookup, null);
    try std.testing.expectEqual(@as(f32, 0), controller.cal.samples(.neural));
    try std.testing.expectEqual(@as(f32, 0), controller.cal.samples(.look));
    // Native execution reaches 20, then 21, then the correction 24. In particular,
    // logits belonging to the rejected native row 99 are not a reached decision.
    controller.observeTree(&selected, &.{ 20, 21, 24 }, .neural, null);
    controller.observeTree(&offered, &.{ 20, 21, 24 }, .lookup, null);
    try std.testing.expectEqual(@as(f32, 3), controller.cal.samples(.neural));
    try std.testing.expectEqual(@as(f32, 2), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look_continuation));
    try std.testing.expectApproxEqAbs(@as(f32, 0.32), controller.cal.q(.look, 0.4, 2), 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.48), controller.cal.q(.look_continuation, 0.6, 2), 0.00001);
    try std.testing.expectEqual(@as(f32, 0.7), controller.cal.q(.look_continuation, 0.7, 2));
    try std.testing.expectEqual(@as(f32, 0.3), controller.cal.q(.look_continuation, 0.3, 2));
    try std.testing.expectEqual(@as(f32, 0.9), controller.cal.q(.neural, 0.9, 0));
    try std.testing.expectEqual(@as(f32, 0.6), controller.cal.q(.look_continuation, 0.6, 1));
}

test "merged selected lookup success is learned once and missing next decision is censored" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const nodes = [_]VoteNode{ vote(2, null, 1, 0.8, 4), vote(3, 0, 2, 0.6, 4) };
    const offered = lookupTree(1, &nodes, 1, &controller.cal, 2);
    var merged = neuralTree(1, &.{2}, &.{-1}, &.{0}, &.{0.7}, &controller.cal, 1);
    merge(&merged, &offered);
    const selected = controller.chooseTreeControlled(&merged, 0, 0, 2, 2, null, .generic);
    try std.testing.expectEqual(policy.NodeSource.both, selected.sources[1]);
    controller.observeTree(&selected, &.{2}, .neural, null);
    controller.observeTree(&offered, &.{2}, .lookup, null);
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.neural));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 0), controller.cal.samples(.look_continuation));
    try std.testing.expectApproxEqAbs(@as(f32, 0.84), controller.cal.q(.look, 0.8, 1), 0.00001);
    // A separately reached bonus can label the unselected next edge, but nothing
    // after the first differing lookup edge is attributed as a failure.
    controller.observeTree(&selected, &.{ 2, 4 }, .neural, null);
    controller.observeTree(&offered, &.{ 2, 4 }, .lookup, null);
    try std.testing.expectEqual(@as(f32, 2), controller.cal.samples(.look));
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look_continuation));
}

test "request lookup restriction preserves excluded sibling mass and raw evidence" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    var nodes: [47]VoteNode = undefined;
    for (&nodes, 0..) |*node, index| node.* = vote(@intCast(100 + index), null, 1, 1, 4);
    const offered = lookupTree(1, &nodes, 2, &controller.cal, 1);
    var trust: policy.RequestTrust = .{};
    const cold = controller.restrictLookup(&offered, &trust);
    try std.testing.expectEqual(offered.len, cold.len);
    try std.testing.expectEqualSlices(u32, offered.tokens[0..offered.len], cold.tokens[0..cold.len]);
    for (cold.scores[1..cold.len]) |reach| try std.testing.expectEqual(@as(f32, 0), reach);
    trust.learn(.look, 1, true);
    const supported = controller.restrictLookup(&offered, &trust);
    for (supported.scores[1..supported.len]) |reach| {
        try std.testing.expectApproxEqAbs(@as(f32, 2.0 / (3.0 * 47.0)), reach, 0.00001);
    }
    try std.testing.expectEqualSlices(f32, offered.look_confidence[0..offered.len], supported.look_confidence[0..supported.len]);
    try std.testing.expectEqualSlices(u32, offered.look_order[0..offered.len], supported.look_order[0..supported.len]);
    try expectClosed(&supported);
}

test "cold request lookup cannot displace neural selection but retains shadow continuation" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const nodes = [_]VoteNode{ vote(11, null, 1, 1, 4), vote(12, null, 1, 1, 4), vote(13, 0, 2, 1, 4) };
    const offered = lookupTree(1, &nodes, 1, &controller.cal, 2);
    var trust: policy.RequestTrust = .{};
    const cold = controller.restrictLookup(&offered, &trust);
    var merged = neuralTree(1, &.{11}, &.{-1}, &.{0}, &.{0.8}, &controller.cal, 1);
    merge(&merged, &cold);
    const selected = controller.chooseTreeControlled(&merged, 0, 0, 2, 2, null, .generic);
    try std.testing.expectEqualSlices(u32, &.{ 1, 11 }, selected.tokens[0..selected.len]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), selected.scores[1], 0.00001);
    controller.observeTree(&selected, &.{ 11, 13 }, .neural, null);
    // The already known bonus teaches the unselected next lookup edge.
    controller.observeTree(&offered, &.{ 11, 13 }, .lookup, &trust);
    try std.testing.expectEqual(@as(f32, 1), controller.cal.samples(.look_continuation));
    try std.testing.expect(trust.q(.look, 1) > 0);
    try std.testing.expect(trust.q(.look_continuation, 1) > 0);
    const supported = controller.restrictLookup(&offered, &trust);
    try std.testing.expect(supported.scores[3] > 0);
    try expectClosed(&supported);
}

test "lookup supported continuation survives zero entry trust and neural parent agreement" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const nodes = [_]VoteNode{ vote(11, null, 1, 1, 4), vote(12, 0, 2, 0.6, 4) };
    const offered = lookupTree(10, &nodes, 2, &controller.cal, 2);
    var trust: policy.RequestTrust = .{};
    trust.learn(.look_continuation, 0.6, true);
    const restricted = controller.restrictLookup(&offered, &trust);
    try std.testing.expectEqual(@as(f32, 0), restricted.scores[1]);
    try std.testing.expectEqual(@as(f32, 0), restricted.scores[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), restricted.conditional_reach[2], 0.00001);
    var neural = neuralTree(10, &.{11}, &.{-1}, &.{0}, &.{0.7}, &controller.cal, 1);
    merge(&neural, &restricted);
    try std.testing.expectApproxEqAbs(@as(f32, 0.42), neural.scores[2], 0.00001);
    const selected = controller.chooseTreeControlled(&neural, 0, 0, 3, 3, null, .generic);
    try std.testing.expectApproxEqAbs(@as(f32, 0.42), selected.scores[2], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), selected.conditional_reach[2], 0.00001);
    try expectClosed(&selected);
}

test "lookup-only root merge cannot bypass request restriction through recalibration" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    const nodes = [_]VoteNode{ vote(11, null, 1, 1, 4), vote(12, 0, 2, 1, 4) };
    const offered = lookupTree(10, &nodes, 2, &controller.cal, 2);
    const trust: policy.RequestTrust = .{};
    const restricted = controller.restrictLookup(&offered, &trust);
    var candidate = Tree.init(10);
    merge(&candidate, &restricted);
    try std.testing.expect(candidate.scores_calibrated);
    const selected = controller.chooseTreeControlled(&candidate, 0, 0, 3, 3, null, .generic);
    for (selected.scores[1..selected.len]) |reach| try std.testing.expectEqual(@as(f32, 0), reach);
}

test "lookup routing cannot bypass the assistant using another kernel variant price" {
    var controller = policy.Policy.init(policy.PriceModel.init(std.testing.allocator));
    defer controller.deinit();
    _ = controller.observeSpecRemainingProbe(20, 1, 0, 0);
    controller.prices.observe(9, 1, .tree, 0, .generic);
    var offer = Tree.chain(10, &.{ 11, 12, 13, 14, 15, 16, 17, 18 }, 1, 1);
    const unknown = route(&controller, &offer, 0, 0, 9, .linear_gdn, null);
    try std.testing.expect(!unknown.decision.skip);
    try std.testing.expect(unknown.lookup_ms == null);
    try std.testing.expectEqual(@as(usize, 9), offer.len);
    controller.prices.observe(9, 1, .tree, 0, .linear_gdn);
    const measured = route(&controller, &offer, 0, 0, 9, .linear_gdn, null);
    try std.testing.expect(measured.decision.skip);
    try std.testing.expectEqual(Alternative.lookup, measured.source);
    try std.testing.expectEqual(@as(u32, 9), measured.lookup_rows);
}
