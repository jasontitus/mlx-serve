//! External token history supplies proposals to the stock MTP verifier/gate.
//! No native head policy, numerical mode, or proposal probability is changed.
const std = @import("std");
const env = @import("../../../env.zig");
const tokens = @import("token_history.zig");
const lookup = @import("mtp_lookup.zig");
const Generator = @import("../generate/generator.zig").Generator;
const Transformer = @import("../transformer.zig").Transformer;
const log = @import("../../../sdk/util/log.zig");

pub fn storePath() ?[]const u8 {
    const path = env.span(.MLX_SERVE_LOOKUP_STORE) orelse return null;
    return if (path.len > 0) path else null;
}

fn corpusPath() ?[]const u8 {
    const path = env.span(.MLX_SERVE_LOOKUP_CORPUS) orelse return null;
    return if (path.len > 0) path else null;
}

pub fn enabled() bool {
    return storePath() != null or corpusPath() != null;
}

pub fn writes() bool {
    return storePath() != null and !env.on(.MLX_SERVE_LOOKUP_READ_ONLY);
}

pub fn cache(xfm: *Transformer) *tokens.HistoryCache {
    if (xfm.token_history == null) xfm.token_history = tokens.HistoryCache.init(xfm.allocator);
    return &xfm.token_history.?;
}

pub const Request = struct {
    allocator: std.mem.Allocator,
    drafter: tokens.Drafter,
    corpus: ?tokens.CorpusLease = null,
    persistent: ?tokens.PersistentLease = null,
    output: std.ArrayList(u32) = .empty,
    query: [tokens.max_order]u32 = undefined,
    query_len: usize = 0,
    query_cap: u32 = 0,
    matched_order: u32 = 0,
    queried: bool = false,
    rounds: u32 = 0,
    accepted: u32 = 0,
    /// DFlash's local tier advances only through engine-committed tokens.
    local_primed: bool = false,
    local_generated: usize = 0,

    pub fn init(gen: *Generator, allocator: std.mem.Allocator) !Request {
        const identity = gen.tok.content_sha256;
        if (enabled() and identity == null) return error.TokenizerRequired;
        var self = Request{ .allocator = allocator, .drafter = tokens.Drafter.init(allocator) };
        errdefer self.deinit();
        self.drafter.vocab_limit = @intCast(gen.tok.definedVocabSize());
        if (corpusPath()) |path| {
            self.corpus = try cache(gen.xfm).acquireCorpus(gen.timer.io, path, identity);
            if (self.corpus) |*lease| self.drafter.corpus = lease.index();
        }
        if (storePath()) |path| {
            if (writes()) try tokens.initializePersistent(gen.timer.io, allocator, path, identity.?);
            self.persistent = try cache(gen.xfm).acquirePersistent(gen.timer.io, path, identity.?);
            self.drafter.persistent = self.persistent.?.index();
        }
        return self;
    }

    pub fn deinit(self: *Request) void {
        self.drafter.deinit();
        self.output.deinit(self.allocator);
        if (self.corpus) |*lease| lease.deinit();
        if (self.persistent) |*lease| lease.deinit();
        self.* = undefined;
    }

    /// Cache only the immutable candidate, never the changing admission gate.
    /// A lazy successor probe and its build can ask the same question twice.
    pub fn match(self: *Request, committed: []const u32, t1: u32, cap: u32) !?lookup.Match {
        const n: usize = @min(committed.len, tokens.max_order - 1);
        var query: [tokens.max_order]u32 = undefined;
        @memcpy(query[0..n], committed[committed.len - n ..]);
        query[n] = t1;
        const length = n + 1;
        if (!self.queried or self.query_cap != cap or self.query_len != length or
            !std.mem.eql(u32, self.query[0..self.query_len], query[0..length]))
        {
            const got = try self.drafter.look(query[0..length], cap, &self.output);
            @memcpy(self.query[0..length], query[0..length]);
            self.query_len = length;
            self.query_cap = cap;
            self.matched_order = got.order;
            self.queried = true;
        }
        if (self.output.items.len == 0) return null;
        return .{ .draft = self.output.items, .suffix = self.matched_order };
    }
};

pub fn match(gen: *Generator, allocator: std.mem.Allocator, committed: []const u32, t1: u32, cap: u32) !?lookup.Match {
    if (!enabled()) return null;
    if (gen.external_history == null) {
        const request = try allocator.create(Request);
        errdefer allocator.destroy(request);
        request.* = try Request.init(gen, allocator);
        gen.external_history = request;
    }
    return gen.external_history.?.match(committed, t1, cap);
}

/// The tree result borrows request scratch until the next query. Callers must
/// turn it into their owned flight geometry before another query or suspension.
/// MTP's cached single-candidate match and terminal publication are unchanged.
pub fn queryTree(gen: *Generator, allocator: std.mem.Allocator, depth: u32) !tokens.Look {
    if (gen.external_history == null) {
        const request = try allocator.create(Request);
        errdefer allocator.destroy(request);
        request.* = try Request.init(gen, allocator);
        gen.external_history = request;
    }
    const request = gen.external_history.?;
    if (!request.local_primed) {
        try request.drafter.prime(gen.prompt_ids_owned);
        request.local_primed = true;
    }
    const generated = gen.generated_ids.items;
    std.debug.assert(request.local_generated <= generated.len);
    try request.drafter.observe(generated[request.local_generated..]);
    request.local_generated = generated.len;
    var query: [tokens.max_order]u32 = undefined;
    const n: usize = @min(gen.prompt_ids_owned.len + generated.len, query.len - 1);
    const from_generated = @min(n, generated.len);
    const from_prompt = n - from_generated;
    @memcpy(query[0..from_prompt], gen.prompt_ids_owned[gen.prompt_ids_owned.len - from_prompt ..]);
    @memcpy(query[from_prompt..n], generated[generated.len - from_generated ..]);
    query[n] = gen.next_token_id;
    request.queried = false;
    return request.drafter.look(query[0 .. n + 1], depth, &request.output);
}

pub fn logStats(gen: *const Generator) void {
    if (gen.external_history) |request| log.info("[history] external_rounds={d} accepted_drafts={d}\n", .{ request.rounds, request.accepted });
}

/// Called only at a terminal slot boundary, never a retry's generator teardown.
pub fn publish(slot: anytype, outcome: @import("history_publication.zig").Outcome) void {
    if (slot.history_publication.finalized) return;
    if (slot.legacy_gen) |*gen| logStats(gen);
    const generated = if (slot.legacy_gen) |*gen| gen.generated_ids.items else &.{};
    const output = slot.history_publication.finish(generated, outcome) catch |err| {
        log.warn("[history] publication refused: {s}\n", .{@errorName(err)});
        return;
    };
    if (!writes()) return;
    const ids = output orelse return;
    const tok = slot.model.tokenizer orelse return;
    const identity = tok.content_sha256 orelse {
        log.warn("[history] publication refused: TokenizerRequired\n", .{});
        return;
    };
    const xfm = slot.model.transformer orelse return;
    cache(xfm).appendFinalTurn(slot.io, storePath().?, identity, slot.full_prompt, ids) catch |err| {
        log.warn("[history] publication failed: {s}\n", .{@errorName(err)});
        return;
    };
    log.info("[history] published prompt_tokens={d} output_tokens={d} outcome={s}\n", .{ slot.full_prompt.len, ids.len, @tagName(outcome) });
}

test "external history query retains maximum suffix and invalidates changed context" {
    const a = std.testing.allocator;
    var ids: [100]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(100 + i);
    var corpus = try tokens.Corpus.build(a, &ids, tokens.max_order);
    defer corpus.deinit();
    var request = Request{ .allocator = a, .drafter = tokens.Drafter.init(a) };
    defer request.deinit();
    request.drafter.corpus = &corpus;
    for ([_]usize{ 62, 63, 64, 90 }) |n| {
        const matched = (try request.match(ids[0..n], ids[n], 7)) orelse return error.MissingContinuation;
        try std.testing.expectEqualSlices(u32, ids[n + 1 ..][0..7], matched.draft);
        try std.testing.expectEqual(@as(u32, @intCast(@min(n + 1, tokens.max_order))), matched.suffix);
        const repeated = (try request.match(ids[0..n], ids[n], 3)) orelse return error.MissingShortContinuation;
        try std.testing.expectEqualSlices(u32, ids[n + 1 ..][0..3], repeated.draft);
    }
}

test "external history tree query joins prompt and emitted suffix at maximum order" {
    const a = std.testing.allocator;
    var ids: [100]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(100 + i);
    var corpus = try tokens.Corpus.build(a, &ids, tokens.max_order);
    defer corpus.deinit();
    var request = Request{ .allocator = a, .drafter = tokens.Drafter.init(a) };
    defer request.deinit();
    request.drafter.corpus = &corpus;
    var gen: Generator = undefined;
    gen.external_history = &request;
    gen.prompt_ids_owned = ids[0..60];
    for ([_]usize{ 62, 63, 64, 90 }) |n| {
        gen.generated_ids = .initBuffer(ids[60..n]);
        gen.generated_ids.items = ids[60..n];
        gen.next_token_id = ids[n];
        const got = try queryTree(&gen, a, 7);
        try std.testing.expectEqualSlices(u32, ids[n + 1 ..][0..7], got.tokens);
        try std.testing.expectEqual(@as(u32, @intCast(@min(n + 1, tokens.max_order))), got.order);
    }
}
