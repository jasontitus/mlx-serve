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

    pub fn init(gen: *Generator, allocator: std.mem.Allocator) !Request {
        const identity = gen.tok.content_sha256 orelse return error.TokenizerRequired;
        var self = Request{ .allocator = allocator, .drafter = tokens.Drafter.init(allocator) };
        errdefer self.deinit();
        self.drafter.vocab_limit = @intCast(gen.tok.definedVocabSize());
        const shared = cache(gen.xfm);
        if (corpusPath()) |path| {
            self.corpus = try shared.acquireCorpus(gen.timer.io, path, identity);
            if (self.corpus) |*lease| self.drafter.corpus = lease.index();
        }
        if (storePath()) |path| {
            if (writes()) try tokens.initializePersistent(gen.timer.io, allocator, path, identity);
            self.persistent = try shared.acquirePersistent(gen.timer.io, path, identity);
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
        const n = @min(committed.len, tokens.max_order - 1);
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
