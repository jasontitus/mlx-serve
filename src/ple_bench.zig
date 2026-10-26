//! Opt-in CPU hash + learned-embedding gather benchmark, not token drafting.
//! No model weights are materialized and no MLX/Metal library is linked.
const std = @import("std");
const builtin = @import("builtin");
const qwen4 = @import("qwen4_exp.zig");
const ple = @import("qwen4_ple.zig");
const model = @import("model.zig");
const tokenizer = @import("tokenizer.zig");
pub const io_util = @import("io_util.zig");
pub const log = @import("log.zig");
// The config/tokenizer sidecar modules share the host's types; none of these FFI calls run.
pub const mlx = @import("mlx.zig");
pub const sushi_exl3 = @import("sushi_exl3");
pub const mtp_acceptance = @import("mtp_acceptance.zig");
pub const dsa_nax = @import("dsa_nax.zig");

const Arm = qwen4.NgramTable.BenchmarkArm;
const widths = [_]usize{ 1, 2, 4, 8 };
const bootstrap_trials = 2048;
const Options = struct {
    model_dir: []const u8 = "",
    output: []const u8 = "",
    text: ?[]const u8 = null,
    samples: usize = 256,
    warm_repeats: usize = 8,
    seed: u64 = 0x504c455f42454e43,
};
const Timing = struct { hash_ns: u64, gather_ns: u64, combined_ns: u64 };
const Pair = struct { serial: Timing, pooled: Timing };
const Block = struct {
    token_offset: usize,
    context_and_tokens: []const u32,
    first_arm: Arm,
    first_pass: Pair,
    warm: []Pair,
    rows_not_seen_earlier_in_run: usize,
    production_pool_rounds: u64,
};
const Distribution = struct { count: usize, min: u64, p50: u64, p90: u64, p95: u64, p99: u64, max: u64, mean: f64 };
const TimingDistribution = struct { hash_ns: Distribution, gather_ns: Distribution, combined_ns: Distribution };
const Interval = struct { ratio_of_means: f64, bootstrap_p025: f64, bootstrap_p975: f64 };
const Result = struct {
    batch_tokens: usize,
    rows_per_call: usize,
    packed_payload_bytes_per_call: u64,
    output_f32_bytes_per_call: u64,
    first_pass_serial: TimingDistribution,
    first_pass_pooled: TimingDistribution,
    warm_serial: TimingDistribution,
    warm_pooled: TimingDistribution,
    warm_serial_over_pooled: Interval,
    production_pool_attempted_blocks: usize,
    blocks: []Block,
};
const FileIdentity = struct { name: []const u8, file_bytes: u64, header_sha256: []const u8 };

fn hashHex(a: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(digest, .lower));
}

fn parseOptions(args: []const []const u8) !Options {
    var options: Options = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.MissingOptionValue;
        const key = args[i];
        const value = args[i + 1];
        if (std.mem.eql(u8, key, "--model")) options.model_dir = value else if (std.mem.eql(u8, key, "--output")) options.output = value else if (std.mem.eql(u8, key, "--text")) options.text = value else if (std.mem.eql(u8, key, "--samples")) options.samples = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, key, "--warm-repeats")) options.warm_repeats = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, key, "--seed")) options.seed = try std.fmt.parseInt(u64, value, 0) else return error.UnknownOption;
    }
    if (options.model_dir.len == 0 or options.output.len == 0) return error.ModelAndOutputRequired;
    if (options.samples < 2 or options.samples % 2 != 0 or options.warm_repeats < 2 or options.warm_repeats % 2 != 0) return error.NeedEvenSamplesAndWarmRepeats;
    return options;
}

fn env(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    return std.mem.sliceTo(raw, 0);
}

/// Independent synthetic TEXT, tokenized with the actual pack tokenizer; never synthetic rows.
fn makeText(a: std.mem.Allocator, samples: usize, seed: u64) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    var rng = std.Random.DefaultPrng.init(seed);
    const places = [_][]const u8{ "harbor", "orchard", "library", "station", "workshop", "garden", "market", "observatory" };
    const items = [_][]const u8{ "copper wire", "blue notebooks", "glass lenses", "wooden crates", "canvas bags", "steel bolts", "paper maps", "cotton cloth" };
    const count = try std.math.mul(usize, samples, 8);
    for (0..count) |i| {
        const line = try std.fmt.allocPrint(a, "Record {d}: at the {s}, order {d} contains {d} units of {s}. The clerk records reference {x} and schedules delivery after {d} days.\n", .{
            i, places[rng.random().uintLessThan(usize, places.len)], rng.random().int(u32),
            rng.random().uintLessThan(u32, 10000), items[rng.random().uintLessThan(usize, items.len)],
            rng.random().int(u64), rng.random().uintLessThan(u32, 90),
        });
        try text.appendSlice(a, line);
        a.free(line);
    }
    return text.toOwnedSlice(a);
}

fn measure(io: std.Io, table: *const qwen4.NgramTable, hash: *const qwen4.NgramHash, arm: Arm, prev: []const u32, tokens: []const u32, rows: []i64, out: []f32) !Timing {
    const sw = io_util.Stopwatch.init(io);
    hash.rowIds(prev, tokens, rows);
    const hash_ns = sw.read();
    try table.gatherForBenchmark(arm, rows, out);
    const combined_ns = sw.read();
    return .{ .hash_ns = hash_ns, .gather_ns = combined_ns - hash_ns, .combined_ns = combined_ns };
}

fn pair(io: std.Io, table: *const qwen4.NgramTable, hash: *const qwen4.NgramHash, first: Arm, prev: []const u32, tokens: []const u32, rows: []i64, serial: []f32, pooled: []f32, digest: *std.crypto.hash.sha2.Sha256) !Pair {
    var result: Pair = undefined;
    if (first == .serial) {
        result.serial = try measure(io, table, hash, .serial, prev, tokens, rows, serial);
        result.pooled = try measure(io, table, hash, .pooled, prev, tokens, rows, pooled);
    } else {
        result.pooled = try measure(io, table, hash, .pooled, prev, tokens, rows, pooled);
        result.serial = try measure(io, table, hash, .serial, prev, tokens, rows, serial);
    }
    // Byte equality deliberately includes signed zero and NaN payloads, unlike float ==.
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(serial), std.mem.sliceAsBytes(pooled))) return error.EmbeddingVectorMismatch;
    digest.update(std.mem.sliceAsBytes(serial));
    return result;
}

fn distribution(a: std.mem.Allocator, values: []u64) !Distribution {
    if (values.len == 0) return error.NoSamples;
    const sorted = try a.dupe(u64, values);
    defer a.free(sorted);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
    var sum: f64 = 0;
    for (values) |value| sum += @floatFromInt(value);
    return .{
        .count = values.len, .min = sorted[0], .p50 = sorted[(sorted.len - 1) * 50 / 100],
        .p90 = sorted[(sorted.len - 1) * 90 / 100], .p95 = sorted[(sorted.len - 1) * 95 / 100],
        .p99 = sorted[(sorted.len - 1) * 99 / 100], .max = sorted[sorted.len - 1], .mean = sum / @as(f64, @floatFromInt(values.len)),
    };
}

fn summarize(a: std.mem.Allocator, blocks: []const Block, arm: Arm, warm: bool) !TimingDistribution {
    var hashes: std.ArrayList(u64) = .empty;
    defer hashes.deinit(a);
    var gathers: std.ArrayList(u64) = .empty;
    defer gathers.deinit(a);
    var combined: std.ArrayList(u64) = .empty;
    defer combined.deinit(a);
    for (blocks) |block| {
        if (!warm and block.first_arm != arm) continue;
        const pairs = if (warm) block.warm else @as([]const Pair, &.{block.first_pass});
        for (pairs) |sample| {
            const t = if (arm == .serial) sample.serial else sample.pooled;
            try hashes.append(a, t.hash_ns);
            try gathers.append(a, t.gather_ns);
            try combined.append(a, t.combined_ns);
        }
    }
    return .{ .hash_ns = try distribution(a, hashes.items), .gather_ns = try distribution(a, gathers.items), .combined_ns = try distribution(a, combined.items) };
}

/// Resample whole context blocks, NOT individual repeated warm calls (pseudoreplication).
fn uncertainty(a: std.mem.Allocator, blocks: []const Block, seed: u64) !Interval {
    const serial = try a.alloc(f64, blocks.len);
    defer a.free(serial);
    const pooled = try a.alloc(f64, blocks.len);
    defer a.free(pooled);
    var serial_sum: f64 = 0;
    var pooled_sum: f64 = 0;
    for (blocks, 0..) |block, i| {
        serial[i] = 0;
        pooled[i] = 0;
        for (block.warm) |sample| {
            serial[i] += @floatFromInt(sample.serial.combined_ns);
            pooled[i] += @floatFromInt(sample.pooled.combined_ns);
        }
        serial_sum += serial[i];
        pooled_sum += pooled[i];
    }
    if (pooled_sum == 0) return error.ClockResolutionInsufficient;
    var rng = std.Random.DefaultPrng.init(seed);
    var ratios: [bootstrap_trials]f64 = undefined;
    for (&ratios) |*ratio| {
        var s: f64 = 0;
        var p: f64 = 0;
        for (0..blocks.len) |_| {
            const i = rng.random().uintLessThan(usize, blocks.len);
            s += serial[i];
            p += pooled[i];
        }
        if (p == 0) return error.ClockResolutionInsufficient;
        ratio.* = s / p;
    }
    std.mem.sort(f64, &ratios, {}, std.sort.asc(f64));
    return .{ .ratio_of_means = serial_sum / pooled_sum, .bootstrap_p025 = ratios[(bootstrap_trials - 1) * 25 / 1000], .bootstrap_p975 = ratios[(bootstrap_trials - 1) * 975 / 1000] };
}

fn identities(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, index: []const u8) ![]FileIdentity {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, index, .{});
    defer parsed.deinit();
    const weights = parsed.value.object.get("weight_map").?.object;
    var result: std.ArrayList(FileIdentity) = .empty;
    var it = weights.iterator();
    while (it.next()) |entry| {
        if (!ple.embeddedTensorName(entry.key_ptr.*)) continue;
        const name = entry.value_ptr.string;
        var found = false;
        for (result.items) |file| if (std.mem.eql(u8, file.name, name)) { found = true; break; };
        if (found) continue;
        const file = try dir.openFile(io, name, .{});
        defer file.close(io);
        const stat = try file.stat(io);
        var prefix: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &prefix, 0) != prefix.len) return error.TruncatedTableHeader;
        const len = std.mem.readInt(u64, &prefix, .little);
        if (stat.size < 8 or len > 64 * 1024 * 1024 or len > stat.size - 8) return error.InvalidTableHeader;
        const header = try a.alloc(u8, @intCast(len));
        defer a.free(header);
        if (try file.readPositionalAll(io, header, 8) != len) return error.TruncatedTableHeader;
        try result.append(a, .{ .name = try a.dupe(u8, name), .file_bytes = stat.size, .header_sha256 = try hashHex(a, header) });
    }
    return result.toOwnedSlice(a);
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = init.io;
    var iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer iter.deinit();
    var args: std.ArrayList([]const u8) = .empty;
    while (iter.next()) |arg| try args.append(a, arg);
    if (args.items.len == 2 and std.mem.eql(u8, args.items[1], "--help")) {
        std.debug.print("ple-bench --model PACK --output RESULT.json [--samples 256] [--warm-repeats 8] [--seed INTEGER] [--text CORPUS.txt]\nCPU-only learned embedding vectors; first-touch OS cache is uncontrolled. Samples/repeats must be even and >=2.\n", .{});
        return;
    }
    const options = try parseOptions(args.items);
    var dir = try std.Io.Dir.cwd().openDir(io, options.model_dir, .{});
    defer dir.close(io);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const pack_path = path_buf[0..path_len];
    const config_bytes = try dir.readFileAlloc(io, "config.json", a, .limited(16 * 1024 * 1024));
    var config = try model.parseConfigFromJson(a, config_bytes);
    defer config.deinit(a);
    if (!config.isQwen4()) return error.ExpectedQwenFlashNextPack;
    const hash = try qwen4.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, config.ngram_eos);
    const spec = try model.qwen4EmbeddedSpec(&config);
    const tokenizer_bytes = try dir.readFileAlloc(io, "tokenizer.json", a, .limited(64 * 1024 * 1024));
    var tok = try tokenizer.loadTokenizer(io, a, pack_path);
    defer tok.deinit();
    if (!tok.id_to_token.contains(config.ngram_eos)) return error.TokenizerEosMismatch;
    const text = if (options.text) |path| try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(256 * 1024 * 1024)) else try makeText(a, options.samples, options.seed);
    const tokens = try tok.encode(a, text);
    for (tokens) |id| if (id >= config.vocab_size or !tok.id_to_token.contains(id)) return error.TokenizerVocabMismatch;
    const ctx: usize = hash.ngram_size - 1;
    const block_count = try std.math.mul(usize, options.samples, widths.len);
    // Nonoverlapping context windows across ALL batch sizes; never resample the same offset.
    if (tokens.len / block_count < ctx + widths[widths.len - 1]) return error.CorpusTooShortForDisjointContexts;
    const stride = tokens.len / block_count;
    const open_timer = io_util.Stopwatch.init(io);
    var table = try qwen4.NgramTable.openEmbedded(pack_path, spec);
    defer table.close();
    const open_ns = open_timer.read();
    if (table.pool == null) return error.BenchmarkPoolUnavailable;
    const index_bytes = try dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 * 1024 * 1024));
    const table_files = try identities(io, a, dir, index_bytes);
    const pool_before = table.benchmarkPoolInfo();
    const row_bytes: u64 = @as(u64, table.wcols) * 4 + @as(u64, table.scols) * ple.scaleBytes(table.fmt) * (ple.partCount(table.fmt) - 1);
    const max_rows = widths[widths.len - 1] * hash.n_heads;
    const row_buffer = try a.alloc(i64, max_rows);
    const serial_buffer = try a.alloc(f32, max_rows * table.dim);
    const pooled_buffer = try a.alloc(f32, max_rows * table.dim);
    const production_buffer = try a.alloc(f32, max_rows * table.dim);
    // Pre-touch output pages outside measured intervals.
    @memset(serial_buffer, 0);
    @memset(pooled_buffer, 0);
    @memset(production_buffer, 0);
    var rng = std.Random.DefaultPrng.init(options.seed ^ 0x434f4e54455854);
    var seen: std.AutoHashMap(i64, void) = .init(a);
    var output_digest = std.crypto.hash.sha2.Sha256.init(.{});
    var results: [widths.len]Result = undefined;
    var checked_pairs: u64 = 0;
    for (widths, 0..) |width, width_index| {
        const rows = row_buffer[0 .. width * hash.n_heads];
        const serial = serial_buffer[0 .. rows.len * table.dim];
        const pooled = pooled_buffer[0 .. serial.len];
        const production = production_buffer[0 .. serial.len];
        const blocks = try a.alloc(Block, options.samples);
        var production_pool_blocks: usize = 0;
        for (blocks, 0..) |*block, i| {
            const partition = width_index * options.samples + i;
            const offset = partition * stride + rng.random().uintLessThan(usize, stride - ctx - width + 1) + ctx;
            const prev = tokens[offset - ctx .. offset];
            const ids = tokens[offset .. offset + width];
            hash.rowIds(prev, ids, rows);
            var new_rows: usize = 0;
            for (rows) |r| {
                const entry = try seen.getOrPut(r);
                if (!entry.found_existing) new_rows += 1;
            }
            const first: Arm = if (i % 2 == 0) .serial else .pooled;
            const first_pass = try pair(io, &table, &hash, first, prev, ids, rows, serial, pooled, &output_digest);
            checked_pairs += 1;
            const warm = try a.alloc(Pair, options.warm_repeats);
            for (warm, 0..) |*sample, repeat| {
                sample.* = try pair(io, &table, &hash, if ((i + repeat) % 2 == 0) .serial else .pooled, prev, ids, rows, serial, pooled, &output_digest);
                checked_pairs += 1;
            }
            // Observe the real production policy at this context length, without forcing it.
            // This untimed parity call cannot contaminate the preceding first-pass sample.
            const rounds_before = table.benchmarkPoolInfo().rounds;
            table.gather(rows, production, offset);
            const production_rounds = table.benchmarkPoolInfo().rounds - rounds_before;
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(serial), std.mem.sliceAsBytes(production))) return error.ProductionEmbeddingMismatch;
            if (production_rounds != 0) production_pool_blocks += 1;
            block.* = .{ .token_offset = offset, .context_and_tokens = tokens[offset - ctx .. offset + width], .first_arm = first, .first_pass = first_pass, .warm = warm, .rows_not_seen_earlier_in_run = new_rows, .production_pool_rounds = production_rounds };
        }
        results[width_index] = .{
            .batch_tokens = width, .rows_per_call = rows.len, .packed_payload_bytes_per_call = rows.len * row_bytes,
            .output_f32_bytes_per_call = serial.len * @sizeOf(f32), .first_pass_serial = try summarize(a, blocks, .serial, false),
            .first_pass_pooled = try summarize(a, blocks, .pooled, false), .warm_serial = try summarize(a, blocks, .serial, true),
            .warm_pooled = try summarize(a, blocks, .pooled, true), .warm_serial_over_pooled = try uncertainty(a, blocks, options.seed +% width),
            .production_pool_attempted_blocks = production_pool_blocks, .blocks = blocks,
        };
    }
    var digest: [32]u8 = undefined;
    output_digest.final(&digest);
    const output_hash = std.fmt.bytesToHex(digest, .lower);
    const artifact = .{
        .schema_version = 1, .component = "QwenFlashNext learned ngram embedding hash + CPU gather", .exact_vectors_matched = true,
        .pack_path = pack_path, .config_sha256 = try hashHex(a, config_bytes), .tokenizer_sha256 = try hashHex(a, tokenizer_bytes),
        .index_sha256 = try hashHex(a, index_bytes), .table_files = table_files,
        .build = .{ .zig_version = builtin.zig_version_string, .optimize = @tagName(builtin.mode), .arch = @tagName(builtin.cpu.arch), .os = @tagName(builtin.os.tag), .git_sha = @import("build_options").git_sha },
        .protocol = .{
            .name = "ple-cpu-component-v1", .seed = options.seed, .samples_per_width = options.samples, .warm_repeats = options.warm_repeats,
            .batch_tokens = widths, .corpus_source = options.text orelse "independent-generated-records-v1",
            .corpus_sha256 = try hashHex(a, text), .token_ids_sha256 = try hashHex(a, std.mem.sliceAsBytes(tokens)), .token_id_encoding = "native u32 (little endian on arm64)", .corpus_tokens = tokens.len,
            .cold = "first pass only; alternate first arm over disjoint token contexts; OS page cache NOT flushed or guaranteed cold; second arm excluded from cold summary",
            .warm = "immediate repeated paired calls on the same context; order alternates each repeat; equality and hashing outside timers",
            .uncertainty = "2048 percentile bootstrap resamples of whole distinct-context blocks; ratio of warm combined means; repeated calls are not independent samples",
            .timing = "preallocated outputs; hash_ns then gather_ns, combined_ns includes intermediate clock read and benchmark bounds checks; no tokenizer/open/equality cost",
            .controls = "explicit existing mmap serial vs existing pooled pread internals; pooled errors fail, never silently fall back; no calibration/background warmer started",
            .production = "untimed gather at each token_offset; production_pool_rounds reports actual pool dispatch attempts; wide forced-pool results are diagnostic, not default-policy speedup",
        },
        .environment = .{ .QWEN4_PLE_PREFETCH = env("QWEN4_PLE_PREFETCH"), .QWEN4_PLE_PREFETCH_PREFILL = env("QWEN4_PLE_PREFETCH_PREFILL"), .QWEN4_PLE_PREFETCH_PREFILL_MIN_KV = env("QWEN4_PLE_PREFETCH_PREFILL_MIN_KV"), .MLX_SERVE_NGRAM_WARM = env("MLX_SERVE_NGRAM_WARM") },
        .table = .{ .format = @tagName(table.fmt), .rows = table.rows, .dim = table.dim, .bits = table.bits, .group_size = table.group_size, .heads_per_token = hash.n_heads, .ngram_size = hash.ngram_size, .layer_index = spec.layer_index, .shards = spec.shards, .global_scale = table.embedded.?.scale, .payload_bytes = table.embeddedPayloadBytes(), .packed_row_bytes = row_bytes, .open_ns = open_ns },
        .pool_before = pool_before, .pool_after = table.benchmarkPoolInfo(), .unique_rows_touched = seen.count(),
        .unique_logical_payload_bytes = @as(u64, seen.count()) * row_bytes, .checked_control_pairs = checked_pairs,
        .checked_production_calls = block_count, .output_sha256 = @as([]const u8, &output_hash), .results = results,
        .limits = &[_][]const u8{
            "Component timing only: not a drafter, token-ID lookup, or end-to-end model speedup; no component is disabled.",
            "Cold means first workload access, not verified disk-cold; overlapping ngrams/pages and earlier processes may have warmed rows.",
            "Packed bytes are actual logical row payload, not measured physical I/O; VM pages, pread syscalls and cache lines can amplify traffic.",
            "Context-block bootstrap does not remove shared-cache, scheduling or thermal dependence; repeat whole exclusive runs for machine-level uncertainty.",
            "File identity covers path, file sizes, safetensors headers, config, tokenizer and index, NOT a full table payload checksum.",
            "No full target, PLE convolution/projection, GPU transfer, BF16 output cast, target verification or generation is timed.",
            "Production dispatch is observed without startup calibration; loaded-model calibration and memory pressure can change the wide-batch arm.",
        },
    };
    const json = try std.json.Stringify.valueAlloc(a, artifact, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = options.output, .data = json });
    std.debug.print("PLE CPU component artifact: {s}; {d} exact control pairs and {d} production checks, {d} unique rows.\n", .{ options.output, checked_pairs, block_count, seen.count() });
}

test "PLE benchmark protocol: cold summary excludes the second already touched arm" {
    const a = std.testing.allocator;
    const fast: Timing = .{ .hash_ns = 1, .gather_ns = 9, .combined_ns = 10 };
    const slow: Timing = .{ .hash_ns = 99, .gather_ns = 901, .combined_ns = 1000 };
    const blocks = [_]Block{
        .{ .token_offset = 2, .context_and_tokens = &.{ 1, 2, 3 }, .first_arm = .serial, .first_pass = .{ .serial = fast, .pooled = slow }, .warm = &.{}, .rows_not_seen_earlier_in_run = 16, .production_pool_rounds = 1 },
        .{ .token_offset = 20, .context_and_tokens = &.{ 4, 5, 6 }, .first_arm = .pooled, .first_pass = .{ .serial = slow, .pooled = fast }, .warm = &.{}, .rows_not_seen_earlier_in_run = 16, .production_pool_rounds = 1 },
    };
    for ([_]Arm{ .serial, .pooled }) |arm| {
        const result = try summarize(a, &blocks, arm, false);
        try std.testing.expectEqual(@as(usize, 1), result.combined_ns.count);
        try std.testing.expectEqual(@as(f64, 10), result.combined_ns.mean);
        try std.testing.expectEqual(@as(u64, 1), result.hash_ns.p50);
    }
}

test "PLE benchmark protocol: uncertainty resamples paired context blocks" {
    var warm_a: [4]Pair = @splat(.{ .serial = .{ .hash_ns = 2, .gather_ns = 18, .combined_ns = 20 }, .pooled = .{ .hash_ns = 1, .gather_ns = 9, .combined_ns = 10 } });
    var warm_b: [4]Pair = @splat(.{ .serial = .{ .hash_ns = 4, .gather_ns = 196, .combined_ns = 200 }, .pooled = .{ .hash_ns = 2, .gather_ns = 98, .combined_ns = 100 } });
    const blocks = [_]Block{
        .{ .token_offset = 2, .context_and_tokens = &.{}, .first_arm = .serial, .first_pass = warm_a[0], .warm = &warm_a, .rows_not_seen_earlier_in_run = 0, .production_pool_rounds = 0 },
        .{ .token_offset = 20, .context_and_tokens = &.{}, .first_arm = .pooled, .first_pass = warm_b[0], .warm = &warm_b, .rows_not_seen_earlier_in_run = 0, .production_pool_rounds = 0 },
    };
    const interval = try uncertainty(std.testing.allocator, &blocks, 42);
    try std.testing.expectEqual(@as(f64, 2), interval.ratio_of_means);
    try std.testing.expectEqual(@as(f64, 2), interval.bootstrap_p025);
    try std.testing.expectEqual(@as(f64, 2), interval.bootstrap_p975);
}

test "PLE benchmark protocol: counterbalanced runs require nonempty even sample counts" {
    try std.testing.expectError(error.NeedEvenSamplesAndWarmRepeats, parseOptions(&.{ "ple-bench", "--model", "/pack", "--output", "/result", "--samples", "3" }));
    try std.testing.expectError(error.NeedEvenSamplesAndWarmRepeats, parseOptions(&.{ "ple-bench", "--model", "/pack", "--output", "/result", "--warm-repeats", "0" }));
    try std.testing.expectError(error.MissingOptionValue, parseOptions(&.{ "ple-bench", "--model" }));
    const options = try parseOptions(&.{ "ple-bench", "--model", "/pack", "--output", "/result", "--samples", "2", "--warm-repeats", "2" });
    try std.testing.expectEqual(@as(usize, 2), options.samples);
}
