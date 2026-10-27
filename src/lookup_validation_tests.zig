//! Deterministic consumer tests: oracle proposals, real target forward and commit.
//! No timing gate and no copied accept/rollback algorithm. The test-only proposal
//! seam changes the offered tree, never the production verifier or state commit.
const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const transformer = @import("transformer.zig");
const generate = @import("generate.zig");
const dflash = @import("dflash.zig");
const lookup = @import("token_history.zig");
const stair = @import("speculation_policy.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Fix = dflash.TinyFix;
const a = testing.allocator;
const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };

// This hybrid has BOTH attention KV and GDN convolution/recurrent state. All
// contracted matrices use the shipped row-exact affine lane, not dense matmul.
const config_json =
    \\{"model_type":"qwen3_5","hidden_size":64,"intermediate_size":128,
    \\"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,
    \\"head_dim":256,"vocab_size":128,"rms_norm_eps":0.00001,
    \\"rope_theta":10000.0,"tie_word_embeddings":true,"full_attention_interval":2,
    \\"linear_num_key_heads":2,"linear_num_value_heads":8,
    \\"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\"attn_output_gate":true,"max_position_embeddings":2048,"torch_dtype":"bfloat16",
    \\"quantization":{"bits":8,"group_size":64}}
;

fn writeHybrid(io: std.Io, dir: std.Io.Dir, path: []const u8, s: mlx.mlx_stream) !void {
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try Fix.putQuantW(map, "model.embed_tokens.weight", 128, 64, 1, s);
    try Fix.putNorm(map, "model.norm.weight", 64, s);
    var buf: [160]u8 = undefined;
    for (0..4) |li| {
        inline for (.{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |name|
            try Fix.putNorm(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ li, name }), 64, s);
        inline for (.{ "gate_proj", "up_proj" }) |name|
            try Fix.putQuantW(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.{s}.weight", .{ li, name }), 128, 64, 50 + li, s);
        try Fix.putQuantW(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.down_proj.weight", .{li}), 64, 128, 70 + li, s);
        if (li % 2 == 0) {
            const names = [_][]const u8{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj" };
            const rows = [_]usize{ 1536, 1024, 8, 8, 64 };
            const cols = [_]usize{ 64, 64, 64, 64, 1024 };
            for (names, rows, cols, 0..) |name, r, c, j|
                try Fix.putQuantW(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.linear_attn.{s}.weight", .{ li, name }), r, c, 80 + li + j, s);
            inline for (.{ "A_log", "dt_bias" }) |name|
                try Fix.putW(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.linear_attn.{s}", .{ li, name }), 8, 0, 90 + li, s);
            try Fix.putNorm(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.linear_attn.norm.weight", .{li}), 128, s);
            const flat = try Fix.bf16Arr(1536, 4, 95 + li, s);
            defer _ = mlx.mlx_array_free(flat);
            var conv = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(conv);
            try mlx.check(mlx.mlx_reshape(&conv, flat, &[_]c_int{ 1536, 4, 1 }, 3, s));
            try Fix.put(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.linear_attn.conv1d.weight", .{li}), conv);
        } else {
            const names = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" };
            const rows = [_]usize{ 2048, 512, 512, 64 };
            const cols = [_]usize{ 64, 64, 64, 1024 };
            for (names, rows, cols, 0..) |name, r, c, j|
                try Fix.putQuantW(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.{s}.weight", .{ li, name }), r, c, 10 + li + j, s);
            inline for (.{ "q_norm", "k_norm" }) |name|
                try Fix.putNorm(map, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.{s}.weight", .{ li, name }), 256, s);
        }
    }
    const file = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{path}, 0);
    defer a.free(file);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
}

fn equalArray(left: mlx.mlx_array, right: mlx.mlx_array, s: mlx.mlx_stream) !void {
    try testing.expectEqual(left.ctx == null, right.ctx == null);
    if (left.ctx == null) return;
    try testing.expectEqualSlices(c_int, mlx.getShape(left), mlx.getShape(right));
    const x = try Fix.readF32(left, a, s);
    defer a.free(x);
    const y = try Fix.readF32(right, a, s);
    defer a.free(y);
    for (x, y) |xx, yy| {
        try testing.expect(std.math.isFinite(xx) and std.math.isFinite(yy));
        // SSM's multirow tree and serial recurrence can differ at float roundoff;
        // emitted ids and pending target token remain exact below.
        try testing.expectApproxEqAbs(xx, yy, 0.0001);
    }
}

// Use the actual serial Generator.next with its last-token synchronous branch:
// no un-emitted lookahead may hide a wrong committed-state comparison.
fn serialTick(gen: *generate.Generator, assistant: ?*dflash.DflashModel) !?u32 {
    const limit = gen.max_tokens;
    gen.max_tokens = gen.completion_tokens + 1;
    defer gen.max_tokens = limit;
    var captures = [_]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (captures) |capture| { _ = mlx.mlx_array_free(capture); };
    var cl = transformer.CaptureLayers{ .ids = if (assistant) |dm| dm.config.captureIds() else &.{}, .out = &captures };
    if (assistant != null) gen.ctx.capture_layers = &cl;
    defer gen.ctx.capture_layers = null;
    const anchor = gen.ctx.moe_seq_offset.*;
    const token = try gen.next(a);
    if (token != null) if (assistant) |dm| {
        try dflash.appendContext(dm, &gen.dflash_ctx.?, &captures, anchor);
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        gen.dflash_ctx.?.appendEvalArrays(vec);
        try mlx.check(mlx.mlx_eval(vec));
    };
    return token;
}

fn equalState(serial: *generate.Generator, tree: *generate.Generator) !void {
    try testing.expectEqualSlices(u32, serial.generated_ids.items, tree.generated_ids.items);
    try testing.expectEqual(serial.completion_tokens, tree.completion_tokens);
    try testing.expectEqual(serial.next_token_id, tree.next_token_id);
    try testing.expectEqual(serial.ctx.cache.step, tree.ctx.cache.step);
    try testing.expectEqual(serial.ctx.moe_seq_offset.*, tree.ctx.moe_seq_offset.*);
    for (serial.ctx.cache.entries, tree.ctx.cache.entries) |left, right| {
        try testing.expectEqual(left.initialized, right.initialized);
        if (!left.initialized) continue;
        try testing.expectEqual(left.offset, right.offset);
        try equalArray(left.key_view, right.key_view, tree.xfm.s);
        try equalArray(left.value_view, right.value_view, tree.xfm.s);
    }
    const left = serial.ctx.ssm_entries orelse return error.MissingSerialSsmState;
    const right = tree.ctx.ssm_entries orelse return error.MissingTreeSsmState;
    for (left, right) |l, r| {
        try testing.expectEqual(l.initialized, r.initialized);
        if (!l.initialized) continue;
        try equalArray(l.conv_state, r.conv_state, tree.xfm.s);
        try equalArray(l.ssm_state, r.ssm_state, tree.xfm.s);
    }
    try testing.expectEqual(tree.ctx.moe_seq_offset.*, tree.dflash_ctx.?.absLen());
    try testing.expectEqual(serial.dflash_ctx.?.absLen(), tree.dflash_ctx.?.absLen());
    for (serial.dflash_ctx.?.cache.entries, tree.dflash_ctx.?.cache.entries) |l, r| {
        try equalArray(l.key_view, r.key_view, tree.xfm.s);
        try equalArray(l.value_view, r.value_view, tree.xfm.s);
    }
}

const Scenario = enum { full, deep_reject, noncontiguous, budget, eos };

fn transition(scenario: Scenario) !void {
    // A missing backend is an explicit unsupported test, never a successful return.
    if (mlx.noGpuBackend() or !mlx.metalKernelsAvailable()) return error.LookupValidationRequiresMetal;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    var trunk = testing.tmpDir(.{});
    defer trunk.cleanup();
    var assistant = testing.tmpDir(.{});
    defer assistant.cleanup();
    var tb: [512]u8 = undefined;
    const tp = tb[0..try trunk.dir.realPath(io, &tb)];
    var ab: [512]u8 = undefined;
    const ap = ab[0..try assistant.dir.realPath(io, &ab)];
    try writeHybrid(io, trunk.dir, tp, s);
    try Fix.writeAssistant(io, assistant.dir, ap, s);
    var cfg = try model.parseConfig(io, a, tp);
    var weights = try model.loadWeights(io, a, tp);
    defer weights.deinit();
    model.resolveWeightPrefix(&cfg, &weights);
    var sx = try transformer.Transformer.init(io, a, cfg, &weights);
    defer sx.deinit();
    var tx = try transformer.Transformer.init(io, a, cfg, &weights);
    defer tx.deinit();
    var sd = try dflash.loadDflash(io, a, s, ap);
    defer sd.deinit();
    var td = try dflash.loadDflash(io, a, s, ap);
    defer td.deinit();
    try sd.bind(&sx);
    try td.bind(&tx);
    if (!tx.specTreeSupported()) return error.LookupValidationRequiresExactTree;
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var oracle: [12]u32 = undefined;
    {
        var ox = try transformer.Transformer.init(io, a, cfg, &weights);
        defer ox.deinit();
        var od = try dflash.loadDflash(io, a, s, ap);
        defer od.deinit();
        try od.bind(&ox);
        var og = try generate.Generator.initWithOptions(io, a, &ox, &tok, &prompt, 32, .{ .temperature = 0 }, &.{}, .{ .skip_lazy_preforward = true });
        defer og.deinit(a);
        for (&oracle) |*token| token.* = (try serialTick(&og, null)) orelse return error.ShortOracle;
    }
    var serial = try generate.Generator.initWithOptions(io, a, &sx, &tok, &prompt, 32, .{ .temperature = 0 }, &.{}, .{ .dflash_enabled = true, .dflash = &sd, .dflash_min_accepted_per_round = 0 });
    defer serial.deinit(a);
    var tree = try generate.Generator.initWithOptions(io, a, &tx, &tok, &prompt, 32, .{ .temperature = 0 }, &.{}, .{ .dflash_enabled = true, .dflash = &td, .dflash_min_accepted_per_round = 0 });
    var tree_alive = true;
    defer if (tree_alive) tree.deinit(a);
    var cut = stair.Policy.init(stair.PriceModel.init(a));
    defer cut.deinit();
    // Equal measured costs isolate verifier/commit behavior from width economics.
    for (1..17) |rows| {
        cut.prices.observe(@intCast(rows), 1, .tree, 0, .generic);
        cut.prices.observe(@intCast(rows), 1, .tree, 0, .linear_gdn);
    }
    cut.observePlain(1, 0);
    tree.lookup_tiers = .{ .drafter = lookup.Drafter.init(a), .cut = &cut };
    try tree.lookup_tiers.?.drafter.prime(&prompt);
    const old_override = generate.lookup_tiers_override;
    generate.lookup_tiers_override = true;
    defer generate.lookup_tiers_override = old_override;
    if (scenario == .eos) {
        @memcpy(oracle[0..5], &[_]u32{ 31, 42, 53, 64, 75 });
        tree.next_token_id = oracle[0];
    }

    var candidate = stair.Tree.init(oracle[0]);
    var parent: usize = 0;
    if (scenario == .budget) {
        _ = candidate.add(0, (oracle[1] + 1) % 128, 0.95, .draft, 0.95).?;
        _ = candidate.add(0, oracle[1], 0.9, .look, 0.9).?;
    } else for (oracle[1..5], 0..) |token, i| {
        if (scenario == .noncontiguous) {
            // Wrong sibling precedes each accepted row, forcing actual KV gather.
            _ = candidate.add(parent, (token + 1) % 128, 0.7, .draft, 0.8).?;
        }
        const value = if (scenario == .deep_reject and i == 2) (token + 1) % 128 else token;
        parent = candidate.add(parent, value, 0.9, .look, 0.9).?;
    }
    tree.lookup_test_candidate = candidate;
    if (scenario == .budget) tree.max_tokens = 2;
    const eos = [_]u32{oracle[2]};
    var target_ids: [5]i32 = undefined;
    if (scenario == .eos) {
        for (&target_ids, 0..) |*id, i| id.* = @intCast(oracle[i + 1]);
        tree.lookup_test_targets = &target_ids;
        tree.eos_token_ids = &eos;
        serial.eos_token_ids = &eos;
    }
    const step = try tree.dflashTreeRound(a, &td, &tree.dflash_ctx.?, null, null, tree.next_token_id, 4, tree.ctx.moe_seq_offset.*, tree.ctx.cache.step, false, &.{}, null, false, null);
    defer a.free(step.tokens);
    const committed: usize = switch (scenario) { .full, .noncontiguous => 5, .deep_reject => 3, .budget, .eos => 2 };
    try testing.expectEqual(committed, step.tokens.len);
    try testing.expectEqualSlices(u32, oracle[0..committed], step.tokens);
    for (0..committed) |i| {
        if (scenario == .eos) serial.next_token_id = oracle[i];
        _ = (try serialTick(&serial, &sd)) orelse return error.ShortSerial;
    }
    if (scenario == .eos) serial.next_token_id = eos[0];
    if (scenario == .budget or scenario == .eos) {
        // Emission clipping must not censor already verified successes.
        try testing.expectEqual(@as(f32, 1), cut.cal.samples(.look));
        try testing.expect(cut.cal.q(.look, 0.9, 0) > 0.9);
        const expected: f32 = if (scenario == .budget) 0 else 3;
        try testing.expectEqual(expected, cut.cal.samples(.look_continuation));
        if (expected > 0) try testing.expect(cut.cal.q(.look_continuation, 0.9, 0) > 0.9);
    }
    if (scenario == .budget) {
        // Both depth-one alternatives must be verified despite a two-token
        // output budget; capping total rows at two selects only the wrong one.
        try testing.expectEqual(@as(u32, 2), tree.dflash_round_width);
        try testing.expectEqual(@as(u64, 2), tree.dflash_verified_drafts);
        try testing.expectEqual(@as(u32, 1), step.accepted_tokens);
        tree.max_tokens = 32;
        tree.done = false;
        tree.finish_reason = "";
    }
    if (scenario != .eos) {
        try equalState(&serial, &tree);
        // Continued decode is part of the gate; matching a clipped reply alone
        // cannot conceal a stale recurrent state, wrong compacted KV or pending id.
        tree.lookup_test_candidate = null;
        for (0..4) |_| {
            const next = stair.Tree.init(tree.next_token_id);
            const anchor = tree.ctx.moe_seq_offset.*;
            tree.lookup_test_candidate = next;
            const r = try tree.dflashTreeRound(a, &td, &tree.dflash_ctx.?, null, null, tree.next_token_id, 0, anchor, tree.ctx.cache.step, false, &.{}, null, false, null);
            defer a.free(r.tokens);
            _ = (try serialTick(&serial, &sd)) orelse return error.ShortSerialContinuation;
            try equalState(&serial, &tree);
        }
    } else {
        try equalState(&serial, &tree);
        try testing.expect(tree.done);
        try testing.expectEqualStrings("stop", tree.finish_reason);
        try testing.expect((try serialTick(&serial, &sd)) == null);
        try testing.expectEqualSlices(u32, serial.generated_ids.items, tree.generated_ids.items);
        // Exercise the real teardown writer too: an HTTP stop must not leave
        // verified-but-unemitted descendants in next-request lookup history.
        const store_path = try std.fmt.allocPrint(a, "{s}/eos-store", .{tp});
        defer a.free(store_path);
        const identity: [32]u8 = @splat(0x53);
        const store = try a.create(lookup.PersistentStore);
        store.* = lookup.PersistentStore.open(io, a, store_path, identity) catch |err| {
            a.destroy(store);
            return err;
        };
        tree.lookup_tiers.?.persistent = store;
        tree.deinit(a);
        tree_alive = false;
        var reopened = try lookup.PersistentStore.open(io, a, store_path, identity);
        defer reopened.deinit();
        try testing.expectEqualSlices(u32, &prompt, reopened.tokens[0..prompt.len]);
        try testing.expectEqualSlices(u32, oracle[0..committed], reopened.tokens[prompt.len .. prompt.len + committed]);
        try testing.expectEqual(prompt.len + committed + 1, reopened.tokens.len);
        try testing.expectEqual(lookup.separator, reopened.tokens[reopened.tokens.len - 1]);
    }
}

test "lookup validation: merged tree full acceptance and state continuation" { try transition(.full); }
test "lookup validation: merged tree deep rejection and state continuation" { try transition(.deep_reject); }
test "lookup validation: merged tree noncontiguous path compaction" { try transition(.noncontiguous); }
test "lookup validation: production budget selection, learning and continuation" { try transition(.budget); }
test "lookup validation: EOS inside verified block excludes stop token" { try transition(.eos); }
