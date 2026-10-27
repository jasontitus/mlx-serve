//! Explicit opt-in controls; native leaves the upstream controllers untouched.
const std = @import("std");
const env = @import("../../../env.zig");

pub const Selection = struct {
    mode: enum { native, adaptive, cut, max, fixed } = .native,
    /// Total verifier rows, including the anchor. Zero asks the price selector.
    rows: u32 = 0,
};

pub fn parse(value: ?[]const u8) Selection {
    const text = value orelse return .{};
    if (std.mem.eql(u8, text, "adaptive")) return .{ .mode = .adaptive };
    if (std.mem.eql(u8, text, "cut")) return .{ .mode = .cut };
    if (std.mem.eql(u8, text, "max")) return .{ .mode = .max, .rows = std.math.maxInt(u32) };
    if (std.mem.startsWith(u8, text, "fixed:")) {
        const rows = std.fmt.parseInt(u32, text[6..], 10) catch return .{};
        if (rows > 0) return .{ .mode = .fixed, .rows = rows };
    }
    return .{};
}

pub fn selection() Selection {
    return parse(env.span(.MLX_SERVE_LOOKUP_SELECTION));
}

pub fn pricesPath() ?[]const u8 {
    const path = env.span(.MLX_SERVE_LOOKUP_PRICES) orelse return null;
    return if (path.len == 0) null else path;
}

pub fn tracing() bool {
    return env.on(.LOOKUP_TRACE);
}

test "economics settings: native default and explicit bounded row controls" {
    try std.testing.expectEqual(Selection{}, parse(null));
    try std.testing.expectEqual(Selection{}, parse("fixed:0"));
    try std.testing.expectEqual(Selection{}, parse("fixed:no"));
    try std.testing.expectEqual(Selection{}, parse("typo"));
    try std.testing.expectEqual(@as(u32, 7), parse("fixed:7").rows);
    try std.testing.expectEqual(.adaptive, parse("adaptive").mode);
    try std.testing.expectEqual(.cut, parse("cut").mode);
    try std.testing.expectEqual(std.math.maxInt(u32), parse("max").rows);
}
