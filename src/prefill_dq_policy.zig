const std = @import("std");

pub const default_min_rows: usize = 2048;

pub const Policy = struct {
    min_rows: ?usize = null,
    max_dense_bytes: ?usize = null,

    pub fn parse(min_raw: ?[]const u8, bytes_raw: ?[]const u8) !Policy {
        const rows = if (min_raw) |raw| decimal(raw) catch return error.InvalidPrefillDqMinRows else null;
        if (rows) |n| if (n < 256 or n > 65536) return error.InvalidPrefillDqMinRows;
        const bytes = if (bytes_raw) |raw| decimal(raw) catch return error.InvalidPrefillDqMaxBytes else null;
        if (bytes) |n| if (n == 0) return error.InvalidPrefillDqMaxBytes;
        return .{ .min_rows = rows, .max_dense_bytes = bytes };
    }

    pub fn allows(self: Policy, bits: u32, rows: usize, n: usize, k: usize, element_bytes: usize) bool {
        if (rows < (self.min_rows orelse defaultMinRows(bits))) return false;
        if (self.max_dense_bytes) |cap| {
            const elements = @mulWithOverflow(n, k);
            const bytes = @mulWithOverflow(elements[0], element_bytes);
            if (elements[1] != 0 or bytes[1] != 0 or bytes[0] > cap) return false;
        }
        return true;
    }
};

fn decimal(raw: []const u8) !usize {
    if (raw.len == 0) return error.InvalidNumber;
    for (raw) |c| if (c < '0' or c > '9') return error.InvalidNumber;
    return std.fmt.parseInt(usize, raw, 10);
}

pub fn defaultMinRows(bits: u32) usize {
    return if (bits == 2) 384 else default_min_rows;
}

test "prefill DQ policy preserves each shipped crossover without overrides" {
    const p = try Policy.parse(null, null);
    for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |bits| {
        const floor = defaultMinRows(bits);
        try std.testing.expect(!p.allows(bits, floor - 1, 5120, 17408, 2));
        try std.testing.expect(p.allows(bits, floor, 5120, 17408, 2));
    }
}

test "prefill DQ policy independently controls crossover and one dense weight's bytes" {
    const p = try Policy.parse("512", "65536");
    try std.testing.expect(!p.allows(4, 511, 128, 256, 2));
    try std.testing.expect(p.allows(4, 512, 128, 256, 2));
    try std.testing.expect(!p.allows(4, 512, 129, 256, 2));
    try std.testing.expect(!p.allows(4, 512, 128, 256, 4));
    const bytes_only = try Policy.parse(null, "65536");
    try std.testing.expect(!bytes_only.allows(4, 512, 128, 256, 2));
    try std.testing.expect(bytes_only.allows(2, 384, 128, 256, 2));
    const rows_only = try Policy.parse("256", null);
    try std.testing.expect(rows_only.allows(4, 256, 5120, 17408, 2));
    for (1..129) |rows| try std.testing.expect(!rows_only.allows(4, rows, 128, 256, 2));
}

test "prefill DQ policy rejects ambiguous or unsafe experiment settings" {
    for ([_][]const u8{ "", "0", "1", "128", "255", "65537", "-1", "+512", "512 ", " 512", "0x200", "1_024", "18446744073709551616" }) |raw| {
        try std.testing.expectError(error.InvalidPrefillDqMinRows, Policy.parse(raw, null));
    }
    for ([_][]const u8{ "", "0", "-1", "+1", "1GiB", "1 ", "18446744073709551616" }) |raw| {
        try std.testing.expectError(error.InvalidPrefillDqMaxBytes, Policy.parse(null, raw));
    }
}

test "prefill DQ policy declines overflowing dense weights under a byte cap" {
    const p = try Policy.parse(null, "65536");
    try std.testing.expect(!p.allows(4, 8192, std.math.maxInt(usize), 2, 2));
    try std.testing.expect(!p.allows(4, 8192, 1, std.math.maxInt(usize), 2));
}
