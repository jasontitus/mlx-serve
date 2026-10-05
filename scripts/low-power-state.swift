import Foundation
import IOKit.ps

enum PowerMode: String, Codable { case low, high, auto }

struct StateOptions {
    var mode: PowerMode
    var samples = 120
    var intervalMS = 1000

    static func parse(_ args: [String]) throws -> StateOptions {
        guard args.count.isMultiple(of: 2) else { throw StateError.usage }
        var values: [String: String] = [:]
        for index in stride(from: 0, to: args.count, by: 2) {
            let key = args[index]
            guard ["--mode", "--samples", "--interval-ms"].contains(key), values[key] == nil else {
                throw StateError.usage
            }
            values[key] = args[index + 1]
        }
        guard let raw = values["--mode"], let mode = PowerMode(rawValue: raw) else { throw StateError.usage }
        func number(_ key: String, _ fallback: Int, _ range: ClosedRange<Int>) throws -> Int {
            guard let raw = values[key] else { return fallback }
            guard !raw.isEmpty, raw.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }),
                  let value = Int(raw), range.contains(value) else { throw StateError.usage }
            return value
        }
        return try StateOptions(mode: mode, samples: number("--samples", 120, 1...86400),
                                intervalMS: number("--interval-ms", 1000, 100...60000))
    }
}

enum StateError: Error { case usage }

struct PowerSample: Encodable {
    let schema = 1
    let sample_index: Int
    let wall_utc: String
    let uptime_seconds: Double
    let collector_pid: Int32
    let declared_mode: PowerMode
    let low_power: Bool
    let thermal_state: String
    let power_source: String

    var mode_check: String {
        if low_power != (declared_mode == .low) { return "mismatch" }
        return low_power ? "consistent" : "low_flag_clear_only"
    }

    enum CodingKeys: String, CodingKey {
        case schema, sample_index, wall_utc, uptime_seconds, collector_pid
        case declared_mode, low_power, thermal_state, power_source, mode_check
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schema, forKey: .schema)
        try c.encode(sample_index, forKey: .sample_index)
        try c.encode(wall_utc, forKey: .wall_utc)
        try c.encode(uptime_seconds, forKey: .uptime_seconds)
        try c.encode(collector_pid, forKey: .collector_pid)
        try c.encode(declared_mode, forKey: .declared_mode)
        try c.encode(low_power, forKey: .low_power)
        try c.encode(thermal_state, forKey: .thermal_state)
        try c.encode(power_source, forKey: .power_source)
        try c.encode(mode_check, forKey: .mode_check)
    }

    static func thermalName(_ value: Int) -> String {
        switch value {
        case 0: return "nominal"
        case 1: return "fair"
        case 2: return "serious"
        case 3: return "critical"
        default: return "unknown"
        }
    }
}

#if !LOW_POWER_STATE_TESTS
@main
struct LowPowerState {
    static func main() throws {
        let options: StateOptions
        do {
            options = try StateOptions.parse(Array(CommandLine.arguments.dropFirst()))
        } catch {
            FileHandle.standardError.write(Data("Usage: low-power-state --mode low|high|auto [--samples 1..86400] [--interval-ms 100..60000]\n".utf8))
            exit(2)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let clock = ISO8601DateFormatter()
        clock.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let process = ProcessInfo.processInfo
        for index in 0..<options.samples {
            let source: String
            if let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
               let raw = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() {
                source = raw as String
            } else {
                source = "unknown"
            }
            let sample = PowerSample(
                sample_index: index, wall_utc: clock.string(from: Date()),
                uptime_seconds: process.systemUptime, collector_pid: process.processIdentifier,
                declared_mode: options.mode, low_power: process.isLowPowerModeEnabled,
                thermal_state: PowerSample.thermalName(process.thermalState.rawValue), power_source: source)
            try FileHandle.standardOutput.write(contentsOf: encoder.encode(sample) + Data([10]))
            if index + 1 < options.samples {
                Thread.sleep(forTimeInterval: Double(options.intervalMS) / 1000)
            }
        }
    }
}
#endif
