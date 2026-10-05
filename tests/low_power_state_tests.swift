import Foundation

@main
struct LowPowerStateTests {
    static func main() throws {
        for mode in ["low", "high", "auto"] {
            let parsed = try StateOptions.parse(["--mode", mode])
            precondition(parsed.mode.rawValue == mode && parsed.samples == 120 && parsed.intervalMS == 1000)
        }
        let bounds = try StateOptions.parse(["--samples", "86400", "--interval-ms", "100", "--mode", "low"])
        precondition(bounds.samples == 86400 && bounds.intervalMS == 100)
        let bad: [[String]] = [[], ["--mode"], ["--mode", "Low"], ["--mode", "low", "--mode", "high"],
            ["--mode", "high", "--bogus", "1"], ["--mode", "low", "--samples", "0"],
            ["--mode", "low", "--samples", "86401"], ["--mode", "auto", "--interval-ms", "99"],
            ["--mode", "auto", "--interval-ms", "60001"]]
        for args in bad {
            do { _ = try StateOptions.parse(args); fatalError("accepted \(args)") }
            catch StateError.usage {}
        }
        for value in ["-1", "+1", " 1", "1 ", "1.5", "", "0x10", "999999999999999999999999"] {
            do { _ = try StateOptions.parse(["--mode", "low", "--samples", value]); fatalError("accepted \(value)") }
            catch StateError.usage {}
        }
        for mode in [PowerMode.low, .high, .auto] {
            for low in [false, true] {
                let sample = PowerSample(sample_index: 3, wall_utc: "fixture", uptime_seconds: 12.5,
                    collector_pid: 42, declared_mode: mode, low_power: low,
                    thermal_state: PowerSample.thermalName(100), power_source: "unknown")
                let data = try JSONEncoder().encode(sample)
                let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                let expected = low != (mode == .low) ? "mismatch" : (low ? "consistent" : "low_flag_clear_only")
                precondition(json["mode_check"] as? String == expected)
                precondition(json["declared_mode"] as? String == mode.rawValue)
                precondition(json["low_power"] as? Bool == low)
                precondition(json["thermal_state"] as? String == "unknown")
                precondition(json["power_source"] as? String == "unknown")
                precondition(json["uptime_seconds"] as? Double == 12.5)
                precondition(json["schema"] as? Int == 1)
            }
        }
        for (raw, name) in ["nominal", "fair", "serious", "critical"].enumerated() {
            precondition(PowerSample.thermalName(raw) == name)
        }
        if CommandLine.arguments.count == 2 {
            func run(_ args: [String]) throws -> (Int32, Data) {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
                process.arguments = args
                let output = Pipe()
                process.standardOutput = output
                process.standardError = FileHandle.nullDevice
                try process.run()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                return (process.terminationStatus, data)
            }
            let (status, data) = try run(["--mode", "auto", "--samples", "2", "--interval-ms", "100"])
            precondition(status == 0)
            let lines = data.split(separator: 10)
            precondition(lines.count == 2)
            var last = -Double.infinity
            for (index, line) in lines.enumerated() {
                let json = try JSONSerialization.jsonObject(with: Data(line)) as! [String: Any]
                let uptime = json["uptime_seconds"] as! Double
                precondition(uptime > last && json["sample_index"] as? Int == index)
                last = uptime
            }
            let rejected = try run(["--mode", "high", "--samples", "0"])
            precondition(rejected.0 == 2)
        }
        print("low-power-state: argument, schema, unknown-state and mode-attribution tests passed")
    }
}
