#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TEST_DIR=$(mktemp -d "${TMPDIR:-/tmp}/mlx-power-state.XXXXXX")
trap 'rm -rf "$TEST_DIR"' EXIT
swiftc -O -parse-as-library scripts/low-power-state.swift -o "$TEST_DIR/low-power-state"
swiftc -O -D LOW_POWER_STATE_TESTS scripts/low-power-state.swift \
  tests/low_power_state_tests.swift -o "$TEST_DIR/tests"
"$TEST_DIR/tests" "$TEST_DIR/low-power-state"
