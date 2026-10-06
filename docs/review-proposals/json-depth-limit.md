# TensorFold review proposal: json-depth-limit

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`. This exact earlier unrequested source edit is preserved separately; reviewed sources have been restored.

## Triage

- Priority: **P2 — normal availability/correctness**.
- Disposition: **Draft; useful limited guard, not universal cycle safety**.
- Finding: JSON ingestion and serialization recurse without a bound; public value APIs and namespace assignment can create cycles for tojson. Client tools and arguments have real template serialization consumers.

## Observed verification

Seven actual Jinja sources compiled with clang++ C++17 and ASan/UBSan. Array/object leaf depth 512 accepted, 513 rejected; 513 containers ending in an empty container accepted because root depth is zero. Real C wrapper returned a recoverable depth error. Self-array, self-object, and rendered namespace cycle returned depth errors; shared acyclic children remained accepted. This verifies the guard, not a full HTTP/server stack-exhaustion scenario.

## Limits and risks

Bounds from_json/value_to_json only, after JSON text parsing. Stats-mode mark_used, repr and other traversals remain unbounded; the comment does not establish full-library cycle safety. Depth is child-edge depth with root zero, not a universal maximum number of containers. Full-stack safety margin and HTTP error handling are unverified.

## Publication boundary

Fork-only migration review against the unchanged baseline. Upstream CONTRIBUTING.md:63 full-tree build and zig build test gate is NOT met: Zig is absent from PATH and MLX submodules are unstaged. Focused smoke is not a substitute. No merge authorized; no full server/app build or real model inference claimed.
