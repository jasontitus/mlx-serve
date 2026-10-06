# TensorFold review proposal: case-unsigned-char

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`. This exact earlier unrequested source edit is preserved separately; reviewed sources have been restored.

## Triage

- Priority: **P3 — low correctness/hardening**.
- Disposition: **Draft; locally appropriate upper/lower correction**.
- Finding: Direct toupper/tolower calls receive signed-char bytes, violating the libc argument domain for negative values other than EOF. Model template lower/upper filters can receive non-ASCII text.

## Observed verification

Built and exercised the seven actual Jinja source files with clang++ C++17, ASan and UBSan on Apple Silicon macOS. In the C locale, upper/lower preserved every byte 0x80..0xff, changed ASCII as expected, retained input provenance, and rendered UTF-8 strings correctly under the existing byte-oriented behavior. Original undefined behavior was not reproduced as a crash, and these sanitizers do not establish libc domain validity.

## Limits and risks

Not Unicode case mapping and not a complete casing-library repair. capitalize suffix still calls tolower unsafely. Preserved patch comment overstates safety of that related method.

## Publication boundary

Fork-only migration review against the unchanged baseline. Upstream CONTRIBUTING.md:63 full-tree build and zig build test gate is NOT met: Zig is absent from PATH and MLX submodules are unstaged. Focused smoke is not a substitute. No merge authorized; no full server/app build or real model inference claimed.
