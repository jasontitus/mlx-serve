# TensorFold review proposal: append-pop-copy

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`. This exact earlier unrequested source edit is preserved separately; reviewed sources have been restored.

## Triage

- Priority: **Rejected — not a valid fix**.
- Disposition: **Closed archival PR; demonstrated regression, do not merge**.
- Finding: The original report mistook Python-compatible shared list mutation for corruption. Copying append/pop breaks that required behavior.

## Observed verification

Actual renderer with copied proposal: alias append expected [1, 2], got [1]; alias pop expected 2|[1], got 2|[1, 2]; repeated pops expected 2|1|[], got 2|2|[1, 2]. Direct tuple append/pop returned successfully instead of rejecting immutable mutation. Five consumer invariants failed. Original alias behavior had already been independently verified in PI_REVIEW_NOTES.md.

## Limits and risks

Exact unrequested edit preserved only as an archive. Adds allocation/copying, changes prompt/list state semantics, and bypasses tuple immutability checks. Revert is already complete in the review checkout; no replacement implementation is requested.

## Publication boundary

Fork-only migration review against the unchanged baseline. Upstream CONTRIBUTING.md:63 full-tree build and zig build test gate is NOT met: Zig is absent from PATH and MLX submodules are unstaged. Focused smoke is not a substitute. No merge authorized; no full server/app build or real model inference claimed.
