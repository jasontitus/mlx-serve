# TensorFold review proposal: kwarg-equivalence

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`. This exact earlier unrequested source edit is preserved separately; reviewed sources have been restored.

## Triage

- Priority: **P3 underlying candidate; proposed fix rejected**.
- Disposition: **Closed archival PR; demonstrated nonreflexive equality, do not merge**.
- Finding: Pointer-identity comparison differs from structural comparison for distinct equal values, but no production consumer of whole-kwarg structural equality was established. Equal hashes for unequal values are legal and do not establish a hash-contract defect.

## Observed verification

Direct actual value API: integer pointee comparison passed, but nested-list kwarg compared with itself returned false; null kwarg compared with itself and two null kwargs returned false. Three equality invariants failed. Null kwargs are accepted by the public constructor, but the smoke deliberately did not hash them; no server null-kwarg crash is claimed.

## Limits and risks

Dereferencing uses payload equality that is not reflexive for nested mutable lists. Existing cast-before-type-check hazard is not repaired. Preserve the exact edit, do not merge; clarify intended equality contract and actual consumer before any new implementation.

## Publication boundary

Fork-only migration review against the unchanged baseline. Upstream CONTRIBUTING.md:63 full-tree build and zig build test gate is NOT met: Zig is absent from PATH and MLX submodules are unstaged. Focused smoke is not a substitute. No merge authorized; no full server/app build or real model inference claimed.
