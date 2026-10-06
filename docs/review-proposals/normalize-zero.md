# TensorFold review proposal: normalize-zero

Review baseline: `3129a772b4a5a46678e44902b82db4bed42fa737`. This exact earlier unrequested source edit is preserved separately; reviewed sources have been restored.

## Triage

- Priority: **P2 underlying geometry issue; proposed fix rejected**.
- Disposition: **Closed archival PR; insufficient geometry guard, do not merge**.
- Finding: Zero normalization produces nonfinite vectors that can evade collapse rejection in the textured Hunyuan3D mesh simplifier. Returning an arbitrary unit-X vector is not a geometry-validity predicate.

## Observed verification

Compiled actual Simplify.h with the native smoke. Zero normalization became finite, but a surviving face collapse with exactly zero area and a Y neighbor still returned flipped=false, deleted=0. A valid direction (0,1e-6,0) became (1,0,0), not unit-Y. Nondegenerate control stayed accepted. Two geometry invariants failed. Full decimator-selected collapse and end-to-end textured 3D inference were not run.

## Limits and risks

Absolute threshold is scale-dependent and the invented direction changes rejection decisions. Exact original edit preserved for audit only. Needs an explicit degenerate-collapse check, not an endorsement based on finite components.

## Publication boundary

Fork-only migration review against the unchanged baseline. Upstream CONTRIBUTING.md:63 full-tree build and zig build test gate is NOT met: Zig is absent from PATH and MLX submodules are unstaged. Focused smoke is not a substitute. No merge authorized; no full server/app build or real model inference claimed.
