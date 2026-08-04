# AcousticScattering.jl

A Julia package for 3D acoustic scattering: exact modal-series solutions for
canonical spheres and spheroids, exterior BEM and interior FEM solvers for
the Helmholtz equation, FEM-BEM hybrid coupling, and far-field/target-strength
post-processing.

`AcousticScattering.jl` is a standalone package. It also serves as the
external numerical-reference engine for the
[`acousticTS`](https://github.com/brandynlucca/acousticTS) R package's
validation suite ([`acousticTSValidation`](https://github.com/brandynlucca/acousticTSValidation)),
but has no dependency on either.

See the project's scoping document,
`docs/acousticTS-plans/ACOUSTICSCATTERING_JL_PLAN.md` in
`acousticTSValidation`, for the full architecture and roadmap.

## Status

This package is in early scaffolding — module structure is in place, solvers
are not yet implemented. See the plan document's phased roadmap for current
progress.

## Pillars

1. **Analytical & Benchmark** — exact modal series (Mie scattering,
   spheroidal wave functions) and high-frequency approximations (KA, KRM).
2. **Numerical Physics Engine** — exterior BEM (Burton-Miller/CHIEF),
   interior FEM, and FEM-BEM hybrid coupling.
3. **Acoustic Post-Processing** — far-field extrapolation, target strength,
   field visualization.
4. **Ecosystem Integration** — thin wrappers over `GeometryBasics.jl`,
   `MeshIO.jl`, `Gmsh.jl`, `LinearAlgebra`, `Krylov.jl`, and optional
   $H$-matrix/FMM acceleration.
