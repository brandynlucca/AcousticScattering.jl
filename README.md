# AcousticScattering.jl

A standalone Julia package for 3D acoustic scattering: exact modal-series
solutions for canonical spheres and spheroids, exterior BEM and interior FEM
solvers for the Helmholtz equation, FEM-BEM hybrid coupling, and far-field /
target-strength post-processing.

This package is **not** specific to any one project. It is, however, also
used as the external numerical-reference engine for the
[`acousticTS`](https://github.com/brandynlucca/acousticTS) R package's
validation suite,
[`acousticTSValidation`](https://github.com/brandynlucca/acousticTSValidation) —
see that repo's `docs/acousticTS-plans/ACOUSTICSCATTERING_JL_PLAN.md` for the
full scoping document, dependency survey, architecture, and phased roadmap
this package follows.

## Status

Early scaffolding. Module structure (`src/analytical/`, `src/engine/`,
`src/postprocessing/`, `src/ecosystem/`) is in place; solvers are not yet
implemented. See the plan document linked above for current progress and
open decisions (BEM/FEM backend selection).

## Scope

1. **Analytical & Benchmark Modules** — exact modal series (Mie scattering,
   spheroidal wave functions) for fluid/rigid/soft/elastic spheres and
   prolate/oblate spheroids; Kirchhoff Approximation and Kirchhoff-Ray-Mode
   high-frequency baselines.
2. **Numerical Physics Engine** — exterior BEM (collocation/Galerkin,
   Burton-Miller/CHIEF), interior FEM for inhomogeneous fluid/viscoelastic
   domains, and FEM-BEM hybrid coupling.
3. **Acoustic Post-Processing & Target Metrics** — Kirchhoff-Helmholtz
   far-field extrapolation, backscatter/bistatic target strength (TS),
   near-field pressure maps and polar radiation patterns.
4. **Ecosystem Integration** — thin wrappers over established Julia
   infrastructure (`GeometryBasics.jl`, `MeshIO.jl`, `Gmsh.jl`,
   `LinearAlgebra`, `Krylov.jl`, optional $H$-matrix/FMM acceleration)
   rather than reimplementing mesh I/O or linear algebra.

## Related packages

- [`SpheroidalWaves.jl`](https://github.com/brandynlucca/SpheroidalWaveFunctions) —
  prolate/oblate angular and radial spheroidal wave functions; direct
  dependency of this package's spheroidal analytical solvers.
- [`acousticTS`](https://github.com/brandynlucca/acousticTS) — R package for
  physics-based target strength models.
- [`acousticTSValidation`](https://github.com/brandynlucca/acousticTSValidation) —
  validation suite for `acousticTS`; consumes this package as its external
  numerical reference.

## Installation

Not yet registered. During active development:

```julia
using Pkg
Pkg.develop(path = "C:/Users/Brandyn/GitHub/AcousticScattering.jl")
```

## License

MIT
