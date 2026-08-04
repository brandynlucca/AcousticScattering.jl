# AcousticScattering.jl

A standalone Julia package for 3D acoustic scattering: exact modal-series
solutions for canonical spheres and spheroids, axisymmetric/Fourier-mode and
full 3D boundary-element and finite-element solvers for the Helmholtz
equation, FEM-BEM hybrid coupling, and far-field / target-strength
post-processing.

## Status

Early scaffolding. Module structure (`src/analytical/`, `src/engine/`,
`src/postprocessing/`, `src/ecosystem/`) is in place; solvers are not yet
implemented.

## Scope

1. **Analytical & Benchmark Modules** — exact modal series (Mie scattering,
   spheroidal wave functions) for fluid/rigid/soft/elastic spheres and
   prolate/oblate spheroids; Kirchhoff Approximation and Kirchhoff-Ray-Mode
   high-frequency baselines.
2. **Numerical Physics Engine** — axisymmetric (Fourier-mode) boundary
   element and radial finite-element solvers for bodies of revolution, full
   3D boundary-element/finite-element solvers (collocation/Galerkin,
   Burton-Miller/CHIEF) for general geometry, and FEM-BEM hybrid coupling
   for elastic-shell/fluid interaction problems.
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

## Installation

Not yet registered. During active development:

```julia
using Pkg
Pkg.develop(path = "C:/Users/Brandyn/GitHub/AcousticScattering.jl")
```

## License

MIT
