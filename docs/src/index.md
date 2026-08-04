# AcousticScattering.jl

A Julia package for 3D acoustic scattering: exact modal-series solutions for
canonical spheres and spheroids, axisymmetric and full 3D BEM/FEM solvers for
the Helmholtz equation, FEM-BEM hybrid coupling, and far-field/target-strength
post-processing.

`AcousticScattering.jl` is a standalone package with no dependency on any
particular downstream application.

## Status

This package is in early scaffolding — module structure is in place, solvers
are not yet implemented.

## Pillars

1. **Analytical & Benchmark** — exact modal series (Mie scattering,
   spheroidal wave functions) and high-frequency approximations (KA, KRM).
2. **Numerical Physics Engine** — axisymmetric (Fourier-mode) BEM and radial
   FEM for bodies of revolution, full 3D BEM (Burton-Miller/CHIEF) and FEM
   for general geometry, and FEM-BEM hybrid coupling.
3. **Acoustic Post-Processing** — far-field extrapolation, target strength,
   field visualization.
4. **Ecosystem Integration** — thin wrappers over `GeometryBasics.jl`,
   `MeshIO.jl`, `Gmsh.jl`, `LinearAlgebra`, `Krylov.jl`, and optional
   $H$-matrix/FMM acceleration.
