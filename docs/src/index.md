# AcousticScattering.jl

AcousticScattering.jl predicts how individual bodies scatter sound in a fluid. Construct a
geometry and material configuration, solve with a modal, Kirchhoff, FEM, BEM, or MFS method,
and extract target strength in dB re 1 m². Compare idealized shapes, explore frequency and
orientation dependence, and check numerical calculations against analytical models.

```@example home
using AcousticScattering

sound_speed = 1477.4 # m/s
frequency = 38000.0 # Hz
wavenumber = 2pi * frequency / sound_speed
solution = modal(Sphere(0.01), Rigid(), wavenumber)
target_strength(solution)
```

![Modal backscatter from a rigid 1 cm sphere across 12–200 kHz.](tutorials/rigid_sphere_frequency.png)

Start with [Getting Started](@ref getting-started), then follow
[Your first frequency sweep](@ref first-sweep) to generate this plot and save its data.
[Choosing a solver](@ref solver-selection) lists supported combinations. The
[API Reference](@ref api-reference) describes current calls.

## Scope and maturity

Implemented methods include sphere/spheroid series, finite-cylinder approximations,
curved-cylinder models, physical-optics surface integrals, axisymmetric and full surface BEM,
radial and meridian FEM, MFS, and coupled structural shells. Supported materials differ by
solver and geometry. Creating a material does not guarantee support in every solver.

This is a development package (`0.1.0-DEV`). These pages describe the current source tree.
Consult the [migration guide](@ref migration) when updating older scripts. Full 3D volume FEM
is not implemented. KRM is not a public solver, and full 3D BEM does not implement
Burton–Miller/CHIEF regularization. Additional visualization recipes are under development.
