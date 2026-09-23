# AcousticScattering.jl

```@raw html
<img class="home-logo" src="assets/logo.svg" alt="AcousticScattering.jl logo"
     width="224" height="224">
```

AcousticScattering.jl predicts how individual bodies scatter sound in a fluid. Construct a geometry and material configuration, solve with the following methods:
- [Modal series solutions for canonical shapes](@ref modal-theory)
- [Kirchoff physical optics and approximation](@ref kirchhoff-theory)
- [Boundary element methods](@ref boundary-theory)
- [Method of fundamental solutions](@ref boundary-theory)
- [Finite element methods](@ref fem-theory)
- [Fourier matching methods](@ref fourier-matching-theory)

Various quantities can then be extracted from these solutions such as the raw scattering amplitude, phase, and target strength (dB re. 1 m²). Compare idealized shapes, explore frequency and orientation dependence, and check numerical calculations against analytical models.

```@example home
using AcousticScattering

sound_speed = 1477.4 # m/s
frequency = 38000.0 # Hz
wavenumber = 2pi * frequency / sound_speed
solution = modal(Sphere(0.01), Rigid(), wavenumber)
target_strength(solution)
```

![Modal backscatter from a rigid 1 cm sphere across 12–200 kHz.](tutorials/rigid_sphere_frequency.png)

Start with [Getting Started](@ref getting-started), then follow [Your first frequency sweep](@ref first-sweep) to generate this plot and save its data. [Choosing a solver](@ref solver-selection) lists supported combinations. The [API Reference](@ref api-reference) describes current calls.

## Try an application

For a quick look at the package's capabilities, open the [Visualization Gallery](@ref gallery): pressure maps, 3D geometry, resonances and angular patterns, with short Julia examples.

| Question | Tutorial |
|:--|:--|
| How does an elastic reference target differ from a rigid sphere? | [Reference-target models](@ref reference-target-tutorial) |
| Where is a gas bubble's scattering resonance? | [Gas-bubble resonance](@ref bubble-resonance-tutorial) |
| How does a coating affect scattering and internal pressure? | [Coated particles](@ref coated-particle-tutorial) |


## Scope and maturity

Implemented methods include sphere/spheroid series, finite-cylinder approximations, curved-cylinder models, physical-optics surface integrals, axisymmetric and full surface BEM, radial and meridian FEM, MFS, and coupled structural shells. Supported materials differ by solver and geometry. Creating a material does not guarantee support in every solver.

Full 3D volume FEM is unavailable. See [BEM and MFS](@ref boundary-theory) for boundary-element formulations and irregular-frequency treatment.
