# [API Reference](@id api-reference)

The public workflow is geometry → boundary/material → solver → result query. Start with the
[tutorial](@ref first-sweep) and check the [support table](@ref solver-selection) before choosing
a model. Numerical inputs use meters, seconds, Hz, and exterior-fluid wavenumber in rad/m.

```@meta
CurrentModule = AcousticScattering
```

## Geometry

```@docs
AbstractBody
Sphere
Spheroid
Cylinder
Shell
Irregular
```

## Boundary conditions and materials

```@docs
Rigid
PressureRelease
FluidFilled
GasFilled
SolidElastic
Shelled
FluidLayer
ElasticLayer
ViscousLayer
LayeredMaterial
VacuumInterior
FluidInterior
```

## Solver families

```@docs
modal
kirchhoff
fem
bem
mfs
fourier
```

## Solutions and post-processing

```@docs
AbstractSolution
ModalSolution
KirchhoffSolution
FEMSolution
BEMSolution
MFSSolution
FMSolution
target_strength
scattering_amplitude
```

## [Pressure at Cartesian points](@id pressure-evaluation)

```@docs
pressure
```

## Solver diagnostics

```@docs
diagnostics
```

## Mesh construction

```@docs
Mesh
mesh
```

## Sampling and visualization

```@docs
components
frequency_sweep
incidence_angle_sweep
bistatic_sweep
bistatic_map
```
