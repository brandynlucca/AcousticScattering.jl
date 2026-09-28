# [API Reference](@id api-reference)

Start with the [tutorial](@ref first-sweep) and check the [support table](@ref solver-selection) before choosing a model. Numerical inputs use meters, seconds, Hz, and exterior-fluid wavenumber in rad/m.

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

## Incident fields

```@docs
IncidentField
PlaneWave
SphericalWave
BesselBeam
```

## Boundary conditions and materials

```@docs
Rigid
PressureRelease
Impedance
FluidFilled
GasFilled
SolidElastic
ViscoelasticSolid
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
tmatrix
kirchhoff
fem
bem
mfs
fourier
free_surface
```

## Solutions and post-processing

```@docs
AbstractSolution
ModalSolution
TMatrixSolution
KirchhoffSolution
FEMSolution
BEMSolution
MFSSolution
FMSolution
FreeSurfaceSolution
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
