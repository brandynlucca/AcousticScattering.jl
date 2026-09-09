# [API Reference](@id api-reference)

The public workflow is geometry → boundary/material → solver → result query. This curated
reference describes current calls without exposing every internal function. Start with the
[tutorial](@ref first-sweep) and check the [support table](@ref solver-selection) before changing
the model. Numerical inputs use meters, seconds, Hz, and exterior-fluid wavenumber in rad/m.

## Geometry and materials

| Constructor | Inputs and role |
|:--|:--|
| `Sphere(radius)` | Positive radius, m |
| `Spheroid(axial_radius, equatorial_radius)` | Positive semi-axes in meters, ordered as axial then equatorial |
| `Cylinder(radius, length; radius_curvature = Inf, endcap_depth = 0.0)` | Radius, length, and optional curvature/cap dimensions, m |
| `Shell(body, thickness)` | Structural FEM geometry over a sphere/spheroid |
| `Rigid()` | Zero normal velocity |
| `PressureRelease()` | Zero total surface pressure |
| `FluidFilled(density_contrast, soundspeed_contrast; coupling = :full)` | Penetrable fluid. Spheroid coupling may also be `:diagonal` |
| `GasFilled(...)` | Alias of `FluidFilled` |
| `SolidElastic(density_contrast, speed_longitudinal_contrast, speed_transversal_contrast)` | Solid elastic body |
| `FluidLayer(density_contrast, soundspeed_contrast)` | Fluid shell material |
| `ElasticLayer(density_contrast, speed_longitudinal_contrast, speed_transversal_contrast; interior_coupling = :generalized)` | Elastic shell material |
| `VacuumInterior()` | Pressure release at a fluid shell's inner surface |
| `FluidInterior(density_contrast, soundspeed_contrast)` | Shell cavity fluid |
| `ViscousLayer(soundspeed_exterior, density_contrast, soundspeed_contrast, kinematic_viscosity_compressional, kinematic_viscosity_shear)` | Viscous outer layer with kinematic viscosity in m²/s |
| `LayeredMaterial(outer, inner, radius_ratio)` | Two layers, currently restricted to a viscous outer layer and elastic inner layer |
| `Shelled(material, interior, radius_ratio)` | Layered boundary with inner/outer radius ratio |
| `Shelled(poisson, density, youngs_modulus)` | Absolute structural FEM material, kg/m³ and Pa |

Geometry types inherit from `AbstractBody`. A `Shelled` constructor is not an arbitrary
material-law registration mechanism. See [Geometry and materials](@ref geometry-materials)
for interface physics, radius ordering, and supported pairings.

## Solver families

```julia
solution = modal(body, boundary, wavenumber; kwargs...)
solution = kirchhoff(body, boundary, wavenumber; kwargs...)
solution = fem(body, boundary, wavenumber; kwargs...)
solution = bem(body, boundary, wavenumber; kwargs...)
solution = mfs(body, boundary, wavenumber; kwargs...)
```

`kwargs...` above summarizes method-specific keywords. Each method has its own options, so do not pass
the same options to every method. Unknown choices can produce `ArgumentError` or a lower-level
`MethodError` on paths whose input validation is still being consolidated.

| Function | Returns | Main options |
|:--|:--|:--|
| `modal` | `ModalSolution` | Sphere `angle`, `m_max`; spheroid `incidence_angle`, `m_max`, `n_max`; cylinder `incidence_angle`, `m_max` |
| `kirchhoff` | `KirchhoffSolution` | `incidence_angle` for nonspherical bodies |
| `fem` | `FEMSolution` | `method`, discretization controls. Sphere radial defaults to `:radial` |
| `bem` | `BEMSolution` | `method = :axisymmetric` or `:full`, incidence, `n` or `meshsize` respectively |
| `mfs` | `MFSSolution` | Incidence, `offset`, axisymmetric `n`, modal cutoff. The bent path has source-grid controls |

Nonspherical incidence defaults to broadside (`pi / 2`). Sphere modal/Kirchhoff does not take
`incidence_angle`. Axisymmetric numerical solvers on a sphere can still choose an incident axis.
For shell structural FEM the required signature is instead
`fem(shell_body, material, ext_density, ext_soundspeed, int_density, int_soundspeed, k;
method = :general, incidence_angle = pi / 2, ...)`.
See [FEM and shell coupling](@ref fem-theory) for restrictions.

## Solution interface and post-processing

All five solution types inherit from `AbstractSolution`. Construct them through their solver,
not by depending on concrete data fields or internal type parameters.

| Result | `target_strength(solution)` | `scattering_amplitude(solution)` | Change observation after solving? |
|:--|:--|:--|:--|
| Modal/Kirchhoff | dB re 1 m² | Complex meters | Rerun with supported solver arguments |
| Ordinary radial/meridian FEM | dB re 1 m² | Throws `ArgumentError` | No |
| Axisymmetric BEM/MFS | Directional dB | Complex meters | `angle`, `azimuth` |
| Full BEM | Backscatter by default | Complex meters | `direction` unit vector |
| Bent MFS | Backscatter | Complex meters | No general direction query |
| Structural shell FEM | Directional dB | Complex meters | `angle`, `azimuth` |

Axisymmetric observation defaults are fixed body coordinates, not incidence-relative
backscatter. Follow [Conventions](@ref conventions). Ordinary FEM does not retain the phase
needed to reconstruct amplitude from target strength alone.

`target_strength(amplitude::Number)` also converts an amplitude using `20 * log10(abs(amplitude))`.
Do not pass an already logarithmic target strength to that overload. For cross section, use
`abs2(scattering_amplitude(solution))` when the amplitude is available.

## Mesh construction

```julia
surface_mesh = mesh(Sphere(0.01); resolution = 32)
surface_mesh = mesh(Spheroid(0.02, 0.01); k = 100.0)
surface_mesh = mesh(Sphere(0.01); method = :full, resolution = 0.003)
```

Each returns `Mesh`. Exactly one of `resolution` or `k` must be supplied. Resolution means
panel count for axisymmetric methods, edge length in meters for full methods. Full surface
generation supports sphere and spheroid. Keep `Panel` and quadrature details out of application
code. Public mesh accessors and access to meshes through solution objects are still being
consolidated.

## Sampling and visualization

The numerical sweep helpers `frequency_sweep`, `incidence_angle_sweep`, `bistatic_sweep`, and
`bistatic_map`, with their result types `FrequencySweep`, `IncidenceAngleSweep`, `BistaticSweep`,
and `BistaticMap`, are being developed alongside plotting recipes. This documentation pass
does not freeze their changing contract. The executable tutorials use explicit solver calls
and CairoMakie, and [the gallery](@ref gallery) tracks the remaining 2D/3D examples.
