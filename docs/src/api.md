# [API Reference](@id api-reference)

The public workflow is geometry → boundary/material → solver → result query. Start with the
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
the same options to every method. Use the supported combinations in the model guide.

| Function | Returns | Main options |
|:--|:--|:--|
| `modal` | `ModalSolution` | Sphere `angle`, `m_max`; spheroid `incidence_angle`, `m_max`, `n_max`; cylinder `incidence_angle`, `m_max` |
| `kirchhoff` | `KirchhoffSolution` | `incidence_angle` for nonspherical bodies |
| `fem` | `FEMSolution` | `method`, discretization controls. Sphere radial defaults to `:radial` |
| `bem` | `BEMSolution` | `method = :axisymmetric` or `:full`, incidence, `n` or `meshsize` respectively |
| `mfs` | `MFSSolution` | Incidence, `offset`, axisymmetric `n`, modal cutoff; bent-cylinder `n_s`, `n_phi` |

Rigid/pressure-release full BEM defaults to `formulation=:burton_miller`;
`:cbie` selects the conventional equation. Full meshes use `mesh_order=2` (curved quadratic
triangles), with `mesh_order=1` for flat triangles and `mesh_order=3` for cubic triangles.
Integration `qorder` defaults to 4. Fluid full BEM defaults to `formulation=:muller`,
a dense two-trace solve; `:cbie` uses four traces. `equilibrate=true` scales rows and
columns before factorization. `condition_limit=512` bounds the condition-number calculation;
zero disables it.
See [BEM and MFS](@ref boundary-theory) for convergence controls and formulation scope.

For several fluid regions use `bem(surfaces, materials, k; parents, ...)`, where each
surface is a full-3D `Mesh` and each material is a `FluidFilled`. Region `i` lies immediately
inside surface `i`; `parents[i]` is the region immediately outside it (`0` is the unbounded
exterior). Parents precede children; omitting `parents` gives the nested chain `[0,1,2,...]`.
Every material contrast refers to the unbounded exterior. The dense coupled Müller solve
supports incidence, equilibration, condition limits, integration `correction`, and geometry
`validation` controls. Mesh size and quadrature belong to the individual `mesh` calls.

`solution.data.interfaces[i]` retains complex total `pressure` and the two
`normal_derivative_interior` / `normal_derivative_exterior` traces. Both derivatives use
the surface normal pointing from region `i` into its parent. The record also contains
`surface`, `interior`, and `exterior`. `diagnostics(solution).interface_residuals` reports
the pressure and density-scaled derivative representation residuals on both sides.
See [Coupled fluid regions](@ref coupled-fluid-regions) for their normalization and limitations.

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

Axisymmetric BEM/MFS and structural shell FEM default to backscatter:
`angle = pi - incidence_angle, azimuth = pi`. Explicit observation angles are measured in
fixed body coordinates. Follow [Conventions](@ref conventions). Ordinary FEM does not retain the phase
needed to reconstruct amplitude from target strength alone.

`target_strength(amplitude::Number)` also converts an amplitude using `20 * log10(abs(amplitude))`.
Do not pass an already logarithmic target strength to that overload. For cross section, use
`abs2(scattering_amplitude(solution))` when the amplitude is available.

## Solver diagnostics

`diagnostics(solution)` returns a named tuple for BEM, MFS and FEM, and `nothing` for
modal and Kirchhoff solutions. A missing report is not a successful-convergence flag.

For rigid/soft full BEM, inspect `converged`, `iterations`, `relative_residual` and
`residual_history`. The report also retains mesh size, quadrature order/node count, system
size, geometry order, formulation/coupling, compression/correction settings and solver options.
Transmission uses a direct solve:
its residual covers the complete coupled system, while `converged` and `iterations` are
`nothing` and its history is empty. `scaled_relative_residual` describes the equilibrated
system; the original-system residual remains available. `condition_number` and
`scaled_condition_number` are `nothing` when the unknown count exceeds `condition_limit`.
Residuals measure the assembled linear system, not
discretization accuracy or physical-model validity. See the [BEM guide](@ref boundary-theory).

Axisymmetric BEM, MFS and FEM reports contain `systems`: per-mode or per-basis solves
with residuals, equation/unknown counts and numerical controls. Summary residuals and
counts are maxima over these systems. `solver_options` records public settings; actual
discretization controls also appear in the per-system reports. Direct solves use
`converged = iterations = nothing`. Adaptive radial FEM separately reports
`refinement.converged`, `change_db`, `target_tol` and the final `n_elements`.

For MFS, `oversampling=2` increases collocation resolution with the source grid fixed.
Each system reports its source offsets/counts, conditioning/rank and a boundary residual
at points excluded from the solve. These checks add assembly and SVD cost. Consult the
[BEM and MFS guide](@ref boundary-theory) for their normalization and limitations.

For a bent cylinder, `mfs(...; n_s = 40, n_phi = 32)` controls the source grid. Counts must
be at least 3. The legacy `n_φ` spelling is accepted, but passing both spellings is an error.
This body-based bent MFS omits end caps. Closed-surface MFS uses
`mfs(surface, boundary, k; source_mesh, check_mesh, offset)` for rigid/pressure-release
boundaries. All three meshes must describe the same closed body. `check_mesh` is optional;
its boundary residual is reported separately from the fitted residual. Refine source spacing,
offset and collocation resolution independently.

## Mesh construction

```julia
surface_mesh = mesh(Sphere(0.01); resolution = 32)
surface_mesh = mesh(Spheroid(0.02, 0.01); k = 100.0)
surface_mesh = mesh(Sphere(0.01); method = :full, resolution = 0.003)
surface_mesh = mesh(Cylinder(0.005, 0.02; radius_curvature = 0.02, endcap_depth = 0.005);
    method = :full, resolution = 0.0032, mesh_order = 3)
```

Each returns `Mesh`. Exactly one of `resolution` or `k` must be supplied. Resolution means
panel count for axisymmetric methods, edge length in meters for full methods. Full surface
generation supports spheres and spheroids.

For a supplied closed surface, use the same `mesh` and `bem` functions:

```julia
surface_mesh = mesh("body.msh"; units = :mm, qorder = 4)
solution = bem(surface_mesh, FluidFilled(1.05, 1.02), 100.0;
    incidence_angle = pi / 3, incidence_azimuth = pi / 6)
amplitude = scattering_amplitude(solution; direction = [0.0, 1.0, 0.0])
```

`mesh(nodes, triangles; units=:m, labels, provenance)` accepts coordinate/connectivity
matrices with one point/triangle per column. Triangles have 3, 6 or 10 nodes in Gmsh ordering.
`mesh(generate; units=:m, qorder=4)` calls `generate(gmsh)` in an owned Gmsh session and imports
the resulting model without a file. Gmsh physical surface tags and names are preserved.
See [Geometry and incidence](@ref geometry-tutorial) for geometry requirements and validation.
The optional `validation` named tuple sets `maxdepth` (default 20, at most 24) and `maxwork`
(default 200000). Bounds that remain unresolved at either limit produce an error.

## Sampling and visualization

`mesh(; semiaxes=(a,b,c), center=(x,y,z), rotation=(axis=(0,1,0), angle=beta))`
constructs a full-3D ellipsoid in metres. `resolution` sets the dimensionless edge size
before stretching a unit sphere; `tip_ratio` refines its local x poles. Geometry order
(`mesh_order`) and quadrature order (`qorder`) are independent.

`frequency_sweep(solve, frequencies, sound_speed)` calls `solve(k)` at frequencies in Hz.
`incidence_angle_sweep(solve, angles)` calls `solve(angle)` with radians.
`bistatic_sweep(solution, angles; azimuth=0)` evaluates retained surface data without
re-solving, including full-3D and coupled BEM. Results contain `target_strength`, complex
`amplitudes` and `labels`; scalar-only FEM results have `amplitudes=nothing`.

`components(solution; labels)` reuses a coupled fluid-region solution and solves each
interface alone in the original exterior medium. Return it from a sweep callback, or
pass it to `bistatic_sweep`, to compare coupled, isolated and coherently summed responses.
Comparison arrays have samples in rows and labeled responses in columns.

With Makie loaded, `plot(sweep, sweeps...)` returns a figure with strength and phase rows
and one column per sweep. `quantity=:target_strength`, `:phase`, `:magnitude`, `:real`
or `:imag` selects a single row. `plot!(axis, sweep; quantity)` adds a saved response to
an existing axis. Phase is wrapped to ±π; none of these plots performs another solve.

The [frequency-sweep tutorial](@ref first-sweep) demonstrates solving across frequencies,
plotting target strength with CairoMakie, and saving the results as CSV. See
[Geometry and incidence](@ref geometry-tutorial) for an angle sweep and
[the gallery](@ref gallery) for the resulting figures.

For coupled fluid-region solutions, `plot(solution; kind=:mesh)` colours each interface
separately. `kind=:surface_field` shows total interface pressure, normalized to the unit
incident wave, with `field=:pressure_magnitude` (default), `:pressure_phase`,
`:pressure_real` or `:pressure_imag`. `interfaces=[1, 2]` selects surface indices;
`interface_colors` sets their geometry colours; `(colour, alpha)` sets opacity.
For geometry plots, `wireframe_interfaces=[1]` draws the selected interface as a mesh
grid, leaving other interfaces filled. This shows an internal organ within the complete
outer body. Field plots share a `colorrange` and
`colormap`. `show_edges=true` overlays the displayed triangles. Standalone `plot` adds
a geometry legend or pressure colourbar automatically; set `legend=false` or
`colorbar=false` to omit it. `interface_labels` names the displayed geometry interfaces.

`cutaway=(normal=(0, 1, 0), offset=0)` clips exterior-adjacent surfaces to
`normal ⋅ x ≤ offset`, leaving internal interfaces whole. Display triangles and
vertex-averaged pressure approximate the curved quadrature surface; the cutaway does
not alter the solve. Both `plot` and `plot!` accept these options. See
[A synthetic fish and swimbladder](@ref fish-tutorial) for geometry, phase-preserving
component comparisons and convergence checks.
