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
| Supported radial FEM spheres (acoustic, solid elastic and layered) | dB re 1 m² | Complex meters | Backscatter only |
| Cylinder radial and meridian FEM | dB re 1 m² | Throws `ArgumentError` | No |
| Axisymmetric BEM/MFS | Directional dB | Complex meters | `angle`, `azimuth` |
| Full BEM | Backscatter by default | Complex meters | `direction` unit vector |
| Bent MFS | Backscatter | Complex meters | No general direction query |
| Structural shell FEM | Directional dB | Complex meters | `angle`, `azimuth` |

Axisymmetric BEM/MFS and structural shell FEM default to backscatter:
`angle = pi - incidence_angle, azimuth = pi`. Explicit observation angles are measured in
fixed body coordinates. Follow [Conventions](@ref conventions). Scalar-only FEM results do not
retain the phase needed to reconstruct amplitude from target strength alone.

`target_strength(amplitude::Number)` also converts an amplitude using `20 * log10(abs(amplitude))`.
Do not pass an already logarithmic target strength to that overload. For cross section, use
`abs2(scattering_amplitude(solution))` when the amplitude is available.

## [Pressure at Cartesian points](@id pressure-evaluation)

`pressure(solution, points; field=:total, region=nothing)` returns complex pressure divided by the incident
pressure amplitude. It supports `modal` and radial `fem` solutions for `Sphere` with
`Rigid`, `PressureRelease`, `FluidFilled`/`GasFilled`, `SolidElastic`, fluid shells with
fluid/vacuum interiors, and elastic shells with fluid interiors. Axisymmetric/full BEM
and axisymmetric MFS support spheres and straight cylinders with rigid, pressure-release
and fluid-filled boundaries; full MFS supports their rigid and pressure-release boundaries.
Full BEM/MFS also support closed bent cylinders and supplied meshes with those respective
boundary conditions. Coupled fluid BEM supports nested, branched and disconnected regions.
Spheres are centered at the origin and straight cylinders extend
along x. Supplied meshes retain their coordinates.
Modal/FEM incidence travels along +x, with pressure `exp(im*k*x)` under the `exp(-iωt)`
convention. BEM/MFS incidence follows the solve's angles. Coordinates are in meters.

```@example pressure_sampling
using AcousticScattering

solution = modal(Sphere(0.01), FluidFilled(1.2, 1.1), 100.0)
pressure(solution, [(0.0, 0.0, 0.0), (0.010001, 0.0, 0.0), (0.02, 0.0, 0.0)])
```

A three-coordinate tuple or vector returns one complex value. An array of such points
returns an array with the same shape. A real `3×N` matrix represents points in columns
and returns a vector of length `N`.

| `field` | Meaning and domain |
|:--|:--|
| `:total` | Incident plus scattered pressure outside; transmitted total pressure in fluid regions inside |
| `:scattered` | Scattered pressure on and outside the body |
| `:incident` | Unperturbed plane wave at any point |
| `:interior` | Total pressure in a homogeneous fluid body, spherical cavity or bounded coupled fluid region, including the inner interface trace |
| `:shell` | Total pressure in a fluid shell, including both of its surface traces |

For layered spheres, `:interior` selects the fluid cavity and `:shell` selects the fluid
wall. The default `:total` selects the fluid region containing the point. At the outer
surface it selects the exterior trace; at the inner interface it selects the fluid cavity,
or the fluid shell's trace when the cavity is vacuum. Points within eight floating-point
spacings of either radius count as interface points. Rigid, pressure-release, vacuum and
elastic regions have no acoustic pressure field. `modal(...; angle)` selects the far-field observation
angle; it does not rotate the incident wave used by `pressure`.

Refine `m_max` to check modal truncation, including agreement between the two surface
traces. For FEM, also refine the radial elements: samples use linear/quadratic interpolation
in the exterior annulus, linear interpolation in a fluid interior, and outgoing spherical
waves beyond the outer radius. Fluid shells use linear FEM interpolation through the wall.
Elastic and layered FEM spheres use their spherical-wave coefficients in the exterior and
fluid cavity. Elastic-shell `interior_coupling=:identical_fluid` uses exterior-fluid properties
in the cavity. Elastic stress/displacement, viscous layers and spheroidal fields are unavailable.

```@example pressure_sampling
boundary = Shelled(FluidLayer(1.04, 1.04), FluidInterior(1.2, 1.1), 0.8)
layered = fem(Sphere(0.01), boundary, 160.0; n_elements=640)
pressure(layered, [(0.0, 0.0, 0.0), (0.009, 0.0, 0.0), (0.02, 0.0, 0.0)])
```

Compare the cavity and shell traces at their shared radius:

```@example pressure_sampling
point = (0.008, 0.0, 0.0)
(cavity=pressure(layered, point; field=:interior),
 shell=pressure(layered, point; field=:shell))
```

The same query samples BEM/MFS fields:

```@example pressure_sampling
boundary_solution = mfs(Sphere(0.01), FluidFilled(1.2, 1.1), 100.0;
    incidence_angle=pi/3, n=96, oversampling=2, offset=0.002, m_max=8)
pressure(boundary_solution, [(0.0, 0.0, 0.0), (0.006, 0.008001, 0.0)])
```

MFS evaluates its solved sources. Refine source spacing, offset, collocation resolution
and the azimuthal cutoff separately. Axisymmetric MFS grades sources toward sharp
flat-cylinder rims: `offset` is a maximum, limited locally to half the distance to
the corner. Full BEM uses density-interpolation quadrature
near the surface; refine `meshsize`, `mesh_order` and `qorder`. A small linear-system
residual alone does not establish field accuracy. Region selection uses the analytic
sphere radius or straight-cylinder dimensions, while the mesh approximates its boundary. Compare both interface traces
under refinement when evaluating at or very close to the boundary.

Axisymmetric BEM uses singularity-subtracted adaptive integration. Its piecewise-constant
surface traces can require many panels for near-surface queries, especially away from
panel midpoints. Check convergence at the actual requested points. A cylinder's
axisymmetric BEM geometry has flat ends; full BEM and MFS use its `endcap_depth`.

```@example pressure_sampling
body = Cylinder(0.005, 0.01; endcap_depth=0.005)
cylinder_solution = mfs(body, FluidFilled(1.2, 1.1), 50.0;
    n=128, oversampling=2, offset=0.0012, incidence_angle=pi/3, m_max=6)
pressure(cylinder_solution, [(0.0, 0.0, 0.0), (0.010001, 0.0, 0.0)])
```

For bent cylinders and supplied meshes, region selection follows the solved curved mesh.
Adaptive ray crossings refine ambiguous elements; an unresolved location raises an error.
The geometric stopping tolerance is 512 floating-point spacings at the largest absolute
Bernstein coordinate. At a surface point, `:total` uses the exterior trace and `:interior`
uses the fluid-side trace. Full MFS requires a closed `Mesh`; the lateral-only bent-cylinder
MFS approximation has no point-pressure query.

```@example pressure_sampling
bent_body = Cylinder(0.005, 0.01; radius_curvature=0.02, endcap_depth=0.005)
bent_solution = bem(bent_body, Rigid(), 50.0;
    method=:full, meshsize=0.005, mesh_order=3, qorder=4)
pressure(bent_solution, [(0.0, 0.01, 0.0), (0.02, 0.0, 0.0)])
```

For coupled fluid BEM, the default query chooses the containing fluid from the solved
curved interfaces. `region=0` selects the unbounded exterior; `region=i` selects the fluid
immediately inside surface `i`, excluding any child regions. Points outside the selected
region raise an error. The same query can sample several regions:

```@example pressure_sampling
outer = mesh(Sphere(0.01); method=:full, resolution=0.006, mesh_order=3, qorder=5)
inner = mesh(Sphere(0.005); method=:full, resolution=0.003, mesh_order=3, qorder=5)
coupled = bem([outer, inner], [FluidFilled(1.2, 1.1), FluidFilled(0.7, 0.8)], 30.0;
    condition_limit=0)
pressure(coupled, [(0.0, 0.0, 0.0), (0.007, 0.0, 0.0), (0.02, 0.0, 0.0)])
```

At a coupled interface, `region` can select either adjacent fluid. Without it, `:total`
uses the parent-side trace and `:interior` uses the child-side trace. In a bounded fluid,
`:interior` and `:total` both give transmitted total pressure; `:scattered` is restricted
to the unbounded exterior. Use `region=i` to select a coupled shell; `:shell` is reserved
for layered spherical solutions. `:incident` remains the unperturbed exterior-medium
plane wave, with domain checking when `region` is explicit. Refine each surface and its
quadrature separately, especially near interfaces and resonances.

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
For fluid/gas full-3D BEM, `incidence_angle_sweep(surface, material, k, angles)` and
`incidence_angle_sweep(surfaces, materials, k, angles; parents)` reuse geometry checks,
operators and factorization at fixed exterior `k`. Pass `incidence_azimuth` and BEM
solver options as keywords. The multiple-interface form accepts `components=true`
and `labels` to include isolated responses and their coherent sum. Reuse is confined
to that call; a subsequent call uses its own geometry, material and numerical options.
For `Rigid()` and `PressureRelease()`, `incidence_angle_sweep(surface, boundary, k, angles)`
reuses the layer and system operators, including their compression. Each angle starts a
fresh GMRES solve. Keywords are `incidence_azimuth`, `formulation`, `compression`,
`correction` and `gmres_kwargs`, with the same defaults as `bem(surface, boundary, k)`.
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
