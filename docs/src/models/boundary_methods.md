# [BEM and MFS](@id boundary-theory)

Both methods solve for a boundary representation of an outgoing wave. BEM discretizes boundary integral equations. MFS fits fields produced by fictitious sources.

## Boundary elements

The exterior Green kernel is

```math
G(\boldsymbol x,\boldsymbol y)=\frac{e^{ik|\boldsymbol x-\boldsymbol y|}}{4\pi|\boldsymbol x-\boldsymbol y|}.
```

Single- and double-layer potentials integrate ``G`` and its normal derivative against surface densities, with a half-identity jump on smooth surfaces.

Axisymmetric BEM expands azimuthal dependence into Fourier modes over a meridian mesh. Panel count, Fourier cutoff and quadrature tolerance each control a separate error source ([Helsing and Karlsson, 2014](https://doi.org/10.1016/j.jcp.2014.04.053)). Fluid shells use paired inner and outer surfaces.

At axial incidence, the direct CBIE ``\tfrac12 p_{\rm scat}-K[p_{\rm scat}]=-G[\partial p_{\rm scat}/\partial n]`` holds for any boundary. `Rigid` and `PressureRelease` set one boundary value and solve for the other. [`Impedance`](@ref)`(zeta)`'s Robin condition, `∂p_scat/∂n = -ik/zeta*(p_scat+p_inc) - ∂p_inc/∂n`, substitutes directly into it, giving a single equation in `p_scat`:

```math
\left(\tfrac12 I-K-\frac{ik}{\zeta}G\right)p_{\rm scat}=\frac{ik}{\zeta}G[p_{\rm inc}]+G\!\left[\frac{\partial p_{\rm inc}}{\partial n}\right],
```

which reduces exactly to `Rigid`'s own equation as `zeta → ∞`. The direct solve becomes ill-conditioned as `zeta → 0`; use `PressureRelease` there instead.

Full BEM accepts any closed triangular surface, including non-axisymmetric and nonconvex geometry, with outward normals and one connected interface per homogeneous region. Curved quadratic and cubic triangles preserve the supplied geometry. The mesh validator bounds each triangle's Jacobian and its separation from neighbors with outward-rounded Bernstein coefficients, raising an error on an inconclusive or self-intersecting mesh ([Johnen et al., 2013](https://doi.org/10.1016/j.jcp.2012.08.051)).

### Irregular frequencies and Burton–Miller coupling

Conventional boundary integral equations can lose uniqueness at fictitious interior eigenfrequencies. For rigid or pressure-release full BEM, `formulation=:burton_miller` (the default) couples the pressure equation to its normal derivative. With outward normals, the scattered pressure ``p`` and its normal derivative ``q`` satisfy

```math
\left(\tfrac12 I-D\right)p+Sq=0,\qquad -Hp+\left(\tfrac12 I+K'\right)q=0,
```

where ``S``, ``D``, ``K'`` and ``H`` are the single-, double-, adjoint-double- and hypersingular-layer operators ([Burton and Miller, 1971](https://doi.org/10.1098/rspa.1971.0097)).

```julia
bem(surface, boundary, k; formulation = :cbie, mesh_order = 3)
```

`formulation=:cbie` selects the conventional pressure equation instead of the Burton–Miller default. `mesh_order` selects flat (`1`), quadratic (`2`, default) or cubic (`3`) triangles. Fluid transmission uses its own coupled equations, covered next. `chief_points` adds an optional CHIEF augmentation, for axial axisymmetric fluid transmission only.

### Fluid transmission and conditioning

Let ``g=\rho_{\rm int}/\rho_{\rm ext}``, with total interface pressure ``p`` and exterior normal derivative ``q`` on the same outward normal. Full BEM's default `formulation=:muller` solves

```math
\begin{bmatrix} I-D_e+D_i & S_e-gS_i\\ -H_e+H_i/g & I+K'_e-K'_i\end{bmatrix}
\begin{bmatrix}p\\q\end{bmatrix}=\begin{bmatrix}p_{\rm inc}\\\partial_n p_{\rm inc}\end{bmatrix},
```

with subscripts `e`/`i` denoting exterior/interior wavenumbers ([van 't Wout et al., 2022](https://doi.org/10.1016/j.camwa.2021.11.021)).

At low frequency, density interpolation reconstructs the normal-derivative operators from the Calderón identities ``SK'=DS`` and ``SH=D^2-\tfrac14 I`` when the enclosing radius ``r`` of every medium satisfies ``kr\le\pi/2``, a numerical guard below the ball-domain Dirichlet bound ``\pi/r`` ([Grebenkov and Nguyen, 2013](https://arxiv.org/abs/1206.1278)). A looser regular-wave
fit instead requires ``kr\le1`` and ``2kr_{\rm rms}\le1`` ([Faria et al., 2021](https://doi.org/10.1016/j.cma.2021.113703)).

```julia
bem(surface, boundary, k; formulation = :muller, equilibrate = true)
```

`equilibrate=true` (default) row- and column-scales the matrix before solving.

Dense single-interface fluid BEM supports `precision=:mixed` with equilibration and double-precision fallback. Dense CBIE also supports `frequency_sweep` with explicit training frequencies and full-solve fallback. See [Performance](@ref performance-tutorial) for both options.

Single-interface Müller supports an opt-in compressed solve:

```julia
solution = bem(surface, boundary, k;
    compression = (method = :hmatrix, tol = 1e-9),
    gmres_kwargs = (reltol = 1e-10, restart = 150, maxiter = 500))
```

Compression uses GMRES. Dense LU remains the default. Compressed CBIE and edge quadrature are unsupported. Refine mesh, compression and iteration tolerances independently.

`formulation=:cbie` avoids hypersingular operators but retains fictitious interior resonances. Both fluid formulations solve for two unknowns per surface node.

## Coupled fluid regions

Several fluid interfaces can share one coupled field. Surface `i` separates region `i` from `parents[i]`, with `0` denoting the unbounded exterior, and normals point out of the volume each
surface encloses.

```julia
bem(surfaces, materials, k; parents = [0, 1, 1]) # two inclusions inside region 1
```

Each region carries its own density contrast ``g_r`` and wavenumber relative to the exterior. Interface `i`, between its own region `r` and parent region `s`, enforces

```math
p_i^{(r)}=p_i^{(s)},\qquad \frac{1}{g_r}\partial_n p_i^{(r)}=\frac{1}{g_s}\partial_n p_i^{(s)},
```

continuity of pressure and the density-scaled normal derivative, without requiring coincident mesh nodes on either side. Assembling these traces over every interface gives the same coupled Müller system as above, one block per interface. `formulation=:muller` (default) assembles that system directly. `formulation=:cbie` enforces the pressure equation separately on each side, trading the hypersingular operator for the fictitious-eigenfrequency protection `:muller` provides.

Every interface must be closed, connected, outward oriented and disjoint from the others, checked numerically at assembly. `diagnostics(solution).interface_residuals` reports interface continuity errors. This formulation covers lossless scalar fluids only.

Coupled `:muller` accepts the same compression options. `incidence_angle_sweep` reuses operators and factorizations across angles. `recycle_dimension=0` disables reuse of previous solutions. Refine tolerances when interfaces are close or near resonance.

## Method of fundamental solutions

MFS approximates the exterior scattered pressure as

```math
p_{\rm scat}(\boldsymbol x)\approx\sum_{j=1}^N a_j G(\boldsymbol x,\boldsymbol y_j),
```

using fictitious sources placed inside the scatterer, with collocation enforcing the boundary condition on the true surface ([Fairweather et al., 2003](https://doi.org/10.1016/S0955-7997(03)00017-1)). Transmission problems need matched exterior and interior source sets. Increasing the source count alone does not guarantee accuracy, since source placement and the resulting matrix conditioning matter just as much.

```julia
mfs(body, boundary, k) # axisymmetric for straight bodies, lateral-only for bent
mfs(Spheroid(1.2, 1.0),
    Shelled(FluidLayer(1.2, 1.1), FluidInterior(0.7, 0.8), 0.55), 1.5;
    incidence_angle = 0.0, n = 40, oversampling = 2) # confocal fluid shell
mfs(surface, boundary, k) # full 3D, rigid, pressure-release or fluid-filled
mfs(bladder_surface, backbone_surface, FluidFilled(g_bone, h_bone), k;
    bladder_offset, backbone_offset_ext, backbone_offset_int) # coupled components
```

`offset` places sources a physical distance from the surface, not a fraction of body length. Sharp cylinder corners make normal-offset placement unreliable, so smooth caps help where geometry permits them. Compare offset and source count against a modal or BEM reference before trusting a new configuration.

Rigid flat-ended cylinders are unsupported by axisymmetric `mfs`. Use `bem` for that geometry, or smooth caps when physically appropriate.

The bladder/backbone overload couples a pressure-release bladder to a disjoint scalar-fluid backbone. Contrasts are relative to exterior water. Refine source meshes and check boundary residuals on independent meshes. This overload provides far-field results only.

Nested-fluid MFS supports axial incidence on spheres and confocal prolate spheroids with one fluid shell and a fluid core. `offset_outer` and `offset_inner` are positive distances in meters. `pressure(...; field=:total)` selects the region by position. Oblique incidence, oblate bodies, multiple shells and elastic layers are unsupported.

Closed-surface fluid MFS uses `offset_ext` and `offset_int` for exterior and interior sources. Refine `source_mesh` and offsets independently of collocation, and use `check_mesh` for boundary residuals.

## Post-processing

The stored boundary fields enter the package's far-field integral.

```julia
target_strength(solution; angle, azimuth) # axisymmetric BEM/MFS
target_strength(solution; direction) # full BEM and full MFS
target_strength(solution) # bent MFS, monostatic only
```

See [Conventions](@ref conventions) before interpreting default angles as backscatter.

## Full-3D solve diagnostics

```@example bem_diagnostics
using AcousticScattering

solution = bem(Sphere(0.01), Rigid(), 100.0; method = :full, meshsize = 0.01)
report = diagnostics(solution)
@assert report.converged
@assert report.relative_residual < 1e-3
(iterations = report.iterations, relative_residual = report.relative_residual)
```

GMRES failure emits a warning and sets `report.converged = false`, with the residual recomputed from the assembled operator including any compression. `solver_options`, `formulation`, `coupling` and `mesh_order` record the solve settings.

`relative_residual` and `scaled_relative_residual` describe the original and equilibrated equations. Dense condition estimates are available up to `condition_limit=512` unknowns. Compressed solves omit them. Mixed precision also reports refinement iterations and backward errors. Small residuals do not establish mesh or compression accuracy.

## Pressure fields and solver diagnostics

### Pressure away from and near the surface

```julia
pressure(solution, points; field = :scattered)
```

[`pressure`](@ref pressure-evaluation) evaluates exterior and transmitted fields for spheres, straight cylinders, and spheroids via axisymmetric BEM/MFS or full BEM. It also supports closed bent or supplied full surfaces. Coupled fluid BEM evaluates every region, including the unbounded exterior. Axisymmetric BEM subtracts a constant-pressure Laplace double layer to resolve the near-singular double-layer peak. Full BEM
uses density-interpolation quadrature, and MFS evaluates its retained source fields directly.

For rigid, pressure-release or fluid rims, full BEM supports an edge correction that factors out the leading wedge singularity in triangles adjoining a sharp edge, where the normal jump exceeds 30 degrees.

```julia
solution = bem(mesh(body; method = :full, mesh_order = 3), boundary, k;
    formulation = :cbie, compression = (method = :none,), correction = (method = :edge,))
```

`rtol`, `atol` and `maxsubdiv` in `correction` control this integration independently of the
linear solve.

Full spheroid, bent and supplied surfaces locate a query point's fluid region from ray crossings against Bernstein-bounded curved patches. In a coupled fluid region, `region=0` is the unbounded exterior and `region=i` is the fluid immediately inside interface `i`.

### Residuals and conditioning

```julia
diagnostics(solution).systems                       # one entry per azimuthal system
diagnostics(solution).systems[i].boundary_residual
diagnostics(solution).systems[i].condition_number
```

Each MFS system reports `boundary_residual` (held-out point residuals), `pressure_residual` and `velocity_residual` (separate transmission checks), and `condition_number`/`numerical_rank` (SVD diagnostics of the unscaled collocation matrix, up to `condition_limit=512` unknowns). `n`/`oversampling` set the source and collocation budget (`n_s`/`n_phi` for bent cylinders).

Use `incidence_angle_sweep(mfs, body, boundary, k, angles)` to reuse axisymmetric MFS assembly and factorization. Add `return_diagnostics=true` for per-angle reports. Bent and full-surface MFS use the callback sweep. Direct solves report `converged = iterations = nothing`.
