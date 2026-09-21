# [BEM and MFS](@id boundary-theory)

Both methods solve for a boundary representation of an outgoing wave. BEM discretizes boundary
integral equations. MFS fits fields produced by fictitious sources.

## Boundary elements

The exterior Green kernel is

```math
G(\boldsymbol x,\boldsymbol y)=\frac{e^{ik|\boldsymbol x-\boldsymbol y|}}{4\pi|\boldsymbol x-\boldsymbol y|}.
```

Single- and double-layer potentials integrate ``G`` and its normal derivative against surface
densities, with a half-identity jump on smooth surfaces.

Axisymmetric BEM expands azimuthal dependence into Fourier modes over a meridian mesh. Panel
count, Fourier cutoff and quadrature tolerance each control a separate error source ([Helsing and
Karlsson, 2014](https://doi.org/10.1016/j.jcp.2014.04.053)). Fluid shells use paired inner and
outer surfaces.

Full BEM accepts any closed triangular surface, including non-axisymmetric and nonconvex
geometry, with outward normals and one connected interface per homogeneous region. Curved
quadratic and cubic triangles preserve the supplied geometry. The mesh validator bounds each
triangle's Jacobian and its separation from neighbors with outward-rounded Bernstein
coefficients, raising an error on an inconclusive or self-intersecting mesh ([Johnen et al.,
2013](https://doi.org/10.1016/j.jcp.2012.08.051)).

### Irregular frequencies and Burton–Miller coupling

Conventional boundary integral equations can lose uniqueness at fictitious interior
eigenfrequencies. For rigid or pressure-release full BEM, `formulation=:burton_miller` (the
default) couples the pressure equation to its normal derivative. With outward normals, the
scattered pressure ``p`` and its normal derivative ``q`` satisfy

```math
\left(\tfrac12 I-D\right)p+Sq=0,\qquad -Hp+\left(\tfrac12 I+K'\right)q=0,
```

where ``S``, ``D``, ``K'`` and ``H`` are the single-, double-, adjoint-double- and
hypersingular-layer operators ([Burton and Miller, 1971](https://doi.org/10.1098/rspa.1971.0097)).

```julia
bem(surface, boundary, k; formulation = :cbie, mesh_order = 3)
```

`formulation=:cbie` selects the conventional pressure equation instead of the Burton–Miller
default. `mesh_order` selects flat (`1`), quadratic (`2`, default) or cubic (`3`) triangles.
Fluid transmission uses its own coupled equations, covered next. `chief_points` adds an optional
CHIEF augmentation, for axial axisymmetric fluid transmission only.

### Fluid transmission and conditioning

Let ``g=\rho_{\rm int}/\rho_{\rm ext}``, with total interface pressure ``p`` and exterior normal
derivative ``q`` on the same outward normal. Full BEM's default `formulation=:muller` solves

```math
\begin{bmatrix} I-D_e+D_i & S_e-gS_i\\ -H_e+H_i/g & I+K'_e-K'_i\end{bmatrix}
\begin{bmatrix}p\\q\end{bmatrix}=\begin{bmatrix}p_{\rm inc}\\\partial_n p_{\rm inc}\end{bmatrix},
```

with subscripts `e`/`i` denoting exterior/interior wavenumbers ([van 't Wout et al.,
2022](https://doi.org/10.1016/j.camwa.2021.11.021)).

At low frequency, density interpolation reconstructs the normal-derivative operators from the
Calderón identities ``SK'=DS`` and ``SH=D^2-\tfrac14 I`` when the enclosing radius ``r`` of every
medium satisfies ``kr\le\pi/2``, a numerical guard below the ball-domain Dirichlet bound
``\pi/r`` ([Grebenkov and Nguyen, 2013](https://arxiv.org/abs/1206.1278)). A looser regular-wave
fit instead requires ``kr\le1`` and ``2kr_{\rm rms}\le1`` ([Faria et al.,
2021](https://doi.org/10.1016/j.cma.2021.113703)).

```julia
bem(surface, boundary, k; formulation = :muller, equilibrate = true)
```

`equilibrate=true` (default) row- and column-scales the matrix before solving.

## [Coupled fluid regions](@id coupled-fluid-regions)

Several fluid interfaces can share one coupled field. Surface `i` separates region `i` from
`parents[i]`, with `0` denoting the unbounded exterior, and normals point out of the volume each
surface encloses.

```julia
bem(surfaces, materials, k; parents = [0, 1, 1])   # two inclusions inside region 1
```

Each region carries its own density contrast ``g_r`` and wavenumber relative to the exterior.
Interface `i`, between its own region `r` and parent region `s`, enforces

```math
p_i^{(r)}=p_i^{(s)},\qquad \frac{1}{g_r}\partial_n p_i^{(r)}=\frac{1}{g_s}\partial_n p_i^{(s)},
```

continuity of pressure and the density-scaled normal derivative, without requiring coincident
mesh nodes on either side. Assembling these traces over every interface gives the same coupled
Müller system as above, one block per interface. `formulation=:muller` (default) assembles that
system directly. `formulation=:cbie` enforces the pressure equation separately on each side,
trading the hypersingular operator for the fictitious-eigenfrequency protection `:muller`
provides.

Every interface must be closed, connected, outward oriented and disjoint from the others, checked
numerically at assembly. `solution.data.interfaces` and
`diagnostics(solution).interface_residuals` retain the per-side pressure, normal derivatives and
residuals. This formulation covers lossless scalar fluids only.

## Method of fundamental solutions

MFS approximates the exterior scattered pressure as

```math
p_{\rm scat}(\boldsymbol x)\approx\sum_{j=1}^N a_j G(\boldsymbol x,\boldsymbol y_j),
```

using fictitious sources placed inside the scatterer, with collocation enforcing the boundary
condition on the true surface ([Fairweather et al.,
2003](https://doi.org/10.1016/S0955-7997(03)00017-1)). Transmission problems need matched
exterior and interior source sets. Increasing the source count alone does not guarantee accuracy,
since source placement and the resulting matrix conditioning matter just as much.

```julia
mfs(body, boundary, k)      # axisymmetric for straight bodies, lateral-only for bent
mfs(surface, boundary, k)   # full 3D, rigid or pressure-release only
```

`offset` places sources a physical distance from the surface, not a fraction of body length.
Sharp cylinder corners make normal-offset placement unreliable, so smooth caps help where
geometry permits them. Compare offset and source count against a modal or BEM reference before
trusting a new configuration.

## Post-processing

The stored boundary fields enter the package's far-field integral.

```julia
target_strength(solution; angle, azimuth)   # axisymmetric BEM/MFS
target_strength(solution; direction)        # full BEM
target_strength(solution)                   # bent MFS, monostatic only
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

GMRES failure emits a warning and sets `report.converged = false`, with the residual recomputed
from the assembled operator including any compression. `solver_options`, `formulation`,
`coupling` and `mesh_order` record the solve settings.

Fluid transmission instead reports its dense coupled system's residual, with no iteration count.
`relative_residual` and `scaled_relative_residual` refer to the original and equilibrated systems
respectively, matching when `equilibrate=false`. `condition_number` and `scaled_condition_number`
are SVD 2-norm condition numbers, computed only up to `condition_limit` (default 512) unknowns.

## Pressure fields and solver diagnostics

### Pressure away from and near the surface

```julia
pressure(solution, points; field = :scattered)
```

[`pressure`](@ref pressure-evaluation) evaluates exterior and transmitted fields for spheres,
straight cylinders, and (via full BEM/MFS) closed bent or supplied meshes. Coupled fluid BEM
evaluates every region, including the unbounded exterior. Axisymmetric BEM subtracts a
constant-pressure Laplace double layer to resolve the near-singular double-layer peak. Full BEM
uses density-interpolation quadrature, and MFS evaluates its retained source fields directly.

For rigid, pressure-release or fluid rims, full BEM supports an edge correction that factors out
the leading wedge singularity in triangles adjoining a sharp edge, where the normal jump exceeds
30 degrees.

```julia
solution = bem(mesh(body; method = :full, mesh_order = 3), boundary, k;
    formulation = :cbie, compression = (method = :none,), correction = (method = :edge,))
```

`rtol`, `atol` and `maxsubdiv` in `correction` control this integration independently of the
linear solve.

Bent and supplied surfaces locate a query point's fluid region from ray crossings against
Bernstein-bounded curved patches. In a coupled fluid region, `region=0` is the unbounded exterior
and `region=i` is the fluid immediately inside interface `i`.

### Residuals and conditioning

```julia
diagnostics(solution).systems                       # one entry per azimuthal system
diagnostics(solution).systems[i].boundary_residual
diagnostics(solution).systems[i].condition_number
```

Each MFS system reports `boundary_residual` (held-out point residuals), `pressure_residual` and
`velocity_residual` (separate transmission checks), and `condition_number`/`numerical_rank` (SVD
diagnostics of the unscaled collocation matrix, up to `condition_limit=512` unknowns).
`n`/`oversampling` set the source and collocation budget (`n_s`/`n_phi` for bent cylinders).
Direct solves leave `converged` and `iterations` as `nothing`.
