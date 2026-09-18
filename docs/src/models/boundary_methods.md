# [BEM and MFS](@id boundary-theory)

Both methods solve for a boundary representation of an outgoing wave. BEM discretizes boundary
integral equations. MFS fits fields produced by fictitious sources. They have different
conditioning and resolution controls, although both can retain data for many observation queries.

## Boundary elements

The exterior Green kernel is

```math
G(\boldsymbol x,\boldsymbol y)=
\frac{e^{ik|\boldsymbol x-\boldsymbol y|}}
{4\pi|\boldsymbol x-\boldsymbol y|}.
```

Single- and double-layer potentials integrate `G` and its normal derivative against surface
densities. Boundary traces include a half-identity jump on smooth surfaces. Which trace is
unknown depends on whether pressure, normal derivative, or a coupled fluid interface is imposed.
Normal orientation and the exterior/interior trace determine signs. Do not mix equations from
references with different normal conventions.

Axisymmetric BEM in `engine/axisymmetric_bem.jl` expands azimuthal dependence into Fourier
modes and integrates ring kernels over a meridian mesh. A rotationally symmetric body at oblique
incidence still requires multiple modes. Panel count, Fourier cutoff, and quadrature tolerance
control distinct errors. Fluid shells use inner and outer surfaces in `engine/shell_bem.jl`.

Near a gas-inclusion resonance, small boundary-integral errors can shift the peak appreciably.
Refine the panel count and quadrature tolerance separately, compare complex amplitudes at
identical frequencies, and check both sides of the peak. A small matrix residual alone does
not bound this error. Ring proximity also matters independently of panel length: nearly
coincident source and observation rings require accurate azimuthal quadrature. For a detailed
treatment of axisymmetric near-singular quadrature, see
[Helsing and Karlsson (2014)](https://arxiv.org/abs/1310.4715).

Full BEM uses Gmsh and Inti surface quadrature, density-interpolation
corrections, and optional compressed operators for rigid/soft conditions. The fluid-transmission
path uses a dense block system. `bem(surface_mesh, boundary, k)` accepts a supplied closed
triangular surface from `mesh`, including non-axisymmetric and nonconvex geometry. Normals
must point out of the body. Coordinates are converted to metres before integration, and `k`
is in inverse metres. One connected interface separates the homogeneous interior from the
exterior; nested surfaces cannot be modeled by assigning several labels to this interface.

Curved quadratic/cubic triangles preserve the supplied geometry. Refine geometry and
quadrature independently, especially near narrow gaps and sharp edges. The mesh validator
checks topology and bounds the polynomial geometry throughout each curved triangle.
Bernstein coefficients enclose the projected Jacobian and the surface in convex bounds;
adaptive subdivision tightens these enclosures. The Jacobian bounding approach follows
[Johnen, Remacle and Geuzaine (2013)](https://doi.org/10.1016/j.jcp.2012.08.051).

Acceptance also requires an injectivity certificate for each element and separation
certificates for element pairs. Shared vertices and curved edges are treated explicitly.
Coefficient arithmetic rounds outward, and neither a distance tolerance nor a subdivision
limit permits an ambiguous pair to pass. An unresolved case raises an error. These are
sufficient conditions: valid geometry with a folded corner-plane projection, or bounds
that remain inconclusive within the work limit, can be rejected. Geometric acceptance
does not establish convergence of the acoustic discretization.

### Irregular frequencies and Burton–Miller coupling

Conventional boundary integral equations can lose uniqueness at fictitious interior
eigenfrequencies even though the exterior scattering problem has a unique solution.
For rigid/pressure-release full BEM, `formulation=:burton_miller` couples the pressure
equation to its normal derivative. With outward body normals and the outgoing kernel above,
the scattered pressure `p` and its normal derivative `q` satisfy

```math
\left(\tfrac12 I-D\right)p+Sq=0,\qquad
-Hp+\left(\tfrac12 I+K'\right)q=0.
```

Here `S`, `D`, `K'`, and `H` are the single-layer, double-layer, adjoint double-layer,
and hypersingular boundary operators. Adding `i/k` times the second equation to the first
gives the Burton–Miller equation. Its nonreal coupling removes the fictitious-frequency
nonuniqueness; it does not remove discretization error. See
[Burton and Miller (1971)](https://doi.org/10.1098/rspa.1971.0097).
The prescribed scattered trace is `q=-∂ₙp_inc` for a rigid surface and `p=-p_inc`
for a pressure-release surface. The other trace is obtained from the coupled equation.

For a pressure-release surface, the equivalent equation for total normal derivative
`q_total = q + ∂ₙp_inc` has the analytic incident-field forcing

```math
\left[S+\frac{i}{k}\left(\tfrac12 I+K'\right)\right]q_{\rm total}
=p_{\rm inc}+\frac{i}{k}\partial_n p_{\rm inc}.
```

Kernel reciprocity gives `K'(x,y)=D(y,x)` without complex conjugation, but singular
quadrature corrections need not preserve that relation as a weighted matrix transpose.
The normal-derivative operators use their own corrected quadrature. Sharp rims require
local mesh refinement and complex-amplitude convergence checks.

Full BEM uses this coupling by default for rigid/pressure-release boundaries at positive
wavenumber. `formulation=:cbie` selects the conventional pressure equation.
`mesh_order=2` uses curved quadratic triangles; `mesh_order=1` uses flat triangles.
`mesh_order=3` uses cubic triangles for finer geometric approximation at the same mesh size.
Refine `meshsize` and integration `qorder` separately: improving integration on a flat
mesh does not recover the exact curved geometry. Supported triangle quadrature orders
include 2, 4, 5 and 7. Check complex amplitude as well as target strength, since a phase
error can be hidden by close agreement in magnitude.

Fluid transmission uses its own coupled equations. The optional `chief_points`
augmentation applies only to axial axisymmetric fluid transmission.

### Fluid transmission and conditioning

Let `g=ρ_interior/ρ_exterior`, let `p` be total interface pressure, and let `q` be
its exterior normal derivative. With the same outward normal on both sides, full BEM's
`formulation=:muller` solves

```math
\begin{bmatrix}
I-D_e+D_i & S_e-gS_i\\
-H_e+H_i/g & I+K'_e-K'_i
\end{bmatrix}
\begin{bmatrix}p\\q\end{bmatrix}
=\begin{bmatrix}p_{\rm inc}\\\partial_n p_{\rm inc}\end{bmatrix}.
```

Subscripts `e` and `i` denote exterior and interior wavenumbers. Interior normal derivative
is `g*q`; subtract the incident traces to obtain the exterior scattered traces.
This is the Müller formulation in [van 't Wout et al., equation (26)](https://arxiv.org/abs/2104.04618),
with the hypersingular sign converted to the convention above.

For low-frequency Müller quadrature, density interpolation uses regular spherical
Helmholtz solutions. Their pressure and normal traces are fitted with row scaling and
a singular-value decomposition on each source element. The radial functions use the
[spherical Bessel power series](https://dlmf.nist.gov/10.53); the correction enforces
Green's representation, following the
[density-interpolation method](https://doi.org/10.1016/j.cma.2021.113703).
The regular basis requires `k*radius <= 1` and `2k*rms_radius <= 1`; target arguments
must be at most two. The enclosing radius is the maximum distance from quadrature
nodes to their mean position. The RMS radius averages squared distances using surface
quadrature areas as weights. In a coupled region, each radius is the maximum over
its individual interfaces and their combined node set. Every medium must satisfy
these bounds before regular-wave interpolation is used. All media
share this choice across the whole solve, preserving cancellation at
weak-contrast interfaces.

For low-frequency self-interactions, the normal-derivative operators are obtained
from the Calderón identities

```math
S K' = D S,\qquad S H = D^2-\tfrac14 I.
```

These relate operators at the **same** wavenumber; see
[van 't Wout et al., equations (28) and (38)](https://arxiv.org/abs/2104.04618).
With density interpolation, fluid BEM uses this reconstruction when regional
`k*radius <= π/2` in every medium. It uses the selected pressure operators with
either regular-wave or direct density interpolation. Both size measures are unchanged
by rigid rotations and translations.
It factors the single-layer matrix and avoids direct hypersingular quadrature for those
self-interactions. Other interactions retain direct derivative quadrature.
For a domain enclosed by a ball of radius `R`, Dirichlet domain monotonicity gives
a first interior wavenumber of at least `π/R`; see
[Grebenkov and Nguyen, sections 2(v) and 3.3](https://arxiv.org/abs/1206.1278).
The `π/2` selection leaves a factor-two margin relative to that ball value.
Here the radius is estimated from quadrature nodes, so this is a numerical guard,
not a certified eigenvalue bound for the curved mesh.
Single-layer inversion is unsuitable near its interior Dirichlet resonances, so this
reconstruction is restricted to low frequency. `diagnostics(solution).derivative_evaluation`
records the selected evaluations. Reconstructed flux residuals use the same pressure
operators and do not independently establish quadrature accuracy.

Both `:muller` and the four-trace `:cbie` alternative use dense direct factorization.
`equilibrate=true` divides each matrix row by its largest absolute entry, then does the
same for columns, transforming the right-hand side and recovering the physical unknowns
after solving. This reduces differences in matrix-entry scales; it does not remove physical
resonances or quadrature errors. Compare complex amplitudes under mesh and quadrature
refinement, especially near a gas-inclusion resonance.

Use [Numerical convergence](@ref convergence-tutorial) to compare axial sphere results against
modal values. Canonical comparisons are described by
[Jech et al. (2015)](https://doi.org/10.1121/1.4937607). Details of this package's ring quadrature,
assembly, and defaults are package-derived numerical choices.

## [Coupled fluid regions](@id coupled-fluid-regions)

Several fluid interfaces require a single coupled field. For `bem(surfaces, materials, k)`,
surface `i` separates region `i` from `parents[i]`; region `0` is the unbounded exterior.
The default parent list `[0,1,2,...]` describes successive nesting. A list `[0,1,1]`
instead places two separate inclusions inside region `1`. Normals always point out of the
volume enclosed by each surface, including the surfaces of inner inclusions.

Each region has density contrast `g_r` and wavenumber `k_r=k/h_r`, relative to the
unbounded exterior. Interface unknowns are total pressure `p_i` and the common scaled
derivative `v_i=(∂ₙp)/g_r`. Thus pressure and normal velocity are continuous without
requiring coincident nodes on different interfaces.

Let `B_r` denote the surfaces bounding region `r`, and let `s_rj=+1` when surface `j`
is its outer boundary, or `-1` when it bounds a hole. At interface `i`, the two trace
equations for either adjoining region are

```math
\frac12 p_i+\sum_{j\in B_r}s_{rj}
  \left(D^r_{ij}p_j-g_r S^r_{ij}v_j\right)=b^p_{ri},
```

```math
\frac12 v_i+\sum_{j\in B_r}s_{rj}
  \left(\frac{H^r_{ij}}{g_r}p_j-K^{\prime r}_{ij}v_j\right)=b^v_{ri}.
```

The right-hand sides are the incident pressure and its normal derivative for `r=0`,
and zero for bounded regions. Adding the equations from the two sides gives the coupled
Müller system. Operators between different surfaces use the same regional wavenumber as
the self-interaction terms. Only interfaces bordering the unbounded exterior contribute
directly to the outgoing far field; their traces already contain the internal interactions.
Adding target strengths from separate body solves does not give this field.

With `formulation=:cbie`, the pressure equation is enforced separately on both sides,
giving two equations for the shared pressure and density-scaled derivative. This avoids
the hypersingular operator and can be preferable for low-frequency gas inclusions.
The conventional formulation is not protected against fictitious eigenfrequencies;
extend its frequency range only with refinement and independent reference checks.
`formulation=:muller` is the default.

`solution.data.interfaces` retains complex total pressure and normal derivatives from
both sides. In `diagnostics(solution).interface_residuals[i]`, `pressure_interior` and
`pressure_exterior` are the norms of the separate pressure-equation residuals divided by
`norm(p_i)`. The two `flux_*` fields divide the scaled-derivative residual norms by
`max(norm(v_i), k*norm(p_i)/g_i)`. Machine epsilon bounds denominators away from zero.
These side residuals need not vanish even when the summed linear system is solved accurately.
They use the assembly quadrature, so check mesh and quadrature refinement as well.
For `:cbie`, the pressure residuals instead belong to the solved equations themselves;
they measure algebraic solution accuracy. The unsampled `flux_*` residuals are `nothing`.

Every interface must be closed, connected, outward oriented and disjoint from all other
interfaces. Curved-element separation uses outward-rounded Bernstein bounds; unresolved
separation raises an error. Adaptive numerical winding integrals check the specified
containment after separation. The containment calculation is a numerical topology check,
not an interval certificate. Close interfaces require near-singular integration and can
need substantially finer quadrature than widely separated interfaces.

For concentric spheres with outer radius `a` and inner radius `b`, material list
`[FluidFilled(g_shell,h_shell), FluidFilled(g_core,h_core)]` describes the same fluids as
`Shelled(FluidLayer(g_shell,h_shell), FluidInterior(g_core,h_core), b/a)`.
General meshes specify the two shapes directly, without a common radius ratio. This
formulation describes lossless scalar fluids. Elastic or viscous walls require their own
constitutive equations and cannot be represented by substituting a sound speed here.

## Method of fundamental solutions

MFS approximates the exterior scattered pressure as

```math
p_{\mathrm{scat}}(\boldsymbol x)\approx
\sum_{j=1}^{N} a_j G(\boldsymbol x,\boldsymbol y_j),
```

with fictitious source points inside the scatterer. Collocation enforces the required boundary
condition. Transmission requires appropriate exterior/interior field representations. Increasing
source count alone does not guarantee accuracy: placement and conditioning matter too.

`mfs(body, ...)` uses an axisymmetric representation for straight bodies and a lateral-only
grid for bent cylinders. `mfs(surface, ...)` instead enforces rigid or pressure-release
conditions on a closed full-3D mesh. Supply a coarser `source_mesh` and an independent
`check_mesh` to inspect residuals between collocation points. Its far field is evaluated
directly from the outgoing point-source coefficients.

`offset` is a physical distance, not a dimensionless
fraction unless you multiply it by a body length yourself. Sharp cylinder corners make normal
offset placement problematic. Smooth caps are useful where the intended geometry permits them.

Check both offset and resolution and compare to modal/BEM reference cases. General background is
[Fairweather, Karageorghis, and Martin (2003)](https://doi.org/10.1016/S0955-7997(03)00017-1).
This reference explains the method but does not independently validate the package's defaults.

## Post-processing

The stored boundary fields enter the package's far-field integral. Axisymmetric BEM/MFS accepts
observation polar angle and azimuth. Full BEM accepts a unit direction vector. Bent MFS currently
returns only its supported monostatic query. See [Conventions](@ref conventions) before
interpreting default angles as backscatter.

## Full-3D solve diagnostics

```@example bem_diagnostics
using AcousticScattering

solution = bem(Sphere(0.01), Rigid(), 100.0; method = :full, meshsize = 0.01)
report = diagnostics(solution)
@assert report.converged
@assert report.relative_residual < 1e-3
(iterations = report.iterations, relative_residual = report.relative_residual)
```

GMRES failure still emits a warning and is retained as `report.converged == false`.
The residual is recomputed using the assembled operator, including any compression.
`report.residual_history` contains GMRES's residual estimates, which may differ from the
recomputed residual, particularly with preconditioning. `solver_options` records the tolerance
and iteration controls; mesh size, quadrature order and correction/compression settings are
also retained. `formulation` and `coupling` identify the rigid/pressure-release equation;
`mesh_order` records the geometric order. Check mesh and quadrature convergence separately
from this linear-solve check.

Fluid transmission reports the residual of its complete dense coupled system. It has no
iterative convergence flag or iteration count, so those fields are `nothing`.
`relative_residual` refers to the original equations, while `scaled_relative_residual`
refers to the scaled equations. With `equilibrate=false`, these coincide.
`condition_number` and `scaled_condition_number` are matrix 2-norm condition numbers,
computed by SVD only when the unknown count is at most `condition_limit` (default 512).
Otherwise both are `nothing` and `conditioning=:not_computed`. These numbers depend
on units and scaling; they do not measure scattering accuracy.

## Pressure fields and solver diagnostics

### Pressure away from and near the surface

[`pressure`](@ref pressure-evaluation) evaluates exterior and transmitted fluid fields for spheres and
straight cylinders; full BEM/MFS also accept closed bent and supplied meshes.
Coupled fluid BEM evaluates all fluid regions, including the unbounded exterior.
Axisymmetric BEM integrates the retained Fourier pressure and normal
derivative traces over the revolved meridian. Subtracting a constant-pressure Laplace
double layer removes the narrow double-layer peak; its closed-surface identity supplies
the selected one-sided limit. Adaptive integration intervals follow the target's distance
from each panel and ring. A failed quadrature error check raises `ArgumentError`.

This treatment resolves the integral's near singularity, but the retained densities remain
piecewise constant. At a shared meridian endpoint, their mean selects the angular
bisector limit of the discrete field. Refine the panels and Fourier cutoff, and compare
pressure at the actual observation points: near-surface errors depend on position within a panel.
Full BEM uses density-interpolation quadrature and MFS evaluates its retained source fields.
In full BEM point evaluation, the nearest element is omitted from the base quadrature;
density interpolation supplies its contribution without subtracting unbounded kernel
values as the target approaches a source node.
For either method, refine the geometric and acoustic discretization separately.

At higher frequencies, check quadrature order as well as elements per wavelength.
At a narrow gas resonance, geometric error can shift the response appreciably even
when increasing quadrature order changes little. Compare complex pressure as well as
its magnitude: a small decibel difference can conceal a larger phase error. A converged
linear solve and an accurate far-field amplitude do not establish near-interface accuracy.
Sharp rims require additional mesh grading and quadrature checks. Inspect the solver
residual before comparing fields, and cross-check axisymmetric geometry with graded
BEM/MFS meshes; full-3D refinement alone is insufficient evidence of rim accuracy.

For rigid and pressure-release rims, full BEM supports `correction=(method=:edge,)` with
`formulation=:cbie` and `compression=(method=:none,)`. Pressure-release boundaries use
the total normal pressure derivative as the single-layer unknown. Rigid boundaries use
an indirect single-layer density, determined by imposing the normal-velocity condition.
On triangles adjoining a sharp edge, interpolation factors out the leading wedge
singularity; the rigid density also includes successive fractional corner powers.
Adaptive integration treats both self and neighbouring-element interactions. Pressure
and far-field queries use the same weighted density. These equations retain their
interior-resonance limitation.

```julia
surface = mesh(Cylinder(0.5, 1.0); method=:full, resolution=0.15,
    mesh_order=3, qorder=5)
solution = bem(surface, PressureRelease(), 0.5; incidence_angle=pi/3,
    formulation=:cbie, compression=(method=:none,), correction=(method=:edge,))
p = pressure(solution, [(0.5, 0.3, 0.4), (0.501, 0.3006, 0.4008)]; field=:scattered)
```

For `FluidFilled`, the same edge correction requires `formulation=:cbie` and uses
total pressure and exterior normal flux as its two unknown traces. Material-dependent
corner powers describe their different behaviour at the rim. The constant-density
Laplace identity removes the constant pressure background from the singular part of
the double-layer operator. Interior and exterior pressure queries and far-field
amplitudes use the same corner interpolation.

This quadrature accepts closed conforming linear, quadratic or cubic triangular meshes.
Normal jumps greater than 30 degrees identify sharp edges; a triangle adjoining more than
one such edge requires refinement. Set `rtol`, `atol` and `maxsubdiv` in `correction` to
control integration independently of the linear solve. Its default tolerances are `1e-7`
and `1e-10`, with at most 16,384 adaptive subdivisions. Mesh and quadrature refinement
remain necessary. The rim comparisons cover flat rigid and pressure-release cylinders
at `ka=0.25` and incidence `pi/3`. For the radius-0.5, length-1 cylinder above, the rigid
comparison uses `resolution=0.1` with cubic geometry and quadrature order 5 to meet the
0.1% complex-pressure comparison at all sampled rim points; its refinement check uses
`resolution=0.12`. Use a tight linear tolerance such as `gmres_kwargs=(reltol=1e-11,)`.
These rigid and pressure-release comparisons do not establish accuracy at other
frequencies, wedge angles or material contrasts.

The fluid rim comparison uses the same cylinder and incidence, `FluidFilled(1.2,1.1)`,
and `k=0.5`. Cubic meshes at `resolution=0.25,0.2` and quadrature order 7 meet the
0.1% complex-pressure and 0.01 dB limits at all sampled rim points against axisymmetric
BEM and MFS. The change between these two meshes also meets both limits. Validation
covers this material contrast and frequency, with exterior scattered pressure,
interior/exterior continuity and far-field consistency checked separately.

Bent and supplied surfaces use oriented ray crossings to select the fluid region.
Bernstein bounds exclude elements away from the ray. A curved patch contributes its
corner triangle's crossing only when its boundary can deform to the triangle edges
without crossing the ray; otherwise the patch subdivides. Alternate rays handle tangencies
and shared edges. Unresolved bounds raise an error. This avoids substituting a fixed
triangulation for the curved surface when classifying a nearby point.

In a coupled fluid region, the representation includes every bounding interface at
that region's wavenumber. With normals pointing out of each child, its enclosing
surface contributes the interior representation and its children contribute exterior
representations. The unbounded exterior sums the root interfaces, subtracting incident
pressure and its normal derivative from their total traces to evaluate scattered pressure.
The incident wave is then added for a total-field query. Interface pressure is continuous;
the appropriate side's normal derivative carries the density contrast.

[`pressure`](@ref pressure-evaluation) locates points in the parent/child hierarchy using the curved meshes.
`region=0` selects the unbounded exterior; `region=i` selects the fluid immediately
inside interface `i`, excluding its children. At an interface both adjacent selections
are valid. Without an explicit region, `:total` uses the parent trace and `:interior`
the child trace. Compare these traces under mesh and quadrature refinement; their
agreement measures the evaluated fields, beyond the continuity built into the unknowns.

### Residuals and conditioning

`diagnostics(solution).systems` contains one report per azimuthal system, including its
mode, equation and unknown counts, quadrature controls, and recomputed residuals.
The summary residuals are the maxima over the individual reports. Direct solves have
`converged = iterations = nothing`; a small linear residual does not establish mesh convergence.

For MFS, `n` controls the source-mesh panel budget and `oversampling` multiplies the
collocation budget. Bent cylinders instead use `n_s` and `n_phi` for the source grid and
multiply both collocation dimensions. The default `oversampling=1` gives square systems;
larger integer values use least squares with the same sources.

At flat-cylinder rims, axisymmetric MFS limits each local source offset to half
the panel midpoint's distance from the corner. The requested `offset` is a maximum;
the source rings approach the edge as the graded meridian mesh is refined. Check
both source count and offset. Convergence in source count with a fixed distance
from the edge can conceal a persistent error in the near field.
Normal-derivative quadrature resolves the angular scale set by source-to-boundary
separation and raises `ArgumentError` if its error estimate exceeds the requested tolerance.

Each MFS system also reports:

- `boundary_residual`: absolute and relative residuals at points excluded from the solve.
  Axisymmetric checks use the quarter and three-quarter points of each straight panel,
  so they test the represented boundary field on the discretized surface, not geometric error.
  Bent checks use a doubled grid in both directions on the lateral surface.
- `pressure_residual` and `velocity_residual`: separate transmission checks, normalized by
  their respective incident-field right-hand sides. The velocity equations use normal
  pressure derivatives with density-contrast scaling. The joint residual mixes equation
  units and depends on their scaling; inspect the separate checks as well.
- `condition_number`, `numerical_rank`, `rank_tolerance`: singular-value diagnostics of
  the unscaled collocation matrix. The rank cutoff is the largest singular value times
  machine epsilon times the larger matrix dimension. These are scaling-dependent checks,
  not estimates of physical accuracy.
  `condition_limit=512` limits SVD computation to systems with at most 512 unknowns.
  Larger systems report `conditioning=:not_computed` and `nothing` in these three fields.
  Raise the limit to obtain an SVD for larger systems; zero skips it for every system.
- Source offsets and `source_count`, `collocation_count`, `check_count`. For transmission,
  `source_count` includes both source sets; each boundary point supplies two equations.

Relative residuals use `norm(A*x-b)/norm(b)`; a zero right-hand side gives zero only for
a zero residual and `Inf` otherwise. Higher modes with tiny incident fields can therefore
have large relative residuals even when their absolute errors are small. Independent
boundary checks and singular-value diagnostics require additional assembly and computation.
