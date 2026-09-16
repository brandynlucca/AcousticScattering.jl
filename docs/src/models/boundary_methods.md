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

Kernel reciprocity gives `K'(x,y)=D(y,x)` without complex conjugation. With positive
quadrature-weight matrix `W`, its discrete reciprocal form is `W⁻¹ Dᵀ W`.
This uses the same double-layer quadrature for both operators. Sharp rims still require
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
its individual interfaces and their combined node set. Pressure and derivative
corrections use the same selection throughout that region.

For low-frequency self-interactions, the normal-derivative operators are obtained
from the Calderón identities

```math
S K' = D S,\qquad S H = D^2-\tfrac14 I.
```

These relate operators at the **same** wavenumber; see
[van 't Wout et al., equations (28) and (38)](https://arxiv.org/abs/2104.04618).
With density interpolation, fluid BEM uses this reconstruction in the regular-wave
range defined above. Both size measures are unchanged by rigid rotations and translations.
It factors the single-layer matrix and avoids direct hypersingular quadrature for those
self-interactions. Other interactions retain direct derivative quadrature.
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

## Axisymmetric BEM and MFS diagnostics

`diagnostics(solution).systems` contains one report per azimuthal system, including its
mode, equation and unknown counts, quadrature controls, and recomputed residuals.
The summary residuals are the maxima over the individual reports. Direct solves have
`converged = iterations = nothing`; a small linear residual does not establish mesh convergence.

For MFS, `n` controls the source-mesh panel budget and `oversampling` multiplies the
collocation budget. Bent cylinders instead use `n_s` and `n_phi` for the source grid and
multiply both collocation dimensions. The default `oversampling=1` gives square systems;
larger integer values use least squares with the same sources.

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
