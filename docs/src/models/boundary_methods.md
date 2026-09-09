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

Full BEM in `engine/full_bem.jl` uses Gmsh and Inti surface quadrature, density-interpolation
corrections, and optional compressed operators for rigid/soft conditions. The fluid-transmission
path uses a dense block system. The public full-mesh route supports spheres and spheroids.

Conventional boundary integral equations can suffer fictitious interior eigenfrequencies.
Full BEM does not implement Burton–Miller or CHIEF regularization. An optional `chief_points`
augmentation exists for the axial axisymmetric fluid-transmission path. Its presence must not
be generalized to rigid/soft or full-3D paths.

Use [Numerical convergence](@ref convergence-tutorial) to compare axial sphere results against
modal values. Canonical comparisons are described by
[Jech et al. (2015)](https://doi.org/10.1121/1.4937607). Details of this package's ring quadrature,
assembly, and defaults are package-derived numerical choices.

## Method of fundamental solutions

MFS approximates the exterior scattered pressure as

```math
p_{\mathrm{scat}}(\boldsymbol x)\approx
\sum_{j=1}^{N} a_j G(\boldsymbol x,\boldsymbol y_j),
```

with fictitious source points inside the scatterer. Collocation enforces the required boundary
condition. Transmission requires appropriate exterior/interior field representations. Increasing
source count alone does not guarantee accuracy: placement and conditioning matter too.

`engine/mfs.jl` implements the axisymmetric method, and `analytical/bent_cylinder.jl` contains
the genuinely 3D bent-cylinder path. `offset` is a physical distance, not a dimensionless
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
