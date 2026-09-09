# [Modal series](@id modal-theory)

Modal solvers use separation of variables or a finite-length reduction, then sum scattering
coefficients. The sphere and spheroid models use separation, while the finite-cylinder model
includes a length approximation.
See [Choosing a solver](@ref solver-selection) and the [materials examples](@ref materials-tutorial).

## Sphere

For radius `a` and scattering angle `theta`, the implemented amplitude is

```math
f_s(\theta)=-\frac{i}{k}\sum_{\ell=0}^{\ell_{\max}}
(2\ell+1)P_\ell(\cos\theta)A_\ell.
```

Rigid and pressure-release coefficients are respectively

```math
A_\ell^{\mathrm{rigid}}=-\frac{j_\ell'(ka)}{{h_\ell^{(1)}}'(ka)},
\qquad
A_\ell^{\mathrm{soft}}=-\frac{j_\ell(ka)}{h_\ell^{(1)}(ka)}.
```

Fluid coefficients match pressure and density-weighted normal derivatives. Elastic coefficients
also match displacement and stress. The implementation is in `analytical/sphere_modal.jl`.
Increase `m_max` to test truncation. The keyword uses “m” although the sum here uses
spherical degree. Highly resonant or extreme-contrast cases can be ill-conditioned.

The [convergence tutorial](@ref convergence-tutorial) compares numerical methods with this
sphere reference. Historical Jech comparisons and limiting-case tests are in `test/runtests.jl`.
Agreement at those test points is not a uniform error bound.

## Spheroid

Separation in prolate/oblate spheroidal coordinates yields angular and radial spheroidal wave
functions. Rigid/soft boundaries decouple appropriate modes. A penetrable spheroid generally
couples angular degrees because interior and exterior wavenumbers differ.

`FluidFilled(...; coupling = :full)` solves the off-diagonal coupling system.
`:diagonal` neglects that coupling and changes the model approximation. The implementation in
`analytical/spheroid_modal.jl` uses SpheroidalWaves. Increase both `m_max` and `n_max`.
Higher orders can become unreliable when radial functions are poorly conditioned.

[Furusawa (1988)](https://www.jstage.jst.go.jp/article/ast1980/9/1/9_1_13/_article)
develops the spheroidal fish-target models behind this treatment. The oblate implementation
and numerical choices must also be checked against the package's current tests.

## Finite and bent cylinders

`analytical/cylinder_modal.jl` combines a circular-cylinder coefficient sum with a finite
axial coherence factor:

```math
f_{\mathrm{bs}}\propto L\,
\frac{\sin(kL\cos\beta)}{kL\cos\beta}
\sum_m B_m(k a\sin\beta).
```

This describes the lateral finite-length approximation and omits end-cap scattering.
It should not be advertised as the exact solution of a closed finite cylinder, particularly
near end-on incidence. Elastic cylinder coefficients live in `cylinder_elastic_modal.jl`.

For curvature radius `rho_c`, `analytical/bent_cylinder.jl` multiplies the straight amplitude
by `L_effective / L`, with

```math
L_{\mathrm{effective}} =
\int_{-L/2}^{L/2}
\exp\left(i\frac{8k z_{\max}}{L^2}x^2\right)\,dx,
\quad
z_{\max}=\rho_c\left(1-\cos\frac{L}{2\rho_c}\right).
```

This correction applies near broadside and does not account for all effects of curvature.
Straight-limit and cross-method tests provide regression evidence.

## Shells and viscosity

Fluid-shell coefficients match two interfaces. Elastic-shell coefficients use coupled
displacement/traction determinants. `ElasticLayer` defaults to `interior_coupling = :generalized`,
using the actual cavity fluid. `:identical_fluid` instead uses the exterior medium in the
original inner-boundary terms, ignoring differing cavity contrasts. Choose it only to reproduce
that restricted formulation. See the Goodman–Stern reference in [References](@ref references).

The viscous-elastic implementation in `analytical/vesm.jl` represents a viscous outer layer,
elastic wall, and fluid core. Viscosity introduces frequency-dependent complex wavenumbers.
Only its monopole is supported. Higher modes have unresolved physical-validation problems.
Use this model only in its low-frequency, monopole regime.
