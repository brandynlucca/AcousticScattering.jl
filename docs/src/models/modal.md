# [Modal series](@id modal-theory)

Modal solvers use separation of variables or a finite-length reduction, then sum scattering coefficients. The sphere and spheroid models use separation, while the finite-cylinder model includes a length approximation. See [Choosing a solver](@ref solver-selection) and the [materials examples](@ref materials-tutorial).

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

Fluid coefficients match pressure and density-weighted normal derivatives ([Anderson, 1950](https://doi.org/10.1121/1.1906621)). Solid-elastic coefficients also match displacement and stress ([Hickling, 1962](https://doi.org/10.1121/1.1909055)).

```julia
modal(Sphere(a), Rigid(), k; m_max = 40)
modal(Sphere(a), FluidFilled(g, h), k)
modal(Sphere(a), SolidElastic(g, hl, ht), k)
```

Increase `m_max` to test truncation. The keyword uses “m” although the sum here uses spherical degree. Highly resonant or extreme-contrast cases can be ill-conditioned. The [convergence tutorial](@ref convergence-tutorial) compares numerical methods with this sphere reference. Check convergence over the frequencies and material contrasts of interest.

## Spheroid

Separation in prolate/oblate spheroidal coordinates yields angular and radial spheroidal wave functions. Rigid/soft boundaries decouple appropriate modes. A penetrable spheroid generally couples angular degrees because interior and exterior wavenumbers differ.

`FluidFilled(...; coupling = :full)` solves the off-diagonal coupling system. `:diagonal` neglects that coupling and changes the model approximation. Spheroidal wave functions are evaluated with SpheroidalWaves. Increase both `m_max` and `n_max`.
Higher orders can become unreliable when radial functions are poorly conditioned. [Furusawa (1988)](https://www.jstage.jst.go.jp/article/ast1980/9/1/9_1_13/_article) develops the prolate spheroidal fish-target models behind this treatment.

## Finite and bent cylinders

The cylinder model combines a circular-cylinder coefficient sum with a finite
axial coherence factor:

```math
f_{\mathrm{bs}}\propto L\,
\frac{\sin(kL\cos\beta)}{kL\cos\beta}
\sum_m B_m(k a\sin\beta).
```

This describes the lateral finite-length approximation and omits end-cap scattering ([Stanton, 1988](https://doi.org/10.1121/1.396184)). It is not an exact solution for a closed finite cylinder, particularly near end-on incidence.

For curvature radius `rho_c`, the bent-cylinder model ([Stanton, 1989](https://doi.org/10.1121/1.398193)) multiplies the straight amplitude by `L_effective / L`, with

```math
L_{\mathrm{effective}} =
\int_{-L/2}^{L/2}
\exp\left(i\frac{8k z_{\max}}{L^2}x^2\right)\,dx,
\quad
z_{\max}=\rho_c\left(1-\cos\frac{L}{2\rho_c}\right).
```

This correction applies near broadside and does not account for all effects of curvature.

```julia
modal(Cylinder(a, L), Rigid(), k) # straight
modal(Cylinder(a, L; radius_curvature = rho_c), Rigid(), k) # bent, near broadside
```

## Shells and viscosity

Fluid-shell coefficients match pressure and normal velocity across two interfaces ([Jech et al., 2015](https://doi.org/10.1121/1.4937607)). Elastic-shell coefficients use the Goodman and Stern boundary-matching determinant ([Goodman and Stern,
1962](https://doi.org/10.1121/1.1928120)), which assumes identical interior and exterior fluids.

```julia
modal(Sphere(a), Shelled(FluidLayer(g, h), FluidInterior(g_i, h_i), b_a), k)
modal(Sphere(a), Shelled(ElasticLayer(g, cl, ct), FluidInterior(g_i, h_i), b_a), k)
```

`ElasticLayer` defaults to `interior_coupling = :generalized`, using the actual cavity fluid ([Stanton, 1990](https://doi.org/10.1121/1.400321)). `:identical_fluid` instead reproduces the original Goodman-Stern inner-boundary terms, ignoring differing cavity contrasts. Choose it only to reproduce that restricted formulation.

The viscous-elastic model ([Feuillade and Nero, 1998](https://doi.org/10.1121/1.423076)) represents a viscous outer layer, elastic wall, and fluid core. Viscosity introduces frequency-dependent complex wavenumbers. Only its monopole is supported. Use this model only in its low-frequency, monopole regime.
