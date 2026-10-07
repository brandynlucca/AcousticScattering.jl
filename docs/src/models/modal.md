# [Modal series](@id modal-theory)

Sphere and spheroid modal solvers separate variables. Finite-cylinder models add a length approximation. See [Choosing a solver](@ref solver-selection).

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

`Impedance(zeta)` is a locally reacting (Robin) boundary, `∂p/∂n=-ik\,p/\mathrm{zeta}` at `r=a`:

```math
A_\ell^{\mathrm{imp}}=-\frac{j_\ell'(ka)+ij_\ell(ka)/\mathrm{zeta}}{{h_\ell^{(1)}}'(ka)+i h_\ell^{(1)}(ka)/\mathrm{zeta}}.
```

`zeta → ∞` and `zeta → 0` recover `A_ℓ^{rigid}` and `A_ℓ^{soft}` above.

```julia
modal(Sphere(a), Rigid(), k; m_max = 40)
modal(Sphere(a), FluidFilled(g, h), k)
modal(Sphere(a), SolidElastic(g, hl, ht), k)
```

Increase `m_max` to check spherical-degree truncation across the required frequencies and contrasts. Resonances and extreme contrasts can cause ill-conditioning. See [Convergence](@ref convergence-tutorial).

### Non-plane-wave incidence

`incident::IncidentField` generalizes the sum above, replacing the plane wave's degree-independent phase with the field's own spherical-harmonic weight `Qₗ/(2ℓ+1)`:

```math
f(\theta)=\frac{1}{k}\sum_{\ell=0}^{\ell_{\max}}Q_\ell(-i)^{\ell+1}P_\ell(\cos\theta)A_\ell,
\qquad
Q_\ell^{\mathrm{plane}}=(2\ell+1)i^\ell.
```

`SphericalWave(range)` is a point source at `-range * direction`, with `range > a`, and propagation toward the origin along `direction` ([Sapozhnikov and Bailey, 2013](https://doi.org/10.1121/1.4773924), Eq. 10):

```math
Q_\ell^{\mathrm{spherical}}=ik(-1)^\ell(2\ell+1)h_\ell^{(1)}(k\cdot\mathrm{range}).
```

```julia
modal(Sphere(a), Rigid(), k; incident = SphericalWave(range))
```

The boundary coefficients are unchanged. The regular incident expansion converges inside the source range, while the scattered expansion remains valid beyond it. `pressure` adds the exact incident field for exterior total-pressure queries. Incident and total pressure are undefined at the source.

`BesselBeam(angle)` is a zeroth-order non-diffracting beam whose plane-wave components make `angle` (its half-conical angle `β`) with the observation axis:

```math
Q_\ell^{\mathrm{Bessel}}=(2\ell+1)i^\ell P_\ell(\cos\beta).
```

`angle=0` recovers a plane wave. Only zeroth-order, axisymmetric Bessel beams are supported.

### Reusing direction, phase and amplitude

Built-in fields accept normalized Cartesian `direction`, complex `amplitude` and `phase` in radians. The scale is `amplitude * cis(phase)`, giving pressure at the origin for plane and Bessel waves. Spherical waves use $p=A\exp(ikR)/R$, without a $1/(4\pi)$ factor.

```julia
beam = BesselBeam(pi / 6; direction = (1, 2, -1), amplitude = 0.4 + 0.7im, phase = 0.3)
k = 100.0
solution = modal(Sphere(0.01), FluidFilled(1.2, 0.9), k; incident = beam)
pressure(solution, (0.02, 0.003, 0.0))
incident_pressure(beam, k, (0.02, 0.003, 0.0))
incident_gradient(beam, k, (0.02, 0.003, 0.0))
incident_coefficient(beam, 2, k)  # Q_2 / 5 about the beam axis
```

Sphere-modal `angle` is measured from the field axis. Full BEM, closed-surface MFS and volume FEM also accept these fields. Set observation directions explicitly because their defaults follow solver incidence angles. `PlaneWave()` uses each solver's angle convention, with `+x` for standalone queries and sphere modal.

Custom fields provide `incident_pressure` and `incident_gradient`, or `IncidentField(pressure, gradient)` callbacks. Rebuild callbacks when frequency changes. Keep sources outside the target and any resolved exterior FEM domain.

## Spheroid

### Incident fields

Rigid, pressure-release and fluid-filled spheroids accept `incident=field`, including built-in fields and pressure/gradient callbacks. Projection retains both azimuthal sectors and complex phase. Fluid transmission uses the selected full or diagonal coupling.

```julia
beam = BesselBeam(pi / 6; direction = (1, 1, 0), amplitude = 1)
solution = modal(Spheroid(0.02, 0.01), Rigid(), 150.0;
    incident = beam, m_max = 8, n_max = 12,
    incident_n_eta = 48, incident_n_phi = 48,
    scatter_angle = 3pi / 4, scatter_azimuth = pi)
scattering_amplitude(solution)
```

Directions use the x-polar body frame. Set `scatter_angle` and `scatter_azimuth` explicitly when supplying an incident field. These solutions return far-field amplitudes only. Sources must lie outside the spheroid.

Refine `incident_n_eta` and `incident_n_phi` separately from modal orders. Their defaults are `max(32,2n_max+12)` and `max(32,4m_max+4)`. The polar count must exceed `n_max`, and the azimuthal count twice the retained order. Full fluid coupling also requires refinement of `n_quad`.

### Plane-wave series

Separation in prolate/oblate spheroidal coordinates yields angular and radial spheroidal wave functions. Rigid/soft boundaries decouple appropriate modes. A penetrable spheroid generally couples angular degrees because interior and exterior wavenumbers differ.

`FluidFilled(...; coupling = :full)` solves the off-diagonal coupling system. `:diagonal` neglects that coupling and changes the model approximation. Spheroidal wave functions are evaluated with SpheroidalWaves. Increase both `m_max` and `n_max`.

`incidence_angle_sweep(body, boundary, k, angles; method=:modal)` reuses the spheroidal basis for a full-coupling fluid spheroid at fixed frequency. Angles are radians from the symmetry axis. The result contains complex backscatter amplitudes and target strengths.

Higher orders can become unreliable when radial functions are poorly conditioned. [Furusawa (1988)](https://www.jstage.jst.go.jp/article/ast1980/9/1/9_1_13/_article) develops the prolate spheroidal fish-target models behind this treatment.

## Finite and bent cylinders

The cylinder model combines a circular-cylinder coefficient sum with a finite
axial coherence factor:

```math
f_{\mathrm{bs}}\propto L\,
\frac{\sin(kL\cos\beta)}{kL\cos\beta}
\sum_m B_m(k a\sin\beta).
```

This lateral finite-length approximation omits end-cap scattering ([Stanton, 1988](https://doi.org/10.1121/1.396184)). Use it near broadside and check short cylinders or oblique incidence against a closed-surface solver. [Jech et al. (2015)](https://doi.org/10.1121/1.4937607) limit its use as a reference to roughly 15 to 20 degrees from broadside.

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

Fluid shells match pressure and normal velocity ([Jech et al., 2015](https://doi.org/10.1121/1.4937607), [Venas and Jenserud, 2022](https://doi.org/10.1016/j.jsv.2022.117263)). `LayeredMaterial` composes spherical fluid/elastic stacks around a fluid or vacuum core. Elastic shells generalize the [Goodman and Stern (1962)](https://doi.org/10.1121/1.1928120) formulation, whose original form assumes identical interior and exterior fluids.

```julia
modal(Sphere(a), Shelled(FluidLayer(g, h), FluidInterior(g_i, h_i), b_a), k)
modal(Sphere(a), Shelled(LayeredMaterial(FluidLayer(g1, h1),
    FluidLayer(g2, h2), interface_ratio), FluidInterior(g_i, h_i), b_a), k)
modal(Sphere(a), Shelled(ElasticLayer(g, cl, ct), FluidInterior(g_i, h_i), b_a), k)
modal(Sphere(a), Shelled(LayeredMaterial(FluidLayer(g1, h1),
    ElasticLayer(g2, cl, ct), interface_ratio), FluidInterior(g_i, h_i), b_a), k)
```

`ElasticLayer` defaults to `interior_coupling=:generalized`, using the actual cavity fluid ([Stanton, 1990](https://doi.org/10.1121/1.400321)). `:identical_fluid` retains the original Goodman-Stern restriction. Mixed spherical stacks require `:generalized`, with pressure queries only in fluid regions. Nonspherical mixed stacks are unsupported.

The [viscous-elastic model](https://doi.org/10.1121/1.423076) has a viscous outer layer, elastic wall and fluid core. Only the low-frequency monopole is supported.