# [Transition-matrix solutions](@id tmatrix-theory)

`tmatrix` solves elastic spheroid and shell scattering in spheroidal coordinates ([Hackman, 1984](https://doi.org/10.1121/1.390297)).

## Field expansions

In prolate spheroidal coordinates ``(\xi,\eta,\phi)`` with focal half-distance ``f`` and size parameter ``c=kf``, the exterior pressure is

```math
p=\sum_{m,n}\left[a_{mn}R_{mn}^{(1)}(c,\xi)+f_{mn}R_{mn}^{(3)}(c,\xi)\right]S_{mn}(c,\eta)\,e^{im\phi},
```

with ``R_{mn}^{(1)}`` the regular radial spheroidal wave function, ``R_{mn}^{(3)}`` the outgoing one, and ``S_{mn}`` the angular one. `m_max` and `n_max` truncate this sum.

## Elastic displacement

Solid displacement satisfies the [Navier equation](@ref fem-theory). Its expansion uses scalar potentials $R_{mn}S_{mn}e^{im\phi}$, one longitudinal family at $k_L$ and two transverse families at $k_T$:

```math
\boldsymbol u=\frac{1}{k_L}\nabla\chi_L
+\frac{1}{\sqrt\lambda}\nabla\chi_T\times\boldsymbol r
+\frac{1}{k_T}\nabla\times\left(\nabla\chi_T\times\boldsymbol r\right),
```

where ``\lambda`` is ``\chi_T``'s own separation constant.

## Transition matrix

Interface matching uses pressure, normal displacement and zero shear traction, with Betti's identity:

```math
\int_S\left(\boldsymbol u\cdot\boldsymbol t(\boldsymbol v)-\boldsymbol v\cdot\boldsymbol t(\boldsymbol u)\right)dS=0.
```

Coupling between incident and test degrees makes $f=Ta$ dense.

## Solid spheroid

Single elastic spheroids and shells accept `incident=field`, including Bessel beams and pressure/gradient callbacks, in the x-polar body frame.

Refine `incident_n_eta` and `incident_n_phi` separately from modal orders. Observation angles `scatter_angle` and `scatter_azimuth` are fixed at solve time. Mixed-layer methods accept plane waves only.

```julia
tmatrix(Spheroid(a, b), SolidElastic(g, hl, ht), k; incidence_angle = pi / 3)
```

The spherical limit recovers elastic-sphere scattering. Large stiffness and density approach the rigid limit.

## Elastic shell

`Shelled(ElasticLayer(...), FluidInterior(...) or VacuumInterior(), radius_ratio)` is a shell with a confocal inner surface, whose equatorial semi-axis is `radius_ratio` times the outer one. An oblate shell needs a `radius_ratio` above the focal ratio so that this surface exists.

Thin or elongated shells need larger `n_max`. Reducing both modal orders by two checks convergence and warns if amplitude changes exceed 1%.

## Mixed confocal layers

Prolate mixed stacks combine `FluidLayer` and `ElasticLayer` around a fluid or vacuum core. Interfaces must be confocal, contrasts relative to the exterior fluid, and elastic coupling `:generalized`.

```julia
body = Spheroid(1.2, 1.0)
layers = Shelled(LayeredMaterial(FluidLayer(1.2, 1.1),
    ElasticLayer(2.7, 4.2, 2.1), 0.8), FluidInterior(0.8, 0.9), 0.55)
solution = tmatrix(body, layers, 1.2)
```

Limits are $a/b \leq 1.25$, $ka \leq 1.5$, `m_max <= 4` and `4 <= n_max <= 8`. The default compares `n_max=8` with 6 and rejects amplitude changes above 3%. Use refined volume FEM outside this range.

### Far-field-generated transition

`method=:farfield` builds a reusable angular transition from volume FEM for mixed prolate stacks with a fluid core ([Ganesh and Hawkins, 2022](https://doi.org/10.1121/10.0009679)).

```julia
body = Spheroid(1.5, 1.0)
layers = Shelled(LayeredMaterial(FluidLayer(1.2, 1.1),
    ElasticLayer(2.7, 4.2, 2.1), 0.8), FluidInterior(0.8, 0.9), 0.55)
transition = tmatrix(body, layers, 1.2; method = :farfield)
f = scattering_amplitude(transition;
    incidence_angle = pi / 2, angle = pi / 2, azimuth = 0.0)
diagnostics(transition).holdout_error
```

Defaults are `polar_order=6` and `m_max=4`. An independent incidence check rejects angular reconstruction errors above 1%. The validated range is $a/b \leq 1.5$ and $ka \leq 1.8$.

Refine angular orders together, fluid mesh size with `points_per_wavelength` or `h`, interfaces with `h_body`, and exterior closure with `domain_radius` and `dtn_order`. Thin layers need explicit `h_body` refinement. Check resonances separately.

Reuse the transition with `scattering_amplitude` or `target_strength` without further FEM solves. Defaults use the original directions. Changing incidence alone selects backscatter. Projected and single-shell solutions retain fixed observation directions.

## Sweeps

`incidence_angle_sweep(body, boundary, k, angles)` computes each azimuthal order once and reuses it for every angle. Pass `method = :volume` to sweep the volume FEM instead.
