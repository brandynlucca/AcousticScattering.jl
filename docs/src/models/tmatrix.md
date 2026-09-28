# [Transition-matrix solutions](@id tmatrix-theory)

`tmatrix` solves scattering by elastic prolate and oblate spheroids and elastic shells in spheroidal coordinates ([Hackman, 1984](https://doi.org/10.1121/1.390297)). It returns a `TMatrixSolution`.

## Field expansions

In prolate spheroidal coordinates ``(\xi,\eta,\phi)`` with focal half-distance ``f`` and size parameter ``c=kf``, the exterior pressure is

```math
p=\sum_{m,n}\left[a_{mn}R_{mn}^{(1)}(c,\xi)+f_{mn}R_{mn}^{(3)}(c,\xi)\right]S_{mn}(c,\eta)\,e^{im\phi},
```

with ``R_{mn}^{(1)}`` the regular radial spheroidal wave function, ``R_{mn}^{(3)}`` the outgoing one, and ``S_{mn}`` the angular one. `m_max` and `n_max` truncate this sum.

## Elastic displacement

Solid displacement satisfies the Navier equation of [FEM and shell coupling](@ref fem-theory). It is expanded in vector spheroidal wave functions built from the same ``R_{mn}S_{mn}e^{im\phi}`` scalar potentials, one longitudinal family at the solid's compressional wavenumber ``k_L`` and two transverse families at its shear wavenumber ``k_T``:

```math
\boldsymbol u=\frac{1}{k_L}\nabla\chi_L
+\frac{1}{\sqrt\lambda}\nabla\chi_T\times\boldsymbol r
+\frac{1}{k_T}\nabla\times\left(\nabla\chi_T\times\boldsymbol r\right),
```

where ``\lambda`` is ``\chi_T``'s own separation constant.

## Transition matrix

Pressure, normal displacement and zero shear traction are matched at the interface, tested against Betti's reciprocal identity for two elastodynamic states of the same frequency:

```math
\int_S\left(\boldsymbol u\cdot\boldsymbol t(\boldsymbol v)-\boldsymbol v\cdot\boldsymbol t(\boldsymbol u)\right)dS=0.
```

Every incident degree pairs with every test degree through this surface integral, so the resulting matrix ``f=Ta`` is dense, unlike the diagonal series of [`modal`](@ref modal-theory).

## Solid spheroid

```julia
tmatrix(Spheroid(a, b), SolidElastic(g, hl, ht), k; incidence_angle = pi / 3)
```

The solution reduces to the elastic sphere as the aspect ratio approaches one and to the rigid spheroid as the solid becomes stiff and dense. It agrees with a full-wave isogeometric reference and with the volume FEM of [FEM and shell coupling](@ref fem-theory).

## Elastic shell

`Shelled(ElasticLayer(...), FluidInterior(...) or VacuumInterior(), radius_ratio)` is a shell with a confocal inner surface, whose equatorial semi-axis is `radius_ratio` times the outer one. An oblate shell needs a `radius_ratio` above the focal ratio so that this surface exists.

Elongated and thin shells converge slowly in `n_max`. Each shell solve is repeated with `m_max` and `n_max` reduced by 2 and warns when the amplitude changes by more than 1%. Use `n_max` near 30 for a 3:1 shell with `radius_ratio = 0.8` and check against `bem` beyond that.

## Sweeps

`incidence_angle_sweep(body, boundary, k, angles)` computes each azimuthal order once and reuses it for every angle. Pass `method = :volume` to sweep the volume FEM instead.
