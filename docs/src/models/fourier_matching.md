# [Fourier matching](@id fourier-matching-theory)

Fourier matching conformally maps an irregular axisymmetric body's meridian profile to a coordinate system in which the mapped surface is exactly circular, then matches boundary conditions there using spherical wave functions. It targets smooth bodies of revolution between the canonical (sphere/spheroid) and general numerical (BEM/MFS/FEM) solvers. See [Reeder and Stanton (2004)](https://doi.org/10.1121/1.1648681), extending [DiPerna and Stanton (1994)](https://doi.org/10.1121/1.411243)'s two-dimensional method.

```julia
fourier(Irregular(profile), boundary, k) # Rigid, PressureRelease or FluidFilled
fourier(Sphere(a), boundary, k) # converts to an equivalent Irregular
incidence_angle_sweep(body::Irregular, boundary, k, angles)
```

`fourier` returns an `FMSolution`, the same calling convention as `modal`, `bem` and `mfs`. Converting a `Sphere`/`Spheroid` to `Irregular` works, but `modal` is exact and cheaper for those bodies. `incidence_angle_sweep` builds the conformal mapping and boundary-matching transition operator once and reuses it across all `angles`, since neither depends on incidence angle.

## Conformal mapping

For a meridian profile ``R(\theta)`` (radial distance from the origin, ``\theta`` the polar angle), expand it in a Fourier series and solve for an angle-correction series ``\theta(w)`` such that the mapped coordinate system becomes exactly circular at the body surface

```math
R(\theta) = a + \sum_{n\ge1} r_n^c\cos(n\theta),
\qquad
\theta(w) = w + \sum_{l\ge1}\left[\delta_l^c\cos(lw) + \delta_l^s\sin(lw)\right].
```

Both poles lie on the axis, so only cosine terms appear. Odd cosine harmonics still allow fore-aft asymmetry. `Irregular` fits the mirrored profile and rejects nonzero sine coefficients.

The ``\delta`` coefficients solve a nonlinear system (`NLsolve.jl`, forward-mode automatic differentiation), using a continuation homotopy for bodies far from circular. The mapping is rejected (`is_admissible`) if its Jacobian vanishes anywhere outside the body.

## Pressure-release boundary matching

The scattered field is expanded in spherical Bessel/Hankel and associated Legendre functions. For each azimuthal order ``m``, matching the total pressure to zero on the mapped surface gives a dense linear system

```math
\sum_n a_{nm} R_n^m + \sum_n b_{nm} Q_n^m = 0,
```

solved for the scattered coefficients ``b_{nm}``, where ``a_{nm}`` are the incident plane-wave coefficients and ``R_n^m``, ``Q_n^m`` are integrals of the mapped surface against spherical Bessel and Hankel functions. Incidence azimuth is fixed at ``0``, and general azimuth is recovered by shifting the observation azimuth instead, matching the rest of this package's axisymmetric solvers.

## Rigid boundary matching

The rigid condition instead matches the normal derivative of the total pressure to zero on the mapped surface, using the coordinate system's normal derivative

```math
\hat{n}\cdot\nabla P(u,w,v) = \frac{1}{\sqrt{f_w^2+f_u^2}}\frac{\partial P}{\partial u}.
```

Differentiating the modal series w.r.t. ``u`` through both the radial argument ``kr(u,w)`` and the Legendre argument ``g(u,w)/r(u,w)`` gives the same linear-system form as the soft case, with ``R_n'^m``, ``Q_n'^m`` in place of ``R_n^m``, ``Q_n^m``.

## Fluid boundary matching

For a penetrable interior with the given density and sound-speed contrast (`FluidFilled`'s own convention), pressure and velocity are both matched across the mapped surface, introducing an interior field expansion with wavenumber ``k_1 = k/h`` (`soundspeed_contrast` is ``h``) and coefficients ``l_{nm}``, regular at the origin. Eliminating ``l_{nm}`` from the two matched linear systems gives ``b_{nm}`` in terms of the exterior matrices above and a third pair of interior matrices ``S_n^m``, ``S_n'^m``. Velocity continuity is weighted by `density_contrast` alone, since the sound-speed part of the impedance ratio is already carried by the ``k``/``k_1`` factors in the normal-derivative matrices.

## Numerical conditioning

`R_n^m`, `Q_n^m` and their primed and interior counterparts span many orders of magnitude across ``n`` from spherical Hankel-function growth alone, not from genuine ill-conditioning. Each per-``m`` linear solve uses a row/column-equilibrated truncated-SVD pseudoinverse (`pinv` on the equilibrated matrix) rather than a plain solve, so truncation compares singular values on a properly-scaled basis instead of against raw Hankel-function magnitude.

## Truncation guard

Fourier matching does not converge monotonically in `n_max` for elongated bodies. Each solve is repeated with `n_max` and `m_max` reduced by 2, and `diagnostics(sol).convergence` reports the largest relative change in the far-field amplitude. A warning is emitted above `1e-2`.

## Accuracy envelope

Worst complex-amplitude error against axisymmetric BEM over backscatter, forward and off-plane directions, at the default settings of `fourier(Spheroid(aspect, 1), ...)`. Prolate spheroids at incidence ``\pi/3``, fluid contrasts `1.05` and `1.02`. `!` marks a solve where the truncation guard warns.

| aspect | ``ka`` | pressure-release | rigid | fluid |
|:--|:--|:--|:--|:--|
| 1.5 | 1 | `3e-5` | `4e-5` | `1e-4` |
| 2 | 1 | `3e-5` | `5e-5` | `2e-4` |
| 3 | 1 | `2e-5` | `8e-4` | `3e-4` |
| 5 | 1 | `4e-5` | `3e-2` | `8e-3` |
| 1.5 | 3 | `1e-4` | `2e-4` | `4e-3` |
| 2 | 3 | `2e-4` | `4e-4` | `4e-4` |
| 3 | 3 | `7e-5` | `5e-4` | `2e-4` |
| 5 | 3 | `2e-4` | `3e-2` `!` | `5e-3` |

At 5:1 the rigid boundary reaches only a few percent and the fluid boundary is marginal. Use `bem` for rigid bodies beyond 4:1.

The usable `n_max` is limited by `mapping_order`. Beyond about ``\mathrm{mapping\_order}/(2\,\mathrm{aspect})+2`` the error grows rapidly, so elongated bodies (aspect above 2.5) default to `mapping_order = 20*aspect` (at most 96) and `n_max = m_max` set by that bound. The truncation guard can pass a result with an error of up to a few percent for a rigid 5:1 body. The quadrature tolerance `rtol` and node cap `maxevals` do not change the error at these settings.

## Validation

Reconstruction and mapping convergence are checked against a sphere (exact) and prolate spheroids up to 10:1 aspect ratio (against the exact ellipse equation). All three boundary conditions are checked against the exact sphere modal solution and, for prolate spheroids at oblique incidence up to 5:1 aspect ratio, against independent axisymmetric BEM, at backscatter,
forward scatter and off-plane bistatic angles. The fluid boundary is checked at both weak (density and sound-speed contrasts near ``1``) and gas (strong) contrast. A fore-aft asymmetric noncanonical body (``R(\theta)=1+0.12\cos 3\theta+0.05\cos 2\theta``) agrees with axisymmetric BEM on the mapped surface within `3e-4` in complex amplitude for all three boundaries, including gas contrast across its resonance. [Reeder et al. (2004)](https://doi.org/10.1121/1.1648318) validate the method against measured fish morphology.
