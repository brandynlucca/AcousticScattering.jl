# [Kirchhoff physical optics](@id kirchhoff-theory)

For the distinction between physical-edge diffraction, smooth-surface creeping
waves and elastic guided waves, see [Surface waves and resonances](@ref wave-diagnostics).

Kirchhoff replaces the true boundary field with a locally reflected incident field on the illuminated surface and suppresses the shadowed contribution. Numerical integration can be accurate at any chosen frequency while the physical-optics approximation remains inaccurate outside its validity regime. Do not equate an “exact integral” with exact wave scattering.

## Reflection and amplitude

The normal-incidence pressure reflection factor used for a fluid interface is

```math
R_c = \frac{gh-1}{gh+1},
\qquad g=\rho_{\mathrm{int}}/\rho_{\mathrm{ext}},
\quad h=c_{\mathrm{int}}/c_{\mathrm{ext}}.
```

Rigid and pressure-release boundaries use `+1` and `-1`. A fluid-shell sphere uses a frequency- and thickness-dependent two-interface reflection approximation. There is no general implemented elastic-material Kirchhoff reflection law. This is the physical-optics approximation used for swimbladdered fish targets ([Foote, 1985](https://doi.org/10.1121/1.392438)). 

For a sphere, the finite-frequency expression used by the solver is

```math
f_s = R_c a\left[\frac{1}{2}e^{-2ika}
-\frac{i(e^{-2ika}-1)}{4ka}\right].
```

```julia
kirchhoff(Sphere(a), Rigid(), k)
```

Its leading magnitude tends to `abs(R_c) * a / 2` at high frequency. The correction term is part of the physical-optics surface integral, not a replacement for the exact modal series at small `ka`.

The time convention is $\exp(-i\omega t)$, with incident pressure
$\exp(ik\mathbf d\cdot\mathbf x)$ and outgoing scattered pressure
$f\exp(ikr)/r$. For outward normal $\mathbf n$, the source-facing surface
has $\mathbf n\cdot\mathbf d<0$. The monostatic physical-optics integral is

$$
f=\frac{ikR_c}{2\pi}\int_{\mathbf n\cdot\mathbf d<0}
(\mathbf n\cdot\mathbf d)\exp(2ik\mathbf d\cdot\mathbf x)\,dS.
$$

Translation by $\mathbf t$ multiplies the amplitude by
$\exp(2ik\mathbf d\cdot\mathbf t)$. Target-strength agreement alone does
not establish phase accuracy; coherent combinations require complex-amplitude
checks as well.

## Geometry-specific formulations

Straight cylinders use a Bessel-series lateral contribution plus an illuminated end-cap contribution. This differs from finite-cylinder modal calculations, which omit the caps. Spheroids use adaptive integration over polar and azimuthal coordinates, with illumination intervals determined geometrically. Bent cylinders use nested integration over the curved
surface.

The quadrature tolerance controls integration error, not physical model error. The [geometry tutorial](@ref geometry-tutorial) runs a bent-cylinder example. The [frequency tutorial](@ref first-sweep) provides the sphere workflow to adapt for comparisons.

`kirchhoff(surface::Mesh, boundary, k)` evaluates the same backscatter surface integral directly over a supplied full-3D mesh (`mesh(path)`, `mesh(generate)`, `mesh(nodes, triangles)`, or `mesh(body; method=:full)`), retaining its actual shape, normals and endcaps rather than a closed-form reduction. It reproduces `kirchhoff(Sphere/Spheroid/Cylinder, ...)` to mesh-discretization accuracy when fed a matching generated mesh. Illumination follows the local outward normal against the incident direction, not a hidden-surface check, so a concave surface (e.g. a real swimbladder) can misclassify a self-shadowed region as illuminated. Check concave geometry with a full-wave solver.

```julia
surface = mesh(path; units = :mm)
kirchhoff(surface, Rigid(), k; incidence_angle = pi / 2)
```

## Validity envelope

Against the exact modal series, a rigid or pressure-release sphere agrees to within `0.1` dB for `ka ≳ 12` and disagrees by `10` dB or more for `ka ≲ 0.5`; the error is not monotonic in between, since Kirchhoff smooths over the modal series' own peaks and nulls. A rigid spheroid at broadside incidence agrees to within `1` dB for `kb ≳ 5` (`b` the minor semi-axis), largely independent of aspect ratio from `1.5` to `8`. At fixed `kb = 10`, error grows from `0.3` dB at broadside to `1.4` dB at end-on. [Jech et al. (2015)](https://doi.org/10.1121/1.4937607) report a similar pattern (agreement near broadside/high `ka`, larger deviations off-broadside) across independently implemented models. Check any case outside these ranges against `modal` or a converged numerical method.

These sampled trends are not uniform error bounds. For an orientation distribution, average scattering powers before converting to dB. Averaging target strengths gives a different result and can conceal large errors at individual angles. Concave self-shadowing and diffraction remain outside the local-normal Kirchhoff approximation.
