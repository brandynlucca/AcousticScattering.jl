# [Kirchhoff physical optics](@id kirchhoff-theory)

Kirchhoff replaces the true boundary field with a locally reflected incident field on the
illuminated surface and suppresses the shadowed contribution. Numerical integration can be
accurate at any chosen frequency while the physical-optics approximation remains inaccurate
outside its validity regime. Do not equate an “exact integral” with exact wave scattering.

## Reflection and amplitude

The normal-incidence pressure reflection factor used for a fluid interface is

```math
R_c = \frac{gh-1}{gh+1},
\qquad g=\rho_{\mathrm{int}}/\rho_{\mathrm{ext}},
\quad h=c_{\mathrm{int}}/c_{\mathrm{ext}}.
```

Rigid and pressure-release boundaries use `+1` and `-1`. A fluid-shell sphere uses a
frequency- and thickness-dependent two-interface reflection approximation. There is no
general implemented elastic-material Kirchhoff reflection law.

For a sphere, the finite-frequency expression used by the solver is

```math
f_s = R_c a\left[-\frac{i}{2}e^{2ika}
+\frac{e^{2ika}-1}{4ka}\right].
```

Its leading magnitude tends to `abs(R_c) * a / 2` at high frequency. The correction term is
part of the physical-optics surface integral, not a replacement for the exact modal series
at small `ka`.

## Geometry-specific formulations

Straight cylinders use a Bessel-series lateral contribution plus an illuminated end-cap
contribution. This differs from finite-cylinder modal calculations, which omit the caps.
Spheroids use adaptive integration over polar and azimuthal coordinates, with illumination
intervals determined geometrically. Bent cylinders use nested integration over the curved
surface.

The quadrature tolerance controls integration error, not physical model error. Check against
modal or converged numerical methods over the frequency and orientation range of interest.
The [geometry tutorial](@ref geometry-tutorial) runs a bent-cylinder example. The
[frequency tutorial](@ref first-sweep) provides the sphere workflow to adapt for comparisons.

[Jech et al. (2015)](https://doi.org/10.1121/1.4937607) provides canonical cross-model benchmarks.
