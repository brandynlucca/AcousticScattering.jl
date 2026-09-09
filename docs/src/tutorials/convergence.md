# [Numerical convergence](@id convergence-tutorial)

Use a modal sphere as a reference for FEM and BEM. These small axial examples run in the docs
build and avoid ambiguity about the observation direction.

## Radial FEM

```@example convergence
using AcousticScattering

body = Sphere(0.01)
boundary = Rigid()
wavenumber = 2pi * 12000.0 / 1477.4
reference = target_strength(modal(body, boundary, wavenumber))
element_counts = [40, 80]
fem_strengths = [target_strength(fem(body, boundary, wavenumber;
    n_elements = count, m_max = 12)) for count in element_counts]
fem_errors = abs.(fem_strengths .- reference)
@assert all(isfinite, fem_strengths)
@assert last(fem_errors) < 0.05
(elements = element_counts, error_db = fem_errors)
```

Element count and modal truncation are separate controls. Fluid-filled radial FEM exposes
`n_elements_int` and `n_elements_ext` instead of a single `n_elements`.

## Axisymmetric BEM

```@example convergence
panel_counts = [24, 48]
bem_strengths = [target_strength(bem(body, boundary, wavenumber;
    incidence_angle = 0.0, n = count)) for count in panel_counts]
bem_errors = abs.(bem_strengths .- reference)
@assert all(isfinite, bem_strengths)
@assert last(bem_errors) < 1.0
(panels = panel_counts, error_db = bem_errors)
```

The assertions are example smoke checks. Scientific error budgets need refinement over the
frequency and parameter range of interest. Quadrature and panel resolution control different
errors. Neither removes the formulation's irregular-frequency limitations.

For oblique axisymmetric incidence `beta` in the x–z plane, query backscatter with
`target_strength(solution; angle = pi - beta, azimuth = pi)`. The default `angle = pi`
always points along negative z, regardless of incidence.
