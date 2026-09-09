# [Geometry and incidence](@id geometry-tutorial)

Compare spheroids and cylinders using a consistent incidence convention, then construct a mesh.

## Spheroids

```@example geometry
using AcousticScattering

wavenumber = 2pi * 12000.0 / 1477.4
prolate = Spheroid(0.02, 0.01)
oblate = Spheroid(0.01, 0.02)
incidence_angles = [0.0, pi / 4, pi / 2]
strengths = [target_strength(modal(body, Rigid(), wavenumber;
    incidence_angle = angle, m_max = 6, n_max = 8))
    for angle in incidence_angles, body in (prolate, oblate)]
@assert all(isfinite, strengths)
strengths
```

Rows correspond to angles, and columns to the prolate and oblate bodies. The first semi-axis follows the symmetry axis.
These small orders keep the example inexpensive. Increase both to establish convergence.

## Cylinder aspect dependence

```@example geometry
using CairoMakie: Figure, Axis, lines!, save

cylinder = Cylinder(0.01, 0.07)
angles = collect(range(0.05, pi / 2; length = 50))
cylinder_strengths = [target_strength(modal(cylinder, Rigid(), wavenumber;
    incidence_angle = angle)) for angle in angles]
figure = Figure(; size = (760, 420))
axis = Axis(figure[1, 1]; xlabel = "Incidence angle (degrees)",
    ylabel = "Target strength (dB re 1 m²)", title = "Finite-cylinder modal approximation")
lines!(axis, rad2deg.(angles), cylinder_strengths; color = :navy)
save("cylinder_incidence.png", figure)
nothing # hide
```

![Finite-cylinder target strength versus incidence at 12 kHz.](cylinder_incidence.png)

Zero is end-on and 90 degrees is broadside. This modal implementation neglects cap scattering,
so the curve is not an end-on accuracy claim for a closed finite cylinder. `endcap_depth`
affects MFS geometry, not the modal calculation.

## Curvature

```@example geometry
bent = Cylinder(0.01, 0.07; radius_curvature = 0.14)
bent_modal = modal(bent, Rigid(), wavenumber; incidence_angle = pi / 2)
bent_kirchhoff = kirchhoff(bent, Rigid(), wavenumber; incidence_angle = pi / 2)
@assert isfinite(target_strength(bent_modal))
@assert isfinite(target_strength(bent_kirchhoff))
(modal = target_strength(bent_modal), kirchhoff = target_strength(bent_kirchhoff))
```

Modal applies a near-broadside Fresnel correction. Kirchhoff integrates the curved illuminated
surface. Their difference includes model error, especially at this modest frequency.
Bent-cylinder MFS is a separate 3D solve. BEM rejects bent cylinders. The FEM cylinder path uses
straight geometry and must not be interpreted as a curved-body calculation.

## Mesh construction

```@example geometry
surface_mesh = mesh(prolate; resolution = 32)
@assert surface_mesh isa Mesh
typeof(surface_mesh)
```

This is an axisymmetric meridian discretization. `resolution` means panel count for
`method = :axisymmetric`, but target edge length in meters for `method = :full`. Supply either
`resolution` or `k`, not both. Full surface meshing currently supports spheres and spheroids.
See the [gallery](@ref gallery) for mesh and surface-field plots.
