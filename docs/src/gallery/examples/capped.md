# [Pressure on a closed fluid cylinder](@id gallery-capped)

A fluid-filled cylinder has hemispherical ends and oblique incidence. Full-surface BEM includes
both ends. The camera looks across the long axis so the illuminated face and pressure variation
remain visible. The gold arrow marks the incident wave direction.

```@example gallery_capped
using AcousticScattering
using CairoMakie: plot, save

body = Cylinder(0.008, 0.024; endcap_depth=0.008)
surface_mesh = mesh(body; method=:full, resolution=0.0035, mesh_order=3, qorder=4)
solution = bem(surface_mesh, FluidFilled(1.5, 0.8), 250.0;
    incidence_angle=pi / 3, incidence_azimuth=pi / 4)
fig = plot(solution; kind=:surface_field, field=:pressure_magnitude,
    colorbar=true, incident_arrow=true,
    figure=(size=(900, 580), figure_padding=(70, 20, 20, 20)),
    axis=(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        azimuth=-0.42pi, elevation=0.28pi))
save("capped_pressure.png", fig)
nothing # hide
```

![Scattered-pressure magnitude on a closed fluid cylinder under oblique incidence.](capped_pressure.png)

The cylinder's long axis is x. Mesh size and quadrature order should be refined for quantitative comparisons.
