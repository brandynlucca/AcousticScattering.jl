# [A supplied faceted body](@id gallery-faceted)

Load a closed, outward-oriented Gmsh surface supplied by the user. This small asymmetric
crystalline target keeps the example focused on the actual workflow: load, solve, and plot.
The visible edges make it clear that the solver uses the supplied connectivity rather than a
built-in body.

```@example gallery_faceted
using AcousticScattering
using CairoMakie: plot, save

mesh_path = joinpath(pkgdir(AcousticScattering), "docs", "src", "gallery",
    "examples", "irregular_target.msh")
surface_mesh = mesh(mesh_path; qorder=4)
solution = bem(surface_mesh, Rigid(), 220.0;
    incidence_angle=pi / 3, incidence_azimuth=0.4)
fig = plot(solution; kind=:surface_field, field=:pressure_magnitude, show_edges=true,
    colormap=:viridis, colorbar=true, incident_arrow=true,
    figure=(size=(900, 580), figure_padding=(70, 20, 20, 20)),
    axis=(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
        azimuth=-0.38pi, elevation=0.24pi))
save("faceted_pressure.png", fig)
nothing # hide
```

![Scattered-pressure magnitude on a supplied asymmetric crystalline mesh.](faceted_pressure.png)

This small mesh demonstrates the input and plotting workflow. Sharp-edge pressure requires local refinement for quantitative accuracy.
