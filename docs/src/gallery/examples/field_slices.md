# [Pressure-field slices in 3D](@id gallery-field-slices)

Two pressure slices are projected onto a floor and a far wall so neither obscures the sphere.
The wave travels along +x. Color shows the real part of total pressure divided by incident
amplitude. The rings mark where each slice plane cuts the sphere, $z=0$ (black) for the floor slice
and $x=6$ mm (gold) for the wall slice.

```@example gallery_field_slices
using AcousticScattering
using CairoMakie: plot, save

radius = 0.018
solution = modal(Sphere(radius), FluidFilled(2.0, 0.65), 250.0)
fig = plot(solution; kind=:field_slices,
    slices=((axis=:z, at=0.0, project_to=-0.060),
        (axis=:x, at=0.006, project_to=0.060)),
    extent=0.055, resolution=151, field=:pressure_real,
    colorrange=(-3, 3))
save("field_slices.png", fig)
nothing # hide
```

![Pressure slices projected onto a floor and wall, with the slice planes marked on the sphere.](field_slices.png)
