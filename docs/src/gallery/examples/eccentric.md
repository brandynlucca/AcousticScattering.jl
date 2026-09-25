# [An off-center fluid inclusion](@id gallery-eccentric)

A displaced, rotated dense inclusion breaks the symmetry of an ellipsoidal host. Both interfaces
show total pressure, with the host translucent so the inclusion is visible through it. The gold
arrow marks the incident wave direction.

```@example gallery_eccentric
using AcousticScattering
using CairoMakie: plot, save

surfaces = [
    mesh(; semiaxes=(0.034, 0.026, 0.022), resolution=0.5, qorder=4),
    mesh(; semiaxes=(0.013, 0.008, 0.010), center=(0.009, 0.004, 0.002),
        rotation=(axis=(0, 0, 1), angle=deg2rad(22)), resolution=0.5, qorder=4)]
solution = bem(surfaces, [FluidFilled(1.05, 0.98), FluidFilled(1.8, 0.7)], 250.0;
    parents=[0, 1], incidence_angle=pi / 3, incidence_azimuth=pi / 5)
fig = plot(solution; kind=:surface_field, interfaces=[1, 2],
    interface_alpha=[0.3, 1], colormap=:viridis, incident_arrow=true,
    figure=(size=(900, 580),),
    axis=(azimuth=-0.38pi, elevation=0.24pi, xlabel="x (m)",
        ylabel="y (m)", zlabel="z (m)"))
save("eccentric_pressure.png", fig)
nothing # hide
```

![Total pressure on an eccentric inclusion seen through its translucent host interface.](eccentric_pressure.png)

Materials use density and sound-speed ratios relative to the exterior. These mesh settings are for visualization (see [numerical convergence](@ref convergence-tutorial) for refinement).
