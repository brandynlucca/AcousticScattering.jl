# [Two gas cavities in one body](@id gallery-cavities)

A branched region tree places both cavities inside the same outer body, `parents=[0, 1, 1]`.
At this frequency the larger cavity is close to a spatially varying response while the smaller
one remains comparatively quiet. A cutaway host gives the colored cavity traces physical
context without laying them over a front-facing wireframe.

```@example gallery_cavities
using AcousticScattering
using CairoMakie: plot, plot!, save

surfaces = [
    mesh(; semiaxes=(0.035, 0.025, 0.022), resolution=0.5, qorder=4),
    mesh(; semiaxes=(0.007, 0.006, 0.005), center=(-0.013, 0.004, 0), resolution=0.5, qorder=4),
    mesh(; semiaxes=(0.005, 0.004, 0.004), center=(0.013, -0.004, 0.002), resolution=0.5, qorder=4)]
materials = [FluidFilled(1.04, 1.04), GasFilled(0.0012, 0.23), GasFilled(0.0012, 0.23)]
solution = bem(surfaces, materials, 70.0; parents=[0, 1, 1], incidence_angle=pi / 3)
fig = plot(solution; kind=:surface_field, interfaces=[2, 3], colormap=:magma,
    colorrange=(0, 0.03), figure=(size=(900, 580),),
    axis=(azimuth=-0.38pi, elevation=0.24pi, xlabel="x (m)",
        ylabel="y (m)", zlabel="z (m)"))
plot!(fig.axis, solution; kind=:mesh, interfaces=[1],
    cutaway=(normal=(0, -1, 0), offset=0.0), interface_colors=[(:steelblue, 0.28)])
save("cavity_pressure.png", fig)
nothing # hide
```

![Internal pressure on two differently sized gas cavities within one fluid body.](cavity_pressure.png)

Color shows $|p|/|p_\mathrm{inc}|$ (dimensionless). This is one illustrative frequency. A
resonance study needs a frequency sweep and mesh refinement.
