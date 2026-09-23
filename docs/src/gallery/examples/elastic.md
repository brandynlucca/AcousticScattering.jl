# [Elastic resonances in a solid sphere](@id gallery-elastic)

Compare a rigid sphere with an elastic sphere of the same 2 cm radius. The illustrative elastic material has density, compressional-speed and shear-speed ratios of 7.8, 4 and 2.

```@example gallery_elastic
using AcousticScattering
using CairoMakie: Figure, Axis, lines!, axislegend, save

body = Sphere(0.02)
frequencies = range(10000.0, 120000.0; length=1101)
rigid = frequency_sweep(k -> modal(body, Rigid(), k), frequencies, 1500.0)
elastic = frequency_sweep(k -> modal(body, SolidElastic(7.8, 4.0, 2.0), k),
    frequencies, 1500.0)
fig = Figure(size=(760, 480))
ax = Axis(fig[1, 1]; xlabel="Frequency (kHz)", ylabel="TS (dB re 1 m²)")
lines!(ax, frequencies ./ 1000, rigid.target_strength; label="Rigid", color=:navy)
lines!(ax, frequencies ./ 1000, elastic.target_strength; label="Elastic", color=:darkorange)
axislegend(ax; position=:lb)
save("elastic.png", fig)
nothing # hide
```

![Rigid and elastic sphere spectra, including narrow elastic resonances.](elastic.png)

Sharp features need local frequency refinement (see [reference-target models](@ref reference-target-tutorial)).
