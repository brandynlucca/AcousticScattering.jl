# [A coating changes the spectrum](@id gallery-coating)

Keep a 7.5 mm fluid core fixed and surround it with a 2.5 mm fluid coating. Material ratios are illustrative.

```@example gallery_coating
using AcousticScattering
using CairoMakie: Figure, Axis, lines!, axislegend, save

core = FluidFilled(0.8, 0.9)
coated = Shelled(FluidLayer(1.4, 1.2), FluidInterior(0.8, 0.9), 0.75)
frequencies = range(10000.0, 120000.0; length=301)
bare = frequency_sweep(k -> modal(Sphere(0.0075), core, k), frequencies, 1500.0)
shell = frequency_sweep(k -> modal(Sphere(0.01), coated, k), frequencies, 1500.0)
fig = Figure(size=(760, 480))
ax = Axis(fig[1, 1]; xlabel="Frequency (kHz)", ylabel="TS (dB re 1 m²)")
lines!(ax, frequencies ./ 1000, bare.target_strength; label="Bare core", color=:navy)
lines!(ax, frequencies ./ 1000, shell.target_strength; label="Coated core", color=:darkorange)
axislegend(ax; position=:rb)
save("coating.png", fig)
nothing # hide
```

![Bare and coated particle spectra.](coating.png)

Explore internal pressure in the [coated-particle tutorial](@ref coated-particle-tutorial).
