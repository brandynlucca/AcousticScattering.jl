# [A millimeter bubble, a strong resonance](@id gallery-bubble)

Sweep a 1 mm gas bubble in water. This idealized fluid model includes radiation damping.

```@example gallery_bubble
using AcousticScattering
using CairoMakie: plot, save

bubble = Sphere(0.001)
gas = GasFilled(1.2 / 1000, 343 / 1500)
frequencies = range(1000.0, 6000.0; length=1001)
response = frequency_sweep(k -> modal(bubble, gas, k), frequencies, 1500.0)
fig = plot(response; legend=false, colors=[:darkorange],
    figure=(size=(760, 560),))
save("bubble.png", fig)
nothing # hide
```

![Bubble resonance in target strength and complex-amplitude phase.](bubble.png)

The phase turns rapidly through resonance. See the [application tutorial](@ref bubble-resonance-tutorial) to locate and resolve the peak.
