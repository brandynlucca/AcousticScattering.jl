# [Your first frequency sweep](@id first-sweep)

Compute rigid-sphere backscatter, check modal truncation, and save a figure and data. Install
AcousticScattering and CairoMakie following [Getting Started](@ref getting-started). The blocks
on this page share a Julia session and execute during every documentation build.

## Define and solve

```@example first_sweep
using AcousticScattering
using CairoMakie: Figure, Axis, lines!, save

body = Sphere(0.01) # radius, m
boundary = Rigid()
sound_speed = 1477.4 # m/s
frequencies = collect(range(12000.0, 200000.0; length = 80)) # Hz
wavenumbers = 2pi .* frequencies ./ sound_speed
solutions = [modal(body, boundary, k) for k in wavenumbers]
strengths = target_strength.(solutions)
@assert all(isfinite, strengths)
nothing # hide
```

`Rigid()` imposes zero total normal fluid velocity at the surface. The modal solver is a useful
starting point because the sphere permits separation of variables. Each solution represents one
frequency. See [Choosing a solver](@ref solver-selection) when changing shape or material.

## Check accuracy

```@example first_sweep
check_wavenumber = last(wavenumbers)
coarse = modal(body, boundary, check_wavenumber; m_max = 40)
fine = modal(body, boundary, check_wavenumber; m_max = 80)
truncation_difference = abs(target_strength(fine) - target_strength(coarse))
@assert truncation_difference < 1e-6
truncation_difference
```

This checks modal truncation at one frequency, not the physical adequacy of a rigid-sphere model.

![Incidence angle beta from the body axis and the opposite backscatter direction.](../assets/directions.svg)

For a sphere the orientation does not change backscatter. For an axisymmetric nonspherical
body, incidence and observation must be specified separately. See [Conventions](@ref conventions).

## Plot and save

```@example first_sweep
figure = Figure(; size = (760, 420))
axis = Axis(figure[1, 1]; xlabel = "Frequency (kHz)",
    ylabel = "Target strength (dB re 1 m²)", title = "Rigid sphere, radius 1 cm")
lines!(axis, frequencies ./ 1000, strengths; color = :navy, linewidth = 2)
save("rigid_sphere_frequency.png", figure)
nothing # hide
```

![Modal backscatter from a rigid 1 cm sphere across 12–200 kHz.](rigid_sphere_frequency.png)

```@example first_sweep
open("rigid_sphere_frequency.csv", "w") do io
    println(io, "frequency_hz,target_strength_db")
    for (frequency, strength) in zip(frequencies, strengths)
        println(io, frequency, ",", strength)
    end
end
nothing # hide
```

Download the [figure](rigid_sphere_frequency.png) and [CSV](rigid_sphere_frequency.csv).
In your own session, these files appear in the current working directory.

Continue with [Materials and shells](@ref materials-tutorial),
[Geometry and incidence](@ref geometry-tutorial), or
[Numerical convergence](@ref convergence-tutorial). Compilation and threading advice lives in
the separate [performance tutorial](@ref performance-tutorial).
