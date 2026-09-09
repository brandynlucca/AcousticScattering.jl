# [Materials and shells](@id materials-tutorial)

These examples demonstrate fluid, solid-elastic, and layered spheres. All contrasts are relative
to the exterior fluid. The example values are illustrative, not calibration specifications.

## Fluid and gas interiors

```@example materials
using AcousticScattering

body = Sphere(0.01)
sound_speed = 1477.4
wavenumber = 2pi * 38000.0 / sound_speed
water_like = FluidFilled(1028.9 / 1026.8, 1480.3 / sound_speed)
gas_like = GasFilled(1.2 / 1026.8, 343.0 / sound_speed)
water_strength = target_strength(modal(body, water_like, wavenumber))
gas_strength = target_strength(modal(body, gas_like, wavenumber))
@assert isapprox(water_strength, -94.278687; atol = 1e-3)
(water_like = water_strength, gas_like = gas_strength)
```

`GasFilled` aliases `FluidFilled`: both impose pressure and normal-velocity continuity.
Gas resonances may require fine frequency sampling. A coarse grid can miss their peaks.

## Solid elastic sphere

```@example materials
solid = SolidElastic(7800.0 / 1026.8, 5900.0 / sound_speed, 3200.0 / sound_speed)
solid_solution = modal(body, solid, wavenumber)
@assert isfinite(target_strength(solid_solution))
target_strength(solid_solution)
```

Inputs are density, longitudinal-speed, and shear-speed contrasts. See
[Modal series](@ref modal-theory) for the elastic boundary conditions.

## Fluid and elastic shells

```@example materials
fluid_shell = Shelled(FluidLayer(1.1, 1.05), VacuumInterior(), 0.9)
filled_shell = Shelled(FluidLayer(1.1, 1.05), FluidInterior(1.0, 1.0), 0.9)
elastic_shell = Shelled(ElasticLayer(7.8, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.9)
shell_strengths = [target_strength(modal(body, configuration, wavenumber))
    for configuration in (fluid_shell, filled_shell, elastic_shell)]
@assert all(isfinite, shell_strengths)
shell_strengths
```

`0.9` is inner radius divided by outer radius. The `Sphere` radius is the outer radius.
`VacuumInterior()` imposes pressure release at the inner fluid-layer surface. It is not a
zero-density `FluidInterior`.

## Viscous flesh and an elastic wall

```@example materials
flesh = ViscousLayer(sound_speed, 1.05, 1.02, 1e-5, 1e-6)
wall = ElasticLayer(1.1, 1.2, 0.2)
layers = LayeredMaterial(flesh, wall, 0.8)
swimbladder = Shelled(layers, FluidInterior(0.0012, 0.23), 0.7)
low_wavenumber = 2pi * 1000.0 / sound_speed
viscous_solution = modal(body, swimbladder, low_wavenumber; m_max = 0)
@assert isfinite(target_strength(viscous_solution))
target_strength(viscous_solution)
```

Both ratios use the outer body radius. The gas core ends at `0.7 * radius`, and the elastic
wall ends at `0.8 * radius`. Viscosity inputs are kinematic viscosities in m²/s. The implemented
model is restricted to the monopole (`m_max = 0`). Higher orders are not supported.

Structural shell FEM instead uses `Shelled(poisson, density, youngs_modulus)` with
`Shell(body, thickness)`. See [FEM and shell coupling](@ref fem-theory) for its absolute units
and distinct interior-fluid arguments.
