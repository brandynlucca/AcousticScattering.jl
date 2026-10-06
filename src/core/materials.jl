# Shared boundary conditions and material laws.

abstract type AbstractBoundaryCondition end

"Interior medium enclosed by a shell layer in a [`Shelled`](@ref) boundary condition."
abstract type AbstractShellInterior end

"Material of the shell layer in a [`Shelled`](@ref) boundary condition."
abstract type AbstractShellMaterial end

"""
    Rigid()

Rigid boundary with zero total normal velocity, for supported geometries and solvers.
"""
struct Rigid <: AbstractBoundaryCondition end

"""
    PressureRelease()

Soft boundary with zero total acoustic pressure, for supported geometries and solvers.
"""
struct PressureRelease <: AbstractBoundaryCondition end

"""
    Impedance(zeta)

Locally reacting boundary with specific acoustic impedance ratio `zeta = Z/(ρc)`, dimensionless
and relative to the exterior fluid, under the `exp(-iωt)` convention `∂p/∂n = -ik p/zeta` at the
surface. `zeta` must have a nonnegative real part (a passive, non-generating boundary) and be
nonzero. [`Rigid`](@ref) and [`PressureRelease`](@ref) are its `zeta → ∞` and `zeta → 0` limits.
Supported by `modal(Sphere, ...)` and by `bem(...; method = :axisymmetric)` on
`Sphere`/`Spheroid`/straight `Cylinder`, at any incidence angle, including
[`incidence_angle_sweep`](@ref)'s factorized reuse. The direct BEM solve becomes
ill-conditioned as `zeta → 0`. Use [`PressureRelease`](@ref) there.
"""
struct Impedance <: AbstractBoundaryCondition
    zeta::ComplexF64
    function Impedance(zeta::Number)
        isfinite(zeta) && !iszero(zeta) ||
            throw(ArgumentError("zeta must be finite and nonzero"))
        real(zeta) >= 0 ||
            throw(ArgumentError("zeta must have a nonnegative real part"))
        return new(ComplexF64(zeta))
    end
end

"""
    FluidFilled(density_contrast, soundspeed_contrast; coupling=:full)

Homogeneous fluid transmission boundary condition (Anderson, 1950). Gas-filled bodies use the
same boundary condition with different material contrasts.

- `density_contrast` is ρ_interior/ρ_exterior.
- `soundspeed_contrast` is c_interior/c_exterior.
- `coupling` affects the spheroid modal series only (the sphere is diagonal by symmetry).
  `:full` solves the complete off-diagonal boundary coupling. `:diagonal` uses a cheaper
  approximation that degrades at higher `ka` and larger eccentricity.

See [Modal series](@ref modal-theory).
"""
struct FluidFilled <: AbstractBoundaryCondition
    density_contrast::Float64
    soundspeed_contrast::Float64
    coupling::Symbol

    function FluidFilled(density_contrast::Real, soundspeed_contrast::Real; coupling::Symbol = :full)
        coupling in (:full, :diagonal) ||
            throw(ArgumentError("coupling must be :full or :diagonal"))
        return new(Float64(density_contrast), Float64(soundspeed_contrast), coupling)
    end
end

"Alias for [`FluidFilled`](@ref), provided for gas-filled spheres, which solve the same boundary-value problem."
const GasFilled = FluidFilled

"""
    VacuumInterior()

A void/vacuum shell interior: pressure-release (`p=0`) at the shell's own inner surface, no
interior medium. Pairs with [`FluidLayer`](@ref) to reproduce the former `ShellSoft` boundary
condition, see [`Shelled`](@ref).
"""
struct VacuumInterior <: AbstractShellInterior end

"""
    FluidInterior(density_contrast, soundspeed_contrast)

A fluid- or gas-filled shell interior. `density_contrast` = ρ_interior/ρ_exterior,
`soundspeed_contrast` = c_interior/c_exterior, both relative to the *exterior* medium (not to the
shell). Pairs with [`FluidLayer`](@ref) (former `ShellFluidFilled`) or [`ElasticLayer`](@ref)
(former `ElasticShell`), see [`Shelled`](@ref).
"""
struct FluidInterior <: AbstractShellInterior
    density_contrast::Float64
    soundspeed_contrast::Float64

    function FluidInterior(density_contrast::Real, soundspeed_contrast::Real)
        density_contrast > 0 || throw(ArgumentError("density_contrast must be positive"))
        soundspeed_contrast > 0 ||
            throw(ArgumentError("soundspeed_contrast must be positive"))
        return new(Float64(density_contrast), Float64(soundspeed_contrast))
    end
end

"""
    FluidLayer(density_contrast, soundspeed_contrast)

A fluid spherical shell layer (no shear waves), e.g. a swim bladder wall modeled as fluid rather
than solid tissue (Jech et al., 2015). `density_contrast` is ρ_shell/ρ_exterior and
`soundspeed_contrast` is c_shell/c_exterior, both relative to the exterior medium. Pair with
[`VacuumInterior`](@ref) for a pressure-release interior or [`FluidInterior`](@ref) for a fluid
interior, via [`Shelled`](@ref).

See [Modal series](@ref modal-theory) for the boundary-matching derivation.
"""
struct FluidLayer <: AbstractShellMaterial
    density_contrast::Float64
    soundspeed_contrast::Float64

    function FluidLayer(density_contrast::Real, soundspeed_contrast::Real)
        density_contrast > 0 || throw(ArgumentError("density_contrast must be positive"))
        soundspeed_contrast > 0 ||
            throw(ArgumentError("soundspeed_contrast must be positive"))
        return new(Float64(density_contrast), Float64(soundspeed_contrast))
    end
end

"""
    ElasticLayer(density_contrast, speed_longitudinal_contrast, speed_transversal_contrast; interior_coupling=:generalized)

An isotropic elastic spherical shell layer (Goodman and Stern, 1962), paired with a
[`FluidInterior`](@ref) via [`Shelled`](@ref). All contrasts are relative to the exterior medium.

- `density_contrast` is ρ_shell/ρ_ext.
- `speed_longitudinal_contrast` and `speed_transversal_contrast` are c_L/c_ext and c_T/c_ext,
  the shell's longitudinal and shear wave speeds.
- `interior_coupling`: `:generalized` (default) uses the interior fluid's own wavenumber and
  density. `:identical_fluid` uses the exterior fluid's properties at the inner radius instead
  (Stanton, 1990), agreeing with `:generalized` only when the interior and exterior fluids match.

See [Modal series](@ref modal-theory) for the boundary-matching determinant.
"""
struct ElasticLayer <: AbstractShellMaterial
    density_contrast::Float64
    speed_longitudinal_contrast::Float64
    speed_transversal_contrast::Float64
    interior_coupling::Symbol

    function ElasticLayer(density_contrast::Real, speed_longitudinal_contrast::Real,
            speed_transversal_contrast::Real; interior_coupling::Symbol = :generalized)
        density_contrast > 0 ||
            throw(ArgumentError("density_contrast must be positive"))
        speed_longitudinal_contrast > 0 ||
            throw(ArgumentError("speed_longitudinal_contrast must be positive"))
        speed_transversal_contrast > 0 ||
            throw(ArgumentError("speed_transversal_contrast must be positive"))
        interior_coupling in (:generalized, :identical_fluid) ||
            throw(ArgumentError("interior_coupling must be :generalized or :identical_fluid"))
        return new(Float64(density_contrast), Float64(speed_longitudinal_contrast),
            Float64(speed_transversal_contrast), interior_coupling)
    end
end

"""
    ViscousLayer(soundspeed_exterior, density_contrast, soundspeed_contrast,
                 kinematic_viscosity_compressional, kinematic_viscosity_shear)

A viscous fluid-like shell layer (Feuillade and Nero, 1998), e.g. the flesh surrounding a fish's
swimbladder. Used as the outer layer of a [`LayeredMaterial`](@ref).

- `soundspeed_exterior` is the exterior medium's sound speed in m/s.
- `density_contrast` and `soundspeed_contrast` are lossless reference density/compressional-speed
  contrasts, `ρ2/ρ1`, `c2/c1`, relative to the exterior medium.
- `kinematic_viscosity_compressional` and `kinematic_viscosity_shear` are the kinematic bulk and
  shear viscosities in m²/s. Set both to `0.0` for the lossless (purely elastic) limit.

See [Modal series](@ref modal-theory).
"""
struct ViscousLayer <: AbstractShellMaterial
    soundspeed_exterior::Float64
    density_contrast::Float64
    soundspeed_contrast::Float64
    kinematic_viscosity_compressional::Float64
    kinematic_viscosity_shear::Float64

    function ViscousLayer(
            soundspeed_exterior::Real, density_contrast::Real, soundspeed_contrast::Real,
            kinematic_viscosity_compressional::Real, kinematic_viscosity_shear::Real)
        soundspeed_exterior > 0 ||
            throw(ArgumentError("soundspeed_exterior must be positive"))
        density_contrast > 0 || throw(ArgumentError("density_contrast must be positive"))
        soundspeed_contrast > 0 ||
            throw(ArgumentError("soundspeed_contrast must be positive"))
        kinematic_viscosity_compressional >= 0 ||
            throw(ArgumentError("kinematic_viscosity_compressional must be nonnegative"))
        kinematic_viscosity_shear >= 0 ||
            throw(ArgumentError("kinematic_viscosity_shear must be nonnegative"))
        return new(Float64(soundspeed_exterior), Float64(density_contrast),
            Float64(soundspeed_contrast),
            Float64(kinematic_viscosity_compressional), Float64(kinematic_viscosity_shear))
    end
end

"""
    LayeredMaterial(outer, inner, radius_ratio)

Two concentric shell material layers. `radius_ratio` is the inner layer's outer radius
divided by the outer layer's outer radius, in (0,1). For spherical modal scattering,
`FluidLayer` and `ElasticLayer` entries can be nested to any finite depth; each material's
density and wave-speed contrasts are relative to the exterior fluid. Fluid layers use
pressure and normal-velocity continuity, while elastic layers retain longitudinal and
shear motion with traction and displacement conditions. Nesting a `LayeredMaterial` in `inner` makes
its radius ratio local to the enclosing layer's outer radius. For example,
`LayeredMaterial(f1, LayeredMaterial(f2, f3, 0.75), 0.8)` has interfaces at 0.8 and 0.6
times the body's outer radius. The viscous-flesh-over-elastic-wall monopole model
(Feuillade and Nero, 1998) remains supported:

```julia
Shelled(LayeredMaterial(ViscousLayer(...), ElasticLayer(...), radius_ratio_wall),
    FluidInterior(...), radius_ratio_core)
```
"""
struct LayeredMaterial{L1 <: AbstractShellMaterial, L2 <: AbstractShellMaterial} <:
       AbstractShellMaterial
    outer::L1
    inner::L2
    radius_ratio::Float64

    function LayeredMaterial(outer::L1,
            inner::L2,
            radius_ratio::Real) where
            {L1 <: AbstractShellMaterial, L2 <: AbstractShellMaterial}
        0 < radius_ratio < 1 || throw(ArgumentError("radius_ratio must be in (0, 1)"))
        return new{L1, L2}(outer, inner, Float64(radius_ratio))
    end
end

# Ratios in a nested material are local to the enclosing layer's outer radius.
# The returned radii are all relative to the body's outer radius.
function _fluid_layers(material::FluidLayer, outer_ratio::Real = 1.0)
    (FluidLayer[material], Float64[outer_ratio])
end

function _fluid_layers(material::LayeredMaterial, outer_ratio::Real = 1.0)
    material.outer isa FluidLayer ||
        throw(ArgumentError("layered spherical modal scattering requires FluidLayer at every shell interface"))
    inner_layers, inner_radii = _fluid_layers(material.inner,
        outer_ratio * material.radius_ratio)
    return (vcat(FluidLayer[material.outer], inner_layers),
        vcat(Float64[outer_ratio], inner_radii))
end

function _fluid_layers(::AbstractShellMaterial, ::Real)
    throw(ArgumentError("layered spherical modal scattering requires FluidLayer at every shell interface"))
end

function _innermost_interface_ratio(material::LayeredMaterial)
    ratio = material.radius_ratio
    inner = material.inner
    return inner isa LayeredMaterial ?
           ratio * _innermost_interface_ratio(inner) : ratio
end

"""
    ElasticFEMLayer(poisson, density, youngs_modulus)

Full through-thickness elastic shell material, using absolute (not contrast) values. Used only by
[`fem`](@ref)`(::Shell, ::Shelled, ...)`'s 2D shell-FEM solve. Construct via
[`Shelled`](@ref)`(poisson, density, youngs_modulus)`.

- `poisson` is Poisson's ratio (dimensionless, `< 0.5`).
- `density` is the absolute shell density in kg/m³.
- `youngs_modulus` is the absolute Young's modulus in Pa.
"""
struct ElasticFEMLayer <: AbstractShellMaterial
    poisson::Float64
    density::Float64
    youngs_modulus::Float64

    function ElasticFEMLayer(poisson::Real, density::Real, youngs_modulus::Real)
        -1.0 < poisson < 0.5 || throw(ArgumentError("poisson must lie in (-1, 0.5)"))
        density > 0 || throw(ArgumentError("density must be positive"))
        youngs_modulus > 0 || throw(ArgumentError("youngs_modulus must be positive"))
        return new(Float64(poisson), Float64(density), Float64(youngs_modulus))
    end
end

"""
    Shelled(material, interior, radius_ratio)
    Shelled(poisson, density, youngs_modulus)

The single boundary condition/material type for every shell in this package.

The three-argument form pairs a `material` layer ([`FluidLayer`](@ref), [`ElasticLayer`](@ref),
or a [`LayeredMaterial`](@ref)) with either a [`VacuumInterior`](@ref) or a
[`FluidInterior`](@ref), for use as a boundary condition with
[`modal`](@ref)/[`kirchhoff`](@ref)/[`fem`](@ref)`(::Sphere/Cylinder, ...)`. Examples:
- `Shelled(FluidLayer(...), VacuumInterior(), radius_ratio)`
- `Shelled(FluidLayer(...), FluidInterior(...), radius_ratio)`
- `Shelled(ElasticLayer(...), FluidInterior(...), radius_ratio)`
- `Shelled(LayeredMaterial(ViscousLayer(...), ElasticLayer(...), radius_ratio_wall),
  FluidInterior(...), radius_ratio_core)`

`radius_ratio` is the inner/outer shell surface radius, in (0,1). `a` in every
`target_strength`/`form_function` call is the shell's outer radius. When `material` is a
`LayeredMaterial`, the innermost material interface must be strictly outside the
enclosing `Shelled`'s interior radius. Spherical `modal` supports arbitrary finite
compositions of `FluidLayer` and `ElasticLayer` with a fluid or vacuum core.
Mixed stacks require `ElasticLayer(interior_coupling = :generalized)`.

The one-argument-triple form `Shelled(poisson, density, youngs_modulus)` instead builds an
`ElasticFEMLayer` with no `interior`/`radius_ratio`, for use with
[`fem`](@ref)`(::Shell, ::Shelled, ...)`.
"""
struct Shelled{M <: AbstractShellMaterial, I <: Union{Nothing, AbstractShellInterior}} <:
       AbstractBoundaryCondition
    material::M
    interior::I
    radius_ratio::Union{Nothing, Float64}

    function Shelled(material::M,
            interior::I,
            radius_ratio::Real) where
            {M <: AbstractShellMaterial, I <: AbstractShellInterior}
        0 < radius_ratio < 1 || throw(ArgumentError("radius_ratio must be in (0, 1)"))
        material isa LayeredMaterial &&
            radius_ratio >= _innermost_interface_ratio(material) &&
            throw(ArgumentError("radius_ratio must be less than the innermost material interface ratio"))
        return new{M, I}(material, interior, Float64(radius_ratio))
    end
    function Shelled(poisson::Real, density::Real, youngs_modulus::Real)
        material = ElasticFEMLayer(poisson, density, youngs_modulus)
        return new{ElasticFEMLayer, Nothing}(material, nothing, nothing)
    end
end

"""
    SolidElastic(density_contrast, speed_longitudinal_contrast, speed_transversal_contrast)

Solid elastic sphere boundary condition (MacLennan 1981), the classical
sonar calibration-sphere model (e.g. tungsten carbide, steel reference
spheres). Material properties are contrasts relative to the exterior
medium, matching [`ElasticLayer`](@ref)'s convention:

- `density_contrast` = ρ_body / ρ_ext
- `speed_longitudinal_contrast` = c_L / c_ext,
  `speed_transversal_contrast` = c_T / c_ext.

Uses Hickling's (1962) resonance form (`sin η`, `cos η`) rather than
a boundary-matrix determinant like [`ElasticLayer`](@ref)/[`Shelled`](@ref): there's no
interior fluid, so it's a direct ratio of tangent terms. For the package's `exp(-iωt)` time
convention and outgoing first-kind Hankel waves, the amplitude is
`-i/k * Σ (2m+1) Pₘ(cosθ) Aₘ` with `Aₘ = sin η (-i cos η - sin η)`, which
maps exactly onto this package's existing `form_function`/`target_strength`
convention, with `target_strength(f) = 20 log10(|f|)`.
"""
struct SolidElastic <: AbstractBoundaryCondition
    density_contrast::Float64
    speed_longitudinal_contrast::Float64
    speed_transversal_contrast::Float64

    function SolidElastic(density_contrast::Real, speed_longitudinal_contrast::Real,
            speed_transversal_contrast::Real)
        density_contrast > 0 || throw(ArgumentError("density_contrast must be positive"))
        speed_longitudinal_contrast > 0 ||
            throw(ArgumentError("speed_longitudinal_contrast must be positive"))
        speed_transversal_contrast > 0 ||
            throw(ArgumentError("speed_transversal_contrast must be positive"))
        return new(Float64(density_contrast), Float64(speed_longitudinal_contrast),
            Float64(speed_transversal_contrast))
    end
end

"""
    ViscoelasticSolid(density_contrast, speed_longitudinal_contrast, speed_transversal_contrast;
        loss_longitudinal=0.0, loss_transversal=0.0)

Solid with hysteretic damping, for example bone, with the contrasts of [`SolidElastic`](@ref).
`loss_longitudinal` and `loss_transversal` are the loss factors `tan δ` of the longitudinal and
shear moduli, which become `M(1 - i loss_longitudinal)` and `μ(1 - i loss_transversal)` for the
package's `exp(-iωt)` convention. Supported as a region material of the volume
[`fem`](@ref)`(bodies, materials, k)`.
"""
struct ViscoelasticSolid <: AbstractBoundaryCondition
    density_contrast::Float64
    speed_longitudinal_contrast::Float64
    speed_transversal_contrast::Float64
    loss_longitudinal::Float64
    loss_transversal::Float64

    function ViscoelasticSolid(density_contrast::Real, speed_longitudinal_contrast::Real,
            speed_transversal_contrast::Real; loss_longitudinal::Real = 0.0,
            loss_transversal::Real = 0.0)
        density_contrast > 0 || throw(ArgumentError("density_contrast must be positive"))
        speed_longitudinal_contrast > 0 ||
            throw(ArgumentError("speed_longitudinal_contrast must be positive"))
        speed_transversal_contrast > 0 ||
            throw(ArgumentError("speed_transversal_contrast must be positive"))
        loss_longitudinal >= 0 ||
            throw(ArgumentError("loss_longitudinal must be nonnegative"))
        loss_transversal >= 0 ||
            throw(ArgumentError("loss_transversal must be nonnegative"))
        return new(Float64(density_contrast), Float64(speed_longitudinal_contrast),
            Float64(speed_transversal_contrast), Float64(loss_longitudinal),
            Float64(loss_transversal))
    end
end
