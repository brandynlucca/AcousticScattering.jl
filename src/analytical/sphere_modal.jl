# Exact modal (Mie) series for rigid, pressure-release, and fluid-filled spheres.
# f(θ) = -i/k * Σ_{m=0}^{m_max} (2m+1) Pₘ(cosθ) Aₘ, backscatter is θ = π.

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

Two concentric shell material layers, `outer` ([`ViscousLayer`](@ref) or [`ElasticLayer`](@ref))
enclosing `inner` ([`ElasticLayer`](@ref)). `radius_ratio` is the inner layer's own outer radius
over the outer layer's outer radius, in (0,1). Used as the `material` of a [`Shelled`](@ref)
whose `interior` is enclosed by two layers rather than one, e.g. a viscous-flesh-over-elastic-wall
swimbladder model (Feuillade and Nero, 1998):

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
or a [`LayeredMaterial`](@ref) of two layers) with either a [`VacuumInterior`](@ref) or a
[`FluidInterior`](@ref), for use as a boundary condition with
[`modal`](@ref)/[`kirchhoff`](@ref)/[`fem`](@ref)`(::Sphere/Cylinder, ...)`. Examples:
- `Shelled(FluidLayer(...), VacuumInterior(), radius_ratio)`
- `Shelled(FluidLayer(...), FluidInterior(...), radius_ratio)`
- `Shelled(ElasticLayer(...), FluidInterior(...), radius_ratio)`
- `Shelled(LayeredMaterial(ViscousLayer(...), ElasticLayer(...), radius_ratio_wall),
  FluidInterior(...), radius_ratio_core)`

`radius_ratio` is the inner/outer shell surface radius, in (0,1). `a` in every
`target_strength`/`form_function` call is the shell's outer radius. When `material` is a
`LayeredMaterial`, its own `radius_ratio` must be strictly greater than the enclosing `Shelled`'s.

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
        material isa LayeredMaterial && radius_ratio >= material.radius_ratio &&
            throw(ArgumentError("radius_ratio must be less than material.radius_ratio (the two " *
                                "shell layers' own interface), got $radius_ratio >= $(material.radius_ratio)"))
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

function _default_mode_count(ka::Real)
    return max(20, ceil(Int, 2.5 * ka + 4 * cbrt(ka + 1) + 10))
end

function _modal_coefficient(::Rigid, m::Integer, k::Real, a::Real)
    ka = k * a
    return -jsd(m, ka) / hsd(m, ka)
end

function _modal_coefficient(::PressureRelease, m::Integer, k::Real, a::Real)
    ka = k * a
    return -js(m, ka) / hs(m, ka)
end

function _modal_coefficient(bc::FluidFilled, m::Integer, k::Real, a::Real)
    return _sphere_fluid_coefficients(bc, m, k, a).scattered
end

function _sphere_fluid_coefficients(bc::FluidFilled, m::Integer, k::Real, a::Real)
    ka = k * a
    ka_interior = ka / bc.soundspeed_contrast
    gh = bc.density_contrast * bc.soundspeed_contrast

    jd_int = jsd(m, ka_interior)
    jd_ext = jsd(m, ka)
    j_int = js(m, ka_interior)
    j_ext = js(m, ka)
    h_ext = hs(m, ka)
    hd_ext = hsd(m, ka)
    denominator = gh * j_int * hd_ext - h_ext * jd_int
    scattered = (j_ext * jd_int - gh * j_int * jd_ext) / denominator
    interior = gh * (j_ext * hd_ext - jd_ext * h_ext) / denominator
    return (; scattered, interior)
end

function _sphere_interface_solution(matrix, rhs)
    columns = maximum(abs, matrix; dims = 1)
    scaled = matrix ./ columns
    rows = maximum(abs, scaled; dims = 2)
    return (scaled ./ rows \ (rhs ./ vec(rows))) ./ vec(columns)
end

function _modal_coefficient(
        bc::Union{Shelled{FluidLayer, VacuumInterior},
            Shelled{FluidLayer, FluidInterior}, Shelled{ElasticLayer, FluidInterior}},
        m::Integer, k::Real, a::Real)
    return _sphere_shell_coefficients(bc, m, k, a).scattered
end

function _sphere_shell_coefficients(bc::Shelled{FluidLayer}, m::Integer, k::Real, a::Real)
    ka = k * a
    k2a = ka / bc.material.soundspeed_contrast
    k2b = k2a * bc.radius_ratio
    gh_shell = bc.material.density_contrast * bc.material.soundspeed_contrast
    n = bc.interior isa FluidInterior ? 4 : 3
    matrix = zeros(ComplexF64, n, n)
    rhs = zeros(ComplexF64, n)
    matrix[1, 1:3] = [hs(m, ka), -js(m, k2a), -ys(m, k2a)]
    matrix[2, 1:3] = [hsd(m, ka), -jsd(m, k2a) / gh_shell, -ysd(m, k2a) / gh_shell]
    matrix[3, 2:3] = [js(m, k2b), ys(m, k2b)]
    rhs[1:2] = [-js(m, ka), -jsd(m, ka)]
    if bc.interior isa FluidInterior
        k3b = ka * bc.radius_ratio / bc.interior.soundspeed_contrast
        gh_int = bc.interior.density_contrast * bc.interior.soundspeed_contrast
        matrix[3, 4] = -js(m, k3b)
        matrix[4, 2:4] = [
            jsd(m, k2b) / gh_shell, ysd(m, k2b) / gh_shell, -jsd(m, k3b) / gh_int]
    end
    x = _sphere_interface_solution(matrix, rhs)
    return (; scattered = x[1], shell = (x[2], x[3]), interior = n == 4 ? x[4] : nothing)
end

# λ/(λ+2G) = 1 - 2β, 2G/(λ+2G) = 2β, where β = (c_T/c_L)²
function _sphere_shell_coefficients(bc::Shelled{ElasticLayer, FluidInterior}, m::Integer, k::Real, a::Real)
    a_in = bc.radius_ratio * a
    ka1s = k * a
    kLs = (k / bc.material.speed_longitudinal_contrast) * a
    kTs = (k / bc.material.speed_transversal_contrast) * a
    kLf = (k / bc.material.speed_longitudinal_contrast) * a_in
    kTf = (k / bc.material.speed_transversal_contrast) * a_in
    β = (bc.material.speed_transversal_contrast / bc.material.speed_longitudinal_contrast)^2
    inv_rho_shell = 1 / bc.material.density_contrast

    # See ElasticLayer's docstring for interior_coupling.
    if bc.material.interior_coupling === :generalized
        k3f = (k / bc.interior.soundspeed_contrast) * a_in
        rho_int_over_shell = bc.interior.density_contrast / bc.material.density_contrast
    else
        k3f = k * a_in
        rho_int_over_shell = inv_rho_shell
    end

    j(x) = js(m, x)
    jd(x) = jsd(m, x)
    jdd(x) = jsdd(m, x)
    y(x) = ys(m, x)
    yd(x) = ysd(m, x)
    ydd(x) = ysdd(m, x)

    a1 = j(ka1s) * inv_rho_shell
    a2 = ka1s * jd(ka1s)
    a11 = hs(m, ka1s) * inv_rho_shell
    a21 = ka1s * hsd(m, ka1s)

    a12 = (1 - 2β) * j(kLs) - 2β * jdd(kLs)
    a22 = kLs * jd(kLs)
    a32 = 2 * (kLs * jd(kLs) - j(kLs))
    a42 = (1 - 2β) * j(kLf) - 2β * jdd(kLf)
    a52 = kLf * jd(kLf)
    a62 = 2 * (kLf * jd(kLf) - j(kLf))

    a13 = -2m * (m + 1) * kTs^(-2) * (kTs * jd(kTs) - j(kTs))
    a23 = m * (m + 1) * j(kTs)
    a33 = kTs^2 * jdd(kTs) + (m + 2) * (m - 1) * j(kTs)
    a43 = -2m * (m + 1) * kTf^(-2) * (kTf * jd(kTf) - j(kTf))
    a53 = m * (m + 1) * j(kTf)
    a63 = kTf^2 * jdd(kTf) + (m + 2) * (m - 1) * j(kTf)

    a14 = (1 - 2β) * y(kLs) - 2β * ydd(kLs)
    a24 = kLs * yd(kLs)
    a34 = 2 * (kLs * yd(kLs) - y(kLs))
    a44 = (1 - 2β) * y(kLf) - 2β * ydd(kLf)
    a54 = kLf * yd(kLf)
    a64 = 2 * (kLf * yd(kLf) - y(kLf))

    a15 = -2m * (m + 1) * kTs^(-2) * (kTs * yd(kTs) - y(kTs))
    a25 = m * (m + 1) * y(kTs)
    a35 = kTs^2 * ydd(kTs) + (m + 2) * (m - 1) * y(kTs)
    a45 = -2m * (m + 1) * kTf^(-2) * (kTf * yd(kTf) - y(kTf))
    a55 = m * (m + 1) * y(kTf)
    a65 = kTf^2 * ydd(kTf) + (m + 2) * (m - 1) * y(kTf)

    # Interior fluid's own regular solution, at its own wavenumber k3f.
    a46 = j(k3f) * rho_int_over_shell
    a56 = k3f * jd(k3f)

    if m == 0
        A_numerator = zeros(ComplexF64, 4, 4)
        A_numerator[1, 1:3] = [a1, a12, a14]
        A_numerator[2, 1:3] = [a2, a22, a24]
        A_numerator[3, 2:4] = [a42, a44, a46]
        A_numerator[4, 2:4] = [a52, a54, a56]
    else
        A_numerator = zeros(ComplexF64, 6, 6)
        A_numerator[1, 1:5] = [a1, a12, a13, a14, a15]
        A_numerator[2, 1:5] = [a2, a22, a23, a24, a25]
        A_numerator[3, 2:5] = [a32, a33, a34, a35]
        A_numerator[4, 2:6] = [a42, a43, a44, a45, a46]
        A_numerator[5, 2:6] = [a52, a53, a54, a55, a56]
        A_numerator[6, 2:5] = [a62, a63, a64, a65]
    end
    A_denominator = copy(A_numerator)
    A_denominator[1:2, 1] = [a11, a21]

    x = _sphere_interface_solution(A_denominator, -A_numerator[:, 1])
    density = bc.material.interior_coupling === :generalized ?
              bc.interior.density_contrast : 1.0
    return (; scattered = x[1], interior = density * x[end])
end

# Resonance form, see SolidElastic's docstring.
function _modal_coefficient(bc::SolidElastic, m::Integer, k::Real, a::Real)
    ka_sw = k * a
    ka_l = (k / bc.speed_longitudinal_contrast) * a
    ka_t = (k / bc.speed_transversal_contrast) * a
    g = bc.density_contrast

    js_sw = js(m, ka_sw)
    jsd_sw = jsd(m, ka_sw)
    ys_sw = ys(m, ka_sw)
    ysd_sw = ysd(m, ka_sw)
    js_l = js(m, ka_l)
    jsd_l = jsd(m, ka_l)
    js_t = js(m, ka_t)
    jsd_t = jsd(m, ka_t)

    tan_sw = -ka_sw * jsd_sw / js_sw
    tan_l = -ka_l * jsd_l / js_l
    tan_t = -ka_t * jsd_t / js_t
    tan_beta = -ka_sw * ysd_sw / ys_sw
    tan_diff = -js_sw / ys_sw

    along_m = m * (m + 1)
    tan_l_add = tan_l + 1
    tan_t_div = along_m - 1 - ka_t^2 / 2 + tan_t
    numerator = tan_l / tan_l_add - along_m / tan_t_div
    denominator1 = (along_m - ka_t^2 / 2 + 2 * tan_l) / tan_l_add
    denominator2 = along_m * (tan_t + 1) / tan_t_div
    denominator = denominator1 - denominator2
    ratio = -0.5 * ka_t^2 * numerator / denominator
    phi = -ratio / g

    eta_tan = tan_diff * (phi + tan_sw) / (phi + tan_beta)
    cos_eta = 1 / sqrt(1 + eta_tan^2)
    sin_eta = eta_tan * cos_eta

    return sin_eta * (-im * cos_eta - sin_eta)
end

"""
    form_function(boundary::AbstractBoundaryCondition, k, a; angle=π, m_max=default)

Far-field scattering amplitude f(θ) in m of a sphere of radius `a` in m in a
medium with wavenumber `k` in 1/m, evaluated at scattering angle `angle`
in rad (default π, backscatter). `m_max` truncates the modal sum. The
default grows with `ka` and can be overridden for tighter/looser
convergence control.
"""
function form_function(boundary::AbstractBoundaryCondition, k::Real, a::Real;
        angle::Real = π, m_max::Integer = _default_mode_count(k * a))
    x = cos(angle)
    total = zero(ComplexF64)
    for m in 0:m_max
        Am = _modal_coefficient(boundary, m, k, a)
        total += (2m + 1) * legendre_p(m, x) * Am
    end
    return -im / k * total
end

"""
    target_strength(boundary::AbstractBoundaryCondition, k, a; angle=π, m_max=default)

Target strength [dB re 1 m²] of a sphere under modal-series scattering
(rigid, pressure-release, or fluid/gas-filled boundary conditions).
"""
function target_strength(boundary::AbstractBoundaryCondition, k::Real, a::Real; kwargs...)
    return target_strength(form_function(boundary, k, a; kwargs...))
end
