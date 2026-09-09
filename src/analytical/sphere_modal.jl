# Exact modal (Mie) series for rigid, pressure-release, and fluid-filled spheres.
# f(θ) = -i/k * Σ_{m=0}^{m_max} (2m+1) Pₘ(cosθ) Aₘ, backscatter is θ = π.

abstract type AbstractBoundaryCondition end

"Interior medium enclosed by a shell layer in a [`Shelled`](@ref) boundary condition."
abstract type AbstractShellInterior end

"Material of the shell layer in a [`Shelled`](@ref) boundary condition."
abstract type AbstractShellMaterial end

"Rigid (fixed, immovable) sphere boundary condition."
struct Rigid <: AbstractBoundaryCondition end

"Pressure-release (soft) sphere boundary condition."
struct PressureRelease <: AbstractBoundaryCondition end

"""
    FluidFilled(density_contrast, soundspeed_contrast; coupling=:full)

Fluid-filled sphere boundary condition (Anderson 1950). Also covers
gas-filled spheres: the boundary-value problem is identical, only the
material contrasts differ.

- `density_contrast`: g = ρ_interior / ρ_exterior
- `soundspeed_contrast`: h = c_interior / c_exterior
- `coupling`: for the spheroid modal series only (ignored for the sphere,
  which is diagonal exactly by symmetry, see sphere_modal.jl and
  spheroid_modal.jl). `:full` solves the complete off-diagonal boundary
  coupling (Furusawa 1988 Eq. 4). `:diagonal` uses the cheaper, less
  accurate closed-form approximation (Eq. 5) that ignores mode coupling
  between the interior and exterior angular functions, with accuracy
  degrading at higher `ka` and larger eccentricity.
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

A *fluid* spherical shell layer (no shear waves, just an ordinary fluid layer with its own
density/sound-speed, e.g. a swim bladder wall modeled as a fluid layer rather than solid tissue).
`density_contrast` = ρ_shell/ρ_exterior, `soundspeed_contrast` = c_shell/c_exterior, both relative
to the *exterior* medium (not to each other). Paired with [`VacuumInterior`](@ref) this is
Jech et al. (2015)'s "fluid shell, pressure release interior" sphere (former `ShellSoft`); paired
with [`FluidInterior`](@ref) it's their "fluid shell, fluid interior" sphere (former
`ShellFluidFilled`, covering both gas-filled-shell and weakly-scattering-shell cases — same
boundary-value problem, different contrasts). See [`Shelled`](@ref).

Derivation (vacuum interior): two fluid regions (exterior `r>a`, shell `b<r<a`), the shell's
general solution `Cⱼₗ(k₂r) + Dyₗ(k₂r)` (both spherical Bessel kinds, since the shell doesn't
include the origin) closed at `r=b` by `p(b)=0`, which gives `D/C` in closed form. Pressure/
velocity continuity at `r=a` then gives `Aₘ` by the same ratio-elimination algebra already used
for [`FluidFilled`](@ref), confirmed directly (not just by construction) to reduce to the exact
[`PressureRelease`](@ref) coefficient in the `radius_ratio → 1` (vanishing shell) limit, an
analytically exact check (a 0/0 cancellation that works out exactly, not merely numerically
close).

Derivation (fluid interior): three fluid regions. The interior (`r<b`) must be regular at the
origin (`jₗ(k₃r)` only, no `yₗ`), giving a closed-form pressure/velocity-continuity ratio at `r=b`
that plays exactly the role the vacuum interior's `p(b)=0` condition does. Both reduce to a single
`D/C` ratio feeding the same outer (`r=a`) algebra, which is why the two interior kinds share their
outer-boundary code. Confirmed directly to reduce to the exact [`FluidFilled`](@ref) coefficient
(with the interior contrasts) in the `radius_ratio → 1` limit.
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

An isotropic elastic spherical shell layer (Goodman & Stern 1962), paired with a
[`FluidInterior`](@ref) via [`Shelled`](@ref) (former `ElasticShell`). All material properties are
*contrasts* relative to the exterior medium (matching [`FluidFilled`](@ref)'s convention), so
`form_function`/`target_strength`'s `k` needs only the exterior medium's wavenumber, not its
absolute density/soundspeed.

- `density_contrast` = ρ_shell / ρ_ext
- `speed_longitudinal_contrast` = c_L / c_ext, `speed_transversal_contrast` = c_T / c_ext, the
  shell's longitudinal and shear wave speeds, related to Lamé parameters `λ, G` and shell density
  `ρ_shell` by `c_L = √((λ+2G)/ρ_shell)`, `c_T = √(G/ρ_shell)`.

Implements the Goodman & Stern (1962) boundary-matching determinant, with the inner-boundary terms
(`a46`, `a56`) evaluated using the interior fluid's own wavenumber/density (not the exterior
medium's), which is what radial-displacement and radial-stress continuity at the shell/interior-
fluid interface actually require. Each mode's coefficient `b_m` is computed as a genuine complex
determinant ratio `det(A_numerator)/det(A_denominator)`, preserving phase for the coherent modal
sum.

Validated (see test/runtests.jl): the stiff/dense-shell limit recovers (to ~0.003 dB at realistic
contrast) the already-validated `Rigid` result regardless of interior-fluid properties. The
complementary thin-shell limit (`radius_ratio → 1` on the enclosing [`Shelled`](@ref), shell
properties matched to make it acoustically near-transparent) trends the right direction but
converges slowly and becomes numerically delicate before fully resolving, the same class of
direct/unnormalized-radial-function fragility documented for the liquid-filled spheroid coupling
in spheroid_modal.jl, not a separate concern.

## `interior_coupling`: generalized vs. original (1962) inner boundary

Goodman & Stern's (1962) own paper states its two fluid media (exterior and interior) "are taken
to be identical ideal fluids" (Sec. II). The published 6×6 determinant's inner-boundary column
(`α₄₆ = jₗ(y)ρI/ρII`, `α₅₆ = y jₗ'(y)`, their Eqs. 7cc-7dd) is evaluated at `y = k(R-Δ)`, the
*exterior* fluid's own wavenumber, and uses the *exterior*/shell density ratio, because interior ≡
exterior was assumed from the outset. It is not a free choice in their formula as published. The
generalization to a distinct interior fluid (interior wavenumber and density used in those same
two terms instead) is a later modification, cited by name in Stanton (1990) ("Sound scattering by
spherical and elongated shelled bodies", JASA 88, 1619-1633, Sec. I.A, referencing Poggio 1969 and
Murphy et al. 1979): "generalized to the case of the inner and outer fluids being different by a
simple modification of the two nonzero terms in the sixth column of each determinant". Stanton's
own numerical results in that paper use the generalized determinants throughout.

`interior_coupling` selects which of these two forms the inner-boundary terms (`a46`, `a56`) use:
- `:generalized` (default): the interior fluid's own `k3f`/density enter `a46`/`a56` (this
  package's existing, already-validated behavior, matching Stanton 1990's generalized
  determinants).
- `:identical_fluid`: reproduces the original 1962 paper exactly. `a46`/`a56` are evaluated using
  the *exterior* medium's wavenumber and density ratio at the inner radius, regardless of what the
  enclosing [`Shelled`](@ref)'s [`FluidInterior`](@ref) contrasts are set to (those are ignored in
  this mode, since the 1962 formula has no place for them, matching the paper's own restriction,
  not an approximation of it).

The two must agree exactly when the interior contrasts are both `1` (interior genuinely is the
exterior fluid), a direct, checkable consistency test, validated in test/runtests.jl alongside a
comparison at real interior/exterior contrast (air- and water-filled shells) quantifying how far
the original, non-generalized formula departs from the physically-correct generalized one whenever
the interior fluid actually differs from the exterior.
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

A viscous fluid-like shell layer (Feuillade & Nero 1998), e.g. the flesh surrounding a fish's
swimbladder. Used as the outer layer of a [`LayeredMaterial`](@ref) (see [`Shelled`](@ref)),
former `ViscoelasticShell`.

- `soundspeed_exterior` [m/s]: the exterior medium's actual sound speed, needed only to recover
  the angular frequency `ω = k·c1` for this layer's frequency-dependent complex wavenumbers
  (Eqs. 6-7); everything else below is a contrast relative to the exterior medium, matching
  [`ElasticLayer`](@ref).
- `density_contrast`, `soundspeed_contrast`: lossless reference density/compressional-speed
  contrasts, `ρ2/ρ1`, `c2/c1`.
- `kinematic_viscosity_compressional` = `ξ2/ρ2` [m²/s] (`ξ2 = η2+4μ2/3`, Eq. 5's combined
  bulk+shear viscosity) and `kinematic_viscosity_shear` = `μ2/ρ2` [m²/s] (shear viscosity alone),
  both *kinematic* (already divided by this layer's density) so they combine with
  `soundspeed_contrast` and `ω` without needing the absolute density separately. Set both to `0.0`
  to recover the lossless (purely elastic, real wavenumbers) limit.
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

Two concentric shell material layers: `outer` ([`ViscousLayer`](@ref) or [`ElasticLayer`](@ref))
enclosing `inner` ([`ElasticLayer`](@ref)), with `radius_ratio` = the inner layer's own outer
radius over the outer layer's outer radius ∈ (0,1). Used as the `material` of a [`Shelled`](@ref)
whose `interior` is enclosed by *two* layers rather than one, e.g. Feuillade & Nero (1998)'s
viscous-flesh-over-elastic-wall swimbladder model (former `ViscoelasticShell`,
`Shelled(LayeredMaterial(ViscousLayer(...), ElasticLayer(...), radius_ratio_wall),
FluidInterior(...), radius_ratio_core)`).
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

Full through-thickness elastic shell material (absolute, not contrast, values), used only by
[`fem`](@ref)`(::Shell, ::Shelled, ...)`'s 2D shell-FEM solve (`method=:thin`/`:general`), where
the shell's own absolute stiffness matters, not a ratio to the exterior fluid. Construct via
[`Shelled`](@ref)`(poisson, density, youngs_modulus)`, former `ShellMaterial`/`ShellFEMMaterial`.

- `poisson`: Poisson's ratio (dimensionless, `< 0.5`).
- `density` [kg/m³]: absolute shell density.
- `youngs_modulus` [Pa]: absolute Young's modulus.
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

Unified thin-shell boundary condition/material — the single public type for every shell concept in
this package, replacing the former `ShellSoft`, `ShellFluidFilled`, `ElasticShell`,
`ViscoelasticShell`, and `ShellMaterial`/`ShellFEMMaterial`.

The three-argument form is a `material` layer ([`FluidLayer`](@ref), [`ElasticLayer`](@ref), or a
[`LayeredMaterial`](@ref) of two layers) enclosing either a [`VacuumInterior`](@ref)
(void/pressure-release interior) or a [`FluidInterior`](@ref) (fluid/gas-filled interior), for use
as a boundary condition with [`modal`](@ref)/[`kirchhoff`](@ref)/[`fem`](@ref)`(::Sphere/Cylinder,
...)`:
- former `ShellSoft`: `Shelled(FluidLayer(...), VacuumInterior(), radius_ratio)`
- former `ShellFluidFilled`: `Shelled(FluidLayer(...), FluidInterior(...), radius_ratio)`
- former `ElasticShell`: `Shelled(ElasticLayer(...), FluidInterior(...), radius_ratio)`
- former `ViscoelasticShell`: `Shelled(LayeredMaterial(ViscousLayer(...), ElasticLayer(...),
  radius_ratio_wall), FluidInterior(...), radius_ratio_core)`

`radius_ratio` = inner/outer shell surface radius ∈ (0,1) (`a` in every
`target_strength`/`form_function` call is the shell's *outer* radius); when `material` is a
`LayeredMaterial`, its own `radius_ratio` (the two layers' interface) must be strictly greater
than the enclosing `Shelled`'s `radius_ratio` (the interior boundary), matching the physical
ordering exterior > outer layer > inner layer > interior.

The one-argument-triple form `Shelled(poisson, density, youngs_modulus)` instead builds an
[`ElasticFEMLayer`](@ref) with no `interior`/`radius_ratio` (`nothing` for both — the full shell
FEM solve gets its geometry from the `Shell` body and its exterior/interior fluid properties from
[`fem`](@ref)'s own arguments, so neither concept applies here), for use with
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

Implements Hickling's (1962) resonance form (`sin η`, `cos η`) rather than
a boundary-matrix determinant like [`ElasticLayer`](@ref)/[`Shelled`](@ref): there's no
interior fluid, so it's a direct ratio of tangent terms. The classical
expression `f_bs = |-2i f_j / (k a)| a/2` with
`f_j = Σ (2m+1) Pₘ(cosθ) sin η (i cos η - sin η)` rearranges to
`-i/k * Σ (2m+1) Pₘ(cosθ) Aₘ` with `Aₘ = sin η (i cos η - sin η)`, which
maps exactly onto this package's existing `form_function`/`target_strength`
convention (taking `abs()` is equivalent to
`target_strength(f) = 20 log10(|f|)` at the end).

Validated (see test/runtests.jl) against the very-high-contrast to `Rigid` limiting case.
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
    ka = k * a
    ka_interior = ka / bc.soundspeed_contrast
    gh = bc.density_contrast * bc.soundspeed_contrast

    jd_int = jsd(m, ka_interior)
    jd_ext = jsd(m, ka)
    j_int = js(m, ka_interior)
    j_ext = js(m, ka)
    y_ext = ys(m, ka)
    yd_ext = ysd(m, ka)

    ratio = jd_int / jd_ext
    numerator = ratio * (y_ext / j_int) - gh * (yd_ext / jd_ext)
    denominator = ratio * (j_ext / j_int) - gh
    C = numerator / denominator

    return -1 / (1 + im * C)
end

# `dc_ratio` is D/C, the shell's two-Bessel-kind solution ratio fixed by the inner surface condition.
function _shell_outer_coefficient(
        m::Integer, ka::Real, k2a::Real, gh_shell::Real, dc_ratio::Number)
    Jb = js(m, k2a) + dc_ratio * ys(m, k2a)
    Jbd = jsd(m, k2a) + dc_ratio * ysd(m, k2a)
    ratio2 = Jbd / (gh_shell * Jb)

    j_ext = js(m, ka)
    jd_ext = jsd(m, ka)
    h_ext = hs(m, ka)
    hd_ext = hsd(m, ka)

    return (ratio2 * j_ext - jd_ext) / (hd_ext - ratio2 * h_ext)
end

function _modal_coefficient(bc::Shelled{FluidLayer, VacuumInterior}, m::Integer, k::Real, a::Real)
    ka = k * a
    k2a = ka / bc.material.soundspeed_contrast
    k2b = k2a * bc.radius_ratio
    gh_shell = bc.material.density_contrast * bc.material.soundspeed_contrast

    dc_ratio = -js(m, k2b) / ys(m, k2b)
    return _shell_outer_coefficient(m, ka, k2a, gh_shell, dc_ratio)
end

function _modal_coefficient(bc::Shelled{FluidLayer, FluidInterior}, m::Integer, k::Real, a::Real)
    ka = k * a
    k2a = ka / bc.material.soundspeed_contrast
    k2b = k2a * bc.radius_ratio
    k3b = (ka / bc.interior.soundspeed_contrast) * bc.radius_ratio
    gh_shell = bc.material.density_contrast * bc.material.soundspeed_contrast
    gh_23 = (bc.interior.density_contrast * bc.interior.soundspeed_contrast) /
            (bc.material.density_contrast * bc.material.soundspeed_contrast)

    j3b = js(m, k3b)
    jd3b = jsd(m, k3b)
    ratio3 = jd3b / (j3b * gh_23)

    j2b = js(m, k2b)
    jd2b = jsd(m, k2b)
    y2b = ys(m, k2b)
    yd2b = ysd(m, k2b)
    dc_ratio = (jd2b - ratio3 * j2b) / (ratio3 * y2b - yd2b)

    return _shell_outer_coefficient(m, ka, k2a, gh_shell, dc_ratio)
end

# λ/(λ+2G) = 1 - 2β, 2G/(λ+2G) = 2β, where β = (c_T/c_L)²
function _modal_coefficient(bc::Shelled{ElasticLayer, FluidInterior}, m::Integer, k::Real, a::Real)
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

    return det(A_numerator) / det(A_denominator)
end

# Hickling (1962) resonance form, see SolidElastic's docstring.
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

    return sin_eta * (im * cos_eta - sin_eta)
end

"""
    form_function(boundary::AbstractBoundaryCondition, k, a; angle=π, m_max=default)

Far-field scattering amplitude f(θ) [m] of a sphere of radius `a` [m] in a
medium with wavenumber `k` [1/m], evaluated at scattering angle `angle`
[rad] (default π = backscatter). `m_max` truncates the modal sum; the
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
