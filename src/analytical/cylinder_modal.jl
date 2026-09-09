# Finite cylinder modal series (soft-ended, no cap scattering), arbitrary aspect angle.
# f_bs(θ) = prefactor * (L/π) * sinc(k L cos θ) * Σ_{m=0}^{m_max} Bₘ, TS = 20*log10(|f_bs|).

# DLMF 10.6.2: f'_n(x) = (n/x)f_n(x) - f_{n+1}(x)
jcd(n::Integer, x::Real) = (n / x) * besselj(n, x) - besselj(n + 1, x)
ycd(n::Integer, x::Real) = (n / x) * bessely(n, x) - bessely(n + 1, x)
hcd(n::Integer, x::Real) = (n / x) * besselh(n, 1, x) - besselh(n + 1, 1, x)

function _cylinder_modal_coefficient(::Rigid, m::Integer, k1a::Real)
    return (-1)^m * neumann_factor(m) * (jcd(m, k1a) / hcd(m, k1a))
end

function _cylinder_modal_coefficient(::PressureRelease, m::Integer, k1a::Real)
    return (-1)^m * neumann_factor(m) * (besselj(m, k1a) / besselh(m, 1, k1a))
end

function _cylinder_modal_coefficient(bc::FluidFilled, m::Integer, k1a::Real)
    k2a = k1a / bc.soundspeed_contrast
    gh = bc.density_contrast * bc.soundspeed_contrast

    jd1 = jcd(m, k1a)
    jd2 = jcd(m, k2a)
    j1 = besselj(m, k1a)
    j2 = besselj(m, k2a)
    y1 = bessely(m, k1a)
    yd1 = ycd(m, k1a)

    ratio = jd2 / (j2 * jd1)
    Cm_num = ratio * y1 - gh * (yd1 / jd1)
    Cm_denom = ratio * j1 - gh
    Cm = Cm_num / Cm_denom

    return im^(m + 1) * (-neumann_factor(m) * im^m / (1 + im * Cm))
end

"""
    form_function(boundary::AbstractBoundaryCondition, k, radius, length; aspect_angle=π/2, m_max=default)

Complex backscattering amplitude f_bs [m] of a finite circular cylinder
(rigid, pressure-release, or fluid/gas-filled boundary condition) of the
given `radius` and `length` [m], in a medium with wavenumber `k` [1/m], at
`aspect_angle` [rad] between the cylinder axis and the incident direction
(default π/2 = broadside incidence). `m_max` truncates the modal sum. See
[`target_strength`](@ref) for the dB-scale result. Exposed separately
because [`bcms_target_strength`](@ref) needs the complex amplitude to
apply its bent-cylinder coherence correction before squaring.
"""
function form_function(
        boundary::Union{Rigid, PressureRelease, FluidFilled,
            Shelled{ElasticLayer, FluidInterior}, SolidElastic},
        k::Real, radius::Real, length::Real;
        aspect_angle::Real = π / 2, m_max::Integer = _default_mode_count(k *
                                                                         sin(aspect_angle) *
                                                                         radius))
    k1a = k * sin(aspect_angle) * radius
    k1L = k * length

    x = k1L * cos(aspect_angle)
    length_term = iszero(x) ? one(x) : sin(x) / x

    total = zero(ComplexF64)
    for m in 0:m_max
        total += _cylinder_modal_coefficient(boundary, m, k1a)
    end

    prefactor = boundary isa FluidFilled ? -one(ComplexF64) : im
    return prefactor * (length / π) * length_term * total
end

"""
    target_strength(boundary::AbstractBoundaryCondition, k, radius, length; aspect_angle=π/2, m_max=default)

Backscatter target strength [dB re 1 m²] of a finite circular cylinder.
See [`form_function`](@ref) for the underlying complex amplitude.
"""
function target_strength(
        boundary::Union{Rigid, PressureRelease, FluidFilled,
            Shelled{ElasticLayer, FluidInterior}, SolidElastic},
        k::Real, radius::Real, length::Real; kwargs...)
    return target_strength(form_function(boundary, k, radius, length; kwargs...))
end
