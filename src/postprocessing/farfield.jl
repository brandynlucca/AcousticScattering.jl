# Kirchhoff-Helmholtz far-field extrapolation from an axisymmetric (m = 0) BEM surface solution
# to f(theta) [m], azimuthal integral done in closed form via Jacobi-Anger.

"""
    far_field(ps::Vector{Panel}, p_scat, dpdn_scat, k, theta)

Bistatic far-field scattering amplitude f(theta) [m] extrapolated from an
axial-incidence axisymmetric BEM surface solution, piecewise-constant
surface pressure `p_scat` and its normal derivative `dpdn_scat` on panels
`ps` (as returned by [`solve_axial`](@ref)), at polar angle `theta` [rad]
from the z-axis measured from the +z (incidence) direction, `theta = π` is
backscatter, `theta = 0` is forward scatter.
"""
function far_field(ps::Vector{Panel}, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number}, k::Real, theta::Real)
    total = zero(ComplexF64)
    sinθ, cosθ = sin(theta), cos(theta)
    for (panel, pj, dpdnj) in zip(ps, p_scat, dpdn_scat)
        integrand = s -> begin
            ρt, zt = _panel_point(panel, s)
            u = k * ρt * sinθ
            term1 = (dpdnj + im * k * cosθ * panel.nz * pj) * besselj(0, u)
            term2 = k * sinθ * panel.nrho * pj * besselj(1, u)
            return (term1 + term2) * ρt * cis(-k * zt * cosθ) * panel.L
        end
        val, _ = quadgk(integrand, 0.0, 1.0; rtol = 1e-8)
        total += val
    end
    return -total / 2
end

"""
    target_strength(ps::Vector{Panel}, p_scat, dpdn_scat, k, theta)

Target strength [dB re 1 m²] of an axial-incidence axisymmetric BEM
solution at scattering angle `theta` (see [`far_field`](@ref)).
"""
function target_strength(ps::Vector{Panel}, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number}, k::Real, theta::Real)
    return target_strength(far_field(ps, p_scat, dpdn_scat, k, theta))
end

# --- Bistatic far field from an oblique (multi-Fourier-mode) solution, closed form doesn't apply ---
# Integrates the exact representation formula numerically over meridional and azimuthal directions.

"""
    far_field(ps::Vector{Panel}, p_scat_modes, dpdn_scat_modes, k, theta, phi)

Bistatic far-field scattering amplitude f(theta,phi) [m] extrapolated from an
oblique-incidence axisymmetric BEM solution, `p_scat_modes[m+1]`,
`dpdn_scat_modes[m+1]` are Fourier mode `m`'s piecewise-constant surface
pressure/normal-derivative on panels `ps` (as returned by
[`solve_oblique`](@ref)), at observation direction `(theta, phi)` [rad]
(`theta` from the z-axis, `phi` azimuth measured from the incidence plane).
"""
function far_field(
        ps::Vector{Panel}, p_scat_modes::AbstractVector{<:AbstractVector{<:Number}},
        dpdn_scat_modes::AbstractVector{<:AbstractVector{<:Number}}, k::Real, theta::Real, phi::Real;
        rtol::Real = 1e-6)
    m_max = length(p_scat_modes) - 1
    sinθ, cosθ = sin(theta), cos(theta)
    total = zero(ComplexF64)
    for (j, panel) in enumerate(ps)
        integrand_s = s -> begin
            ρt, zt = _panel_point(panel, s)
            integrand_φ = φ′ -> begin
                p_val = sum(p_scat_modes[m + 1][j] * cos(m * φ′) for m in 0:m_max)
                dpdn_val = sum(dpdn_scat_modes[m + 1][j] * cos(m * φ′) for m in 0:m_max)
                xdotn = sinθ * panel.nrho * cos(φ′ - phi) + cosθ * panel.nz
                xdoty = ρt * sinθ * cos(φ′ - phi) + zt * cosθ
                return (im * k * xdotn * p_val + dpdn_val) * cis(-k * xdoty) * ρt
            end
            val, _ = quadgk(integrand_φ, 0.0, 2π; rtol = rtol)
            return val * panel.L
        end
        val, _ = quadgk(integrand_s, 0.0, 1.0; rtol = rtol)
        total += val
    end
    return -total / (4π)
end

"""
    target_strength(ps::Vector{Panel}, p_scat_modes, dpdn_scat_modes, k, theta, phi)

Target strength [dB re 1 m²] of an oblique-incidence axisymmetric BEM
solution at scattering direction `(theta, phi)` (see [`far_field`](@ref)).
"""
function target_strength(
        ps::Vector{Panel}, p_scat_modes::AbstractVector{<:AbstractVector{<:Number}},
        dpdn_scat_modes::AbstractVector{<:AbstractVector{<:Number}}, k::Real, theta::Real, phi::Real)
    return target_strength(far_field(ps, p_scat_modes, dpdn_scat_modes, k, theta, phi))
end
