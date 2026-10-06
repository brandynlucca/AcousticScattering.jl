# Kirchhoff-Helmholtz far-field extrapolation from an axisymmetric (m = 0) BEM surface solution
# to f(theta) [m], azimuthal integral done in closed form via Jacobi-Anger.

"""
    far_field(ps::Vector{Panel}, p_scat, dpdn_scat, k, theta)

Bistatic far-field scattering amplitude f(theta) [m] extrapolated from an
axial-incidence axisymmetric BEM surface solution, piecewise-constant
surface pressure `p_scat` and its normal derivative `dpdn_scat` on panels
`ps` (as returned by [`solve_axial`](@ref)), at polar angle `theta` [rad]
from the x-axis measured from the +x (incidence) direction, `theta = π` is
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
solution at scattering angle `theta` (see `far_field`).
"""
function target_strength(ps::Vector{Panel}, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number}, k::Real, theta::Real)
    return target_strength(far_field(ps, p_scat, dpdn_scat, k, theta))
end

# Bistatic far field from cosine Fourier modes, with the azimuth integrated exactly.

"""
    far_field(ps::Vector{Panel}, p_scat_modes, dpdn_scat_modes, k, theta, phi)

Bistatic far-field scattering amplitude f(theta,phi) [m] extrapolated from an
oblique-incidence axisymmetric BEM solution, `p_scat_modes[m+1]`,
`dpdn_scat_modes[m+1]` are Fourier mode `m`'s piecewise-constant surface
pressure/normal-derivative on panels `ps` (as returned by
[`solve_oblique`](@ref)), at observation direction `(theta, phi)` [rad]
(`theta` from the x-axis, `phi` azimuth measured from the incidence plane).
"""
function far_field(
        ps::Vector{Panel}, p_scat_modes::AbstractVector{<:AbstractVector{<:Number}},
        dpdn_scat_modes::AbstractVector{<:AbstractVector{<:Number}}, k::Real, theta::Real, phi::Real;
        rtol::Real = 1e-6)
    return _far_field_modes(ps, p_scat_modes, dpdn_scat_modes, k, theta, phi, 0; rtol)
end

# The same representation for a contiguous batch of Fourier orders. The public
# evaluator starts at zero; streamed sweeps must retain the actual mode indices.
function _far_field_modes(ps, p_scat_modes, dpdn_scat_modes, k, theta, phi, first_mode;
        rtol = 1e-6, return_error = false)
    m_max = first_mode + length(p_scat_modes) - 1
    sinθ, cosθ = sin(theta), cos(theta)
    weights = [(-im)^m * cos(m * phi) for m in first_mode:m_max]
    total = zero(ComplexF64)
    error = 0.0
    for (j, panel) in enumerate(ps)
        integrand_s = s -> begin
            ρt, zt = _panel_point(panel, s)
            u = k * ρt * sinθ
            # Integral cos(m*φ′)*exp(-im*u*cos(φ′-phi)) dφ′
            # = 2π*(-im)^m*cos(m*phi)*J_m(u). The radial-normal term
            # follows by differentiating in u. Adjacent orders avoid division
            # by u at the poles; evaluate J_m directly, since upward recurrence
            # is unstable when the retained order substantially exceeds |u|.
            previous = first_mode == 0 ? zero(u) : besselj(first_mode - 1, u)
            current, following = besselj(first_mode, u), besselj(first_mode + 1, u)
            value = zero(ComplexF64)
            for m in first_mode:m_max
                derivative = m == 0 ? -following : (previous - following) / 2
                index = m - first_mode + 1
                p, dp = p_scat_modes[index][j], dpdn_scat_modes[index][j]
                value += weights[index] *
                         ((dp + im*k*cosθ*panel.nz*p)*current -
                          k*sinθ*panel.nrho*p*derivative)
                if m < m_max
                    previous, current = current, following
                    following = besselj(m + 2, u)
                end
            end
            return value * ρt * cis(-k*zt*cosθ) * panel.L
        end
        val, estimate = quadgk(integrand_s, 0.0, 1.0; rtol = rtol)
        total += val
        error += estimate
    end
    return return_error ? (-total / 2, error / 2) : -total / 2
end

"""
    target_strength(ps::Vector{Panel}, p_scat_modes, dpdn_scat_modes, k, theta, phi)

Target strength [dB re 1 m²] of an oblique-incidence axisymmetric BEM
solution at scattering direction `(theta, phi)` (see `far_field`).
"""
function target_strength(
        ps::Vector{Panel}, p_scat_modes::AbstractVector{<:AbstractVector{<:Number}},
        dpdn_scat_modes::AbstractVector{<:AbstractVector{<:Number}}, k::Real, theta::Real, phi::Real)
    return target_strength(far_field(ps, p_scat_modes, dpdn_scat_modes, k, theta, phi))
end
