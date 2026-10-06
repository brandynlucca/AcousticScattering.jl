"""
    fourier(body::Irregular, boundary::AbstractBoundaryCondition, k;
           incidence_angle=π/2, continuation_steps=8, mapping_order=length(body.rc),
           m_max=_default_mode_count(k*body.a), n_max=m_max, rtol=1e-6, maxevals=1000)

Fourier-matching result, returns an [`FMSolution`](@ref). Conformally maps `body`'s meridian
profile to a coordinate system where the mapped surface is exactly circular, then matches the
boundary condition using spherical wave functions. See [Fourier matching](@ref
fourier-matching-theory). Supports [`Rigid`](@ref), [`PressureRelease`](@ref) and
[`FluidFilled`](@ref) boundaries. `mapping_order` and `continuation_steps` control the conformal
mapping (see `solve_mapping`). `m_max`/`n_max` truncate the modal series. `rtol` and `maxevals` set the tolerance and maximum
node count of the boundary-matching quadrature. Post-process with
[`target_strength`](@ref)`(sol; angle, azimuth)` or [`scattering_amplitude`](@ref)`(sol; angle,
azimuth)`, defaulting to backscatter.

Each solve is repeated at `n_max` and `m_max` reduced by 2. A warning is emitted when the far-field
amplitude changes by more than `1e-2` of its peak, and the change is reported by
[`diagnostics`](@ref) as `convergence`. See [Fourier matching](@ref fourier-matching-theory).
"""
function fourier(body::Irregular, boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, continuation_steps::Integer = 8,
        mapping_order::Integer = max(length(body.rc), 1),
        m_max::Integer = _default_mode_count(k * body.a), n_max::Integer = m_max,
        rtol::Real = 1e-6, maxevals::Integer = 1000)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive, got $k"))
    mapping = solve_mapping(body, mapping_order; continuation_steps)
    is_admissible(mapping) || throw(ArgumentError(
        "Irregular's conformal mapping is inadmissible (Jacobian vanishes somewhere). " *
        "Try a higher mapping_order or more continuation_steps"))
    transition = _boundary_transition(mapping, k, boundary; m_max, n_max, rtol, maxevals)
    a = _incident_coefficients(n_max, m_max, k, incidence_angle)
    b = _apply_transition(transition, a, n_max, m_max)
    b_check = _check_coefficients(transition, k, incidence_angle, n_max, m_max)
    _warn_fm_convergence(_fm_convergence(b, b_check, k))
    return FMSolution(
        body, boundary, Float64(k), mapping, b, Float64(incidence_angle), b_check)
end

"""
    fourier(body::Sphere, boundary, k; kwargs...)
    fourier(body::Spheroid, boundary, k; mapping_order=..., kwargs...)

Convenience overloads for the canonical bodies, matching [`bem`](@ref)/[`mfs`](@ref)'s body-type
coverage. Both convert `body` to an equivalent [`Irregular`](@ref) internally and solve with
[`fourier`](@ref). [`modal`](@ref) is exact and cheaper for these bodies. These overloads exist
for interface consistency and cross-checking, not as the recommended solver. The returned
[`FMSolution`](@ref) keeps the original `Sphere`/`Spheroid` in its `body` field.

A spheroid with aspect ratio above 2.5 defaults to a higher `mapping_order` and a matching `m_max`/`n_max`.
See [Fourier matching](@ref fourier-matching-theory) for the validated aspect-ratio range.
"""
function fourier(body::Sphere, boundary::AbstractBoundaryCondition, k::Real; kwargs...)
    irregular_body = Irregular(body.radius, Float64[], Float64[])
    sol = fourier(irregular_body, boundary, k; kwargs...)
    return FMSolution(body, sol.boundary, sol.k, sol.mapping, sol.b, sol.incidence_angle,
        sol.b_check)
end

const _FM_ELONGATED_ASPECT = 2.5

# Mapping order and mode count for a spheroid. The usable `n_max` is limited by the mapping order, so elongated bodies take a high order and a matching mode count.
function _spheroid_fourier_defaults(body::Spheroid, mapping_order, options)
    aspect = max(body.a, body.b) / min(body.a, body.b)
    aspect > _FM_ELONGATED_ASPECT || return (
        something(mapping_order, clamp(round(Int, 6aspect), 8, 64)), options)
    order = something(mapping_order, clamp(round(Int, 20aspect), 16, 96))
    modes = round(Int, order / (2aspect)) + 2
    (haskey(options, :m_max) || haskey(options, :n_max)) && return order, options
    return order, (; m_max = modes, options...)
end

function fourier(body::Spheroid, boundary::AbstractBoundaryCondition, k::Real;
        mapping_order::Union{Nothing, Integer} = nothing, kwargs...)
    mapping_order, options = _spheroid_fourier_defaults(body, mapping_order, kwargs)
    irregular_body = Irregular(mapping_order) do theta
        1 / sqrt((cos(theta) / body.a)^2 + (sin(theta) / body.b)^2)
    end
    sol = fourier(irregular_body, boundary, k; mapping_order, options...)
    return FMSolution(body, sol.boundary, sol.k, sol.mapping, sol.b, sol.incidence_angle,
        sol.b_check)
end
