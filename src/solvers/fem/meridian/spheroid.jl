# Spheroid geometry and mode sampling for the shared meridian FEM assembly.

"""
    _spheroid_r_inner(θ, a, b)

Distance from the center to a prolate/oblate spheroid's own surface along
the ray at angle `θ` from the x-axis (axis of symmetry, semi-axis `a`), `b` is the equatorial semi-axis, matching [`Spheroid`](@ref)'s convention.
From `(z/a)² + (ρ/b)² = 1` with `z = r cosθ, ρ = r sinθ`:
`r(θ) = 1/√(cos²θ/a² + sin²θ/b²)`.
"""
function _spheroid_r_inner(θ::Real, a::Real, b::Real)
    return 1 / sqrt(cos(θ)^2 / a^2 + sin(θ)^2 / b^2)
end

"""
    _spheroid_meridian_fem_modes(boundary, k, a, b, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Complex exterior surface traces of a rigid/pressure-release prolate/oblate
spheroid (semi-axes `a` along the symmetry axis, `b` equatorial, matching
[`Spheroid`](@ref)) at `incidence_angle` [rad] from the x-axis (`0` =
axial/end-on, `π/2` = broadside), via the 2D `(ρ,z)` meridian FEM described
in the module comment above, decomposed into azimuthal Fourier modes
`m = 0, …, m_max`.
"""
function _spheroid_meridian_fem_modes(
        boundary::Union{Rigid, PressureRelease}, k::Real,
        a::Real, b::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max),
        solve_reports = nothing, retain_fields::Bool = false)
    β = incidence_angle
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    fields = Vector{_MeridianFEMModeField}(undef, m_max + 1)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    # Each mode's FEM assembly+solve is fully independent, safe to parallelize when threads exist.
    if Threads.nthreads() > 1
        Threads.@threads for m in 0:m_max
            _, pR, dpdnR, field = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache; solve_reports)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
            fields[m + 1] = field
        end
    else
        for m in 0:m_max
            _, pR, dpdnR, field = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache; solve_reports)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
            fields[m + 1] = field
        end
    end
    θ = collect(range(0.0, π; length = n_theta + 1))

    mesh_R = MeridianMesh(R .* sin.(θ), R .* cos.(θ))
    ps_R = panels(mesh_R)
    nt1 = size(θ, 1)
    p_panel_modes = [ComplexF64[0.5 * (pm[j] + pm[j + 1]) for j in 1:(nt1 - 1)]
                     for pm in p_modes]
    dpdn_panel_modes = [ComplexF64[0.5 * (dm[j] + dm[j + 1]) for j in 1:(nt1 - 1)]
                        for dm in dpdn_modes]

    return retain_fields ? (ps_R, p_panel_modes, dpdn_panel_modes, fields) :
           (ps_R, p_panel_modes, dpdn_panel_modes)
end

"""
    _spheroid_meridian_fem_modes(boundary::FluidFilled, k, a, b, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Complex exterior surface traces of a fluid/gas-filled prolate/oblate spheroid
at `incidence_angle` [rad] from the x-axis, via the coupled interior/
exterior 2D meridian FEM described in the module comment above.
"""
function _spheroid_meridian_fem_modes(boundary::FluidFilled, k::Real,
        a::Real, b::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max),
        solve_reports = nothing, retain_fields::Bool = false)
    β = incidence_angle
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    fields = Vector{_MeridianFEMModeField}(undef, m_max + 1)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    # See the Rigid/PressureRelease method above for why this is safe to
    # thread and why `θ` is discarded here and recomputed afterward.
    if Threads.nthreads() > 1
        Threads.@threads for m in 0:m_max
            _, pR, dpdnR, field = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache; solve_reports)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
            fields[m + 1] = field
        end
    else
        for m in 0:m_max
            _, pR, dpdnR, field = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache; solve_reports)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
            fields[m + 1] = field
        end
    end
    θ = collect(range(0.0, π; length = n_theta + 1))

    mesh_R = MeridianMesh(R .* sin.(θ), R .* cos.(θ))
    ps_R = panels(mesh_R)
    nt1 = size(θ, 1)
    p_panel_modes = [ComplexF64[0.5 * (pm[j] + pm[j + 1]) for j in 1:(nt1 - 1)]
                     for pm in p_modes]
    dpdn_panel_modes = [ComplexF64[0.5 * (dm[j] + dm[j + 1]) for j in 1:(nt1 - 1)]
                        for dm in dpdn_modes]

    return retain_fields ? (ps_R, p_panel_modes, dpdn_panel_modes, fields) :
           (ps_R, p_panel_modes, dpdn_panel_modes)
end

function _spheroid_fem_mode_trace(boundary::Union{Rigid, PressureRelease, FluidFilled},
        m::Integer, k::Real, β::Real, a::Real, b::Real, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64}; solve_reports = nothing)
    θ = collect(range(0.0, pi; length = n_theta + 1))
    r_inner = [_spheroid_r_inner(t, a, b) for t in θ]
    return _meridian_fem_mode_trace(boundary, m, k, β, θ, r_inner, R,
        n_r, n_theta, l_max, hs_cache, hsd_cache; solve_reports)
end
