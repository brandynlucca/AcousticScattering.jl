# Full 2D meridian (ρ,z) axisymmetric acoustic FEM for a prolate/oblate spheroid, reusing
# `cylinder_meridian_fem.jl`'s building blocks with a plain uniform θ grid.

"""
    _spheroid_r_inner(θ, a, b)

Distance from the center to a prolate/oblate spheroid's own surface along
the ray at angle `θ` from the z-axis (axis of symmetry, semi-axis `a`), `b` is the equatorial semi-axis, matching [`Spheroid`](@ref)'s convention.
From `(z/a)² + (ρ/b)² = 1` with `z = r cosθ, ρ = r sinθ`:
`r(θ) = 1/√(cos²θ/a² + sin²θ/b²)`.
"""
function _spheroid_r_inner(θ::Real, a::Real, b::Real)
    return 1 / sqrt(cos(θ)^2 / a^2 + sin(θ)^2 / b^2)
end

# Mode `m`'s r=R nodal trace for a rigid/pressure-release spheroid, structurally identical to
# `cylinder_meridian_fem.jl`'s `_cylinder_fem_mode_trace`, with a plain uniform θ grid.
function _spheroid_fem_mode_trace(
        boundary::Union{Rigid, PressureRelease}, m::Integer, k::Real, β::Real,
        a::Real, b::Real, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64})
    nr1 = n_r + 1
    nt1 = n_theta + 1
    N = nr1 * nt1
    gidx(i, j) = (i - 1) * nt1 + j

    θ = collect(range(0.0, π; length = nt1))
    r_inner = [_spheroid_r_inner(θq, a, b) for θq in θ]
    s_grid = _graded_radial_s(n_r)

    ρmat = Matrix{Float64}(undef, nr1, nt1)
    zmat = Matrix{Float64}(undef, nr1, nt1)
    for j in 1:nt1, i in 1:nr1

        s = s_grid[i]
        rij = r_inner[j] + s * (R - r_inner[j])
        ρmat[i, j] = rij * sin(θ[j])
        zmat[i, j] = rij * cos(θ[j])
    end

    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    b_vec = zeros(ComplexF64, N)
    m2 = Float64(m)^2

    _cylinder_fem_bulk!(Is, Js, Vs, ρmat, zmat, n_r, n_theta, gidx, k, 1.0, m2)

    nl = l_max - m + 1
    Peval = Matrix{Float64}(undef, nl, nt1)
    Q = Matrix{ComplexF64}(undef, nl, nt1)
    for (idx, l) in enumerate(m:l_max)
        Q[idx, :] = _theta_hat_integral(θ, θq -> legendre_p(l, m, cos(θq)))
        Peval[idx, :] = [legendre_p(l, m, cos(θq)) for θq in θ]
    end
    dtn_diag = [(2l + 1) / 2 * _legendre_norm_ratio(l, m) *
                (k * hsd_cache[l + 1] / hs_cache[l + 1]) for l in m:l_max]
    K_DtN = R^2 .* (Q' * (dtn_diag .* Q))
    idxR = [gidx(nr1, j) for j in 1:nt1]
    for jl in 1:nt1, il in 1:nt1

        push!(Is, idxR[il])
        push!(Js, idxR[jl])
        push!(Vs, -K_DtN[il, jl])
    end

    K = sparse(Is, Js, Vs, N, N)

    idxa = [gidx(1, j) for j in 1:nt1]
    ρb_ = [ρmat[1, j] for j in 1:nt1]
    zb_ = [zmat[1, j] for j in 1:nt1]
    if boundary isa Rigid
        v = zeros(ComplexF64, nt1)
        gq = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
        for e in 1:(nt1 - 1)
            ρ1, z1 = ρb_[e], zb_[e]
            ρ2, z2 = ρb_[e + 1], zb_[e + 1]
            Δρ, Δz = ρ2 - ρ1, z2 - z1
            L = hypot(Δρ, Δz)
            nρ_seg, nz_seg = -Δz / L, Δρ / L
            for (xi, w) in gq
                t = (xi + 1) / 2
                N1, N2 = 1 - t, t
                ρq = ρ1 + t * Δρ
                zq = z1 + t * Δz
                jac = L / 2
                dpdn_inc = _dpdn_inc_mode(m, k, β, ρq, zq, nρ_seg, nz_seg)
                val = dpdn_inc * ρq
                v[e] += w * jac * N1 * val
                v[e + 1] += w * jac * N2 * val
            end
        end
        b_vec[idxa] .+= v
    else
        p_bc = ComplexF64[-_p_inc_mode(m, k, β, ρb_[j], zb_[j]) for j in 1:nt1]
        for (row, gi) in enumerate(idxa)
            b_vec .-= K[:, gi] .* p_bc[row]
        end
        for gi in idxa
            K[:, gi] .= 0
            K[gi, :] .= 0
        end
        for (row, gi) in enumerate(idxa)
            K[gi, gi] = 1
            b_vec[gi] = p_bc[row]
        end
    end

    if m >= 1
        for i in 1:nr1, j in (1, nt1)

            gi = gidx(i, j)
            K[:, gi] .= 0
            K[gi, :] .= 0
            K[gi, gi] = 1
            b_vec[gi] = 0
        end
    end

    p = _dtn_fem_solve(K, b_vec, m, l_max, k * R)

    pR = p[idxR]
    Bl = ComplexF64[(2l + 1) / 2 * _legendre_norm_ratio(l, m) * dot(Q[idx, :], pR) /
                    hs_cache[l + 1]
                    for (idx, l) in enumerate(m:l_max)]
    dpdnR = ComplexF64[sum(Bl[idx] * k * hsd_cache[l + 1] * Peval[idx, j]
                       for (idx, l) in enumerate(m:l_max))
                       for j in 1:nt1]

    return θ, pR, dpdnR
end

# `hs(l,kR)`/`hsd(l,kR)` for l=0:l_max don't depend on Fourier mode `m`, cached once instead of
# recomputed for every one of the `m_max+1` modes.
function _spherical_hankel_cache(l_max::Integer, kR::Real)
    hs_cache = ComplexF64[hs(l, kR) for l in 0:l_max]
    hsd_cache = ComplexF64[hsd(l, kR) for l in 0:l_max]
    return hs_cache, hsd_cache
end

"""
    spheroid_meridian_fem_target_strength(boundary, k, a, b, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Target strength [dB re 1 m²] of a rigid/pressure-release prolate/oblate
spheroid (semi-axes `a` along the symmetry axis, `b` equatorial, matching
[`Spheroid`](@ref)) at `incidence_angle` [rad] from the z-axis (`0` =
axial/end-on, `π/2` = broadside), via the 2D `(ρ,z)` meridian FEM described
in the module comment above, decomposed into azimuthal Fourier modes
`m = 0, …, m_max`.
"""
function spheroid_meridian_fem_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real,
        a::Real, b::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max))
    β = incidence_angle
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    # Each mode's FEM assembly+solve is fully independent, safe to parallelize when threads exist.
    if Threads.nthreads() > 1
        Threads.@threads for m in 0:m_max
            _, pR, dpdnR = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
        end
    else
        for m in 0:m_max
            _, pR, dpdnR = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
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

    return target_strength(ps_R, p_panel_modes, dpdn_panel_modes, k, π - β, π)
end

# Mode `m`'s r=R trace for a FluidFilled spheroid, structurally identical to the cylinder version.
function _spheroid_fem_mode_trace(boundary::FluidFilled, m::Integer, k::Real, β::Real,
        a::Real, b::Real, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64})
    g, h = boundary.density_contrast, boundary.soundspeed_contrast
    k_int = k / h

    nt1 = n_theta + 1
    θ = collect(range(0.0, π; length = nt1))
    r_inner = [_spheroid_r_inner(θq, a, b) for θq in θ]
    m2 = Float64(m)^2

    nr1_int = n_r + 1
    nr1_ext = n_r + 1

    interface_base = 1 + (n_r - 1) * nt1
    n_ext_only = nr1_ext - 1
    N = interface_base + nt1 + n_ext_only * nt1

    gidx_int(i, j) = i == 1 ? 1 :
                     (i == nr1_int ? interface_base + j : 1 + (i - 2) * nt1 + j)
    gidx_ext(i, j) = i == 1 ? interface_base + j : interface_base + nt1 + (i - 2) * nt1 + j

    s_int = _graded_radial_s_interior(n_r)
    s_ext = _graded_radial_s(n_r)

    ρmat_int = Matrix{Float64}(undef, nr1_int, nt1)
    zmat_int = Matrix{Float64}(undef, nr1_int, nt1)
    for j in 1:nt1, i in 1:nr1_int

        rij = s_int[i] * r_inner[j]
        ρmat_int[i, j] = rij * sin(θ[j])
        zmat_int[i, j] = rij * cos(θ[j])
    end

    ρmat_ext = Matrix{Float64}(undef, nr1_ext, nt1)
    zmat_ext = Matrix{Float64}(undef, nr1_ext, nt1)
    for j in 1:nt1, i in 1:nr1_ext

        rij = r_inner[j] + s_ext[i] * (R - r_inner[j])
        ρmat_ext[i, j] = rij * sin(θ[j])
        zmat_ext[i, j] = rij * cos(θ[j])
    end

    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    b_vec = zeros(ComplexF64, N)

    _cylinder_fem_bulk!(
        Is, Js, Vs, ρmat_int, zmat_int, n_r, n_theta, gidx_int, k_int, 1 / g, m2)
    _cylinder_fem_bulk!(Is, Js, Vs, ρmat_ext, zmat_ext, n_r, n_theta, gidx_ext, k, 1.0, m2)
    _cylinder_fem_interior_source!(
        b_vec, ρmat_int, zmat_int, n_r, n_theta, gidx_int, (k_int^2 - k^2) / g, m, k, β)

    idx_iface = [gidx_ext(1, j) for j in 1:nt1]
    ρ_iface = [ρmat_ext[1, j] for j in 1:nt1]
    z_iface = [zmat_ext[1, j] for j in 1:nt1]
    b_vec[idx_iface] .+= _cylinder_fem_interface_forcing(
        ρ_iface, z_iface, 1 - 1 / g, m, k, β)

    nl = l_max - m + 1
    Peval = Matrix{Float64}(undef, nl, nt1)
    Q = Matrix{ComplexF64}(undef, nl, nt1)
    for (idx, l) in enumerate(m:l_max)
        Q[idx, :] = _theta_hat_integral(θ, θq -> legendre_p(l, m, cos(θq)))
        Peval[idx, :] = [legendre_p(l, m, cos(θq)) for θq in θ]
    end
    dtn_diag = [(2l + 1) / 2 * _legendre_norm_ratio(l, m) *
                (k * hsd_cache[l + 1] / hs_cache[l + 1]) for l in m:l_max]
    K_DtN = R^2 .* (Q' * (dtn_diag .* Q))
    idxR = [gidx_ext(nr1_ext, j) for j in 1:nt1]
    for jl in 1:nt1, il in 1:nt1

        push!(Is, idxR[il])
        push!(Js, idxR[jl])
        push!(Vs, -K_DtN[il, jl])
    end

    K = sparse(Is, Js, Vs, N, N)

    if m >= 1
        for i in 1:nr1_int, j in (1, nt1)

            gi = gidx_int(i, j)
            K[:, gi] .= 0
            K[gi, :] .= 0
            K[gi, gi] = 1
            b_vec[gi] = 0
        end
        for i in 1:nr1_ext, j in (1, nt1)

            gi = gidx_ext(i, j)
            K[:, gi] .= 0
            K[gi, :] .= 0
            K[gi, gi] = 1
            b_vec[gi] = 0
        end
    end

    p = _dtn_fem_solve(K, b_vec, m, l_max, k * R)

    pR = p[idxR]
    Bl = ComplexF64[(2l + 1) / 2 * _legendre_norm_ratio(l, m) * dot(Q[idx, :], pR) /
                    hs_cache[l + 1]
                    for (idx, l) in enumerate(m:l_max)]
    dpdnR = ComplexF64[sum(Bl[idx] * k * hsd_cache[l + 1] * Peval[idx, j]
                       for (idx, l) in enumerate(m:l_max))
                       for j in 1:nt1]

    return θ, pR, dpdnR
end

"""
    spheroid_meridian_fem_target_strength(boundary::FluidFilled, k, a, b, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Target strength [dB re 1 m²] of a fluid/gas-filled prolate/oblate spheroid
at `incidence_angle` [rad] from the z-axis, via the coupled interior/
exterior 2D meridian FEM described in the module comment above.
"""
function spheroid_meridian_fem_target_strength(boundary::FluidFilled, k::Real,
        a::Real, b::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max))
    β = incidence_angle
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    # See the Rigid/PressureRelease method above for why this is safe to
    # thread and why `θ` is discarded here and recomputed afterward.
    if Threads.nthreads() > 1
        Threads.@threads for m in 0:m_max
            _, pR, dpdnR = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
        end
    else
        for m in 0:m_max
            _, pR, dpdnR = _spheroid_fem_mode_trace(
                boundary, m, k, β, a, b, R, n_r, n_theta, l_max, hs_cache, hsd_cache)
            p_modes[m + 1] = pR
            dpdn_modes[m + 1] = dpdnR
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

    return target_strength(ps_R, p_panel_modes, dpdn_panel_modes, k, π - β, π)
end
