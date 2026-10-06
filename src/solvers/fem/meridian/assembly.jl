# `hs(l,kR)`/`hsd(l,kR)` for l=0:l_max don't depend on Fourier mode `m`, cached once instead of
# recomputed for every one of the `m_max+1` modes.
function _spherical_hankel_cache(l_max::Integer, kR::Real)
    hs_cache = ComplexF64[hs(l, kR) for l in 0:l_max]
    hsd_cache = ComplexF64[hsd(l, kR) for l in 0:l_max]
    return hs_cache, hsd_cache
end

# Shared meridian assembly and boundary-dispatched acoustic solves.

function _dtn_fem_solve(K, b, m::Integer, l_max::Integer, kR::Real;
        solve_reports = nothing, kwargs...)
    try
        return _solve_reported(K, b, solve_reports; mode = m, l_max, kR, kwargs...)
    catch e
        e isa SingularException || rethrow()
        error("meridian FEM linear system is singular at Fourier mode m=$m, l_max=$l_max, " *
              "k*R=$(round(kR, digits=3)): this is the expected double-precision breakdown of " *
              "the spherical Hankel recursion once l_max exceeds roughly 4-5x k*R (verified " *
              "directly: hs/hsd's real and imaginary parts differ by 100+ orders of magnitude " *
              "there), not a fixable solve, reduce l_max/m_max.")
    end
end

# Interior perturbation-field source, q_int = p_int_total - p_inc.
function _meridian_fem_interior_source!(
        b::Vector{ComplexF64}, ρmat::Matrix{Float64}, zmat::Matrix{Float64},
        n_r::Integer, n_theta::Integer, gidx::F,
        coef::Number, m::Integer, k::Real, β::Real) where {F}
    gq1 = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
    gq2 = [(x, y, wx * wy) for (x, wx) in gq1, (y, wy) in gq1]
    for ei in 1:n_r, ej in 1:n_theta

        nodes = (gidx(ei, ej), gidx(ei + 1, ej), gidx(ei + 1, ej + 1), gidx(ei, ej + 1))
        ρc = (ρmat[ei, ej], ρmat[ei + 1, ej], ρmat[ei + 1, ej + 1], ρmat[ei, ej + 1])
        zc = (zmat[ei, ej], zmat[ei + 1, ej], zmat[ei + 1, ej + 1], zmat[ei, ej + 1])
        bloc = zeros(ComplexF64, 4)
        for (xi, eta, w) in gq2
            Ns = ((1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4,
                (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4)
            dNdxi = (-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4)
            dNdeta = (-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4)
            ρq = sum(Ns[m′] * ρc[m′] for m′ in 1:4)
            zq = sum(Ns[m′] * zc[m′] for m′ in 1:4)
            dρ_dxi = sum(dNdxi[m′] * ρc[m′] for m′ in 1:4)
            dρ_deta = sum(dNdeta[m′] * ρc[m′] for m′ in 1:4)
            dz_dxi = sum(dNdxi[m′] * zc[m′] for m′ in 1:4)
            dz_deta = sum(dNdeta[m′] * zc[m′] for m′ in 1:4)
            detJ = dρ_dxi * dz_deta - dρ_deta * dz_dxi
            val = coef * _p_inc_mode(m, k, β, ρq, zq) * ρq
            for il in 1:4
                bloc[il] += w * abs(detJ) * Ns[il] * val
            end
        end
        for il in 1:4
            b[nodes[il]] += bloc[il]
        end
    end
end

# Bilinear-quadrilateral bulk assembly for one azimuthal mode.
function _meridian_fem_bulk!(Is::Vector{Int}, Js::Vector{Int}, Vs::Vector{ComplexF64},
        ρmat::Matrix{Float64}, zmat::Matrix{Float64},
        n_r::Integer, n_theta::Integer, gidx::F,
        k_region::Real, weight::Real, m2::Real) where {F}
    gq1 = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
    gq2 = [(x, y, wx * wy) for (x, wx) in gq1, (y, wy) in gq1]
    for ei in 1:n_r, ej in 1:n_theta

        nodes = (gidx(ei, ej), gidx(ei + 1, ej), gidx(ei + 1, ej + 1), gidx(ei, ej + 1))
        ρc = (ρmat[ei, ej], ρmat[ei + 1, ej], ρmat[ei + 1, ej + 1], ρmat[ei, ej + 1])
        zc = (zmat[ei, ej], zmat[ei + 1, ej], zmat[ei + 1, ej + 1], zmat[ei, ej + 1])

        Kloc = zeros(ComplexF64, 4, 4)
        for (xi, eta, w) in gq2
            Ns = ((1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4,
                (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4)
            dNdxi = (-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4)
            dNdeta = (-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4)

            ρq = sum(Ns[m′] * ρc[m′] for m′ in 1:4)
            dρ_dxi = sum(dNdxi[m′] * ρc[m′] for m′ in 1:4)
            dρ_deta = sum(dNdeta[m′] * ρc[m′] for m′ in 1:4)
            dz_dxi = sum(dNdxi[m′] * zc[m′] for m′ in 1:4)
            dz_deta = sum(dNdeta[m′] * zc[m′] for m′ in 1:4)
            detJ = dρ_dxi * dz_deta - dρ_deta * dz_dxi

            for jl in 1:4, il in 1:4

                dNi_dρ = (dz_deta * dNdxi[il] - dz_dxi * dNdeta[il]) / detJ
                dNi_dz = (-dρ_deta * dNdxi[il] + dρ_dxi * dNdeta[il]) / detJ
                dNj_dρ = (dz_deta * dNdxi[jl] - dz_dxi * dNdeta[jl]) / detJ
                dNj_dz = (-dρ_deta * dNdxi[jl] + dρ_dxi * dNdeta[jl]) / detJ
                Kloc[il, jl] += weight * w * abs(detJ) *
                                (
                                    ρq * (dNi_dρ * dNj_dρ + dNi_dz * dNj_dz -
                                     k_region^2 * Ns[il] * Ns[jl])
                                    +
                                    (m2 / ρq) * Ns[il] * Ns[jl]
                                )
            end
        end
        for jl in 1:4, il in 1:4

            push!(Is, nodes[il])
            push!(Js, nodes[jl])
            push!(Vs, Kloc[il, jl])
        end
    end
end

"""
    _graded_radial_s(n_r; p=2.0)

Radial layer parameter `s ∈ [0,1]` (`0` = inner boundary, `1` = outer
DtN sphere) with `n_r+1` nodes clustered near `s=0`, where the corner
singularity's influence (and the field gradient generally, close to the
scattering body) is strongest.
"""
function _graded_radial_s(n_r::Integer; p::Real = 2.0)
    return [(i / n_r)^p for i in 0:n_r]
end

"""
    _graded_radial_s_interior(n_r; p=2.0)

Radial layer parameter `s ∈ [0,1]` (`0` = the domain's own center, `1` =
its outer edge) with `n_r+1` nodes clustered near `s=1`, the mirror image
of [`_graded_radial_s`](@ref), for [`FluidFilled`](@ref)'s interior region,
where the boundary of interest (the fluid-fluid interface, and the same
corner singularity as the exterior mesh) is the *outer* edge rather than
the inner one.
"""
function _graded_radial_s_interior(n_r::Integer; p::Real = 2.0)
    return [1 - (1 - i / n_r)^p for i in 0:n_r]
end

function _meridian_fem_interface_forcing(ρb::Vector{Float64}, zb::Vector{Float64},
        coef::Number, m::Integer, k::Real, β::Real)
    nt1 = length(ρb)
    v = zeros(ComplexF64, nt1)
    gq = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
    for e in 1:(nt1 - 1)
        ρ1, z1 = ρb[e], zb[e]
        ρ2, z2 = ρb[e + 1], zb[e + 1]
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
            val = coef * dpdn_inc * ρq
            v[e] += w * jac * N1 * val
            v[e + 1] += w * jac * N2 * val
        end
    end
    return v
end

# Shared meridian mode solves. Geometry supplies the angular grid and interface radii.

function _meridian_fem_mode_trace(
        boundary::Union{Rigid, PressureRelease}, m::Integer, k::Real, β::Real,
        θ::Vector{Float64}, r_inner::Vector{Float64}, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64}; solve_reports = nothing)
    nr1 = n_r + 1

    nt1 = size(θ, 1)
    N = nr1 * nt1
    gidx(i, j) = (i - 1) * nt1 + j

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
    b = zeros(ComplexF64, N)
    m2 = Float64(m)^2

    _meridian_fem_bulk!(Is, Js, Vs, ρmat, zmat, n_r, n_theta, gidx, k, 1.0, m2)

    # Outer boundary (r=R): associated-Legendre DtN (reduces exactly to the
    # m=0 case's plain-Legendre DtN above at m=0).
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

    # Inner boundary: line-integral treatment, mode-m incident field via _dpdn_inc_mode/_p_inc_mode.
    idxa = [gidx(1, j) for j in 1:nt1]
    ρb = [ρmat[1, j] for j in 1:nt1]
    zb = [zmat[1, j] for j in 1:nt1]
    if boundary isa Rigid
        v = zeros(ComplexF64, nt1)
        gq = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
        for e in 1:(nt1 - 1)
            ρ1, z1 = ρb[e], zb[e]
            ρ2, z2 = ρb[e + 1], zb[e + 1]
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
        b[idxa] .+= v
    else
        p_bc = ComplexF64[-_p_inc_mode(m, k, β, ρb[j], zb[j]) for j in 1:nt1]
        for (row, gi) in enumerate(idxa)
            b .-= K[:, gi] .* p_bc[row]
        end
        for gi in idxa
            K[:, gi] .= 0
            K[gi, :] .= 0
        end
        for (row, gi) in enumerate(idxa)
            K[gi, gi] = 1
            b[gi] = p_bc[row]
        end
    end

    # Axis regularity (m >= 1): p_m ≡ 0 at ρ=0 on both rays, applied last so it wins at shared nodes.
    if m >= 1
        for i in 1:nr1, j in (1, nt1)

            gi = gidx(i, j)
            K[:, gi] .= 0
            K[gi, :] .= 0
            K[gi, gi] = 1
            b[gi] = 0
        end
    end

    p = _dtn_fem_solve(K, b, m, l_max, k * R; solve_reports, n_r, n_theta)

    pR = p[idxR]
    Bl = ComplexF64[(2l + 1) / 2 * _legendre_norm_ratio(l, m) * dot(Q[idx, :], pR) /
                    hs_cache[l + 1]
                    for (idx, l) in enumerate(m:l_max)]
    dpdnR = ComplexF64[sum(Bl[idx] * k * hsd_cache[l + 1] * Peval[idx, j]
                       for (idx, l) in enumerate(m:l_max))
                       for j in 1:nt1]

    field = _MeridianFEMModeField(θ, s_grid, r_inner, Float64(R),
        permutedims(reshape(p, nt1, nr1)), nothing, nothing, Bl)
    return θ, pR, dpdnR, field
end

function _meridian_fem_mode_trace(boundary::FluidFilled, m::Integer, k::Real, β::Real,
        θ::Vector{Float64}, r_inner::Vector{Float64}, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64}; solve_reports = nothing)
    g, h = boundary.density_contrast, boundary.soundspeed_contrast
    k_int = k / h

    nt1 = size(θ, 1)

    m2 = Float64(m)^2

    nr1_int = n_r + 1
    nr1_ext = n_r + 1

    # Global DOF layout: origin, interior-only layers, shared interface layer, exterior-only layers.
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
    b = zeros(ComplexF64, N)

    _meridian_fem_bulk!(
        Is, Js, Vs, ρmat_int, zmat_int, n_r, n_theta, gidx_int, k_int, 1 / g, m2)
    _meridian_fem_bulk!(Is, Js, Vs, ρmat_ext, zmat_ext, n_r, n_theta, gidx_ext, k, 1.0, m2)
    _meridian_fem_interior_source!(
        b, ρmat_int, zmat_int, n_r, n_theta, gidx_int, (k_int^2 - k^2) / g, m, k, β)

    # Interface velocity-continuity forcing, the interface curve is r_inner(θ), i.e. ρmat_ext[1,:]/zmat_ext[1,:].
    idx_iface = [gidx_ext(1, j) for j in 1:nt1]
    ρ_iface = [ρmat_ext[1, j] for j in 1:nt1]
    z_iface = [zmat_ext[1, j] for j in 1:nt1]
    b[idx_iface] .+= _meridian_fem_interface_forcing(ρ_iface, z_iface, 1 - 1 / g, m, k, β)

    # Outer boundary (r=R): associated-Legendre DtN, as in the Rigid/PressureRelease case above.
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

    # Axis regularity (m ≥ 1): p_m ≡ 0 at ρ=0, in both the interior and
    # exterior meshes (the shared origin DOF included).
    if m >= 1
        for i in 1:nr1_int, j in (1, nt1)

            gi = gidx_int(i, j)
            K[:, gi] .= 0
            K[gi, :] .= 0
            K[gi, gi] = 1
            b[gi] = 0
        end
        for i in 1:nr1_ext, j in (1, nt1)

            gi = gidx_ext(i, j)
            K[:, gi] .= 0
            K[gi, :] .= 0
            K[gi, gi] = 1
            b[gi] = 0
        end
    end

    p = _dtn_fem_solve(K, b, m, l_max, k * R; solve_reports, n_r, n_theta)

    pR = p[idxR]
    Bl = ComplexF64[(2l + 1) / 2 * _legendre_norm_ratio(l, m) * dot(Q[idx, :], pR) /
                    hs_cache[l + 1]
                    for (idx, l) in enumerate(m:l_max)]
    dpdnR = ComplexF64[sum(Bl[idx] * k * hsd_cache[l + 1] * Peval[idx, j]
                       for (idx, l) in enumerate(m:l_max))
                       for j in 1:nt1]

    exterior = ComplexF64[p[gidx_ext(i, j)] for i in 1:nr1_ext, j in 1:nt1]
    interior = ComplexF64[p[gidx_int(i, j)] for i in 1:nr1_int, j in 1:nt1]
    field = _MeridianFEMModeField(θ, s_ext, r_inner, Float64(R), exterior,
        s_int, interior, Bl)
    return θ, pR, dpdnR, field
end
