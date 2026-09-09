# Full 2D meridian (ρ,z) axisymmetric acoustic FEM for a finite cylinder, an independent
# numerical cross-check of `cylinder_modal.jl`'s FCMS and of `axisymmetric_bem.jl`'s BEM.

# Wraps Julia's opaque `SingularException` (from `l_max` running well past `k*R`) with context.
function _dtn_fem_solve(K, b, m::Integer, l_max::Integer, kR::Real)
    try
        return K \ b
    catch e
        e isa SingularException || rethrow()
        error("meridian FEM linear system is singular at Fourier mode m=$m, l_max=$l_max, " *
              "k*R=$(round(kR, digits=3)): this is the expected double-precision breakdown of " *
              "the spherical Hankel recursion once l_max exceeds roughly 4-5x k*R (verified " *
              "directly: hs/hsd's real and imaginary parts differ by 100+ orders of magnitude " *
              "there), not a fixable solve, reduce l_max/m_max.")
    end
end

"""
    _graded_theta_grid(n_theta, θ_cap; p=2.5)

Angular grid on `[0,π]` with `n_theta+1` nodes, clustered near the two
corners `θ_cap` and `π-θ_cap` where the cylinder's flat cap meets its
curved side, the finite-element solution has a local singularity there
(a sharp reentrant edge in the meridian cross-section), so uniform
spacing converges very slowly under mesh refinement; power-law grading
within each of the three smooth sub-arcs (cap/side/cap), each clustering
toward the corner(s) it borders, restores good convergence without an
excessive node count (standard treatment for a re-entrant-corner
singularity, see e.g. R. B. Kellogg 1974, "Singularities in interface
problems").
"""
function _graded_theta_grid(n_theta::Integer, θ_cap::Real; p::Real = 2.5)
    n1 = max(1, round(Int, n_theta * θ_cap / π))
    n3 = n1
    n2 = max(1, n_theta - n1 - n3)

    θ = Float64[]
    for i in 0:n1
        t = i / n1
        push!(θ, θ_cap * (1 - (1 - t)^p))
    end
    for i in 1:n2
        t = i / n2
        push!(θ, θ_cap + (π - 2θ_cap) * 0.5 * (1 - cos(t * π)))
    end
    for i in 1:n3
        t = i / n3
        push!(θ, (π - θ_cap) + θ_cap * t^p)
    end
    return θ
end

# Volumetric RHS load for the interior fluid's perturbation field q_int = p_int_total - p_inc.
function _cylinder_fem_interior_source!(
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

# Isoparametric bilinear-quad bulk assembly for one azimuthal Fourier mode `m`'s weak form, shared
# by every region (exterior fluid, and interior fluid via `weight = 1/g`).
function _cylinder_fem_bulk!(Is::Vector{Int}, Js::Vector{Int}, Vs::Vector{ComplexF64},
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

"""
    _cylinder_r_inner(θ, radius, length)

Distance from the center to the finite cylinder's own surface along the
ray at angle `θ` from the z-axis (axis of symmetry), flat cap for `θ`
near the axis, curved side otherwise, matching the same meridian profile
`cylinder_mesh` builds for the BEM (traversed top cap → side → bottom
cap).
"""
function _cylinder_r_inner(θ::Real, radius::Real, length::Real)
    halfL = length / 2
    θ_cap = atan(radius / halfL)
    if θ <= θ_cap
        return halfL / cos(θ)
    elseif θ >= π - θ_cap
        return halfL / (-cos(θ))
    else
        return radius / sin(θ)
    end
end

"""
    cylinder_meridian_fem_target_strength(boundary, k, radius, length, R; n_r=30, n_theta=60, l_max=default)

Backscatter target strength [dB re 1 m²] of a finite rigid/pressure-release
cylinder at *axial* (end-on, `p_inc = e^{ikz}`) incidence, computed via a
genuine 2D `(ρ,z)` isoparametric finite element mesh over the cylinder's
actual exterior shape (flat caps + curved side, exact spherical DtN
truncation at `r=R`), see the module preamble. Cross-checks
[`solve_axial`](@ref)'s BEM (same real geometry, already validated against
`Francis_BEM_TS`) and `cylinder_modal.jl`'s FCMS modal series (same
physics, no end-cap diffraction) as two independent, different-method
comparisons.
"""
function cylinder_meridian_fem_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real,
        radius::Real, length::Real, R::Real;
        n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max))
    nr1 = n_r + 1

    halfL = length / 2
    θ_cap = atan(radius / halfL)
    θ = _graded_theta_grid(n_theta, θ_cap)
    nt1 = size(θ, 1)
    N = nr1 * nt1
    gidx(i, j) = (i - 1) * nt1 + j
    r_inner = [_cylinder_r_inner(θq, radius, length) for θq in θ]

    s_grid = _graded_radial_s(n_r)

    # Node (i,j): physical (ρ,z) at radial layer i, angular layer j.
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

            ρq = sum(Ns[m] * ρc[m] for m in 1:4)
            dρ_dxi = sum(dNdxi[m] * ρc[m] for m in 1:4)
            dρ_deta = sum(dNdeta[m] * ρc[m] for m in 1:4)
            dz_dxi = sum(dNdxi[m] * zc[m] for m in 1:4)
            dz_deta = sum(dNdeta[m] * zc[m] for m in 1:4)
            detJ = dρ_dxi * dz_deta - dρ_deta * dz_dxi

            for jl in 1:4, il in 1:4

                dNi_dρ = (dz_deta * dNdxi[il] - dz_dxi * dNdeta[il]) / detJ
                dNi_dz = (-dρ_deta * dNdxi[il] + dρ_dxi * dNdeta[il]) / detJ
                dNj_dρ = (dz_deta * dNdxi[jl] - dz_dxi * dNdeta[jl]) / detJ
                dNj_dz = (-dρ_deta * dNdxi[jl] + dρ_dxi * dNdeta[jl]) / detJ
                Kloc[il, jl] += w * abs(detJ) * ρq *
                                (
                                    dNi_dρ * dNj_dρ + dNi_dz * dNj_dz -
                                    k^2 * Ns[il] * Ns[jl]
                                )
            end
        end
        for jl in 1:4, il in 1:4

            push!(Is, nodes[il])
            push!(Js, nodes[jl])
            push!(Vs, Kloc[il, jl])
        end
    end

    # Outer boundary (r=R, exactly a sphere): exact nonlocal DtN, identical in form to meridian_fem.jl.
    Q = Matrix{ComplexF64}(undef, l_max + 1, nt1)
    for l in 0:l_max
        Q[l + 1, :] = _theta_hat_integral(θ, θq -> legendre_p(l, cos(θq)))
    end
    dtn_diag = [(2l + 1) / 2 * (k * hsd(l, k * R) / hs(l, k * R)) for l in 0:l_max]
    K_DtN = R^2 .* (Q' * (dtn_diag .* Q))
    idxR = [gidx(nr1, j) for j in 1:nt1]
    for jl in 1:nt1, il in 1:nt1

        push!(Is, idxR[il])
        push!(Js, idxR[jl])
        push!(Vs, -K_DtN[il, jl])
    end

    K = sparse(Is, Js, Vs, N, N)

    # Inner boundary: line-integral treatment using each segment's true (ρ,z) geometry and outward normal.
    idxa = [gidx(1, j) for j in 1:nt1]
    ρb = [ρmat[1, j] for j in 1:nt1]
    zb = [zmat[1, j] for j in 1:nt1]
    if boundary isa Rigid
        # Per-segment normal-weighted RHS via 2-point Gauss/linear-hat quadrature.
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
                dpdn_inc = im * k * nz_seg * cis(k * zq)
                val = dpdn_inc * ρq
                v[e] += w * jac * N1 * val
                v[e + 1] += w * jac * N2 * val
            end
        end
        b[idxa] .+= v
    else
        p_bc = ComplexF64[-cis(k * zb[j]) for j in 1:nt1]
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

    p = K \ b

    pR = p[idxR]
    total = zero(ComplexF64)
    for l in 0:l_max
        cl = (2l + 1) / 2 * dot(Q[l + 1, :], pR)
        Bl = cl / hs(l, k * R)
        total += Bl * (-im)^l * legendre_p(l, -1.0)
    end
    f = -im / k * total
    return target_strength(f)
end

# Oblique (general Fourier mode m) incidence: same Jacobi-Anger decomposition as solve_oblique,
# applied to this file's 2D (ρ,z) FEM; outer DtN uses associated-Legendre Pₗᵐ, p_m ≡ 0 enforced on-axis for m >= 1.

# Mode m's r=R nodal trace (θ, p, ∂p/∂n), generalizing the axial (m=0) solve above.
function _cylinder_fem_mode_trace(
        boundary::Union{Rigid, PressureRelease}, m::Integer, k::Real, β::Real,
        radius::Real, length::Real, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64})
    nr1 = n_r + 1

    halfL = length / 2
    θ_cap = atan(radius / halfL)
    θ = _graded_theta_grid(n_theta, θ_cap)
    nt1 = size(θ, 1)
    N = nr1 * nt1
    gidx(i, j) = (i - 1) * nt1 + j
    r_inner = [_cylinder_r_inner(θq, radius, length) for θq in θ]

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

    _cylinder_fem_bulk!(Is, Js, Vs, ρmat, zmat, n_r, n_theta, gidx, k, 1.0, m2)

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

    p = _dtn_fem_solve(K, b, m, l_max, k * R)

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
    cylinder_meridian_fem_target_strength(boundary, k, radius, length, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Target strength [dB re 1 m²] of a finite rigid/pressure-release cylinder
at `incidence_angle` [rad] from the z-axis (`0` = axial, matching the
4-positional-argument method above; `π/2` = broadside, matching the
`Figure_06_Rigid-Cylinder.csv` `Macaulay_FEM_TS`/`Francis_BEM_TS` reference
convention), computed via the same 2D `(ρ,z)` meridian FEM decomposed into
azimuthal Fourier modes `m = 0, …, m_max` (see the module comment above).
Backscatter is observed at the [`solve_oblique`](@ref)-established
convention `target_strength(ps, p_modes, dpdn_modes, k, π - incidence_angle, π)`.
"""
function cylinder_meridian_fem_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real,
        radius::Real, length::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max))
    β = incidence_angle
    θ = Float64[]
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    for m in 0:m_max
        θ, pR, dpdnR = _cylinder_fem_mode_trace(
            boundary, m, k, β, radius, length, R, n_r, n_theta, l_max, hs_cache, hsd_cache)
        p_modes[m + 1] = pR
        dpdn_modes[m + 1] = dpdnR
    end

    mesh_R = MeridianMesh(R .* sin.(θ), R .* cos.(θ))
    ps_R = panels(mesh_R)
    nt1 = size(θ, 1)
    p_panel_modes = [ComplexF64[0.5 * (pm[j] + pm[j + 1]) for j in 1:(nt1 - 1)]
                     for pm in p_modes]
    dpdn_panel_modes = [ComplexF64[0.5 * (dm[j] + dm[j + 1]) for j in 1:(nt1 - 1)]
                        for dm in dpdn_modes]

    return target_strength(ps_R, p_panel_modes, dpdn_panel_modes, k, π - β, π)
end

# Fluid-filled (and gas-filled) cylinder: interior fluid domain coupled at r_inner(θ), pole handled
# via a collapsed-quadrilateral element, interface via shared-DOF pressure continuity plus an explicit velocity-continuity forcing term (g = ρ_int/ρ_ext ≠ 1).

# Interface forcing (1-1/g)·∮v·(∂p_inc/∂n)·ρ ds along the r_inner(θ) interface curve.
function _cylinder_fem_interface_forcing(ρb::Vector{Float64}, zb::Vector{Float64},
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

# Mode `m`'s r=R scattered-field nodal trace for a FluidFilled cylinder,
# solving the coupled interior/exterior system described above.
function _cylinder_fem_mode_trace(boundary::FluidFilled, m::Integer, k::Real, β::Real,
        radius::Real, length::Real, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64})
    g, h = boundary.density_contrast, boundary.soundspeed_contrast
    k_int = k / h

    halfL = length / 2
    θ_cap = atan(radius / halfL)
    θ = _graded_theta_grid(n_theta, θ_cap)
    nt1 = size(θ, 1)
    r_inner = [_cylinder_r_inner(θq, radius, length) for θq in θ]
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

    _cylinder_fem_bulk!(
        Is, Js, Vs, ρmat_int, zmat_int, n_r, n_theta, gidx_int, k_int, 1 / g, m2)
    _cylinder_fem_bulk!(Is, Js, Vs, ρmat_ext, zmat_ext, n_r, n_theta, gidx_ext, k, 1.0, m2)
    _cylinder_fem_interior_source!(
        b, ρmat_int, zmat_int, n_r, n_theta, gidx_int, (k_int^2 - k^2) / g, m, k, β)

    # Interface velocity-continuity forcing, the interface curve is r_inner(θ), i.e. ρmat_ext[1,:]/zmat_ext[1,:].
    idx_iface = [gidx_ext(1, j) for j in 1:nt1]
    ρ_iface = [ρmat_ext[1, j] for j in 1:nt1]
    z_iface = [zmat_ext[1, j] for j in 1:nt1]
    b[idx_iface] .+= _cylinder_fem_interface_forcing(ρ_iface, z_iface, 1 - 1 / g, m, k, β)

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

    p = _dtn_fem_solve(K, b, m, l_max, k * R)

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
    cylinder_meridian_fem_target_strength(boundary::FluidFilled, k, radius, length, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Target strength [dB re 1 m²] of a finite fluid-filled (or gas-filled)
cylinder at `incidence_angle` [rad] from the z-axis, via the coupled
interior/exterior 2D meridian FEM described above.
"""
function cylinder_meridian_fem_target_strength(boundary::FluidFilled, k::Real,
        radius::Real, length::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max))
    β = incidence_angle
    θ = Float64[]
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    for m in 0:m_max
        θ, pR, dpdnR = _cylinder_fem_mode_trace(
            boundary, m, k, β, radius, length, R, n_r, n_theta, l_max, hs_cache, hsd_cache)
        p_modes[m + 1] = pR
        dpdn_modes[m + 1] = dpdnR
    end

    mesh_R = MeridianMesh(R .* sin.(θ), R .* cos.(θ))
    ps_R = panels(mesh_R)
    nt1 = size(θ, 1)
    p_panel_modes = [ComplexF64[0.5 * (pm[j] + pm[j + 1]) for j in 1:(nt1 - 1)]
                     for pm in p_modes]
    dpdn_panel_modes = [ComplexF64[0.5 * (dm[j] + dm[j + 1]) for j in 1:(nt1 - 1)]
                        for dm in dpdn_modes]

    return target_strength(ps_R, p_panel_modes, dpdn_panel_modes, k, π - β, π)
end
