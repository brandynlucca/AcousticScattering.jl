# Full 2D meridian (r,θ) axisymmetric acoustic FEM for a sphere. Unlike radial_fem.jl (exact
# angular separability), this meshes both r and θ, carrying genuine angular discretization error.

# 1D FE (linear hat-function) integral ∫ φⱼ(θ) g(θ) sinθ dθ, via 3-point Gauss-Legendre per segment.
function _theta_hat_integral(θ::Vector{Float64}, g::F) where {F}
    nt1 = length(θ)
    v = zeros(ComplexF64, nt1)
    gq = ((-sqrt(3 / 5), 5 / 9), (0.0, 8 / 9), (sqrt(3 / 5), 5 / 9))
    for e in 1:(nt1 - 1)
        θ1, θ2 = θ[e], θ[e + 1]
        h = θ2 - θ1
        for (xi, w) in gq
            θq = (θ1 + θ2) / 2 + (h / 2) * xi
            jac = h / 2
            N1 = (θ2 - θq) / h
            N2 = (θq - θ1) / h
            val = g(θq) * sin(θq)
            v[e] += w * jac * N1 * val
            v[e + 1] += w * jac * N2 * val
        end
    end
    return v
end

"""
    meridian_fem_target_strength(boundary, k, a, R; n_r=40, n_theta=60, l_max=default)

Backscatter target strength [dB re 1 m²] of a rigid or pressure-release
sphere, computed via a genuine 2D (r,θ) meridian finite element mesh, see
the module preamble for why this is a meaningfully different method from
[`radial_fem_target_strength`](@ref) (which never discretizes θ), and what
it's for. `n_r`/`n_theta` are the number of elements in each direction;
`R` is the DtN truncation radius (exact, zero truncation error, see
preamble).
"""
function meridian_fem_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real, a::Real, R::Real;
        n_r::Integer = 40, n_theta::Integer = 60,
        l_max::Integer = _default_mode_count(k * R))
    nr1 = n_r + 1
    nt1 = n_theta + 1
    N = nr1 * nt1
    gidx(i, j) = (i - 1) * nt1 + j

    r = collect(range(a, R; length = nr1))
    θ = collect(range(0.0, π; length = nt1))

    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    sizehint!(Is, 16 * n_r * n_theta)
    sizehint!(Js, 16 * n_r * n_theta)
    sizehint!(Vs, 16 * n_r * n_theta)
    b = zeros(ComplexF64, N)

    gq1 = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
    gq2 = [(x, y, wx * wy) for (x, wx) in gq1, (y, wy) in gq1]

    for ei in 1:n_r, ej in 1:n_theta

        r1, r2 = r[ei], r[ei + 1]
        θ1, θ2 = θ[ej], θ[ej + 1]
        hr = r2 - r1
        hth = θ2 - θ1
        nodes = (gidx(ei, ej), gidx(ei + 1, ej), gidx(ei + 1, ej + 1), gidx(ei, ej + 1))
        Kloc = zeros(ComplexF64, 4, 4)
        for (xi, eta, w) in gq2
            rq = (r1 + r2) / 2 + (hr / 2) * xi
            θq = (θ1 + θ2) / 2 + (hth / 2) * eta
            jac = (hr / 2) * (hth / 2)
            sinθq = sin(θq)

            Ns = ((1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4,
                (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4)
            dNdxi = (-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4)
            dNdeta = (-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4)
            dNdr = dNdxi .* (2 / hr)
            dNdθ = dNdeta .* (2 / hth)

            for jl in 1:4, il in 1:4

                Kloc[il, jl] += w * jac *
                                (
                                    (dNdr[il] * dNdr[jl] * rq^2 + dNdθ[il] * dNdθ[jl]) *
                                    sinθq
                                    -
                                    k^2 * rq^2 * sinθq * Ns[il] * Ns[jl]
                                )
            end
        end
        for jl in 1:4, il in 1:4

            push!(Is, nodes[il])
            push!(Js, nodes[jl])
            push!(Vs, Kloc[il, jl])
        end
    end

    # Outer boundary (r=R): exact nonlocal DtN via Legendre-mode projection.
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

    # Inner boundary (r=a): p_inc(r,θ) = exp(ikr cosθ) (Rayleigh plane-wave
    # sum in closed form), evaluated pointwise instead of mode-by-mode.
    idxa = [gidx(1, j) for j in 1:nt1]
    if boundary isa Rigid
        dpdn_inc = θq -> im * k * cos(θq) * cis(k * a * cos(θq))
        b[idxa] .+= a^2 .* _theta_hat_integral(θ, dpdn_inc)
    else
        p_bc = ComplexF64[-cis(k * a * cos(θq)) for θq in θ]
        # Eliminate all Dirichlet columns' contribution to the RHS using
        # the original (unmodified) K first, then zero rows/columns.
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

    # Far field: project the r=R trace onto Legendre modes (reusing Q),
    # then the same Rayleigh-expansion far-field sum as radial_fem.jl.
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
