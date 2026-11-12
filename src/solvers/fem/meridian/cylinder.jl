# Full 2D meridian (ρ,z) axisymmetric acoustic FEM for a finite cylinder, an independent
# numerical cross-check of the finite-cylinder modal series and axisymmetric BEM.

"""
    _graded_theta_grid(n_theta, θ_cap; p=2.5)

Angular grid on `[0,π]` with `n_theta+1` nodes, clustered near the two
corners `θ_cap` and `π-θ_cap` where the cylinder's flat cap meets its
curved side, the finite-element solution has a local singularity there
(a sharp reentrant edge in the meridian cross-section), so uniform
spacing converges very slowly under mesh refinement; power-law grading
within each of the three smooth sub-arcs (cap/side/cap), each clustering
toward the corner(s) it borders, restores good convergence without an
excessive node count. See Kellogg (1974), "Singularities in interface problems," for
the treatment of interface singularities.
"""
function _graded_theta_grid(n_theta::Integer, θ_cap::Real; p::Real = 2.5)
    n_theta >= 3 || throw(ArgumentError("n_theta must be at least three"))
    # Reserve a side element even when flat cylinders allocate almost every
    # angular interval to their caps.
    n1 = clamp(round(Int, n_theta * θ_cap / π), 1, (n_theta - 1) ÷ 2)
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

"""
    _cylinder_r_inner(θ, radius, length)

Distance from the center to the finite cylinder's own surface along the
ray at angle `θ` from the x-axis (axis of symmetry), flat cap for `θ`
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

function _cylinder_hp_profile(theta, radius, length)
    r = _cylinder_r_inner(theta, radius, length)
    cap_angle = atan(radius / (length / 2))
    derivative = theta <= cap_angle || theta >= pi - cap_angle ?
                 r * tan(theta) : -r * cos(theta) / sin(theta)
    return r, derivative
end

function _meridian_hp_geometry(body::Cylinder, boundary, R, nr, nt, order; half,
        quadrature_extra::Integer = 0)
    half && throw(ArgumentError("cylinder higher-order geometry uses the full meridian"))
    edges = _graded_theta_grid(nt, atan(body.radius / (body.length / 2)))
    return _spheroid_hp_geometry(boundary, body.length / 2, body.radius, R, nr, nt, order;
        condensed = true, half = false, graded = true, angular_edges = edges, quadrature_extra,
        profile = theta -> _cylinder_hp_profile(theta, body.radius, body.length))
end

function _meridian_hp_dimensions(body::Cylinder)
    (body.length / 2, body.radius, hypot(body.length / 2, body.radius))
end
function _meridian_hp_radius(body::Cylinder, theta)
    _cylinder_r_inner(theta, body.radius, body.length)
end
_meridian_hp_half(body::Cylinder, nt, beta) = false

"""
    cylinder_meridian_fem_target_strength(boundary, k, radius, length, R; n_r=30, n_theta=60, l_max=default)

Backscatter target strength [dB re 1 m²] of a finite rigid/pressure-release
cylinder at *axial* (end-on, `p_inc = e^{ikz}`) incidence, computed via a
genuine 2D `(ρ,z)` isoparametric finite element mesh over the cylinder's
actual exterior shape (flat caps + curved side, exact spherical DtN
truncation at `r=R`). The geometry includes end-cap diffraction.
"""
function cylinder_meridian_fem_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real,
        radius::Real, length::Real, R::Real;
        n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = _default_mode_count(k * R), solve_reports = nothing)
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

    p = _solve_reported(K, b, solve_reports; mode = 0, n_r, n_theta, l_max, R)

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

function _cylinder_fem_mode_workers(ndof, n_modes, mode_threads)
    # Apply the same conservative mesh-size cap as higher-order spheroid FEM.
    # Sparse LU fill varies with geometry, so this is not a process-memory bound.
    memory_workers = max(1, floor(Int, 4 * (50_000 / ndof)^1.5))
    return min(mode_threads, Threads.nthreads(), n_modes, memory_workers)
end

"""
    _cylinder_meridian_fem_modes(boundary, k, radius, length, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default, solve_reports=nothing, retain_fields=false, mode_threads=1)

Panels and per-Fourier-mode complex surface pressure/normal-derivative traces
(`ps, p_modes, dpdn_modes`) of a finite cylinder's oblique 2D `(ρ,z)` meridian FEM solve, shared by
every boundary type here. [`cylinder_meridian_fem_target_strength`](@ref) reduces this to a
single backscatter number; [`far_field`](@ref) reconstructs the complex amplitude at any
observation direction from the same traces. `retain_fields=true` also returns
the nodal exterior/interior fields and outgoing coefficients for point evaluation.
"""
function _cylinder_meridian_fem_modes(
        boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real,
        radius::Real, length::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max),
        solve_reports = nothing, retain_fields::Bool = false, mode_threads::Integer = 1,
        dtn_form::Symbol = :assembled)
    m_max >= 0 || throw(ArgumentError("m_max must be nonnegative"))
    l_max >= m_max || throw(ArgumentError("l_max must be at least m_max"))
    n_r >= 1 || throw(ArgumentError("n_r must be positive"))
    n_theta >= 3 || throw(ArgumentError("n_theta must be at least three"))
    mode_threads >= 1 || throw(ArgumentError("mode_threads must be positive"))
    β = incidence_angle
    θ = _graded_theta_grid(n_theta, atan(radius / (length / 2)))
    r_inner = [_cylinder_r_inner(t, radius, length) for t in θ]
    setup = _meridian_fem_mode_setup(boundary, k, θ, r_inner, R, n_r, n_theta; dtn_form)
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    fields = Vector{_MeridianFEMModeField}(undef, retain_fields ? m_max + 1 : 0)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    workers = _cylinder_fem_mode_workers(size(setup.matrix, 1), m_max + 1, mode_threads)
    function solve_mode(m)
        _, pR, dpdnR, field = _meridian_fem_mode_trace(
            boundary, m, k, β, θ, r_inner, R, n_r, n_theta, l_max, hs_cache, hsd_cache;
            solve_reports, retain_fields, setup, mode_workers = workers, dtn_form)
        p_modes[m + 1] = pR
        dpdn_modes[m + 1] = dpdnR
        retain_fields && (fields[m + 1] = field)
        return nothing
    end
    if workers == 1
        for m in 0:m_max
            solve_mode(m)
        end
    else
        @sync for worker in 1:workers
            Threads.@spawn for m in (worker - 1):workers:m_max
                solve_mode(m)
            end
        end
    end

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
    cylinder_meridian_fem_target_strength(boundary, k, radius, length, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Target strength [dB re 1 m²] of a finite rigid/pressure-release cylinder
at `incidence_angle` [rad] from the x-axis (`0` = axial, matching the
4-positional-argument method above; `π/2` = broadside), computed via the same
2D `(ρ,z)` meridian FEM decomposed into
azimuthal Fourier modes `m = 0, …, m_max` (see the module comment above).
Backscatter is observed at the [`solve_oblique`](@ref)-established
convention `target_strength(ps, p_modes, dpdn_modes, k, π - incidence_angle, π)`.
"""
function cylinder_meridian_fem_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real,
        radius::Real, length::Real, R::Real, incidence_angle::Real; kwargs...)
    ps_R, p_panel_modes, dpdn_panel_modes = _cylinder_meridian_fem_modes(
        boundary, k, radius, length, R, incidence_angle; kwargs...)
    return target_strength(ps_R, p_panel_modes, dpdn_panel_modes, k, π - incidence_angle, π)
end

# Fluid-filled (and gas-filled) cylinder: interior fluid domain coupled at r_inner(θ), pole handled
# via a collapsed-quadrilateral element, interface via shared-DOF pressure continuity plus an explicit velocity-continuity forcing term (g = ρ_int/ρ_ext ≠ 1).

# Mode `m`'s r=R scattered-field nodal trace for a FluidFilled cylinder,
# solving the coupled interior/exterior system described above.

"""
    cylinder_meridian_fem_target_strength(boundary::FluidFilled, k, radius, length, R, incidence_angle; m_max, n_r=30, n_theta=60, l_max=default)

Target strength [dB re 1 m²] of a finite fluid-filled (or gas-filled)
cylinder at `incidence_angle` [rad] from the x-axis, via the coupled
interior/exterior 2D meridian FEM described above.
"""
function cylinder_meridian_fem_target_strength(boundary::FluidFilled, k::Real,
        radius::Real, length::Real, R::Real, incidence_angle::Real; kwargs...)
    ps_R, p_panel_modes, dpdn_panel_modes = _cylinder_meridian_fem_modes(
        boundary, k, radius, length, R, incidence_angle; kwargs...)
    return target_strength(ps_R, p_panel_modes, dpdn_panel_modes, k, π - incidence_angle, π)
end

function _cylinder_fem_mode_trace(boundary::Union{Rigid, PressureRelease, FluidFilled},
        m::Integer, k::Real, β::Real, radius::Real, length::Real, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64}; solve_reports = nothing,
        retain_fields::Bool = true)
    θ = _graded_theta_grid(n_theta, atan(radius / (length / 2)))
    r_inner = [_cylinder_r_inner(t, radius, length) for t in θ]
    return _meridian_fem_mode_trace(boundary, m, k, β, θ, r_inner, R,
        n_r, n_theta, l_max, hs_cache, hsd_cache; solve_reports, retain_fields)
end
