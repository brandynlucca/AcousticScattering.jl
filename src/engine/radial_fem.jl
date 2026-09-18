# Axisymmetric acoustic FEM for a sphere, exploiting exact separability into independent 1D radial
# ODEs per degree l, closed at r = R with an exact Dirichlet-to-Neumann condition.

"""
    radial_fem_coefficient(boundary, l, k, a, R; n_elements=200, order=1)

Scattered-field expansion coefficient `Bₗ` for spherical-harmonic degree
`l`, defined by `p_scat(r,θ) = Σₗ Bₗ hₗ(kr) Pₗ(cosθ)` (the standard Rayleigh-
expansion convention, *not* `sphere_modal.jl`'s `Aₗ`, related by
`Bₗ = (2l+1)iˡ Aₗ`; see [`radial_fem_target_strength`](@ref) for the
matching far-field sum), computed by solving the 1D radial Helmholtz
equation on `[a, R]` via finite elements instead of `sphere_modal.jl`'s
closed-form spherical Bessel/Hankel formula, a genuinely different method
reaching the same physics. `R` is the truncation radius (`R > a`); because
the outer boundary condition is an *exact* DtN closure, accuracy is
controlled by `n_elements` (mesh resolution) and `order`, not by how far
out `R` is. `order=1` uses linear (2-node) elements; `order=2` uses
quadratic (3-node) Lagrangian elements.

For `FluidFilled`, use `n_elements_int=100` and `n_elements_ext=100` to
control the linear meshes on `[0,a]` and `[a,R]`. The interface enforces
pressure and normal-velocity continuity. The coefficient convention is unchanged.
"""
function radial_fem_coefficient(
        boundary::Union{Rigid, PressureRelease, FluidFilled}, l::Integer,
        k::Real, a::Real, R::Real; kwargs...)
    return _radial_fem_mode(boundary, l, k, a, R; kwargs...).coefficient
end

function _radial_fem_mode(
        boundary::Union{Rigid, PressureRelease}, l::Integer, k::Real, a::Real, R::Real;
        n_elements::Integer = 200, order::Integer = 1, solve_reports = nothing)
    if order == 1
        return _radial_fem_mode_linear(
            boundary, l, k, a, R; n_elements, solve_reports)
    elseif order == 2
        return _radial_fem_mode_quadratic(
            boundary, l, k, a, R; n_elements, solve_reports)
    else
        throw(ArgumentError("order must be 1 or 2, got $order"))
    end
end

function _radial_fem_mode_linear(
        boundary::Union{Rigid, PressureRelease}, l::Integer,
        k::Real, a::Real, R::Real; n_elements::Integer = 200, solve_reports = nothing)
    r = collect(range(a, R; length = n_elements + 1))
    n = length(r)

    K = zeros(ComplexF64, n, n)
    b = zeros(ComplexF64, n)

    gq = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
    for e in 1:(n - 1)
        r1, r2 = r[e], r[e + 1]
        h = r2 - r1
        Ig = 0.0   # ∫ r² dr over the element (for the gradient term)
        M11 = 0.0
        M12 = 0.0
        M22 = 0.0  # ∫ N_i N_j r² dr
        for (xi, w) in gq
            rq = (r1 + r2) / 2 + (h / 2) * xi
            jac = h / 2
            N1 = (r2 - rq) / h
            N2 = (rq - r1) / h
            Ig += w * jac * rq^2
            M11 += w * jac * N1 * N1 * rq^2
            M12 += w * jac * N1 * N2 * rq^2
            M22 += w * jac * N2 * N2 * rq^2
        end
        m11p, m12p, m22p = h / 3, h / 6, h / 3  # ∫ N_i N_j dr, closed form

        K[e, e] += Ig / h^2 - k^2 * M11 + l * (l + 1) * m11p
        K[e + 1, e + 1] += Ig / h^2 - k^2 * M22 + l * (l + 1) * m22p
        K[e, e + 1] += -Ig / h^2 - k^2 * M12 + l * (l + 1) * m12p
        K[e + 1, e] += -Ig / h^2 - k^2 * M12 + l * (l + 1) * m12p
    end

    # Outer boundary (r=R): exact DtN closure, R²p'(R)v(R) moved to the LHS.
    dtn = k * hsd(l, k * R) / hs(l, k * R)
    K[n, n] -= R^2 * dtn

    # Inner boundary (r=a).
    ka = k * a
    if boundary isa Rigid
        dpdn_inc_l = im^l * (2l + 1) * k * jsd(l, ka)
        b[1] += a^2 * dpdn_inc_l
    else
        p_inc_l = im^l * (2l + 1) * js(l, ka)
        p_a = -p_inc_l
        for i in 2:n
            b[i] -= K[i, 1] * p_a
            K[i, 1] = 0
        end
        K[1, :] .= 0
        K[1, 1] = 1
        b[1] = p_a
    end

    p = _solve_reported(K, b, solve_reports; mode = l, n_elements, order = 1, R)
    return (; coefficient = p[n] / hs(l, k * R), order = 1,
        exterior = (; radii = r, pressure = p), interior = nothing)
end

# 4-point Gauss-Legendre on [-1,1], exact for the quadratic-element mass matrix below.
const _GQ4 = (
    (-0.8611363115940526, 0.3478548451374538),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538)
)

function _radial_fem_mode_quadratic(
        boundary::Union{Rigid, PressureRelease}, l::Integer,
        k::Real, a::Real, R::Real; n_elements::Integer = 200, solve_reports = nothing)
    r_corner = collect(range(a, R; length = n_elements + 1))
    n = 2 * n_elements + 1  # corner + midside nodes

    K = zeros(ComplexF64, n, n)
    b = zeros(ComplexF64, n)

    for e in 1:n_elements
        r1, r2 = r_corner[e], r_corner[e + 1]
        h = r2 - r1
        idx = (2e - 1, 2e + 1, 2e)  # (left corner, right corner, midside)
        Kloc = zeros(ComplexF64, 3, 3)
        for (xi, w) in _GQ4
            rq = (r1 + r2) / 2 + (h / 2) * xi
            jac = h / 2
            Ns = (xi * (xi - 1) / 2, xi * (xi + 1) / 2, 1 - xi^2)
            dNs = ((2xi - 1) / h, (2xi + 1) / h, -4xi / h)
            for j in 1:3, i in 1:3

                Kloc[i, j] += w * jac *
                              (dNs[i] * dNs[j] * rq^2 -
                               (k^2 * rq^2 - l * (l + 1)) * Ns[i] * Ns[j])
            end
        end
        for j in 1:3, i in 1:3

            K[idx[i], idx[j]] += Kloc[i, j]
        end
    end

    # Outer boundary (r=R, node n): exact DtN closure.
    dtn = k * hsd(l, k * R) / hs(l, k * R)
    K[n, n] -= R^2 * dtn

    # Inner boundary (r=a, node 1).
    ka = k * a
    if boundary isa Rigid
        dpdn_inc_l = im^l * (2l + 1) * k * jsd(l, ka)
        b[1] += a^2 * dpdn_inc_l
    else
        p_inc_l = im^l * (2l + 1) * js(l, ka)
        p_a = -p_inc_l
        for i in 2:n
            b[i] -= K[i, 1] * p_a
            K[i, 1] = 0
        end
        K[1, :] .= 0
        K[1, 1] = 1
        b[1] = p_a
    end

    p = _solve_reported(K, b, solve_reports; mode = l, n_elements, order = 2, R)
    return (; coefficient = p[n] / hs(l, k * R), order = 2,
        exterior = (; radii = collect(range(a, R; length = n)), pressure = p),
        interior = nothing)
end

"""
    radial_fem_target_strength(boundary, k, a, R; m_max=default, n_elements=200)

Backscatter target strength [dB re 1 m²] of an acoustic sphere, computed
from radial finite-element solutions with an exact modal DtN boundary.
The Rayleigh coefficients give `f(θ) = -i/k · Σₗ Bₗ(-i)ˡ Pₗ(cosθ)`;
backscatter uses `θ = π`.
"""
function radial_fem_target_strength(
        boundary::Union{Rigid, PressureRelease, FluidFilled},
        k::Real, a::Real, R::Real; kwargs...)
    return target_strength(_radial_fem_amplitude(
        _radial_fem_modes(boundary, k, a, R; kwargs...), k))
end

function _radial_fem_modes(boundary, k, a, R;
        m_max::Integer = _default_mode_count(k * a), kwargs...)
    return NamedTuple[_radial_fem_mode(boundary, l, k, a, R; kwargs...) for l in 0:m_max]
end

function _radial_fem_amplitude(modes, k)
    total = zero(ComplexF64)
    for (i, mode) in enumerate(modes)
        l = i - 1
        total += mode.coefficient * (-im)^l * legendre_p(l, -1.0)
    end
    return -im / k * total
end

# --- Fluid-filled / transmission boundary: two coupled 1D domains, interior [0, a] and exterior ---
# [a, R], coupled at r=a by pressure and velocity continuity as two extra rows in a block system.

# Bulk (n×n) FEM stiffness matrix for a single domain's radial Helmholtz operator, no boundary
# conditions applied (those are added by the caller).
function _radial_fem_bulk_matrix(l::Integer, k::Real, r::Vector{Float64})
    n = length(r)
    K = zeros(ComplexF64, n, n)
    gq = ((-1 / sqrt(3), 1.0), (1 / sqrt(3), 1.0))
    for e in 1:(n - 1)
        r1, r2 = r[e], r[e + 1]
        h = r2 - r1
        Ig = 0.0
        M11 = 0.0
        M12 = 0.0
        M22 = 0.0
        for (xi, w) in gq
            rq = (r1 + r2) / 2 + (h / 2) * xi
            jac = h / 2
            N1 = (r2 - rq) / h
            N2 = (rq - r1) / h
            Ig += w * jac * rq^2
            M11 += w * jac * N1 * N1 * rq^2
            M12 += w * jac * N1 * N2 * rq^2
            M22 += w * jac * N2 * N2 * rq^2
        end
        m11p, m12p, m22p = h / 3, h / 6, h / 3

        K[e, e] += Ig / h^2 - k^2 * M11 + l * (l + 1) * m11p
        K[e + 1, e + 1] += Ig / h^2 - k^2 * M22 + l * (l + 1) * m22p
        K[e, e + 1] += -Ig / h^2 - k^2 * M12 + l * (l + 1) * m12p
        K[e + 1, e] += -Ig / h^2 - k^2 * M12 + l * (l + 1) * m12p
    end
    return K
end

function _radial_fem_mode(
        boundary::FluidFilled, l::Integer, k::Real, a::Real, R::Real;
        n_elements_int::Integer = 100, n_elements_ext::Integer = 100, solve_reports = nothing)
    g = boundary.density_contrast
    h = boundary.soundspeed_contrast
    k_int = k / h

    r_int = collect(range(0.0, a; length = n_elements_int + 1))
    r_ext = collect(range(a, R; length = n_elements_ext + 1))
    n_i = length(r_int)
    n_e = length(r_ext)

    K_int = _radial_fem_bulk_matrix(l, k_int, r_int)
    K_ext = _radial_fem_bulk_matrix(l, k, r_ext)

    # Exact DtN closure at r=R (same as the single-domain solvers).
    dtn = k * hsd(l, k * R) / hs(l, k * R)
    K_ext[n_e, n_e] -= R^2 * dtn

    # Unknowns: [p_int(1:n_i); p_scat_ext(1:n_e); f_int; f_ext].
    off_pe = n_i
    i_fi = n_i + n_e + 1
    i_fe = n_i + n_e + 2
    ntot = n_i + n_e + 2

    A = zeros(ComplexF64, ntot, ntot)
    bvec = zeros(ComplexF64, ntot)

    # Only the monopole has nonzero pressure at the origin.
    A[1:n_i, 1:n_i] .= K_int
    A[n_i, i_fi] -= 1.0
    if l > 0
        A[1, :] .= 0.0
        A[1, 1] = 1.0
    end

    # Exterior weak form: (K_ext p_scat_ext)[first] = -f_ext.
    A[(n_i + 1):(n_i + n_e), (off_pe + 1):(off_pe + n_e)] .= K_ext
    A[n_i + 1, i_fe] += 1.0

    ka = k * a
    p_inc_l = im^l * (2l + 1) * js(l, ka)
    dpdn_inc_l = im^l * (2l + 1) * k * jsd(l, ka)

    # Pressure continuity: p_int(a) - p_scat_ext(a) = p_inc(a).
    row = n_i + n_e + 1
    A[row, n_i] = 1.0
    A[row, off_pe + 1] = -1.0
    bvec[row] = p_inc_l

    # Velocity continuity: (1/g)f_int - f_ext = a²·∂p_inc/∂r|_a.
    row = n_i + n_e + 2
    A[row, i_fi] = 1.0 / g
    A[row, i_fe] = -1.0
    bvec[row] = a^2 * dpdn_inc_l

    x = _solve_reported(A, bvec, solve_reports; mode = l, n_elements_int, n_elements_ext, R)
    return (; coefficient = x[off_pe + n_e] / hs(l, k * R), order = 1,
        exterior = (; radii = r_ext, pressure = x[(off_pe + 1):(off_pe + n_e)],
            interface_flux = x[i_fe]),
        interior = (; radii = r_int, pressure = x[1:n_i], interface_flux = x[i_fi]))
end

"""
    radial_fem_target_strength_adaptive(boundary, k, a, R; target_tol=0.01, kwargs...)

Double radial element counts until successive target strengths differ by
less than `target_tol` dB. Rigid/pressure-release spheres default to
`n_elements_start=100`, `max_n_elements=8000`, and `order=1`.
Fluid-filled spheres default to 50 and 4000 elements per domain, refining
both linear meshes together. `m_max` controls the modal cutoff separately.
The stopping criterion does not certify complex-amplitude or phase convergence.
"""
function radial_fem_target_strength_adaptive(
        boundary::Union{Rigid, PressureRelease, FluidFilled},
        k::Real, a::Real, R::Real; kwargs...)
    return target_strength(_radial_fem_amplitude(
        _radial_fem_modes_adaptive(boundary, k, a, R; kwargs...), k))
end

function _radial_fem_modes_adaptive(
        boundary::FluidFilled, k::Real, a::Real, R::Real;
        target_tol::Real = 0.01, n_elements_start::Integer = 50,
        max_n_elements::Integer = 4000,
        m_max::Integer = _default_mode_count(k * a), solve_reports = nothing)
    n = n_elements_start
    change_db = nothing
    modes = _radial_fem_modes(
        boundary, k, a, R; m_max = m_max, n_elements_int = n, n_elements_ext = n, solve_reports)
    ts_prev = target_strength(_radial_fem_amplitude(modes, k))
    while n < max_n_elements
        n = min(max_n_elements, 2n)
        modes = _radial_fem_modes(
            boundary, k, a, R; m_max = m_max, n_elements_int = n, n_elements_ext = n, solve_reports)
        ts_new = target_strength(_radial_fem_amplitude(modes, k))
        change_db = abs(ts_new - ts_prev)
        if change_db < target_tol
            _record_refinement!(solve_reports, true, change_db, target_tol, n)
            return modes
        end
        ts_prev = ts_new
        n == max_n_elements && break
    end
    _record_refinement!(solve_reports, false, change_db, target_tol, n)
    @warn "radial_fem_target_strength_adaptive (FluidFilled) did not converge to target_tol=$target_tol dB within max_n_elements=$max_n_elements (ka=$(k*a)), returning the finest solve tried"
    return modes
end

function _radial_fem_modes_adaptive(
        boundary::Union{Rigid, PressureRelease}, k::Real, a::Real, R::Real;
        target_tol::Real = 0.01, n_elements_start::Integer = 100,
        max_n_elements::Integer = 8000,
        m_max::Integer = _default_mode_count(k * a), order::Integer = 1, solve_reports = nothing)
    n = n_elements_start
    change_db = nothing
    modes = _radial_fem_modes(
        boundary, k, a, R; m_max = m_max, n_elements = n, order = order, solve_reports)
    ts_prev = target_strength(_radial_fem_amplitude(modes, k))
    while n < max_n_elements
        n = min(max_n_elements, 2n)
        modes = _radial_fem_modes(
            boundary, k, a, R; m_max = m_max, n_elements = n, order = order, solve_reports)
        ts_new = target_strength(_radial_fem_amplitude(modes, k))
        change_db = abs(ts_new - ts_prev)
        if change_db < target_tol
            _record_refinement!(solve_reports, true, change_db, target_tol, n)
            return modes
        end
        ts_prev = ts_new
        n == max_n_elements && break
    end
    _record_refinement!(solve_reports, false, change_db, target_tol, n)
    @warn "radial_fem_target_strength_adaptive did not converge to target_tol=$target_tol dB within max_n_elements=$max_n_elements (ka=$(k*a)), returning the finest solve tried"
    return modes
end
