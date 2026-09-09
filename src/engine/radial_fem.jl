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
"""
function radial_fem_coefficient(
        boundary::Union{Rigid, PressureRelease}, l::Integer, k::Real, a::Real, R::Real;
        n_elements::Integer = 200, order::Integer = 1)
    if order == 1
        return _radial_fem_coefficient_linear(boundary, l, k, a, R; n_elements = n_elements)
    elseif order == 2
        return _radial_fem_coefficient_quadratic(
            boundary, l, k, a, R; n_elements = n_elements)
    else
        throw(ArgumentError("order must be 1 or 2, got $order"))
    end
end

function _radial_fem_coefficient_linear(
        boundary::Union{Rigid, PressureRelease}, l::Integer,
        k::Real, a::Real, R::Real; n_elements::Integer = 200)
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

    p = K \ b
    return p[n] / hs(l, k * R)
end

# 4-point Gauss-Legendre on [-1,1], exact for the quadratic-element mass matrix below.
const _GQ4 = (
    (-0.8611363115940526, 0.3478548451374538),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538)
)

function _radial_fem_coefficient_quadratic(
        boundary::Union{Rigid, PressureRelease}, l::Integer,
        k::Real, a::Real, R::Real; n_elements::Integer = 200)
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

    p = K \ b
    return p[n] / hs(l, k * R)
end

"""
    radial_fem_target_strength(boundary, k, a, R; m_max=default, n_elements=200)

Backscatter target strength [dB re 1 m²] of a rigid or pressure-release
sphere, computed via [`radial_fem_coefficient`](@ref) mode by mode instead
of the analytical modal series, see the module preamble for why this is a
genuinely independent numerical method (exact DtN truncation, ordinary FEM
discretization error only), not just a re-implementation.

Far-field sum in terms of `radial_fem_coefficient`'s `Bₗ` convention:
asymptotically `hₗ(kr) ~ (-i)^{l+1} e^{ikr}/(kr)`, so matching
`p_scat ~ f(θ)e^{ikr}/r` gives `f(θ) = -i/k · Σₗ Bₗ(-i)ˡ Pₗ(cosθ)`, *not*
`sphere_modal.jl`'s `Σₗ(2l+1)Pₗ(cosθ)Aₗ` form directly, since `Bₗ` already
carries the `(2l+1)iˡ` factor that convention keeps separate (double-
counting it here was an early bug, caught before trusting this against the
modal series, see the validation in test/runtests.jl).
"""
function radial_fem_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real, a::Real, R::Real;
        m_max::Integer = _default_mode_count(k * a), n_elements::Integer = 200, order::Integer = 1)
    total = zero(ComplexF64)
    for l in 0:m_max
        Bl = radial_fem_coefficient(
            boundary, l, k, a, R; n_elements = n_elements, order = order)
        total += Bl * (-im)^l * legendre_p(l, -1.0)
    end
    f = -im / k * total
    return target_strength(f)
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

"""
    radial_fem_coefficient(boundary::FluidFilled, l, k, a, R; n_elements_int=100, n_elements_ext=100)

Coupled two-domain version of [`radial_fem_coefficient`](@ref) for a
fluid-filled sphere, see the module section above this method for the
interior/exterior domain setup and the two coupling conditions at `r=a`.
Returns the same `Bₗ` (exterior Rayleigh-expansion coefficient) convention
as the single-domain method.
"""
function radial_fem_coefficient(
        boundary::FluidFilled, l::Integer, k::Real, a::Real, R::Real;
        n_elements_int::Integer = 100, n_elements_ext::Integer = 100)
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

    # Interior weak form: (K_int p_int)[last] = f_int, elsewhere natural
    # (the r=0 end has no boundary term at all, see the module note).
    A[1:n_i, 1:n_i] .= K_int
    A[n_i, i_fi] -= 1.0

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

    x = A \ bvec
    p_scat_ext_a = x[off_pe + n_e]
    return p_scat_ext_a / hs(l, k * R)
end

"""
    radial_fem_target_strength(boundary::FluidFilled, k, a, R; m_max=default, n_elements_int=100, n_elements_ext=100)

Coupled two-domain version of [`radial_fem_target_strength`](@ref) for a
fluid-filled sphere; see [`radial_fem_coefficient`](@ref)'s `FluidFilled`
method.
"""
function radial_fem_target_strength(boundary::FluidFilled, k::Real, a::Real, R::Real;
        m_max::Integer = _default_mode_count(k * a),
        n_elements_int::Integer = 100, n_elements_ext::Integer = 100)
    total = zero(ComplexF64)
    for l in 0:m_max
        Bl = radial_fem_coefficient(boundary, l, k, a, R; n_elements_int = n_elements_int,
            n_elements_ext = n_elements_ext)
        total += Bl * (-im)^l * legendre_p(l, -1.0)
    end
    f = -im / k * total
    return target_strength(f)
end

"""
    radial_fem_target_strength_adaptive(boundary::FluidFilled, k, a, R; target_tol=0.01, n_elements_start=50, max_n_elements=4000, m_max=default)

Self-checking resolution-doubling wrapper for the `FluidFilled` case,
mirroring [`radial_fem_target_strength_adaptive`](@ref)'s Rigid/
PressureRelease version, doubles both domains' element counts together
each step (the interior domain's own wavenumber `k/soundspeed_contrast`
can be several times larger than the exterior one, e.g. ~4.3× for a
gas-filled sphere, so it is not under-resolved by reusing the same element
*count* even though it needs a finer element *density*).
"""
function radial_fem_target_strength_adaptive(
        boundary::FluidFilled, k::Real, a::Real, R::Real;
        target_tol::Real = 0.01, n_elements_start::Integer = 50,
        max_n_elements::Integer = 4000,
        m_max::Integer = _default_mode_count(k * a))
    n = n_elements_start
    ts_prev = radial_fem_target_strength(
        boundary, k, a, R; m_max = m_max, n_elements_int = n, n_elements_ext = n)
    while n < max_n_elements
        n = min(max_n_elements, 2n)
        ts_new = radial_fem_target_strength(
            boundary, k, a, R; m_max = m_max, n_elements_int = n, n_elements_ext = n)
        abs(ts_new - ts_prev) < target_tol && return ts_new
        ts_prev = ts_new
        n == max_n_elements && break
    end
    @warn "radial_fem_target_strength_adaptive (FluidFilled) did not converge to target_tol=$target_tol dB within max_n_elements=$max_n_elements (ka=$(k*a)), returning the finest solve tried"
    return ts_prev
end

"""
    radial_fem_target_strength_adaptive(boundary, k, a, R; target_tol=0.01, n_elements_start=100, max_n_elements=8000, m_max=default, order=1)

Self-checking resolution-doubling wrapper around
[`radial_fem_target_strength`](@ref): doubles `n_elements` until
successive solves agree to within `target_tol` dB, mirroring
`axisymmetric_bem.jl`'s [`solve_axial_adaptive`](@ref) so neither solver
needs a hand-picked, frequency-independent mesh resolution, a single
fixed `n_elements` that is enough at low `ka` is routinely insufficient at
high `ka` (linear-element FEM error grows with `ka` at fixed resolution),
and unnecessarily fine at low `ka`.
"""
function radial_fem_target_strength_adaptive(
        boundary::Union{Rigid, PressureRelease}, k::Real, a::Real, R::Real;
        target_tol::Real = 0.01, n_elements_start::Integer = 100,
        max_n_elements::Integer = 8000,
        m_max::Integer = _default_mode_count(k * a), order::Integer = 1)
    n = n_elements_start
    ts_prev = radial_fem_target_strength(
        boundary, k, a, R; m_max = m_max, n_elements = n, order = order)
    while n < max_n_elements
        n = min(max_n_elements, 2n)
        ts_new = radial_fem_target_strength(
            boundary, k, a, R; m_max = m_max, n_elements = n, order = order)
        abs(ts_new - ts_prev) < target_tol && return ts_new
        ts_prev = ts_new
        n == max_n_elements && break
    end
    @warn "radial_fem_target_strength_adaptive did not converge to target_tol=$target_tol dB within max_n_elements=$max_n_elements (ka=$(k*a)), returning the finest solve tried"
    return ts_prev
end
