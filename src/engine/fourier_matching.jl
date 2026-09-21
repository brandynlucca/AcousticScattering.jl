# Fourier matching, extending Reeder & Stanton 2004 (doi:10.1121/1.1648681) and DiPerna & Stanton 1994 (doi:10.1121/1.411243).

using NLsolve: nlsolve
using ADTypes: AutoForwardDiff
using QuadGK: gauss, quadgk

"""
    Irregular(r, order; npoints=8*order+16)
    Irregular(a, rc, rs)

A body of revolution about the z axis, for smooth irregular shapes between the canonical
([`Sphere`](@ref), [`Spheroid`](@ref)) and general numerical (`bem`/`mfs`) bodies. Solve with
[`fourier`](@ref).

The first form Fourier-fits a harmonic `order` series to a user-supplied meridian profile
function `r(theta)` [m] (radial distance from the origin, not from the symmetry axis, at polar
angle `theta`), evaluated over a full `2*pi` period even though only `theta in [0,pi]` is
physically part of the meridian (`theta=0` and `theta=pi` are the two poles on the axis of
symmetry). The full-period definition lets the fit use ordinary (mutually orthogonal on
`[0,2*pi]`) trigonometric projection, via direct numerical quadrature with `npoints` nodes. The
default oversamples relative to `order` so the fit itself is not the dominant source of
reconstruction error.

The second form supplies the Fourier series `R(theta) = a + sum_n [rc[n]*cos(n*theta) +
rs[n]*sin(n*theta)]` (Eq. (21)) directly. `a` is the mean radius, and `rc[n]`/`rs[n]` are the
deviation coefficients for harmonic order `n = 1:length(rc)`.
"""
struct Irregular <: AbstractBody
    a::Float64
    rc::Vector{Float64}
    rs::Vector{Float64}
    function Irregular(a::Real, rc::AbstractVector{<:Real}, rs::AbstractVector{<:Real})
        length(rc) == length(rs) ||
            throw(ArgumentError("rc and rs must have the same length"))
        isfinite(a) && a > 0 ||
            throw(ArgumentError("a must be finite and positive, got $a"))
        return new(Float64(a), Float64.(rc), Float64.(rs))
    end
end

function Irregular(r, order::Integer; npoints::Integer = 8 * order + 16)
    order >= 0 || throw(ArgumentError("order must be nonnegative, got $order"))
    nodes, weights = _full_period_quadrature(npoints)
    values = r.(nodes)
    a = sum(weights .* values) / (2π)
    rc = zeros(Float64, order)
    rs = zeros(Float64, order)
    for n in 1:order
        rc[n] = sum(weights .* values .* cos.(n .* nodes)) / π
        rs[n] = sum(weights .* values .* sin.(n .* nodes)) / π
    end
    return Irregular(a, rc, rs)
end

"""
    profile_radius(body, theta)

Evaluate an [`Irregular`](@ref)'s meridian profile `R(theta)` [m] (Eq. (21)) at polar angle
`theta` [rad].
"""
function profile_radius(body::Irregular, theta::Real)
    r = body.a
    for n in eachindex(body.rc)
        r += body.rc[n] * cos(n * theta) + body.rs[n] * sin(n * theta)
    end
    return r
end

function _full_period_quadrature(npoints::Integer)
    nodes, weights = gauss(npoints, 0, 2π)
    return nodes, weights
end

"""
    ConformalMapping(body, delta_c, delta_s, c)

The solved conformal mapping (Eqs. (24)-(30)) for an [`Irregular`](@ref), holding the
angle-correction series coefficients `delta_c[l]`, `delta_s[l]` (`l = 1:length(delta_c)`) in
`theta(w) = w + sum_l [delta_c[l]*cos(l*w) + delta_s[l]*sin(l*w)]` (Eq. (29)), and the resulting
exterior mapping coefficients `c[n+2]` for `n = -1:(length(c)-2)` (Eq. (25)), indexed this way
so `c[1]` is `c_{-1}` and `c[n+2]` is `c_n` for `n >= 0`.
"""
struct ConformalMapping
    body::Irregular
    delta_c::Vector{Float64}
    delta_s::Vector{Float64}
    c::Vector{ComplexF64}
end

"""
    mapping_theta(mapping, w)

Evaluate `theta(w)` [rad] (Eq. (29)) for the solved [`ConformalMapping`](@ref).
"""
function mapping_theta(mapping::ConformalMapping, w::Real)
    theta = w
    for l in eachindex(mapping.delta_c)
        theta += mapping.delta_c[l] * cos(l * w) + mapping.delta_s[l] * sin(l * w)
    end
    return theta
end

"""
    mapping_surface(mapping, w; u=0.0)

Evaluate the mapped surface `(g(u,w), f(u,w))` (Eqs. (26)-(27)) at radial mapping coordinate `u`
(default `0`, the scatterer surface) and angular coordinate `w` [rad]. Returns `(g, f)`, where `g`
is the axial (z) coordinate and `f` is the radial (distance-from-axis) coordinate, matching
[`Irregular`](@ref)'s `(z,rho) = (R*cos(theta), R*sin(theta))` convention.
"""
function mapping_surface(mapping::ConformalMapping, w::Real; u::Real = 0.0)
    c = mapping.c
    G = c[1] * cis(w) * exp(u)
    for n in 0:(length(c) - 2)
        G += c[n + 2] * cis(-n * w) * exp(-n * u)
    end
    return real(G), imag(G)
end

# G'(rho) = dG/du (Eqs. (26)-(27)'s derivative w.r.t. u), shared by mapping_jacobian_squared and the rigid boundary's normal-derivative terms.
function _mapping_derivative(mapping::ConformalMapping, w::Real; u::Real = 0.0)
    c = mapping.c
    dG = c[1] * cis(w) * exp(u)
    for n in 0:(length(c) - 2)
        dG -= n * c[n + 2] * cis(-n * w) * exp(-n * u)
    end
    return dG
end

"""
    mapping_jacobian_squared(mapping, w; u=0.0)

Squared modulus of the conformal mapping's derivative, `|G'(rho)|^2` (Eq. (31)), at `(u,w)`. The
mapping is only admissible where this stays strictly positive for `u >= 0` (Eq. (31)). It is
identically the (nonnegative) Jacobian of the coordinate transformation (Eq. (21) of DiPerna and
Stanton 1994).
"""
function mapping_jacobian_squared(mapping::ConformalMapping, w::Real; u::Real = 0.0)
    return abs2(_mapping_derivative(mapping, w; u))
end

# Eq. (30)'s boundary-value function. `scale<1` is DiPerna Appendix A's continuation homotopy.
function _boundary_value(body::Irregular, delta_c, delta_s, w::Real, scale::Real)
    theta = w
    for l in eachindex(delta_c)
        theta += delta_c[l] * cos(l * w) + delta_s[l] * sin(l * w)
    end
    eith = cis(theta)
    value = body.a * eith
    for n in eachindex(body.rc)
        Rn = complex(body.rc[n], body.rs[n]) / 2
        value += scale * (conj(Rn) * eith^(1 + n) + Rn * eith^(1 - n))
    end
    return value
end

# Real-vector NLsolve residual packing Eq. (30)'s j>1 complex constraints I_j(delta)=0.
function _mapping_residual!(F, x, body::Irregular, order::Integer,
        scale::Real, nodes, weights)
    delta_c = @view x[1:order]
    delta_s = @view x[(order + 1):(2order)]
    for j in 2:(order + 1)
        total = zero(ComplexF64)
        for (w, wt) in zip(nodes, weights)
            total += wt * cis(-j * w) * _boundary_value(body, delta_c, delta_s, w, scale)
        end
        total /= 2π
        k = 2 * (j - 2)
        F[k + 1] = real(total)
        F[k + 2] = imag(total)
    end
    return F
end

"""
    solve_mapping(body, order; continuation_steps=8, npoints=8*order+16,
                  nlsolve_kwargs=(; autodiff=AutoForwardDiff()))

Solve for the exterior conformal mapping of an [`Irregular`](@ref), returning a
[`ConformalMapping`](@ref). `order` truncates both the angle-correction series `delta_c`/`delta_s`
(Eq. (29)) and the mapping coefficients `c` at the same order (matching the `order` complex
constraints `j=2:(order+1)` to the `order` complex unknowns `delta_l`).

Solves the nonlinear system (Eq. (30)) by Newton-Raphson (`NLsolve.jl`, forward-mode automatic
differentiation for the Jacobian), using the continuation/homotopy procedure of DiPerna and
Stanton (1994), Appendix A, Eq. (A9) for profiles far from circular. The harmonic (non-mean-radius)
part of the target profile is scaled by `l/continuation_steps` for `l=1:continuation_steps`,
each step using the previous step's converged `delta` as its initial guess (the first step starts
from `delta=0`). `continuation_steps=1` recovers plain Newton-Raphson from a zero initial guess,
adequate only for profiles already close to circular.

Once `delta` is converged at full strength (`scale=1`), the mapping coefficients `c` follow
directly (no further solve) from the same boundary-value integral's `j<=1` branch (Eq. (30)).
"""
function solve_mapping(body::Irregular, order::Integer;
        continuation_steps::Integer = 8, npoints::Integer = 8 * order + 16,
        nlsolve_kwargs = (; autodiff = AutoForwardDiff()))
    order >= 1 || throw(ArgumentError("order must be at least 1, got $order"))
    continuation_steps >= 1 ||
        throw(ArgumentError("continuation_steps must be at least 1, got $continuation_steps"))
    nodes, weights = _full_period_quadrature(npoints)
    x = zeros(Float64, 2order)
    for step in 1:continuation_steps
        scale = step / continuation_steps
        result = nlsolve(
            (F, y) -> _mapping_residual!(F, y, body, order, scale, nodes, weights),
            x; nlsolve_kwargs...)
        result.f_converged || result.x_converged ||
            throw(ArgumentError(
                "conformal mapping failed to converge at continuation step $step/$continuation_steps " *
                "(scale=$scale); try more continuation_steps or a lower mapping order"))
        x = result.zero
    end
    delta_c = x[1:order]
    delta_s = x[(order + 1):(2order)]
    c = Vector{ComplexF64}(undef, order + 2)
    for (idx, j) in enumerate(1:-1:(-(order)))
        total = zero(ComplexF64)
        for (w, wt) in zip(nodes, weights)
            total += wt * cis(-j * w) * _boundary_value(body, delta_c, delta_s, w, 1.0)
        end
        c[idx] = total / 2π
    end
    return ConformalMapping(body, delta_c, delta_s, c)
end

"""
    is_admissible(mapping; u=0.0, nsamples=360)

Check the conformal mapping's admissibility (Eq. (31)). Returns `true` if `mapping_jacobian_squared`
stays strictly positive at `u` over `nsamples` points spanning `w in [0,2*pi]`, `false` otherwise.
A `false` result means the mapping should be rejected, so do not use its geometry for scattering.
This is only a finite-sample check, not a proof for all `w`.
"""
function is_admissible(mapping::ConformalMapping; u::Real = 0.0, nsamples::Integer = 360)
    for w in range(0, 2π; length = nsamples + 1)[1:(end - 1)]
        mapping_jacobian_squared(mapping, w; u) > 0 || return false
    end
    return true
end

# --- Pressure-release boundary matching (Eqs. 33-54): incidence azimuth fixed at 0, m>=0 only ---

"""
    _incident_coefficients(n_max, m_max, k, incidence_angle)

Incident-wave modal coefficients `a[n+1,m+1]` (Eq. (35)) for a unit-amplitude plane wave from
polar angle `incidence_angle` [rad] (`theta_0`, measured from the axis of symmetry, where `0` is
end-on incidence along `+z`) at azimuth `0`, truncated to `n=0:n_max`, `m=0:m_max`
(`m_max <= n_max`, and entries with `m>n` are `0`).
"""
function _incident_coefficients(n_max::Integer, m_max::Integer, k::Real, incidence_angle::Real)
    a = zeros(ComplexF64, n_max + 1, m_max + 1)
    x0 = cos(incidence_angle)
    for n in 0:n_max, m in 0:min(n, m_max)
        # _legendre_norm_ratio(n,m) is already Gamma(n-m+1)/Gamma(n+m+1) (Eq. (35)), so multiply.
        a[n + 1, m + 1] = im^n * neumann_factor(m) * (2n + 1) * _legendre_norm_ratio(n, m) *
                          legendre_p(n, m, x0)
    end
    return a
end

"""
    _boundary_matrices(mapping, k, m; n_max, rtol=1e-8)

Assemble the pressure-release boundary matrices `R`, `Q` (Eqs. (52)-(53)) for one azimuthal
order `m`, truncated to test/source orders `n, n2 = m:n_max`. `R[i,j]`/`Q[i,j]` correspond to
test order `n = m+i-1` and source order `n2 = m+j-1`. Eqs. (52)-(53) print both indices as "n",
but Eq. (51)'s explicit sum over `n` and Eq. (54)'s explicit matrix inverse only make sense if
these are two independent (test, source) indices, so that is how they are implemented here.
"""
function _boundary_matrices(mapping::ConformalMapping, k::Real, m::Integer;
        n_max::Integer, rtol::Real = 1e-6, maxevals::Integer = 1000)
    ns = m:n_max
    nn = length(ns)
    R = zeros(ComplexF64, nn, nn)
    Q = zeros(ComplexF64, nn, nn)
    for (i, n) in enumerate(ns), (j, n2) in enumerate(ns)

        (RQ, _) = quadgk(0.0, π; rtol, maxevals) do w
            g, f = mapping_surface(mapping, w)
            r = hypot(f, g)
            # Clamp roundoff at the poles (w=0,pi), where legendre_p needs |g/r|<=1 exactly.
            costheta = clamp(g / r, -1.0, 1.0)
            weight = legendre_p(n, m, cos(w)) * sin(w) * legendre_p(n2, m, costheta)
            kr = k * r
            SVector(js(n2, kr) * weight, hs(n2, kr) * weight)
        end
        R[i, j], Q[i, j] = RQ
    end
    return R, Q
end

"""
    _equilibrated_solve(A, rhs)

Row/column-equilibrated truncated-SVD solve of `A*x=rhs`, extending
`_sphere_interface_solution`'s scaling (`sphere_modal.jl`) with a `pinv` in place of the final
backslash. `R_n^m`, `Q_n^m`, `S_n^m` and their primed counterparts span many orders of magnitude
across `n` purely from spherical Hankel-function growth, which is physical, not numerical. An
unscaled `pinv` would wrongly truncate the small-but-dominant low-`n` terms as noise, since it
only compares singular values against the matrix's own largest one.
"""
function _equilibrated_solve(A::AbstractMatrix, rhs)
    col_scale = vec(maximum(abs, A; dims = 1))
    scaled = A ./ col_scale'
    row_scale = vec(maximum(abs, scaled; dims = 2))
    equilibrated = scaled ./ row_scale
    return (pinv(equilibrated) * (rhs ./ row_scale)) ./ col_scale
end

"""
    _pressure_release_transition(mapping, k; m_max, n_max, rtol=1e-6, maxevals=1000)

The pressure-release transition operator (Eq. (54), one `T_m = -Q^+ R` matrix per azimuthal
order `m`, `b[:,m] = T_m * a[:,m]`), independent of incidence angle. Assembling this once and
reusing it across an incidence-angle sweep is this method's whole "transition matrix" appeal
(DiPerna and Stanton 1994's own framing, `docs/DEVELOPMENT_PRIORITIES.md` #6): it is the
expensive part, since it needs the boundary-matching quadrature, while [`_incident_coefficients`](@ref)
is cheap.
"""
function _pressure_release_transition(mapping::ConformalMapping, k::Real;
        m_max::Integer, n_max::Integer, rtol::Real = 1e-6, maxevals::Integer = 1000)
    return map(0:m_max) do m
        R, Q = _boundary_matrices(mapping, k, m; n_max, rtol, maxevals)
        -_equilibrated_solve(Q, R)
    end
end

"""
    solve_pressure_release(mapping, k, incidence_angle; m_max=_default_mode_count(k*mapping.body.a),
                           n_max=m_max, rtol=1e-6, maxevals=1000)

Solve the pressure-release (soft/Dirichlet) Fourier-matching scattering problem (Eqs. (33)-(54))
for a unit-amplitude plane wave incident at polar angle `incidence_angle` [rad], azimuth `0` (see
the module-level convention notes above). Returns the scattered-field coefficient matrix
`b[n+1,m+1]` for `n=0:n_max`, `m=0:m_max` (entries with `n<m` are `0`), for use with
[`fourier_matching_amplitude`](@ref). Solves each azimuthal order's system with
[`_equilibrated_solve`](@ref), since these matrices span many orders of magnitude across `n`
from spherical Hankel-function growth alone, not from genuine ill-conditioning. For repeated
incidence angles at the same `mapping`/`k`, build [`_pressure_release_transition`](@ref) once
instead of calling this repeatedly.
"""
function solve_pressure_release(mapping::ConformalMapping, k::Real, incidence_angle::Real;
        m_max::Integer = _default_mode_count(k * mapping.body.a),
        n_max::Integer = m_max, rtol::Real = 1e-6, maxevals::Integer = 1000)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive, got $k"))
    n_max >= m_max >= 0 ||
        throw(ArgumentError("require n_max >= m_max >= 0, got n_max=$n_max, m_max=$m_max"))
    transition = _pressure_release_transition(mapping, k; m_max, n_max, rtol, maxevals)
    a = _incident_coefficients(n_max, m_max, k, incidence_angle)
    return _apply_transition(transition, a, n_max, m_max)
end

# Apply a per-m transition operator (Vector{Matrix{ComplexF64}}) to incident coefficients a[n+1,m+1].
function _apply_transition(transition, a, n_max::Integer, m_max::Integer)
    b = zeros(ComplexF64, n_max + 1, m_max + 1)
    for m in 0:m_max
        av = @view a[(m + 1):(n_max + 1), m + 1]
        b[(m + 1):(n_max + 1), m + 1] = transition[m + 1] * av
    end
    return b
end

"""
    fourier_matching_amplitude(b, k, angle, azimuth=0.0)

Far-field scattering amplitude [m] (Eq. (43)) at observation polar angle `angle` [rad] and
azimuth `azimuth` [rad], from the coefficient matrix `b` returned by any Fourier-matching solver
([`solve_pressure_release`](@ref), [`solve_rigid`](@ref), ...). Eq. (43) is the same formula
for every boundary condition, and only `b` differs. The `1/k` here comes from the standard
large-argument spherical Hankel asymptotic `h_n^(1)(kr) ~ (-i)^(n+1) e^(ikr)/(kr)` (the argument is
`kr`, not `r`) used to extract `f_s` from Eq. (40)'s `P^scat ~ P^inc e^(ikr)/r * f_s` (Eq. (42)).
Eq. (43) as printed omits it, which is inconsistent with Eq. (42)'s own definition. This is
confirmed empirically (not just by this re-derivation) against the exact sphere modal solution,
where its absence showed up as an exact, angle-independent `actual/expected = k` ratio.
"""
function fourier_matching_amplitude(
        b::AbstractMatrix{ComplexF64}, k::Real, angle::Real, azimuth::Real = 0.0)
    n_max, m_max = size(b) .- 1
    costheta = cos(angle)
    total = zero(ComplexF64)
    for n in 0:n_max
        phase = ComplexF64(im)^(-n - 1)
        for m in 0:min(n, m_max)
            coeff = b[n + 1, m + 1]
            iszero(coeff) && continue
            # No extra m>0 doubling: neumann_factor(m) in a_nm (Eq. (35)) already accounts for it.
            total += coeff * phase * legendre_p(n, m, costheta) * cos(m * azimuth)
        end
    end
    return total / k
end

# --- Rigid boundary matching (Eqs. 55-71): incidence azimuth fixed at 0, m>=0 only ---

"""
    _rigid_boundary_matrices(mapping, k, m; n_max, rtol=1e-6, maxevals=1000)

Assemble the rigid boundary matrices `R'`, `Q'` (Eqs. (69)-(70)) for one azimuthal order `m`,
truncated to test/source orders `n, n2 = m:n_max`, mirroring [`_boundary_matrices`](@ref)'s
(test, source) matrix reinterpretation of the paper's single-`n` notation. Each entry is the
normal derivative (Eq. (66)) of the source term `j_n2(kr)*P_n2^m(g/r)` (or `h_n2^(1)` for `Q'`)
w.r.t. `u`, by the product rule through both `r(u,w)` and the Legendre argument `g(u,w)/r(u,w)`.
"""
function _rigid_boundary_matrices(mapping::ConformalMapping, k::Real, m::Integer;
        n_max::Integer, rtol::Real = 1e-6, maxevals::Integer = 1000)
    ns = m:n_max
    nn = length(ns)
    R = zeros(ComplexF64, nn, nn)
    Q = zeros(ComplexF64, nn, nn)
    for (i, n) in enumerate(ns), (j, n2) in enumerate(ns)

        (RQ, _) = quadgk(0.0, π; rtol, maxevals) do w
            g, f = mapping_surface(mapping, w)
            dG = _mapping_derivative(mapping, w)
            g_u, f_u = real(dG), imag(dG)
            r = hypot(f, g)
            r_u = (f * f_u + g * g_u) / r
            # Clamp roundoff at the poles (w=0,pi), where legendre_p needs |g/r|<=1 exactly.
            costheta = clamp(g / r, -1.0, 1.0)
            dcosdu = (r * g_u - g * r_u) / r^2
            test = legendre_p(n, m, cos(w)) * sin(w) / abs(dG)
            dPdx = ForwardDiff.derivative(x -> legendre_p(n2, m, x), costheta)
            Pval = legendre_p(n2, m, costheta)
            kr = k * r
            js_term = (js(n2, kr) * dPdx * dcosdu + jsd(n2, kr) * k * r_u * Pval) * test
            hs_term = (hs(n2, kr) * dPdx * dcosdu + hsd(n2, kr) * k * r_u * Pval) * test
            SVector(js_term, hs_term)
        end
        R[i, j], Q[i, j] = RQ
    end
    return R, Q
end

"""
    _rigid_transition(mapping, k; m_max, n_max, rtol=1e-6, maxevals=1000)

The rigid transition operator (Eq. (71), one `T_m = -Q'^+ R'` matrix per azimuthal order `m`,
`b[:,m] = T_m * a[:,m]`), independent of incidence angle. See
[`_pressure_release_transition`](@ref) for why this is worth building once and reusing.
"""
function _rigid_transition(mapping::ConformalMapping, k::Real;
        m_max::Integer, n_max::Integer, rtol::Real = 1e-6, maxevals::Integer = 1000)
    return map(0:m_max) do m
        R, Q = _rigid_boundary_matrices(mapping, k, m; n_max, rtol, maxevals)
        -_equilibrated_solve(Q, R)
    end
end

"""
    solve_rigid(mapping, k, incidence_angle; m_max=_default_mode_count(k*mapping.body.a),
               n_max=m_max, rtol=1e-6, maxevals=1000)

Solve the rigid (hard/Neumann) Fourier-matching scattering problem (Eqs. (55)-(71)) for a
unit-amplitude plane wave incident at polar angle `incidence_angle` [rad], azimuth `0` (see the
module-level convention notes above). Returns the scattered-field coefficient matrix
`b[n+1,m+1]` for `n=0:n_max`, `m=0:m_max` (entries with `n<m` are `0`), for use with
[`fourier_matching_amplitude`](@ref). Solves each azimuthal order's system with
[`_equilibrated_solve`](@ref), since these matrices span many orders of magnitude across `n`
from spherical Hankel-function growth alone, not from genuine ill-conditioning. For repeated
incidence angles at the same `mapping`/`k`, build [`_rigid_transition`](@ref) once instead of
calling this repeatedly.
"""
function solve_rigid(mapping::ConformalMapping, k::Real, incidence_angle::Real;
        m_max::Integer = _default_mode_count(k * mapping.body.a),
        n_max::Integer = m_max, rtol::Real = 1e-6, maxevals::Integer = 1000)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive, got $k"))
    n_max >= m_max >= 0 ||
        throw(ArgumentError("require n_max >= m_max >= 0, got n_max=$n_max, m_max=$m_max"))
    transition = _rigid_transition(mapping, k; m_max, n_max, rtol, maxevals)
    a = _incident_coefficients(n_max, m_max, k, incidence_angle)
    return _apply_transition(transition, a, n_max, m_max)
end

# --- Fluid (Cauchy) boundary matching (Eqs. 72-80): incidence azimuth fixed at 0, m>=0 only ---

"""
    _interior_boundary_matrices(mapping, k1, m; n_max, rtol=1e-6, maxevals=1000)

Assemble the fluid boundary's interior matrices `S`, `S'` (the third integrals of Eqs. (76)-(77))
for one azimuthal order `m`, at the interior wavenumber `k1`, mirroring `_boundary_matrices` and
`_rigid_boundary_matrices`'s (test, source) matrix structure but with `j_n2(k1*r)` (and its
`u`-derivative) in place of both `j_n2(kr)` and `h_n2^(1)(kr)`, since the interior field must
stay regular at the origin (Eq. (41)).
"""
function _interior_boundary_matrices(mapping::ConformalMapping, k1::Real, m::Integer;
        n_max::Integer, rtol::Real = 1e-6, maxevals::Integer = 1000)
    ns = m:n_max
    nn = length(ns)
    S = zeros(ComplexF64, nn, nn)
    Sprime = zeros(ComplexF64, nn, nn)
    for (i, n) in enumerate(ns), (j, n2) in enumerate(ns)

        (SS, _) = quadgk(0.0, π; rtol, maxevals) do w
            g, f = mapping_surface(mapping, w)
            dG = _mapping_derivative(mapping, w)
            g_u, f_u = real(dG), imag(dG)
            r = hypot(f, g)
            r_u = (f * f_u + g * g_u) / r
            costheta = clamp(g / r, -1.0, 1.0)
            dcosdu = (r * g_u - g * r_u) / r^2
            test_p = legendre_p(n, m, cos(w)) * sin(w)
            Pval = legendre_p(n2, m, costheta)
            k1r = k1 * r
            j_val = js(n2, k1r)
            dPdx = ForwardDiff.derivative(x -> legendre_p(n2, m, x), costheta)
            S_term = j_val * Pval * test_p
            Sprime_term = (j_val * dPdx * dcosdu + jsd(n2, k1r) * k1 * r_u * Pval) *
                          test_p /
                          abs(dG)
            SVector(S_term, Sprime_term)
        end
        S[i, j], Sprime[i, j] = SS
    end
    return S, Sprime
end

"""
    _fluid_transition(mapping, k, density_contrast, soundspeed_contrast; m_max, n_max,
                      rtol=1e-6, maxevals=1000)

The fluid transition operator (Eq. (80), one `T_m` matrix per azimuthal order `m`,
`b[:,m] = T_m * a[:,m]`), independent of incidence angle. See
[`_pressure_release_transition`](@ref) for why this is worth building once and reusing.
"""
function _fluid_transition(mapping::ConformalMapping, k::Real,
        density_contrast::Real, soundspeed_contrast::Real;
        m_max::Integer, n_max::Integer, rtol::Real = 1e-6, maxevals::Integer = 1000)
    k1 = k / soundspeed_contrast
    return map(0:m_max) do m
        R, Q = _boundary_matrices(mapping, k, m; n_max, rtol, maxevals)
        Rp, Qp = _rigid_boundary_matrices(mapping, k, m; n_max, rtol, maxevals)
        S, Sp = _interior_boundary_matrices(mapping, k1, m; n_max, rtol, maxevals)
        SinvQ = _equilibrated_solve(S, Q)
        SinvR = _equilibrated_solve(S, R)
        M1 = density_contrast .* Qp .- Sp * SinvQ
        M2 = Sp * SinvR .- density_contrast .* Rp
        _equilibrated_solve(M1, M2)
    end
end

"""
    solve_fluid(mapping, k, incidence_angle, density_contrast, soundspeed_contrast;
               m_max=_default_mode_count(k*mapping.body.a), n_max=m_max, rtol=1e-6,
               maxevals=1000)

Solve the fluid (Cauchy) Fourier-matching scattering problem (Eqs. (72)-(80)) for a body whose
interior medium has the given density and sound-speed contrasts relative to the surrounding
fluid, matching [`FluidFilled`](@ref)'s convention, for a unit-amplitude plane wave incident at
polar angle `incidence_angle` [rad], azimuth `0` (see the module-level convention notes above).
Returns the scattered-field coefficient matrix `b[n+1,m+1]`, for use with
[`fourier_matching_amplitude`](@ref).

Eliminating the interior coefficients `l_nm` from Eqs. (78)-(79) by direct substitution (rather
than trusting Eq. (80) as printed, which cross-multiplies as if `R`, `Q`, `S` were scalars) gives
`b = (Q' - S'*S^-1*Q)^-1 * (S'*S^-1*R - R')*a`. This reduces to Eq. (80) exactly when everything
commutes (checked symbolically), the same generalization already used for the collapsed
single-index sums in Eqs. (51)-(54) and (69)-(71). Velocity continuity (Eq. (73) as printed) is
weighted by `density_contrast` alone, not the `gh = density_contrast*soundspeed_contrast` used
with the dimensionless Bessel-derivative convention in [`FluidFilled`](@ref)'s own sphere
formula. `R'`, `Q'`, `S'` here already carry the dimensional `k`/`k1` chain-rule factor from
differentiating `j_n(kr(u,w))`/`j_n(k1*r(u,w))` w.r.t. `u`, which absorbs the soundspeed part of
the impedance ratio and leaves only the density part. This is confirmed against this package's
own validated [`FluidFilled`](@ref) sphere coefficients (weak and gas contrast) to `1e-16`.
Every matrix inverse here (`S`, and the final elimination matrix) uses
[`_equilibrated_solve`](@ref), since these matrices span many orders of magnitude across `n`
from spherical Hankel-function growth alone, not from genuine ill-conditioning. For repeated
incidence angles at the same `mapping`/`k`/contrasts, build [`_fluid_transition`](@ref) once
instead of calling this repeatedly.
"""
function solve_fluid(mapping::ConformalMapping, k::Real, incidence_angle::Real,
        density_contrast::Real, soundspeed_contrast::Real;
        m_max::Integer = _default_mode_count(k * mapping.body.a),
        n_max::Integer = m_max, rtol::Real = 1e-6, maxevals::Integer = 1000)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive, got $k"))
    n_max >= m_max >= 0 ||
        throw(ArgumentError("require n_max >= m_max >= 0, got n_max=$n_max, m_max=$m_max"))
    transition = _fluid_transition(mapping, k, density_contrast, soundspeed_contrast;
        m_max, n_max, rtol, maxevals)
    a = _incident_coefficients(n_max, m_max, k, incidence_angle)
    return _apply_transition(transition, a, n_max, m_max)
end

"""
    _boundary_transition(mapping, k, boundary; m_max, n_max, rtol=1e-6, maxevals=1000)

Dispatch to [`_pressure_release_transition`](@ref), [`_rigid_transition`](@ref) or
[`_fluid_transition`](@ref) by `boundary`'s type, shared by [`fourier`](@ref) and
`incidence_angle_sweep` so the boundary-condition dispatch is written once.
"""
function _boundary_transition(mapping::ConformalMapping, k::Real,
        boundary::AbstractBoundaryCondition; m_max::Integer, n_max::Integer,
        rtol::Real = 1e-6, maxevals::Integer = 1000)
    if boundary isa PressureRelease
        _pressure_release_transition(mapping, k; m_max, n_max, rtol, maxevals)
    elseif boundary isa Rigid
        _rigid_transition(mapping, k; m_max, n_max, rtol, maxevals)
    elseif boundary isa FluidFilled
        _fluid_transition(
            mapping, k, boundary.density_contrast, boundary.soundspeed_contrast;
            m_max, n_max, rtol, maxevals)
    else
        throw(ArgumentError(
            "fourier does not support boundary condition $(typeof(boundary))"))
    end
end
