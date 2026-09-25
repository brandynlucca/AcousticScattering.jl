# Fourier matching, extending Reeder & Stanton 2004 (doi:10.1121/1.1648681) and DiPerna & Stanton 1994 (doi:10.1121/1.411243).

using NLsolve: nlsolve
using ADTypes: AutoForwardDiff
using QuadGK: gauss

"""
    Irregular(r, order; npoints=8*order+16)
    Irregular(a, rc, rs)

A body of revolution about the z axis, for smooth irregular shapes between the canonical
([`Sphere`](@ref), [`Spheroid`](@ref)) and general numerical (`bem`/`mfs`) bodies. Solve with
[`fourier`](@ref).

The first form Fourier-fits a harmonic `order` series to a user-supplied meridian profile
function `r(theta)` in m, the radial distance from the origin, not from the symmetry axis, at
polar angle `theta`. It is evaluated over a full `2*pi` period even though only `theta in [0,pi]`
is
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

Evaluate an [`Irregular`](@ref)'s meridian profile `R(theta)` in m (Eq. (21)) at polar angle
`theta` in rad.
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

Evaluate `theta(w)` in rad (Eq. (29)) for the solved `ConformalMapping`.
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

# Eq. (30)'s boundary-value function at every node. `scale<1` is DiPerna Appendix A's continuation homotopy.
function _boundary_values(body::Irregular, delta_c, delta_s, nodes, cosl, sinl, scale::Real)
    harmonics = complex.(body.rc, body.rs) ./ 2
    return map(eachindex(nodes)) do q
        theta = nodes[q]
        for l in eachindex(delta_c)
            theta += delta_c[l] * cosl[l, q] + delta_s[l] * sinl[l, q]
        end
        eith = cis(theta)
        eith_inv = conj(eith)
        value = body.a * eith
        up = eith
        down = eith
        for n in eachindex(harmonics)
            up *= eith
            down *= eith_inv
            value += scale * (conj(harmonics[n]) * up + harmonics[n] * down)
        end
        value
    end
end

# Real-vector NLsolve residual packing Eq. (30)'s j>1 complex constraints I_j(delta)=0. `kernel[j+order+1,:]` is `weights*cis(-j*nodes)/(2*pi)`.
function _mapping_residual!(F, x, body::Irregular, order::Integer,
        scale::Real, nodes, cosl, sinl, kernel)
    values = _boundary_values(body, @view(x[1:order]), @view(x[(order + 1):(2order)]),
        nodes, cosl, sinl, scale)
    for j in 2:(order + 1)
        total = sum(q -> kernel[j + order + 1, q] * values[q], eachindex(values))
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
    cosl = [cos(l * w) for l in 1:order, w in nodes]
    sinl = [sin(l * w) for l in 1:order, w in nodes]
    kernel = [wt * cis(-j * w) / 2π for j in (-order):(order + 1), (w, wt) in zip(nodes, weights)]
    x = zeros(Float64, 2order)
    for step in 1:continuation_steps
        scale = step / continuation_steps
        result = nlsolve(
            (F, y) -> _mapping_residual!(F, y, body, order, scale, nodes, cosl, sinl, kernel),
            x; nlsolve_kwargs...)
        result.f_converged || result.x_converged ||
            throw(ArgumentError(
                "conformal mapping failed to converge at continuation step $step/$continuation_steps " *
                "(scale=$scale); try more continuation_steps or a lower mapping order"))
        x = result.zero
    end
    delta_c = x[1:order]
    delta_s = x[(order + 1):(2order)]
    values = _boundary_values(body, delta_c, delta_s, nodes, cosl, sinl, 1.0)
    c = Vector{ComplexF64}(undef, order + 2)
    for (idx, j) in enumerate(1:-1:(-(order)))
        c[idx] = sum(q -> kernel[j + order + 1, q] * values[q], eachindex(values))
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
polar angle `incidence_angle` in rad (`theta_0`, measured from the axis of symmetry, where `0` is
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

# --- Fixed-node boundary-matching quadrature (Eqs. 52-53, 69-70, 76-77) ---

"""
    _FMNodes

The conformal mapping's geometry at fixed Gauss-Legendre nodes `w in (0,pi)` (`weight`, `cosw`,
`sinw`), shared by every matrix entry, order `m` and boundary. `r` is the distance from the origin,
`r_u` its `u`-derivative, `costheta` the Legendre argument `g/r`, `dcosdu` its `u`-derivative and
`jacobian` the mapping's `|G'|` (Eq. (31)).
"""
struct _FMNodes
    weight::Vector{Float64}
    cosw::Vector{Float64}
    sinw::Vector{Float64}
    r::Vector{Float64}
    r_u::Vector{Float64}
    costheta::Vector{Float64}
    dcosdu::Vector{Float64}
    jacobian::Vector{Float64}
end

function _fm_nodes(mapping::ConformalMapping, npoints::Integer)
    w, weight = gauss(npoints, 0.0, π)
    g = similar(w)
    f = similar(w)
    g_u = similar(w)
    f_u = similar(w)
    for (q, wq) in enumerate(w)
        g[q], f[q] = mapping_surface(mapping, wq)
        dG = _mapping_derivative(mapping, wq)
        g_u[q], f_u[q] = real(dG), imag(dG)
    end
    r = hypot.(f, g)
    r_u = (f .* f_u .+ g .* g_u) ./ r
    # Clamp roundoff at the poles (w=0,pi), where the Legendre argument needs |g/r|<=1 exactly.
    costheta = clamp.(g ./ r, -1.0, 1.0)
    dcosdu = (r .* g_u .- g .* r_u) ./ r .^ 2
    return _FMNodes(weight, cos.(w), sin.(w), r, r_u, costheta, dcosdu, hypot.(g_u, f_u))
end

# Node count that resolves the highest test order through the mapping's angular stretch, and the radial oscillation.
function _initial_node_count(mapping::ConformalMapping, k::Real, n_max::Integer)
    w = range(0, π; length = 181)
    surface = [mapping_surface(mapping, wi) for wi in w]
    theta = [atan(abs(f), g) for (g, f) in surface]
    stretch = maximum(abs, diff(theta)) / step(w)
    kr = k * maximum(hypot(g, f) for (g, f) in surface)
    return max(64, ceil(Int, 4 * (n_max + 1) * max(stretch, 1.0) + 2 * kr))
end

# Spherical Bessel and Hankel functions of orders `0:n_max` (and their argument derivatives) at `k*r` on the nodes, one row per order.
function _radial_table(nodes::_FMNodes, k::Real, n_max::Integer)
    x = k .* nodes.r
    j = [js(n, xq) for n in 0:(n_max + 1), xq in x]
    h = j .+ im .* [ys(n, xq) for n in 0:(n_max + 1), xq in x]
    slope(f) = [(n / x[q]) * f[n + 1, q] - f[n + 2, q] for n in 0:n_max, q in eachindex(x)]
    return (; j = j[1:(n_max + 1), :], h = h[1:(n_max + 1), :], jd = slope(j), hd = slope(h))
end

# `Pₗᵐ(x)` for `l = m:n_max` by the same recurrence as `legendre_p(l, m, x)`.
function _legendre_column!(col, m::Integer, n_max::Integer, x::Real)
    pmm = 1.0
    if m > 0
        somx2 = sqrt(1 - x^2)
        fact = 1.0
        for _ in 1:m
            pmm *= -fact * somx2
            fact += 2
        end
    end
    col[1] = pmm
    n_max == m && return col
    col[2] = x * (2m + 1) * pmm
    for l in (m + 2):n_max
        col[l - m + 1] = ((2l - 1) * x * col[l - m] - (l + m - 1) * col[l - m - 1]) / (l - m)
    end
    return col
end

# Test functions `Pₙᵐ(cos w) sin w`, source functions `Pₙᵐ(g/r)` and their derivative in the argument, one row per order `n = m:n_max`.
function _legendre_tables(nodes::_FMNodes, m::Integer, n_max::Integer)
    npoints = length(nodes.weight)
    nn = n_max - m + 1
    test = Matrix{Float64}(undef, nn, npoints)
    source = Matrix{Float64}(undef, nn, npoints)
    dsource = Matrix{Float64}(undef, nn, npoints)
    for q in 1:npoints
        _legendre_column!(view(test, :, q), m, n_max, nodes.cosw[q])
        test[:, q] .*= nodes.sinw[q]
        x = nodes.costheta[q]
        _legendre_column!(view(source, :, q), m, n_max, x)
        for i in 1:nn
            l = m + i - 1
            previous = i > 1 ? source[i - 1, q] : 0.0
            dsource[i, q] = ((l + m) * previous - l * x * source[i, q]) / (1 - x^2)
        end
    end
    return (; test, source, dsource)
end

# Normal-derivative source terms of Eq. (66), the product rule through `r(u,w)` and the Legendre argument.
function _normal_terms(radial_value, radial_slope, nodes::_FMNodes, tables, k::Real)
    return radial_value .* tables.dsource .* nodes.dcosdu' .+
           radial_slope .* tables.source .* (k .* nodes.r_u)'
end

# Matrices for one azimuthal order `m` (test rows, source columns) on the given nodes.
function _fm_blocks(kind::Symbol, nodes::_FMNodes, exterior, interior, k::Real, k1::Real,
        m::Integer, n_max::Integer)
    tables = _legendre_tables(nodes, m, n_max)
    order = (m + 1):(n_max + 1)
    weighted = tables.test .* nodes.weight'
    normal = weighted ./ nodes.jacobian'
    soft(radial) = weighted * transpose(radial[order, :] .* tables.source)
    function hard(value, slope, kk)
        terms = _normal_terms(value[order, :], slope[order, :], nodes, tables, kk)
        return normal * transpose(terms)
    end
    if kind === :pressure_release
        return (; R = soft(exterior.j), Q = soft(exterior.h))
    elseif kind === :rigid
        return (; R = hard(exterior.j, exterior.jd, k), Q = hard(exterior.h, exterior.hd, k))
    elseif kind === :interior
        return (; S = soft(exterior.j), Sp = hard(exterior.j, exterior.jd, k))
    else
        return (; R = soft(exterior.j), Q = soft(exterior.h),
            Rp = hard(exterior.j, exterior.jd, k), Qp = hard(exterior.h, exterior.hd, k),
            S = soft(interior.j), Sp = hard(interior.j, interior.jd, k1))
    end
end

function _fm_assemble(kind::Symbol, mapping::ConformalMapping, k::Real, k1::Real,
        orders::AbstractRange, n_max::Integer, npoints::Integer)
    nodes = _fm_nodes(mapping, npoints)
    exterior = _radial_table(nodes, k, n_max)
    interior = kind === :fluid ? _radial_table(nodes, k1, n_max) : nothing
    return [_fm_blocks(kind, nodes, exterior, interior, k, k1, m, n_max) for m in orders]
end

# Largest entrywise change between two assemblies, on the row/column-equilibrated scale the solves use.
function _blocks_change(new, old)
    return maximum(zip(new, old)) do (block_new, block_old)
        maximum(keys(block_new)) do name
            A = block_new[name]
            col_scale, row_scale = _equilibration(A)
            maximum(abs.(A .- block_old[name]) ./ col_scale' ./ row_scale)
        end
    end
end

# Doubles the Gauss-Legendre node count until successive assemblies agree to `rtol` or `maxevals` nodes is reached.
function _fm_blocks_converged(kind::Symbol, mapping::ConformalMapping, k::Real, k1::Real,
        orders::AbstractRange, n_max::Integer, rtol::Real, maxevals::Integer)
    npoints = min(_initial_node_count(mapping, max(k, k1), n_max), maxevals)
    blocks = _fm_assemble(kind, mapping, k, k1, orders, n_max, npoints)
    while npoints < maxevals
        npoints = min(2npoints, maxevals)
        refined = _fm_assemble(kind, mapping, k, k1, orders, n_max, npoints)
        agreed = _blocks_change(refined, blocks) <= rtol
        blocks = refined
        agreed && break
    end
    return blocks
end

"""
    _boundary_matrices(mapping, k, m; n_max, rtol=1e-6, maxevals=1000)

Assemble the pressure-release boundary matrices `R`, `Q` (Eqs. (52)-(53)) for one azimuthal
order `m`, truncated to test/source orders `n, n2 = m:n_max`. `R[i,j]`/`Q[i,j]` correspond to
test order `n = m+i-1` and source order `n2 = m+j-1`. Eqs. (52)-(53) print both indices as "n",
but Eq. (51)'s explicit sum over `n` and Eq. (54)'s explicit matrix inverse only make sense if
these are two independent (test, source) indices, so that is how they are implemented here.

The integrals over `w in (0,pi)` use fixed Gauss-Legendre nodes shared by every entry, and the node
count doubles until the equilibrated matrices change by less than `rtol`, up to `maxevals` nodes.
"""
function _boundary_matrices(mapping::ConformalMapping, k::Real, m::Integer;
        n_max::Integer, rtol::Real = 1e-6, maxevals::Integer = 1000)
    (; R, Q) = only(_fm_blocks_converged(
        :pressure_release, mapping, k, k, m:m, n_max, rtol, maxevals))
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
    col_scale, row_scale = _equilibration(A)
    equilibrated = A ./ col_scale' ./ row_scale
    return (pinv(equilibrated) * (rhs ./ row_scale)) ./ col_scale
end

# Column then row maximum-magnitude scales that equilibrate `A`.
function _equilibration(A::AbstractMatrix)
    col_scale = vec(maximum(abs, A; dims = 1))
    row_scale = vec(maximum(abs, A ./ col_scale'; dims = 2))
    return col_scale, row_scale
end

# `n_max` and `m_max` are reduced by this step for the consistency solve behind `_fm_convergence`.
const _FM_CHECK_STEP = 2
const _FM_CONVERGENCE_TOLERANCE = 1e-2

"""
    _FMTransition

Per-`m` transition matrices `full` at the requested `(n_max, m_max)`, and `check` at
`(n_max - 2, min(m_max, n_max - 2))` (`nothing` when `n_max` is too small to reduce). Both come from
leading blocks of the same assembled matrices, so `check` costs no extra quadrature.
"""
struct _FMTransition
    full::Vector{Matrix{ComplexF64}}
    check::Union{Nothing, Vector{Matrix{ComplexF64}}}
end

# `assemble(m)` returns `nn -> T`, the transition matrix from the leading `nn x nn` blocks of the quadrature matrices at order `m`.
function _transition_operators(assemble, m_max::Integer, n_max::Integer)
    n_check = n_max - _FM_CHECK_STEP
    m_check = min(m_max, n_check)
    full = Matrix{ComplexF64}[]
    check = Matrix{ComplexF64}[]
    for m in 0:m_max
        solve = assemble(m)
        push!(full, solve(n_max - m + 1))
        n_check >= 1 && m <= m_check && push!(check, solve(n_check - m + 1))
    end
    return _FMTransition(full, n_check >= 1 ? check : nothing)
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
    blocks = _fm_blocks_converged(
        :pressure_release, mapping, k, k, 0:m_max, n_max, rtol, maxevals)
    return _transition_operators(m_max, n_max) do m
        (; R, Q) = blocks[m + 1]
        nn -> -_equilibrated_solve(Q[1:nn, 1:nn], R[1:nn, 1:nn])
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
function _apply_transition(transition::_FMTransition, a, n_max::Integer, m_max::Integer)
    return _apply_transition(transition.full, a, n_max, m_max)
end

function _apply_transition(transition::AbstractVector, a, n_max::Integer, m_max::Integer)
    b = zeros(ComplexF64, n_max + 1, m_max + 1)
    for m in 0:m_max
        av = @view a[(m + 1):(n_max + 1), m + 1]
        b[(m + 1):(n_max + 1), m + 1] = transition[m + 1] * av
    end
    return b
end

# Coefficients of the reduced-truncation solve, or `nothing` when `n_max` is too small to reduce.
function _check_coefficients(transition::_FMTransition, k::Real, incidence_angle::Real,
        n_max::Integer, m_max::Integer)
    transition.check === nothing && return nothing
    n_check = n_max - _FM_CHECK_STEP
    m_check = min(m_max, n_check)
    a = _incident_coefficients(n_check, m_check, k, incidence_angle)
    return _apply_transition(transition.check, a, n_check, m_check)
end

"""
    _fm_convergence(b, b_check, k)

Largest change in the far-field amplitude between the solution `b` and the reduced-truncation
solution `b_check`, over 13 polar angles at azimuths `0`, `pi/2` and `pi`, relative to the peak
amplitude of `b` over those directions. `NaN` when there is no reduced solution.
"""
function _fm_convergence(b::AbstractMatrix, b_check, k::Real)
    b_check === nothing && return NaN
    directions = [(angle, azimuth)
                  for angle in range(0, π; length = 13), azimuth in (0.0, π / 2, π)]
    full = [fourier_matching_amplitude(b, k, angle, azimuth) for (angle, azimuth) in directions]
    reduced = [fourier_matching_amplitude(b_check, k, angle, azimuth)
               for (angle, azimuth) in directions]
    peak = maximum(abs, full)
    return iszero(peak) ? 0.0 : maximum(abs.(full .- reduced)) / peak
end

function _warn_fm_convergence(convergence::Real)
    convergence > _FM_CONVERGENCE_TOLERANCE || return nothing
    @warn "Fourier matching is not converged in its truncation orders. Reducing n_max and m_max by $_FM_CHECK_STEP changes the far-field amplitude by $(round(convergence; sigdigits = 2)) of its peak (tolerance $_FM_CONVERGENCE_TOLERANCE). Larger n_max does not always help for elongated bodies, so vary n_max and mapping_order and compare against bem before trusting the result."
    return nothing
end

"""
    fourier_matching_amplitude(b, k, angle, azimuth=0.0)

Far-field scattering amplitude in m (Eq. (43)) at observation polar angle `angle` in rad and
azimuth `azimuth` in rad, from the coefficient matrix `b` returned by any Fourier-matching solver
(`solve_pressure_release`, `solve_rigid`, ...). Eq. (43) is the same formula
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
    (; R, Q) = only(_fm_blocks_converged(:rigid, mapping, k, k, m:m, n_max, rtol, maxevals))
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
    blocks = _fm_blocks_converged(:rigid, mapping, k, k, 0:m_max, n_max, rtol, maxevals)
    return _transition_operators(m_max, n_max) do m
        (; R, Q) = blocks[m + 1]
        nn -> -_equilibrated_solve(Q[1:nn, 1:nn], R[1:nn, 1:nn])
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
    (; S, Sp) = only(_fm_blocks_converged(:interior, mapping, k1, k1, m:m, n_max, rtol, maxevals))
    return S, Sp
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
    blocks = _fm_blocks_converged(:fluid, mapping, k, k1, 0:m_max, n_max, rtol, maxevals)
    return _transition_operators(m_max, n_max) do m
        (; R, Q, Rp, Qp, S, Sp) = blocks[m + 1]
        function (nn)
            ix = 1:nn
            SinvQ = _equilibrated_solve(S[ix, ix], Q[ix, ix])
            SinvR = _equilibrated_solve(S[ix, ix], R[ix, ix])
            M1 = density_contrast .* Qp[ix, ix] .- Sp[ix, ix] * SinvQ
            M2 = Sp[ix, ix] * SinvR .- density_contrast .* Rp[ix, ix]
            _equilibrated_solve(M1, M2)
        end
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
