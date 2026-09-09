# Prolate/oblate spheroidal modal series, built on SpheroidalWaves.jl. Liquid-filled (also covers
# gas-filled) spheroids support `coupling=:diagonal` (Furusawa 1988 Eq. 5) or `:full` (Eq. 4, default).

using QuadGK: gauss

"""
    Spheroid(a, b)

Spheroid of revolution with semi-axis `a` [m] along the axis of symmetry
and equatorial semi-axis `b` [m]. Prolate if `a > b`, oblate if `a < b`.
"""
struct Spheroid <: AbstractBody
    a::Float64
    b::Float64
    kind::Symbol
    xi0::Float64
    q::Float64

    function Spheroid(a::Real, b::Real)
        a == b &&
            throw(ArgumentError("Spheroid requires a ≠ b; use the sphere modal series (sphere_modal.jl) for a sphere"))
        a > 0 && b > 0 || throw(ArgumentError("Spheroid semi-axes must be positive"))
        if a > b
            kind = :prolate
            q = sqrt(a^2 - b^2)
        else
            kind = :oblate
            q = sqrt(b^2 - a^2)
        end
        xi0 = a / q
        return new(Float64(a), Float64(b), kind, xi0, q)
    end
end

function _default_spheroid_orders(k::Real, body::Spheroid)
    order = _default_mode_count(k * max(body.a, body.b))
    return order
end

function _spheroid_modal_coefficient(
        ::Rigid, m::Integer, n::Integer, c::Real, xi0::Real, kind::Symbol;
        precision::Symbol = :double)
    r1 = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 1)
    r3 = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 3)
    return -r1.derivative[1] / r3.derivative[1]
end

function _spheroid_modal_coefficient(
        ::PressureRelease, m::Integer, n::Integer, c::Real, xi0::Real, kind::Symbol;
        precision::Symbol = :double)
    r1 = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 1)
    r3 = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 3)
    return -r1.value[1] / r3.value[1]
end

function _spheroid_modal_coefficient(
        bc::FluidFilled, m::Integer, n::Integer, c::Real, xi0::Real, kind::Symbol;
        precision::Symbol = :double)
    g = bc.density_contrast
    c_int = c / bc.soundspeed_contrast
    r1e = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 1)
    r3e = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 3)
    r1i = SpheroidalWaves.rmn(
        m, n, c_int, [xi0]; spheroid = kind, precision = precision, kind = 1)

    # Pre-multiplied by r1i.derivative rather than divided through by it, which crosses zero at ordinary mode orders.
    E1 = r1e.value[1] * r1i.derivative[1] - g * r1i.value[1] * r1e.derivative[1]
    E3 = r3e.value[1] * r1i.derivative[1] - g * r1i.value[1] * r3e.derivative[1]
    return -E1 / E3
end

# Largest leading-submatrix size s (1 <= s <= size(K3,1)) with cond(K3[1:s,1:s]) <= max_cond.
function _safe_coupling_size(K3::AbstractMatrix; max_cond::Real = 1e10)
    n = size(K3, 1)
    (n <= 1 || cond(K3) <= max_cond) && return n
    lo, hi = 1, n
    while lo < hi
        mid = (lo + hi + 1) ÷ 2
        cond(@view K3[1:mid, 1:mid]) <= max_cond ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

# Full off-diagonal boundary coupling (Furusawa 1988 Eq. 4) for a single azimuthal order m.
# `precision=:quad` runs the quad-precision backend on top of the BigFloat matrix solve.
function _liquid_full_coefficients(bc::FluidFilled, m::Integer, n_max::Integer,
        c::Real, c_int::Real, xi0::Real, kind::Symbol,
        eta_i::Real; n_quad::Integer = 64, max_cond::Real = 1e60,
        precision_bits::Integer = 256, precision::Symbol = :double)
    g = bc.density_contrast
    degrees = m:n_max
    size_ = length(degrees)

    Smn_inc = [SpheroidalWaves.smn(m, n, c, eta_i; spheroid = kind,
                   precision = precision, normalize = true).value[1] for n in degrees]
    if all(iszero, Smn_inc)
        # m >= 1 contributes nothing at exact end-on incidence, Sₘₙ(c,±1) = 0 identically.
        return zeros(ComplexF64, size_)
    end

    nodes, weights = gauss(n_quad)
    Sext_nodes = [SpheroidalWaves.smn(m, n, c, nodes; spheroid = kind,
                      precision = precision, normalize = true).value for n in degrees]
    Sint_nodes = [SpheroidalWaves.smn(m, l, c_int, nodes; spheroid = kind,
                      precision = precision, normalize = true).value for l in degrees]

    R1e = [SpheroidalWaves.rmn(
               m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 1)
           for n in degrees]
    R3e = [SpheroidalWaves.rmn(
               m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 3)
           for n in degrees]
    R1i = [SpheroidalWaves.rmn(
               m, l, c_int, [xi0]; spheroid = kind, precision = precision, kind = 1)
           for l in degrees]

    A = setprecision(BigFloat, precision_bits) do
        alpha = Matrix{BigFloat}(undef, size_, size_)
        for li in 1:size_, ni in 1:size_

            alpha[li, ni] = sum(big.(Sext_nodes[ni]) .* big.(Sint_nodes[li]) .*
                                big.(weights))
        end

        g_big = big(g)
        R1i_v = [big(R1i[li].value[1]) for li in 1:size_]
        R1i_d = [big(R1i[li].derivative[1]) for li in 1:size_]
        Smn_inc_big = big.(Smn_inc)

        # Each row li of K1/K3 pre-scaled by R1i_d[li] rather than divided by it, which crosses zero at ordinary mode orders.
        K1 = Matrix{Complex{BigFloat}}(undef, size_, size_)
        K3 = Matrix{Complex{BigFloat}}(undef, size_, size_)
        for li in 1:size_, ni in 1:size_

            n = degrees[ni]
            jn = Complex{BigFloat}(0, 1)^n
            R1e_v = Complex{BigFloat}(R1e[ni].value[1])
            R1e_d = Complex{BigFloat}(R1e[ni].derivative[1])
            R3e_v = Complex{BigFloat}(R3e[ni].value[1])
            R3e_d = Complex{BigFloat}(R3e[ni].derivative[1])
            E1 = R1e_v * R1i_d[li] - g_big * R1i_v[li] * R1e_d
            E3 = R3e_v * R1i_d[li] - g_big * R1i_v[li] * R3e_d
            K1[li, ni] = jn * Smn_inc_big[ni] * alpha[li, ni] * E1
            K3[li, ni] = jn * Smn_inc_big[ni] * alpha[li, ni] * E3
        end

        s = _safe_coupling_size(K3; max_cond = max_cond)
        K3_safe = K3[1:s, 1:s]
        if s == 1 && cond(K3_safe) > max_cond
            # At exact end-on incidence and large c, Sₘₙ(c,η=±1) for the lowest n can underflow before higher n does.
            @warn "Full liquid-filled spheroid coupling: no numerically safe mode truncation found for m=$m (c=$c, incidence η=$eta_i). This m's contribution may be inaccurate, consider a slightly off-axis incidence angle or coupling=:diagonal." maxlog=10
        end
        b_safe = -vec(sum(K1[1:s, 1:s], dims = 2))
        A_safe = pinv(K3_safe) * b_safe

        A_big = zeros(Complex{BigFloat}, size_)
        A_big[1:s] = A_safe
        return A_big
    end
    return ComplexF64.(A)
end

# Rigid/PressureRelease and FluidFilled(coupling=:diagonal) share this diagonal expansion.
function _form_function_diagonal(
        boundary::AbstractBoundaryCondition, k::Real, body::Spheroid;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        scatter_angle::Real = π - incidence_angle,
        scatter_azimuth::Real = incidence_azimuth + π,
        m_max::Integer = _default_spheroid_orders(k, body),
        n_max::Integer = _default_spheroid_orders(k, body),
        precision::Symbol = :double)
    c = k * body.q
    eta_i = cos(incidence_angle)
    eta_s = cos(scatter_angle)
    dphi = incidence_azimuth - scatter_azimuth

    total = zero(ComplexF64)
    for m in 0:m_max
        azimuth_term = cos(m * dphi)
        iszero(azimuth_term) && continue
        nu_m = neumann_factor(m)
        for n in m:n_max
            Smn_i = SpheroidalWaves.smn(m, n, c, eta_i; spheroid = body.kind,
                precision = precision, normalize = true).value[1]
            Smn_s = SpheroidalWaves.smn(m, n, c, eta_s; spheroid = body.kind,
                precision = precision, normalize = true).value[1]
            Amn = _spheroid_modal_coefficient(
                boundary, m, n, c, body.xi0, body.kind; precision = precision)
            total += nu_m * Smn_i * Smn_s * Amn * azimuth_term
        end
    end
    return -2im / k * total
end

# FluidFilled(coupling=:full) path, see _liquid_full_coefficients for the boundary coupling solve.
function _form_function_full_fluid(bc::FluidFilled, k::Real, body::Spheroid;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        scatter_angle::Real = π - incidence_angle,
        scatter_azimuth::Real = incidence_azimuth + π,
        m_max::Integer = _default_spheroid_orders(k, body),
        n_max::Integer = _default_spheroid_orders(k, body),
        n_quad::Integer = 64, precision::Symbol = :double)
    c = k * body.q
    c_int = c / bc.soundspeed_contrast
    eta_i = cos(incidence_angle)
    eta_s = cos(scatter_angle)
    dphi = incidence_azimuth - scatter_azimuth

    total = zero(ComplexF64)
    for m in 0:m_max
        azimuth_term = cos(m * dphi)
        iszero(azimuth_term) && continue
        nu_m = neumann_factor(m)
        n_max_m = max(n_max, m)
        A = _liquid_full_coefficients(bc, m, n_max_m, c, c_int, body.xi0, body.kind,
            eta_i; n_quad = n_quad, precision = precision)
        for (li, n) in enumerate(m:n_max_m)
            Smn_i = SpheroidalWaves.smn(m, n, c, eta_i; spheroid = body.kind,
                precision = precision, normalize = true).value[1]
            Smn_s = SpheroidalWaves.smn(m, n, c, eta_s; spheroid = body.kind,
                precision = precision, normalize = true).value[1]
            total += nu_m * Smn_i * Smn_s * A[li] * azimuth_term
        end
    end
    return -2im / k * total
end

"""
    form_function(boundary, k, body::Spheroid;
                  incidence_angle=0.0, incidence_azimuth=0.0,
                  scatter_angle=π-incidence_angle, scatter_azimuth=incidence_azimuth+π,
                  m_max=default, n_max=default)

Far-field scattering amplitude f [m] of a prolate/oblate spheroid under a
rigid, pressure-release, or fluid/gas-filled modal series. Angles are
measured from the spheroid's axis of symmetry (`incidence_angle = 0` is
end-on incidence, `π/2` is broadside); the defaults give monostatic
backscatter at the requested incidence angle.

For `FluidFilled`, `boundary.coupling` (`:full` by default, or `:diagonal`)
selects between the complete off-diagonal boundary coupling and the
cheaper, less accurate diagonal approximation, see the module preamble.

`precision` (`:double`, the default, or `:quad`) controls whether
`SpheroidalWaves.jl`'s radial/angular spheroidal wave function evaluations
run at ordinary double precision or its separate quad-precision Fortran
backend. `:quad` is several times slower but needed at large size
parameters or high mode counts, where cancellation in the boundary
coupling can push past double precision's noise floor for any boundary
condition (not just `FluidFilled`). This is a high-`ka` concern only:
at modest size parameters double and quad agree to well under 0.01 dB.
Not a silent default, callers who hit accuracy trouble at high `ka`
should try `precision=:quad` explicitly rather than pay its cost
everywhere.
"""
function form_function(boundary::Union{Rigid, PressureRelease}, k::Real, body::Spheroid; kwargs...)
    return _form_function_diagonal(boundary, k, body; kwargs...)
end

function form_function(boundary::FluidFilled, k::Real, body::Spheroid; kwargs...)
    if boundary.coupling === :diagonal
        return _form_function_diagonal(boundary, k, body; kwargs...)
    end
    return _form_function_full_fluid(boundary, k, body; kwargs...)
end

"""
    target_strength(boundary, k, body::Spheroid; kwargs...)

Target strength [dB re 1 m²] of a prolate/oblate spheroid under a rigid,
pressure-release, or fluid/gas-filled modal series. See
[`form_function`](@ref) for keyword arguments.
"""
function target_strength(boundary::AbstractBoundaryCondition, k::Real, body::Spheroid; kwargs...)
    return target_strength(form_function(boundary, k, body; kwargs...))
end
