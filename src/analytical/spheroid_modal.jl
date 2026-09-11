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

# Degree ranges share native expansions when the dependency supports them.
# Installed versions with only scalar-degree methods remain supported.
function _spheroid_degree_values(f, m, degrees, c, points; kwargs...)
    if applicable(f, m, degrees, c, points)
        return f(m, degrees, c, points; kwargs...)
    end
    results = [f(m, n, c, points; kwargs...) for n in degrees]
    return (; value = hcat((r.value for r in results)...),
        derivative = hcat((r.derivative for r in results)...))
end

function _spheroid_wavefunctions(m, n_max, c, xi, eta; spheroid, precision, radial_kind = 3)
    angular = _spheroid_degree_values(SpheroidalWaves.smn, m, m:n_max, c, eta;
        spheroid, precision, normalize = true)
    radial = _spheroid_degree_values(SpheroidalWaves.rmn, m, m:n_max, c, [xi];
        spheroid, precision, kind = radial_kind)
    return (; angular = angular.value,
        r1 = real.(vec(radial.value)), dr1 = real.(vec(radial.derivative)),
        r2 = imag.(vec(radial.value)), dr2 = imag.(vec(radial.derivative)))
end

function _default_spheroid_orders(k::Real, body::Spheroid)
    order = _default_mode_count(k * max(body.a, body.b))
    return order
end

function _spheroid_modal_coefficient(
        ::Rigid, m::Integer, n::Integer, c::Real, xi0::Real, kind::Symbol;
        precision::Symbol = :double)
    r3 = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 3)
    return -real(r3.derivative[1]) / r3.derivative[1]
end

function _spheroid_modal_coefficient(
        ::PressureRelease, m::Integer, n::Integer, c::Real, xi0::Real, kind::Symbol;
        precision::Symbol = :double)
    r3 = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 3)
    return -real(r3.value[1]) / r3.value[1]
end

function _spheroid_modal_coefficient(
        bc::FluidFilled, m::Integer, n::Integer, c::Real, xi0::Real, kind::Symbol;
        precision::Symbol = :double)
    g = bc.density_contrast
    c_int = c / bc.soundspeed_contrast
    r3e = SpheroidalWaves.rmn(
        m, n, c, [xi0]; spheroid = kind, precision = precision, kind = 3)
    r1i = SpheroidalWaves.rmn(
        m, n, c_int, [xi0]; spheroid = kind, precision = precision, kind = 1)

    # Pre-multiplied by r1i.derivative rather than divided through by it, which crosses zero at ordinary mode orders.
    E1 = real(r3e.value[1]) * r1i.derivative[1] - g * r1i.value[1] * real(r3e.derivative[1])
    E3 = r3e.value[1] * r1i.derivative[1] - g * r1i.value[1] * r3e.derivative[1]
    return -E1 / E3
end

# Solve for the scattered amplitudes i^n S_mn(eta_i) A_mn directly. Putting
# S_mn(eta_i) in the matrix instead makes it singular at angular nodes, even
# though the physical boundary-value problem is regular there.
function _liquid_full_coefficients(bc::FluidFilled, m::Integer, n_max::Integer,
        c::Real, c_int::Real, xi0::Real, kind::Symbol,
        eta_i::Real; eta_s::Real = -eta_i, n_quad::Integer = 64, precision::Symbol = :double)
    degrees = m:n_max
    size_ = length(degrees)
    if m > 0 && abs(eta_i) == 1
        return zeros(ComplexF64, size_), zeros(Float64, size_)
    end
    nodes, weights = gauss(n_quad)
    # Same-parity angular products are even; opposite-parity overlaps vanish.
    half = (n_quad ÷ 2 + 1):n_quad
    nodes = nodes[half]
    weights = 2 .* weights[half]
    isodd(n_quad) && (weights[1] /= 2)
    exterior = _spheroid_wavefunctions(m, n_max, c, xi0, [eta_i; eta_s; nodes];
        spheroid = kind, precision)
    interior = _spheroid_wavefunctions(m, n_max, c_int, xi0, nodes;
        spheroid = kind, precision, radial_kind = 1)
    Sext = [[exterior.angular[1, i]; exterior.angular[3:end, i]] for i in 1:size_]
    Sint = collect(eachcol(interior.angular))
    R3e = [(; value = (complex(exterior.r1[i], exterior.r2[i]),),
               derivative = (complex(exterior.dr1[i], exterior.dr2[i]),)) for i in 1:size_]
    R1i = [(; value = (interior.r1[i],), derivative = (interior.dr1[i],)) for i in 1:size_]
    A = if precision === :quad
        setprecision(BigFloat, 256) do
            _solve_fluid_coupling(BigFloat, bc.density_contrast, degrees,
                Sext, Sint, weights, R3e, R1i)
        end
    else
        _solve_fluid_coupling(Float64, bc.density_contrast, degrees,
            Sext, Sint, weights, R3e, R1i)
    end
    return A, exterior.angular[2, :]
end

function _solve_fluid_coupling(::Type{T}, g, degrees, Sext, Sint, weights,
        R3e, R1i) where {T <: AbstractFloat}
    size_ = length(degrees)
    ext = T.(hcat(Sext...))
    int = T.(hcat(Sint...))
    w = T.(weights)
    # Each weighted interior function is reused for every exterior degree.
    int .*= w
    incident = [Complex{T}(im^n) * ext[1, ni] for (ni, n) in enumerate(degrees)]
    K = Matrix{Complex{T}}(undef, size_, size_)
    rhs = zeros(Complex{T}, size_)
    for li in 1:size_
        vi = T(real(R1i[li].value[1]))
        di = T(real(R1i[li].derivative[1]))
        # Normalize before multiplying to avoid extreme radial magnitudes.
        scale = max(abs(di), abs(T(g) * vi))
        iszero(scale) &&
            throw(ArgumentError("Interior spheroidal radial function underflow; try precision=:quad or fewer modes"))
        di /= scale
        gvi = T(g) * vi / scale
        for ni in 1:size_
            # Opposite parity angular functions have exactly zero overlap.
            alpha = zero(T)
            if iseven(degrees[li] - degrees[ni])
                for qi in eachindex(w)
                    alpha += ext[qi + 1, ni] * int[qi, li]
                end
            end
            ve = Complex{T}(R3e[ni].value[1])
            de = Complex{T}(R3e[ni].derivative[1])
            e1 = real(ve) * di - gvi * real(de)
            e3 = ve * di - gvi * de
            K[li, ni] = alpha * e3
            rhs[li] -= alpha * e1 * incident[ni]
        end
    end
    # Equilibrate the radial column scales, then the boundary-equation rows.
    column_scale = [maximum(abs, @view K[:, ni]) for ni in 1:size_]
    for ni in 1:size_
        iszero(column_scale[ni]) &&
            throw(ArgumentError("Singular spheroidal coupling column; try precision=:quad or fewer modes"))
        K[:, ni] ./= column_scale[ni]
    end
    for li in 1:size_
        row_scale = maximum(abs, @view K[li, :])
        iszero(row_scale) &&
            throw(ArgumentError("Singular spheroidal coupling row; try precision=:quad or fewer modes"))
        K[li, :] ./= row_scale
        rhs[li] /= row_scale
    end
    # Solve the two independent blocks without dropping any same-parity off-diagonal 
    # coupling or degree.
    scattered = similar(rhs)
    for first_index in 1:2
        indices = first_index:2:size_
        isempty(indices) && continue
        scattered[indices] = K[indices, indices] \ rhs[indices]
    end
    scattered ./= column_scale
    # Return S_mn(eta_i) A_mn, which remains defined at incident angular nodes.
    return ComplexF64[scattered[ni] / im^n for (ni, n) in enumerate(degrees)]
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
    for m in 0:min(m_max, n_max)
        m > 0 && (abs(eta_i) == 1 || abs(eta_s) == 1) && continue
        azimuth_term = cos(m * dphi)
        iszero(azimuth_term) && continue
        nu_m = neumann_factor(m)
        if body.kind !== :prolate || precision !== :quad || boundary isa FluidFilled
            for n in m:n_max
                angular = SpheroidalWaves.smn(
                    m, n, c, [eta_i, eta_s]; spheroid = body.kind,
                    precision, normalize = true).value
                Smn_i, Smn_s = angular
                (iszero(Smn_i) || iszero(Smn_s)) && continue
                Amn = _spheroid_modal_coefficient(
                    boundary, m, n, c, body.xi0, body.kind; precision)
                total += nu_m * Smn_i * Smn_s * Amn * azimuth_term
            end
            continue
        end
        exterior = _spheroid_wavefunctions(m, n_max, c, body.xi0, [eta_i, eta_s];
            spheroid = body.kind, precision)
        for (i, n) in enumerate(m:n_max)
            Smn_i, Smn_s = exterior.angular[:, i]
            (iszero(Smn_i) || iszero(Smn_s)) && continue
            r3 = complex(exterior.r1[i], exterior.r2[i])
            d3 = complex(exterior.dr1[i], exterior.dr2[i])
            Amn = if boundary isa Rigid
                -exterior.dr1[i] / d3
            else
                -exterior.r1[i] / r3
            end
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
    for m in 0:min(m_max, n_max)
        m > 0 && (abs(eta_i) == 1 || abs(eta_s) == 1) && continue
        azimuth_term = cos(m * dphi)
        iszero(azimuth_term) && continue
        nu_m = neumann_factor(m)
        A, Smn_s = _liquid_full_coefficients(bc, m, n_max, c, c_int, body.xi0, body.kind,
            eta_i; eta_s, n_quad, precision)
        for (li, n) in enumerate(m:n_max)
            total += nu_m * Smn_s[li] * A[li] * azimuth_term
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
backend. Full fluid coupling uses a scaled direct solve in Float64 for
`:double` and BigFloat for `:quad`, retaining all requested degrees up to
`n_max` (only azimuthal orders `m <= n_max` exist). `:quad` is several times
slower but needed at large size parameters or high mode counts, where cancellation in the boundary
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
