# Prolate/oblate spheroidal modal series, built on SpheroidalWaves.jl. Liquid-filled (also covers
# gas-filled) spheroids support `coupling=:diagonal` or `:full` (default).

using QuadGK: gauss
using LinearAlgebra: lu!

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
function form_function(boundary::Union{Rigid, PressureRelease, FluidFilled},
        k::Real, body::Spheroid; incidence_angle::Real = 0.0, kwargs...)
    return _spheroid_modal_solution(body, boundary, k; incidence_angle, kwargs...).f
end

"""
    target_strength(boundary, k, body::Spheroid; kwargs...)

Target strength [dB re 1 m²] of a prolate/oblate spheroid under a rigid,
pressure-release, or fluid/gas-filled modal series. See
`form_function` for keyword arguments.
"""
function target_strength(boundary::AbstractBoundaryCondition, k::Real, body::Spheroid; kwargs...)
    return target_strength(form_function(boundary, k, body; kwargs...))
end

# Unit-normalized Ferrers coefficients avoid the factorial scales of the raw
# associated Legendre basis. Form the conversion in wide arithmetic even when
# the requested coupling system uses Float64.
function _spheroid_normalized_coefficients(m, n, c, kind, norms)
    terms = max(512, (n - m) ÷ 2 + 2ceil(Int, abs(c)) + 32)
    data = SpheroidalWaves.dmn(m, n, c;
        spheroid = kind, precision = :quad, normalize = true, max_terms = terms)
    values = map(zip(data.degrees, data.coefficients)) do (l, coefficient)
        norm = get!(norms, l) do
            norm2 = BigFloat(2) / (2l + 1)
            for j in (l - m + 1):(l + m)
                norm2 *= j
            end
            sqrt(norm2)
        end
        coefficient * norm
    end
    return (; degrees = data.degrees, values)
end

function _spheroid_angular_overlaps(::Type{T}, m, degrees, c, c_int, kind;
        n_quad, precision, overlap) where {T}
    overlap in (:auto, :coefficients, :quadrature) ||
        throw(ArgumentError("overlap must be :auto, :coefficients or :quadrature"))
    available = isdefined(SpheroidalWaves, :dmn)
    overlap === :coefficients && !available &&
        throw(ArgumentError(
            "coefficient overlaps require SpheroidalWaves 0.6 or later"))
    count = length(degrees)
    # Coefficient extraction costs more than native quadrature for modest
    # bandwidths. Use it automatically where the prolate angular safeguard
    # would otherwise evaluate long expansions at many quadrature nodes.
    use_coefficients = available && kind === :prolate && n_quad >= 32 &&
                       max(abs(c), abs(c_int)) >=
                       max(8, (first(degrees) + last(degrees)) / 2)
    if overlap === :coefficients || (overlap === :auto && use_coefficients)
        return _with_spheroid_precision(:quad) do
            # Basis norms depend only on m and the Ferrers degree, not on n or c.
            norms = Dict{Int, BigFloat}()
            exterior = [_spheroid_normalized_coefficients(m, n, c, kind, norms)
                        for n in degrees]
            interior = [_spheroid_normalized_coefficients(m, n, c_int, kind, norms)
                        for n in degrees]
            result = zeros(T, count, count)
            for i in 1:count, j in 1:count

                isodd(degrees[i] - degrees[j]) && continue
                ext, int = exterior[j], interior[i]
                # Equal parity expansions start at the same Ferrers degree.
                length_ = min(length(ext.values), length(int.values))
                result[i, j] = T(sum(ext.values[q] * int.values[q] for q in 1:length_))
            end
            return result
        end
    end
    nodes, weights = gauss(n_quad)
    half = (n_quad ÷ 2 + 1):n_quad
    nodes, weights = nodes[half], 2 .* weights[half]
    isodd(n_quad) && (weights[1] /= 2)
    exterior = _spheroid_angular_values(m, degrees, c, nodes;
        spheroid = kind, precision)
    interior = _spheroid_angular_values(m, degrees, c_int, nodes;
        spheroid = kind, precision)
    result = transpose(T.(interior)) * (T.(weights) .* T.(exterior))
    for i in 1:count, j in 1:count

        isodd(degrees[i] - degrees[j]) && (result[i, j] = zero(T))
    end
    return result
end

# Assemble once and solve all incidence right-hand sides through the same two
# parity factorizations. Scaling and the retained off-diagonal coupling match
# the scalar quadrature formulation.
function _spheroid_coupled_modes(::Type{T}, g, degrees, overlap, radial, interior,
        angular) where {T}
    count = length(degrees)
    scattered = zeros(Complex{T}, count, size(angular, 1))
    for parity in 0:1
        indices = findall(n -> mod(n, 2) == parity, degrees)
        isempty(indices) && continue
        incoming = @view angular[:, indices]
        all(iszero, incoming) && continue
        size_ = length(indices)
        K = Matrix{Complex{T}}(undef, size_, size_)
        forcing = Matrix{T}(undef, size_, size_)
        for (i, ni) in enumerate(indices)
            vi, di = T(real(interior.value[ni])), T(real(interior.derivative[ni]))
            scale = max(abs(di), abs(T(g) * vi))
            iszero(scale) && throw(ArgumentError(
                "Interior spheroidal radial function underflow; try precision=:quad or fewer modes"))
            di, gvi = di / scale, T(g) * vi / scale
            for (j, nj) in enumerate(indices)
                ve, de = Complex{T}(radial.value[nj]), Complex{T}(radial.derivative[nj])
                K[i, j] = overlap[ni, nj] * (ve * di - gvi * de)
                forcing[i, j] = -overlap[ni, nj] * (real(ve) * di - gvi * real(de))
            end
        end
        columns = vec(maximum(abs, K; dims = 1))
        all(>(0), columns) || throw(ArgumentError("Singular spheroidal coupling column"))
        K ./= transpose(columns)
        rows = vec(maximum(abs, K; dims = 2))
        all(>(0), rows) || throw(ArgumentError("Singular spheroidal coupling row"))
        K ./= rows
        # Within a parity block i^n has a common phase times a real sign.
        # The common phase cancels when converting the solved coefficients
        # back, so the incident multiplication needs only real arithmetic.
        signs = [isodd((degrees[j] - parity) ÷ 2) ? -1 : 1 for j in indices]
        incident = T.(transpose(incoming)) .* signs
        rhs = (forcing * incident) ./ rows
        scattered[indices, :] = ((lu!(K) \ rhs) ./ columns) .* signs
    end
    return scattered
end

# Preserve an exact equatorial direction instead of cos(pi/2)'s rounding error.
function _spheroid_direction_cosine(angle)
    half_pi = oftype(float(angle), pi) / 2
    return abs(rem(angle, 2half_pi)) == half_pi ? zero(float(angle)) : cos(angle)
end

# First-kind angular functions have parity (-1)^(n-m). Evaluate each reflected
# coordinate once, retaining the dependency's contiguous-degree batch interface.
function _spheroid_angular_values(m, degrees, c, eta; spheroid, precision)
    points = unique(abs.(eta))
    lookup = Dict(x => i for (i, x) in enumerate(points))
    indices = [lookup[abs(x)] for x in eta]
    full_degrees = first(degrees):last(degrees)
    angular = _spheroid_degree_values(SpheroidalWaves.smn, m, full_degrees, c, points;
        spheroid, precision, normalize = true).value
    values = angular[indices, degrees .- first(degrees) .+ 1]
    for (j, n) in enumerate(degrees)
        isodd(n - m) || continue
        for (i, x) in enumerate(eta)
            if iszero(x)
                values[i, j] = zero(eltype(values))
            elseif x < 0
                values[i, j] = -values[i, j]
            end
        end
    end
    return values
end

function _with_spheroid_precision(f, precision)
    # Coefficient normalization uses wide intermediates even for double output.
    # Keep a reduced caller context from degrading either precision setting.
    bits = precision === :quad ? 256 : max(256, Base.precision(BigFloat))
    return Base.precision(BigFloat) == bits ? f() : setprecision(f, BigFloat, bits)
end

function _spheroid_modal_modes(body, boundary, k, eta_i, eta_s;
        precision::Symbol = :double, kwargs...)
    return _with_spheroid_precision(precision) do
        _spheroid_modal_modes_impl(body, boundary, k, eta_i, eta_s; precision, kwargs...)
    end
end

function _spheroid_modal_modes_impl(body, boundary, k, eta_i, eta_s;
        m_max::Integer = _default_spheroid_orders(k, body),
        n_max::Integer = _default_spheroid_orders(k, body), n_quad::Integer = 64,
        precision::Symbol = :double, overlap::Symbol = :auto)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    boundary isa Union{Rigid, PressureRelease, FluidFilled} || throw(ArgumentError(
        "acoustic spheroid modal solutions require Rigid, PressureRelease or FluidFilled"))
    m_max >= 0 && n_max >= 0 || throw(ArgumentError("modal orders must be nonnegative"))
    n_quad >= 2 || throw(ArgumentError("n_quad must be at least two"))
    precision in (:double, :quad) ||
        throw(ArgumentError("precision must be :double or :quad"))
    overlap in (:auto, :coefficients, :quadrature) ||
        throw(ArgumentError("overlap must be :auto, :coefficients or :quadrature"))
    T = precision === :quad ? BigFloat : Float64
    c = k * body.q
    count = length(eta_i)
    modes = Matrix{Complex{T}}[]
    receivers = Matrix{T}[]
    for m in 0:min(m_max, n_max)
        degrees = m:n_max
        if m > 0 && all(x -> abs(x) == 1, eta_i)
            push!(modes, zeros(Complex{T}, length(degrees), count))
            push!(receivers, zeros(T, length(eta_s), length(degrees)))
            continue
        end
        angular = _spheroid_angular_values(m, degrees, c,
            [eta_i; eta_s]; spheroid = body.kind, precision)
        incoming = @view angular[1:count, :]
        outgoing = T.(angular[(count + 1):end, :])
        radial = _spheroid_degree_values(SpheroidalWaves.rmn, m, degrees, c,
            [body.xi0]; spheroid = body.kind, precision, kind = 3)
        interior = if boundary isa FluidFilled
            _spheroid_degree_values(SpheroidalWaves.rmn, m, degrees,
                c / boundary.soundspeed_contrast, [body.xi0];
                spheroid = body.kind, precision, kind = 1)
        else
            nothing
        end
        coefficients = if boundary isa FluidFilled && boundary.coupling === :full
            # A broadside plane wave has no odd (n-m) forcing. The two parity
            # blocks are exactly decoupled, so only the even block is needed.
            active = all(iszero, eta_i) ? (1:2:length(degrees)) : (1:length(degrees))
            selected_degrees = degrees[active]
            alpha = _spheroid_angular_overlaps(T, m, selected_degrees, c,
                c / boundary.soundspeed_contrast, body.kind; n_quad, precision, overlap)
            selected_radial = (; value = radial.value[active],
                derivative = radial.derivative[active])
            selected_interior = (; value = interior.value[active],
                derivative = interior.derivative[active])
            result = zeros(Complex{T}, length(degrees), count)
            result[active, :] = _spheroid_coupled_modes(T, boundary.density_contrast,
                selected_degrees, alpha, selected_radial, selected_interior,
                incoming[:, active])
            result
        else
            diagonal = map(eachindex(degrees)) do j
                ve, de = radial.value[j], radial.derivative[j]
                if boundary isa Rigid
                    -real(de) / de
                elseif boundary isa PressureRelease
                    -real(ve) / ve
                else
                    vi, di = interior.value[j], interior.derivative[j]
                    -(real(ve) * di - boundary.density_contrast * vi * real(de)) /
                    (ve * di - boundary.density_contrast * vi * de)
                end
            end
            Complex{T}.(diagonal .* transpose(incoming))
        end
        push!(modes, coefficients)
        push!(receivers, outgoing)
    end
    return modes, receivers
end

function _spheroid_modal_solution(body, boundary, k;
        incidence_angle = pi / 2, incidence_azimuth = 0.0,
        scatter_angle = pi - incidence_angle, scatter_azimuth = incidence_azimuth + pi,
        precision = :double, kwargs...)
    all(isfinite, (incidence_angle, incidence_azimuth, scatter_angle, scatter_azimuth)) ||
        throw(ArgumentError("incidence and observation angles must be finite"))
    return _with_spheroid_precision(precision) do
        eta_i = _spheroid_direction_cosine(incidence_angle)
        eta_s = scatter_angle == pi - incidence_angle ? -eta_i :
                _spheroid_direction_cosine(scatter_angle)
        modes, receivers = _spheroid_modal_modes(body, boundary, k,
            [eta_i], [eta_s]; precision, kwargs...)
        total = sum(neumann_factor(m) * cos(m * (incidence_azimuth - scatter_azimuth)) *
                    sum(receivers[m + 1][1, :] .* modes[m + 1][:, 1])
        for m in 0:(length(modes) - 1))
        data = _SpheroidModalData([vec(mode) for mode in modes], precision,
            Float64(incidence_angle), Float64(incidence_azimuth),
            Float64(scatter_angle), Float64(scatter_azimuth))
        return ModalSolution(body, boundary, Float64(k), ComplexF64(-2im / k * total), data)
    end
end

function _spheroid_modal_amplitudes(sol, angles, azimuths)
    data = sol.data
    data isa _SpheroidModalData || throw(ArgumentError(
        "bistatic modal sampling requires a retained plane-wave spheroid solution"))
    return _with_spheroid_precision(data.precision) do
        _spheroid_modal_amplitudes_impl(sol, angles, azimuths)
    end
end

function _spheroid_modal_amplitudes_impl(sol, angles, azimuths)
    data = sol.data
    length(angles) == length(azimuths) && !isempty(angles) &&
    all(isfinite, angles) && all(isfinite, azimuths) ||
        throw(ArgumentError("supply nonempty, equally sized finite direction arrays"))
    cosines = _spheroid_direction_cosine.(angles)
    eta = unique(cosines)
    lookup = Dict(x => i for (i, x) in enumerate(eta))
    indices = [lookup[x] for x in cosines]
    T = data.precision === :quad ? BigFloat : Float64
    amplitudes = zeros(Complex{T}, length(angles))
    for m in 0:(length(data.coefficients) - 1)
        coefficients = data.coefficients[m + 1]
        all(iszero, coefficients) && continue
        angular = _spheroid_angular_values(m,
            m:(m + length(coefficients) - 1), sol.k * sol.body.q, eta;
            spheroid = sol.body.kind, precision = data.precision)
        polar = angular * coefficients
        for i in eachindex(angles)
            amplitudes[i] += neumann_factor(m) *
                             cos(m * (data.incidence_azimuth - azimuths[i])) *
                             polar[indices[i]]
        end
    end
    return ComplexF64.((-2im / sol.k) .* amplitudes)
end
