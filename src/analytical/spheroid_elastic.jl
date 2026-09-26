# Fluid-loaded solid elastic or shelled spheroid, transition matrix in spheroidal coordinates
# (Hackman, J. Acoust. Soc. Am. 75, 35-45, 1984), built on SpheroidalWaves.jl.

using QuadGK: gauss

# Spheroidal coordinates (xi, eta, phi) of a Cartesian point, for focal half-distance f.
function _spheroidal_coordinates(p, f, spheroid::Symbol)
    x, y, z = p
    if spheroid === :prolate
        r1 = sqrt(x^2 + y^2 + (z - f)^2)
        r2 = sqrt(x^2 + y^2 + (z + f)^2)
        return (r1 + r2) / (2f), (r2 - r1) / (2f), atan(y, x)
    end
    excess = x^2 + y^2 + z^2 - f^2
    xi = sqrt((excess + sqrt(excess^2 + 4f^2 * z^2)) / (2f^2))
    return xi, z / (f * xi), atan(y, x)
end

# Distance from the axis at (xi, eta).
function _axis_distance(f, xi, eta, spheroid::Symbol)
    return f * sqrt((spheroid === :prolate ? xi^2 - 1 : xi^2 + 1) * (1 - eta^2))
end

# Metric factor of the xi direction, and the surface element per d(eta) d(phi), on the surface xi = const.
function _surface_metric(f, xi, eta, spheroid::Symbol)
    if spheroid === :prolate
        return f * sqrt((xi^2 - eta^2) / (xi^2 - 1)),
        f^2 * sqrt((xi^2 - 1) * (xi^2 - eta^2))
    end
    return f * sqrt((xi^2 + eta^2) / (xi^2 + 1)), f^2 * sqrt((xi^2 + 1) * (xi^2 + eta^2))
end

# Third-order Taylor polynomial with coefficients (f, f', f'', f''') about x0.
function _taylor3(c, x, x0)
    d = x - x0
    return ((c[4] / 6 * d + c[3] / 2) * d + c[2]) * d + c[1]
end

# Value and first three derivatives of the radial function of the given kind (1 or 2), and its separation constant.
function _elastic_radial_jet(m::Integer, n::Integer, c::Real, xi::Real, spheroid::Symbol;
        radial_kind::Integer = 1)
    lambda = SpheroidalWaves.eigenvalue(m, n, c; spheroid)
    r = SpheroidalWaves.rmn(
        m, n, c, [xi]; spheroid, kind = radial_kind, second_derivative = true)
    R, R1, R2 = real(r.value[1]), real(r.derivative[1]), real(r.second_derivative[1])
    D, sign = spheroid === :prolate ? (xi^2 - 1, -1) : (xi^2 + 1, 1)
    Q = c^2 * xi^2 - lambda + sign * m^2 / D
    dQ = 2c^2 * xi - 2sign * m^2 * xi / D^2
    R3 = -(4xi * R2 + 2R1 + Q * R1 + dQ * R) / D
    return (R, R1, R2, R3), lambda
end

# Value and first three derivatives of the unit-norm angular function at every node.
function _elastic_angular_jets(m::Integer, n::Integer, c::Real, etas::AbstractVector,
        lambda::Real, spheroid::Symbol)
    s = SpheroidalWaves.smn(
        m, n, c, etas; spheroid, normalize = true, second_derivative = true)
    sign = spheroid === :prolate ? -1 : 1
    return map(eachindex(etas)) do q
        eta = etas[q]
        S, S1, S2 = s.value[q], s.derivative[q], s.second_derivative[q]
        P = lambda + sign * c^2 * eta^2 - m^2 / (1 - eta^2)
        dP = 2sign * c^2 * eta - 2m^2 * eta / (1 - eta^2)^2
        S3 = (4eta * S2 + 2S1 - P * S1 - dP * S) / (1 - eta^2)
        (S, S1, S2, S3)
    end
end

_curl_of_jacobian(J) = [J[3, 2] - J[2, 3], J[1, 3] - J[3, 1], J[2, 1] - J[1, 2]]

# Displacement field of one basis function, as a function of the Cartesian point. `tau` 3 is longitudinal, 1 and 2 are the two transverse types.
function _elastic_basis_field(tau::Integer, m::Integer, f::Real, xi0::Real, eta0::Real,
        Rjet, Sjet, lambda::Real, wavenumber::Real, parity::Symbol, spheroid::Symbol)
    chi = p -> begin
        xi, eta, phi = _spheroidal_coordinates(p, f, spheroid)
        angle = parity === :even ? cos(m * phi) : sin(m * phi)
        _taylor3(Rjet, xi, xi0) * _taylor3(Sjet, eta, eta0) * angle
    end
    tau == 3 && return p -> ForwardDiff.gradient(chi, p) / wavenumber
    toroidal = p -> cross(ForwardDiff.gradient(chi, p), p) / sqrt(lambda)
    tau == 1 && return toroidal
    return p -> _curl_of_jacobian(ForwardDiff.jacobian(toroidal, p)) / wavenumber
end

# Local frame at a surface node, with the outward normal first.
function _elastic_node_frame(f::Real, xi0::Real, eta::Real, phi::Real, spheroid::Symbol)
    rho = _axis_distance(f, xi0, eta, spheroid)
    p = [rho * cos(phi), rho * sin(phi), f * xi0 * eta]
    gxi = ForwardDiff.gradient(q -> _spheroidal_coordinates(q, f, spheroid)[1], p)
    geta = ForwardDiff.gradient(q -> _spheroidal_coordinates(q, f, spheroid)[2], p)
    return p, gxi / norm(gxi), geta / norm(geta), [-sin(phi), cos(phi), 0.0]
end

# Basis functions (tau, parity, degree) that couple to a pressure wave for azimuthal order m.
function _elastic_basis_list(m::Integer, n_max::Integer)
    degrees = m:n_max
    basis = [(3, :even, l) for l in degrees]
    append!(basis, [(2, :even, l) for l in degrees if !(m == 0 && l == 0)])
    m > 0 && append!(basis, [(1, :odd, l) for l in degrees])
    return basis
end

# Amplitudes (u_xi, u_eta, u_phi, t_xi, t_eta, t_phi) of every basis function at every node, with the azimuthal factors removed.
function _elastic_surface_samples(m::Integer, n_max::Integer, f::Real, xi0::Real, etas,
        k::Real, material::NTuple{3, Real}, spheroid::Symbol; radial_kind::Integer = 1)
    rho, cL, cT = material
    kL, kT = k / cL, k / cT
    lame_lambda, mu = rho * (cL^2 - 2cT^2), rho * cT^2
    phi = m == 0 ? 0.0 : π / (4m)
    cos_m, sin_m = m == 0 ? (1.0, 1.0) : (cos(m * phi), sin(m * phi))
    basis = _elastic_basis_list(m, n_max)
    samples = zeros(6, length(basis), length(etas))
    frames = [_elastic_node_frame(f, xi0, eta, phi, spheroid) for eta in etas]
    for (b, (tau, parity, l)) in enumerate(basis)
        c = tau == 3 ? kL * f : kT * f
        wavenumber = tau == 3 ? kL : kT
        Rjet, lambda = _elastic_radial_jet(m, l, c, xi0, spheroid; radial_kind)
        Sjets = _elastic_angular_jets(m, l, c, etas, lambda, spheroid)
        for (q, eta) in enumerate(etas)
            p, exi, eeta, ephi = frames[q]
            u = _elastic_basis_field(
                tau, m, f, xi0, eta, Rjet, Sjets[q], lambda, wavenumber, parity, spheroid)
            value = u(p)
            J = ForwardDiff.jacobian(u, p)
            traction = (lame_lambda * (J[1, 1] + J[2, 2] + J[3, 3]) * I(3) +
                        mu * (J + J')) * exi
            samples[:, b, q] .= (dot(value, exi) / cos_m, dot(value, eeta) / cos_m,
                m == 0 ? 0.0 : dot(value, ephi) / sin_m,
                dot(traction, exi) / cos_m, dot(traction, eeta) / cos_m,
                m == 0 ? 0.0 : dot(traction, ephi) / sin_m)
        end
    end
    return basis, samples
end

# Transition matrix of a confocal elastic layer from Betti's identity on both interfaces.
# `interior` is nothing for a solid body, :vacuum for an empty shell, or (density, soundspeed) contrasts of the interior fluid.
function _elastic_layered_transition(m::Integer, n_max::Integer, k::Real, f::Real,
        xi_outer::Real, xi_inner, material::NTuple{3, Real}, interior, spheroid::Symbol;
        n_quad::Integer = 2n_max + 20)
    shell = xi_inner !== nothing
    fluid_interior = interior isa Tuple
    nodes, weights = gauss(n_quad)
    degrees = m:n_max
    nb = length(_elastic_basis_list(m, n_max))
    ne = length(degrees)
    h = k * f
    phi_normal = m == 0 ? 2π : π
    phi_tangent = m == 0 ? 0.0 : π
    sample(xi, kind) = _elastic_surface_samples(
        m, n_max, f, xi, nodes, k, material, spheroid; radial_kind = kind)[2]
    outer = (; regular = sample(xi_outer, 1))
    outer = shell ? (; outer..., outgoing = outer.regular .+ im .* sample(xi_outer, 2)) :
            outer
    inner = if shell
        regular = sample(xi_inner, 1)
        (; regular, outgoing = regular .+ im .* sample(xi_inner, 2))
    end
    # Unknowns are the regular and outgoing shell coefficients, then the exterior and interior normal-displacement coefficients.
    columns_b = 1:nb
    columns_d = shell ? ((nb + 1):(2nb)) : (1:0)
    n_shell = shell ? 2nb : nb
    columns_c = (n_shell + 1):(n_shell + ne)
    columns_e = fluid_interior ? ((n_shell + ne + 1):(n_shell + 2ne)) : (1:0)
    n_unknown = n_shell + ne + (fluid_interior ? ne : 0)
    tests = shell ? 2nb : nb
    rows_ext = (tests + 1):(tests + ne)
    rows_int = fluid_interior ? ((last(rows_ext) + 1):(last(rows_ext) + ne)) : (1:0)
    n_rows = last(fluid_interior ? rows_int : rows_ext)
    n_rows == n_unknown || error("unbalanced elastic layer system")
    A = zeros(ComplexF64, n_rows, n_unknown)
    Rhat = zeros(ComplexF64, ne, n_unknown)
    exterior = map(degrees) do l
        r1 = SpheroidalWaves.rmn(m, l, h, [xi_outer]; spheroid, kind = 1)
        r3 = SpheroidalWaves.rmn(m, l, h, [xi_outer]; spheroid, kind = 3)
        (; j = real(r1.value[1]), dj = real(r1.derivative[1]),
            h = r3.value[1], dh = r3.derivative[1],
            S = SpheroidalWaves.smn(m, l, h, nodes; spheroid, normalize = true).value)
    end
    if fluid_interior
        rho_i, c_i = interior
        k_i = k / c_i
        bulk = rho_i * c_i^2
        interior_modes = map(degrees) do l
            r1 = SpheroidalWaves.rmn(m, l, k_i * f, [xi_inner]; spheroid, kind = 1)
            (; j = real(r1.value[1]), dj = real(r1.derivative[1]),
                S = SpheroidalWaves.smn(
                    m, l, k_i * f, nodes; spheroid, normalize = true).value)
        end
    end
    unknown_sets(surface) = shell ?
                            ((columns_b, surface.regular), (columns_d, surface.outgoing)) :
                            ((columns_b, surface.regular),)
    for q in 1:n_quad
        eta = nodes[q]
        h_xi, element = _surface_metric(f, xi_outer, eta, spheroid)
        area_outer = weights[q] * element
        tests_outer = shell ? (outer.regular, outer.outgoing) : (outer.regular,)
        # Betti identity on the outer interface, one block of test functions per family.
        for (family, test) in enumerate(tests_outer), r in 1:nb

            row = (family - 1) * nb + r
            for (columns, unknown) in unknown_sets(outer), j in 1:nb

                A[row, columns[j]] += area_outer * (phi_normal *
                                       (unknown[4, j, q] * test[1, r, q] -
                                        test[5, r, q] * unknown[2, j, q]) -
                                       phi_tangent * test[6, r, q] * unknown[3, j, q])
            end
            for (e, mode) in enumerate(exterior)
                A[row, columns_c[e]] -= area_outer * phi_normal * test[4, r, q] *
                                        mode.dj * mode.S[q] / (k * h_xi)
            end
        end
        # Exterior Huygens relation for the incident and scattered coefficients.
        for (e, mode) in enumerate(exterior)
            for (columns, unknown) in unknown_sets(outer), j in 1:nb

                A[rows_ext[e], columns[j]] += area_outer * phi_normal * k *
                                              (unknown[4, j, q] * mode.dh * mode.S[q] /
                                               (k * h_xi) +
                                               k * mode.h * mode.S[q] * unknown[1, j, q])
                Rhat[e, columns[j]] += area_outer * phi_normal * k *
                                       (unknown[4, j, q] * mode.dj * mode.S[q] /
                                        (k * h_xi) +
                                        k * mode.j * mode.S[q] * unknown[1, j, q])
            end
        end
        shell || continue
        h_xi_i, element_inner = _surface_metric(f, xi_inner, eta, spheroid)
        area_inner = weights[q] * element_inner
        for (family, test) in enumerate((inner.regular, inner.outgoing)), r in 1:nb

            row = (family - 1) * nb + r
            if fluid_interior
                for (columns, unknown) in unknown_sets(inner), j in 1:nb

                    A[row, columns[j]] -= area_inner * (phi_normal *
                                           (unknown[4, j, q] * test[1, r, q] -
                                            test[5, r, q] * unknown[2, j, q]) -
                                           phi_tangent * test[6, r, q] * unknown[3, j, q])
                end
                for (e, mode) in enumerate(interior_modes)
                    A[row, columns_e[e]] += area_inner * phi_normal * test[4, r, q] *
                                            mode.dj * mode.S[q] / (k_i * h_xi_i)
                end
            else
                for (columns, unknown) in unknown_sets(inner), j in 1:nb

                    A[row, columns[j]] += area_inner * (phi_normal *
                                           (test[4, r, q] * unknown[1, j, q] +
                                            test[5, r, q] * unknown[2, j, q]) +
                                           phi_tangent * test[6, r, q] * unknown[3, j, q])
                end
            end
        end
        if fluid_interior
            for (n, mode_n) in enumerate(interior_modes)
                for (columns, unknown) in unknown_sets(inner), j in 1:nb

                    A[rows_int[n], columns[j]] += area_inner * phi_normal *
                                                  unknown[4, j, q] *
                                                  mode_n.dj * mode_n.S[q] / (k_i * h_xi_i)
                end
                for (e, mode_e) in enumerate(interior_modes)
                    A[rows_int[n], columns_e[e]] += area_inner * phi_normal * bulk * k_i *
                                                    mode_n.j * mode_n.S[q] * mode_e.dj *
                                                    mode_e.S[q] / (k_i * h_xi_i)
                end
            end
        end
    end
    rhs = zeros(ComplexF64, n_rows, ne)
    for e in 1:ne
        rhs[rows_ext[e], e] = -im
    end
    row_scale = vec(maximum(abs, A; dims = 2))
    row_scale[row_scale .== 0] .= 1
    A, rhs = A ./ row_scale, rhs ./ row_scale
    column_scale = vec(maximum(abs, A; dims = 1))
    column_scale[column_scale .== 0] .= 1
    X = (pinv(A ./ column_scale'; rtol = 1e-13) * rhs) ./ column_scale
    return -im * (Rhat * X)
end

# Far-field amplitude from a per-order transition matrix, with `transition(m)` returning T for degrees m:n_max.
function _elastic_spheroid_amplitude(transition, k::Real, body::Spheroid, incidence_angle,
        incidence_azimuth, scatter_angle, scatter_azimuth, m_max::Integer, n_max::Integer)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    m_max >= 0 && n_max >= 0 || throw(ArgumentError("m_max and n_max must be nonnegative"))
    h = k * body.q
    eta_i, eta_s = cos(incidence_angle), cos(scatter_angle)
    dphi = incidence_azimuth - scatter_azimuth
    total = zero(ComplexF64)
    for m in 0:min(m_max, n_max)
        m > 0 && (abs(eta_i) == 1 || abs(eta_s) == 1) && continue
        azimuth_term = cos(m * dphi)
        iszero(azimuth_term) && continue
        T = transition(m)
        degrees = m:n_max
        S_i = [SpheroidalWaves.smn(
                   m, l, h, [eta_i]; spheroid = body.kind, normalize = true).value[1]
               for l in degrees]
        S_s = [SpheroidalWaves.smn(
                   m, l, h, [eta_s]; spheroid = body.kind, normalize = true).value[1]
               for l in degrees]
        incident = [im^l * S_i[i] for (i, l) in enumerate(degrees)]
        scattered = (T * incident) ./ [im^l for l in degrees]
        total += neumann_factor(m) * sum(S_s .* scattered) * azimuth_term
    end
    return -2im / k * total
end

# Interior surface of the confocal shell, with the equatorial semi-axis scaled by the radius ratio.
function _confocal_inner_xi(body::Spheroid, radius_ratio::Real)
    equatorial = radius_ratio * body.b
    if body.kind === :prolate
        return sqrt(equatorial^2 + body.q^2) / body.q
    end
    equatorial > body.q || throw(ArgumentError(
        "radius_ratio must exceed $(round(body.q / body.b; sigdigits = 4)) for the confocal inner surface of this oblate spheroid to exist"))
    return sqrt(equatorial^2 - body.q^2) / body.q
end

const _ElasticSpheroidBoundary = Union{
    SolidElastic, Shelled{ElasticLayer, <:Union{FluidInterior, VacuumInterior}}}

const _ELASTIC_SHELL_CONVERGENCE_TOLERANCE = 1e-2

function _default_elastic_orders(::SolidElastic, k::Real, body::Spheroid)
    return max(8, ceil(Int, k * max(body.a, body.b)) + 6)
end

function _default_elastic_orders(::Shelled, k::Real, body::Spheroid)
    return max(14, ceil(Int, k * max(body.a, body.b)) + 12)
end

# Transition matrices depend on the frequency, body and truncation but not on any direction, so each order is computed once.
function _memoized_transition(transition)
    cache = Dict{Int, Any}()
    return m -> get!(() -> transition(m), cache, m)
end

function _elastic_transition_function(
        boundary::SolidElastic, k::Real, body::Spheroid, n_max::Integer)
    material = (boundary.density_contrast, boundary.speed_longitudinal_contrast,
        boundary.speed_transversal_contrast)
    return _memoized_transition(m -> _elastic_layered_transition(
        m, n_max, k, body.q, body.xi0, nothing, material, nothing, body.kind))
end

function _elastic_transition_function(
        boundary::Shelled{ElasticLayer, <:Union{FluidInterior, VacuumInterior}},
        k::Real, body::Spheroid, n_max::Integer)
    layer = boundary.material
    material = (layer.density_contrast, layer.speed_longitudinal_contrast,
        layer.speed_transversal_contrast)
    interior = if boundary.interior isa FluidInterior
        layer.interior_coupling === :identical_fluid ? (1.0, 1.0) :
        (boundary.interior.density_contrast, boundary.interior.soundspeed_contrast)
    else
        :vacuum
    end
    xi_inner = _confocal_inner_xi(body, boundary.radius_ratio)
    return _memoized_transition(m -> _elastic_layered_transition(
        m, n_max, k, body.q, body.xi0, xi_inner, material, interior, body.kind))
end

function _warn_elastic_truncation(change::Real)
    change > _ELASTIC_SHELL_CONVERGENCE_TOLERANCE &&
        @warn("Elastic shell amplitude changes by $(round(change; sigdigits = 2)) of its magnitude when m_max and n_max are reduced by 2. Increase m_max and n_max, and compare against bem before trusting the result.")
    return nothing
end

"""
    form_function(boundary::SolidElastic, k, body::Spheroid;
                  incidence_angle=0.0, incidence_azimuth=0.0,
                  scatter_angle=π-incidence_angle, scatter_azimuth=incidence_azimuth+π,
                  m_max=default, n_max=default)
    form_function(boundary::Shelled{ElasticLayer}, k, body::Spheroid; ..., check=true)

Far-field scattering amplitude f [m] of a fluid-loaded solid elastic spheroid, or of an
elastic shell with a fluid or empty interior, from the transition matrix in spheroidal
coordinates. The shell's inner surface is confocal with the outer surface and its equatorial
semi-axis is `radius_ratio` times the outer one. Angles are measured from the axis of symmetry,
and the defaults give monostatic backscatter. `m_max` and `n_max` truncate the azimuthal orders
and degrees. A shell solve is repeated with both reduced by 2, and a warning is emitted when the
amplitude changes by more than 1%. Pass `check = false` to skip the repeat.
"""
function form_function(boundary::_ElasticSpheroidBoundary, k::Real, body::Spheroid;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        scatter_angle::Real = π - incidence_angle,
        scatter_azimuth::Real = incidence_azimuth + π,
        m_max::Integer = _default_elastic_orders(boundary, k, body),
        n_max::Integer = _default_elastic_orders(boundary, k, body), check::Bool = true)
    amplitude(m_limit, n_limit) = _elastic_spheroid_amplitude(
        _elastic_transition_function(boundary, k, body, n_limit), k, body,
        incidence_angle, incidence_azimuth, scatter_angle, scatter_azimuth, m_limit, n_limit)
    result = amplitude(m_max, n_max)
    if check && boundary isa Shelled && n_max > 2 && m_max > 2
        _warn_elastic_truncation(abs(result - amplitude(m_max - 2, n_max - 2)) /
                                 abs(result))
    end
    return result
end
