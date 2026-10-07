# Project Cauchy data on the target's confocal surface. Angular functions are
# orthonormal in eta; cosine/sine Fourier coefficients include the Neumann factor.
# Coordinates here use the public x-polar body frame, not the elastic kernel's z frame.
function _spheroid_incident_traces(body, k, incident, m_max, n_max;
        incidence_angle, incidence_azimuth, incident_n_eta, incident_n_phi, precision)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    m_max >= 0 && n_max >= 0 || throw(ArgumentError("modal orders must be nonnegative"))
    incident_n_eta > n_max || throw(ArgumentError("incident_n_eta must exceed n_max"))
    incident_n_phi > 2min(m_max, n_max) ||
        throw(ArgumentError("incident_n_phi must exceed twice the retained azimuthal order"))
    if incident isa SphericalWave
        source = incident.range * incident.direction
        (source[1] / body.a)^2 + (source[2]^2 + source[3]^2) / body.b^2 > 1 ||
            throw(ArgumentError("spherical source must lie outside the spheroid"))
    end
    _, pinc, gradinc = _incident_callbacks(k, incidence_angle, incidence_azimuth; incident)
    eta, weights = gauss(incident_n_eta)
    phi = 2pi .* (0:(incident_n_phi - 1)) ./ incident_n_phi
    p = zeros(ComplexF64, incident_n_eta, incident_n_phi)
    dp = similar(p)
    xi, q = body.xi0, body.q
    radial = sqrt(xi^2 + (body.kind === :prolate ? -1 : 1))
    for j in eachindex(phi), i in eachindex(eta)

        transverse = sqrt(1 - eta[i]^2)
        cp, sp = cos(phi[j]), sin(phi[j])
        x = q * SVector(xi * eta[i], radial * transverse * cp, radial * transverse * sp)
        tangent = q *
                  SVector(eta[i], xi / radial * transverse * cp, xi / radial * transverse *
                                                                 sp)
        p[i, j] = pinc(x)
        gradient = gradinc(x)
        length(gradient) == 3 && all(isfinite, gradient) ||
            throw(ArgumentError("incident gradient must have three finite components"))
        dp[i, j] = sum(gradient[d] * tangent[d] for d in 1:3)
    end
    all(isfinite, p) && all(isfinite, dp) ||
        throw(ArgumentError("incident traces must be finite on the spheroid"))
    return map(0:min(m_max, n_max)) do m
        angular = _spheroid_degree_values(SpheroidalWaves.smn, m, m:n_max,
            k * q, eta; spheroid = body.kind, precision, normalize = true).value
        fourier = hcat(cos.(m .* phi), sin.(m .* phi)) .*
                  (neumann_factor(m) / incident_n_phi)
        projection = transpose(angular) .* transpose(weights)
        (; p = projection * (p * fourier), dp = projection * (dp * fourier))
    end
end

function _spheroid_fluid_incident_scattered(boundary, body, k, m, n_max, waves, data;
        n_quad, precision)
    n_quad > n_max || throw(ArgumentError("n_quad must exceed n_max for fluid coupling"))
    nodes, weights = gauss(n_quad)
    ext = _spheroid_degree_values(SpheroidalWaves.smn, m, m:n_max, k * body.q, nodes;
        spheroid = body.kind, precision, normalize = true).value
    interior = _spheroid_wavefunctions(m, n_max, k * body.q / boundary.soundspeed_contrast,
        body.xi0, nodes; spheroid = body.kind, precision, radial_kind = 1)
    overlap = transpose(interior.angular) * (weights .* ext)
    count = n_max - m + 1
    K = zeros(eltype(data.p), count, count)
    rhs = zeros(eltype(data.p), count, 2)
    for i in 1:count
        scale = max(abs(interior.dr1[i]), abs(boundary.density_contrast * interior.r1[i]))
        scale > 0 || throw(ArgumentError("interior radial underflow; reduce modal orders"))
        di, gvi = interior.dr1[i] / scale,
        boundary.density_contrast * interior.r1[i] / scale
        for j in 1:count
            alpha = boundary.coupling === :diagonal ? (i == j ? one(scale) : zero(scale)) :
                    (iseven(i - j) ? overlap[i, j] : zero(scale))
            K[i, j] = alpha * (complex(waves.r1[j], waves.r2[j]) * di -
                       gvi * complex(waves.dr1[j], waves.dr2[j]))
            rhs[i, :] .-= alpha .* (di .* data.p[j, :] - gvi .* data.dp[j, :])
        end
    end
    columns = vec(maximum(abs, K; dims = 1))
    all(>(0), columns) || throw(ArgumentError("singular fluid coupling columns"))
    K ./= transpose(columns)
    rows = vec(maximum(abs, K; dims = 2))
    all(>(0), rows) || throw(ArgumentError("singular fluid coupling rows"))
    return ((K ./ rows) \ (rhs ./ rows)) ./ columns
end

function _spheroid_incident_amplitude(boundary, k, body;
        incident::IncidentField, incidence_angle::Real = pi / 2, incidence_azimuth::Real = 0,
        scatter_angle::Real = pi - incidence_angle, scatter_azimuth::Real = incidence_azimuth +
                                                                            pi,
        m_max::Integer = _default_spheroid_orders(k, body),
        n_max::Integer = _default_spheroid_orders(k, body),
        incident_n_eta::Integer = max(32, 2n_max + 12),
        incident_n_phi::Integer = max(32, 4m_max + 4), precision::Symbol = :double,
        n_quad::Integer = 64, transition = nothing)
    boundary isa Union{Rigid, PressureRelease, FluidFilled} || transition !== nothing ||
        throw(ArgumentError("custom incident fields on spheroids require acoustic modal or elastic tmatrix"))
    all(isfinite, (incidence_angle, incidence_azimuth, scatter_angle, scatter_azimuth)) ||
        throw(ArgumentError("incidence and observation angles must be finite"))
    traces = _spheroid_incident_traces(body, k, incident, m_max, n_max;
        incidence_angle, incidence_azimuth, incident_n_eta, incident_n_phi, precision)
    total = zero(ComplexF64)
    for m in 0:min(m_max, n_max)
        waves = _spheroid_wavefunctions(
            m, n_max, k * body.q, body.xi0, [cos(scatter_angle)];
            spheroid = body.kind, precision)
        data = traces[m + 1]
        scattered = if boundary isa Rigid
            -data.dp ./ complex.(waves.dr1, waves.dr2)
        elseif boundary isa PressureRelease
            -data.p ./ complex.(waves.r1, waves.r2)
        elseif boundary isa FluidFilled
            _spheroid_fluid_incident_scattered(boundary, body, k, m, n_max, waves, data;
                n_quad, precision)
        else
            # Use both regular traces to avoid division by a radial value or derivative zero.
            scale = max.(abs.(waves.r1), abs.(waves.dr1))
            all(>(0), scale) ||
                throw(ArgumentError("regular radial underflow; reduce modal orders"))
            v, d = waves.r1 ./ scale, waves.dr1 ./ scale
            coefficients = (v .* data.p + d .* data.dp) ./ ((v .^ 2 + d .^ 2) .* scale)
            transition(m) * coefficients
        end
        azimuth = SVector(cos(m * scatter_azimuth), sin(m * scatter_azimuth))
        for (i, n) in enumerate(m:n_max)
            total += waves.angular[1, i] * sum(scattered[i, j] * azimuth[j] for j in 1:2) /
                     im^n
        end
    end
    return ComplexF64(-im / k * total)
end
