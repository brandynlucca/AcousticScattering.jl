# A finite angular transition reconstructed from FEM far fields at several incidences.
# The body and every confocal layer are symmetric under reflection across the
# equatorial plane, so only positive incident polar quadrature nodes need FEM solves.

struct _FarFieldTMatrixData
    nodes::Vector{Float64}
    blocks::Vector{Matrix{ComplexF64}}
    report::NamedTuple
    incidence_angle::Float64
    incidence_azimuth::Float64
    scatter_angle::Float64
    scatter_azimuth::Float64
end

function _farfield_angular_basis(m::Integer, x::Real, nmax::Integer)
    return [sqrt((2l + 1) / 2 * _legendre_norm_ratio(l, m)) *
            legendre_p(l, m, x) for l in m:nmax]
end

function _farfield_transition_blocks(samples::AbstractArray{<:Complex, 3},
        nodes::AbstractVector{<:Real}, weights::AbstractVector{<:Real},
        mmax::Integer)
    ntheta = length(nodes)
    nphi = size(samples, 3)
    size(samples, 1) == ntheta && size(samples, 2) == ntheta &&
    length(weights) == ntheta && nphi >= 2mmax + 1 ||
        throw(ArgumentError("far-field samples need one scatter and incident polar grid and at least 2m_max+1 azimuths"))
    blocks = Matrix{ComplexF64}[]
    for m in 0:mmax
        fourier = [sum(samples[s, i, p] * cis(-2π * m * (p - 1) / nphi)
                   for p in 1:nphi) / nphi for s in 1:ntheta, i in 1:ntheta]
        basis = reduce(vcat, [permutedims(_farfield_angular_basis(m, x, ntheta - 1))
                              for x in nodes])
        push!(blocks, basis' * Diagonal(weights) * fourier *
                      Diagonal(weights) * basis)
    end
    return blocks
end

function _farfield_transition_amplitude(blocks, nodes, incidence_angle,
        incidence_azimuth, scatter_angle, scatter_azimuth)
    0 <= incidence_angle <= π && 0 <= scatter_angle <= π &&
    isfinite(incidence_azimuth) && isfinite(scatter_azimuth) ||
        throw(ArgumentError("polar angles must lie in [0, π] and azimuths must be finite"))
    xi, xs = cos(incidence_angle), cos(scatter_angle)
    difference = scatter_azimuth - incidence_azimuth
    total = 0.0 + 0.0im
    for m in 0:(length(blocks) - 1)
        incident = _farfield_angular_basis(m, xi, length(nodes) - 1)
        scattered = _farfield_angular_basis(m, xs, length(nodes) - 1)
        contribution = transpose(scattered) * blocks[m + 1] * incident
        total += m == 0 ? contribution : 2contribution * cos(m * difference)
    end
    return total
end

function _farfield_incidence_samples(system::_VolumeSystem, angle, nodes, phis)
    data = _volume_solution(system, angle, 0.0)
    return [_volume_amplitude(data, system.k; angle = acos(x), azimuth = phi)
            for x in nodes, phi in phis]
end

function _farfield_tmatrix(body::Spheroid, boundary::Shelled{<:LayeredMaterial},
        k::Real; polar_order::Integer = 6, m_max::Integer = 4,
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        scatter_angle::Real = π - incidence_angle,
        scatter_azimuth::Real = incidence_azimuth + π,
        points_per_wavelength::Real = 6,
        h::Union{Nothing, Real} = nothing,
        h_body::Union{Nothing, Real} = nothing,
        domain_radius::Union{Nothing, Real} = nothing,
        dtn_order::Union{Nothing, Integer} = nothing, check::Bool = true)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    body.kind === :prolate && body.a / body.b <= 1.5 && k * body.a <= 1.8 ||
        throw(ArgumentError("far-field mixed spheroid T-matrix requires a prolate body, aspect ratio ≤ 1.5 and ka ≤ 1.8"))
    boundary.interior isa FluidInterior || throw(ArgumentError(
        "far-field mixed spheroid T-matrix requires a fluid core"))
    layers, _ = _layered_materials(boundary.material)
    any(layer -> layer isa FluidLayer, layers) &&
    any(layer -> layer isa ElasticLayer, layers) &&
    all(layer -> layer isa Union{FluidLayer, ElasticLayer}, layers) ||
        throw(ArgumentError("far-field mixed spheroid T-matrix requires fluid and elastic layers"))
    iseven(polar_order) && 6 <= polar_order <= 12 &&
    0 <= m_max <= polar_order - 2 || throw(ArgumentError(
        "polar_order must be even and in 6:12, with 0 ≤ m_max ≤ polar_order-2"))
    isfinite(points_per_wavelength) && points_per_wavelength >= 6 ||
        throw(ArgumentError("points_per_wavelength must be at least 6"))
    h === nothing || (isfinite(h) && h > 0) ||
        throw(ArgumentError("h must be finite and positive when specified"))
    h_body === nothing || (isfinite(h_body) && h_body > 0) ||
        throw(ArgumentError("h_body must be finite and positive when specified"))
    dtn_order === nothing || dtn_order >= 1 ||
        throw(ArgumentError("dtn_order must be positive when specified"))

    0 <= incidence_angle <= π && 0 <= scatter_angle <= π &&
    isfinite(incidence_azimuth) && isfinite(scatter_azimuth) ||
        throw(ArgumentError("polar angles must lie in [0, π] and azimuths must be finite"))
    nodes, weights = gauss(polar_order)
    nphi = 2m_max + 1
    phis = [2π * (p - 1) / nphi for p in 1:nphi]
    bodies, materials = _layered_spheroid_volume_regions(body, boundary)
    system = _volume_system_regions(bodies, materials, k;
        closure = :dtn, points_per_wavelength, h, h_body, domain_radius, dtn_order)
    samples = zeros(ComplexF64, polar_order, polar_order, nphi)
    for i in (polar_order ÷ 2 + 1):polar_order
        samples[:, i, :] .= _farfield_incidence_samples(system,
            acos(nodes[i]), nodes, phis)
        GC.gc()
    end
    for i in 1:(polar_order ÷ 2), s in 1:polar_order, p in 1:nphi
        samples[s, i, p] = samples[polar_order + 1 - s,
            polar_order + 1 - i, p]
    end
    blocks = _farfield_transition_blocks(samples, nodes, weights, m_max)

    holdout_error = nothing
    if check
        reference = _farfield_incidence_samples(system, π / 2,
            [-1.0, 0.0, 1.0], [0.0, π])
        directions = ((0.0, 0.0, reference[3, 1]),
            (π / 2, 0.0, reference[2, 1]),
            (π / 2, π, reference[2, 2]),
            (π, 0.0, reference[1, 1]))
        holdout_error = maximum(abs(_farfield_transition_amplitude(blocks, nodes,
                                    π / 2, 0.0, angle, azimuth) - expected) /
                                max(abs(expected), 1e-6 * body.b)
        for (angle, azimuth, expected) in directions)
        holdout_error <= 0.01 || throw(ArgumentError(
            "far-field T-matrix held-out angular error is $(round(100holdout_error; sigdigits=3))%; increase polar_order and m_max or use volume FEM"))
    end
    report = (; method = :fem_farfield, polar_order, m_max,
        points_per_wavelength, closure = :dtn, holdout_error,
        fem_solves = polar_order ÷ 2 + Int(check), fem_assemblies = 1,
        h = system.diagnostics.h,
        surface_meshsize = system.diagnostics.h_body, domain_radius = system.R,
        dtn_order = system.L, cells = system.diagnostics.cells,
        dofs = system.diagnostics.dofs)
    return _farfield_tmatrix_solution(body, boundary, k, nodes, blocks, report;
        incidence_angle, incidence_azimuth, scatter_angle, scatter_azimuth)
end

function _farfield_tmatrix_solution(body, boundary, k, nodes, blocks, report;
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        scatter_angle::Real = π - incidence_angle,
        scatter_azimuth::Real = incidence_azimuth + π)
    amplitude = _farfield_transition_amplitude(blocks, nodes,
        incidence_angle, incidence_azimuth, scatter_angle, scatter_azimuth)
    data = _FarFieldTMatrixData(nodes, blocks, report, incidence_angle,
        incidence_azimuth, scatter_angle, scatter_azimuth)
    return TMatrixSolution(body, boundary, k, amplitude, data)
end
