struct _FullMFSSurfaceData
    sources::Vector{NTuple{3, Float64}}
    coefficients::Vector{ComplexF64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    diagnostics::NamedTuple
end

function _surface_mfs_system(quad, sources, boundary, k, incident)
    points = [Tuple(q.coords) for q in quad]
    normals = [Tuple(q.normal) for q in quad]
    matrix = Matrix{ComplexF64}(undef, length(quad), length(sources))
    rhs = Vector{ComplexF64}(undef, length(quad))
    Threads.@threads for i in eachindex(points)
        pressure = cis(k * _dot3(incident, points[i]))
        rhs[i] = boundary isa PressureRelease ? -pressure :
                 -im * k * _dot3(incident, normals[i]) * pressure
        for j in eachindex(sources)
            matrix[i, j] = boundary isa PressureRelease ?
                           _green3d(k, points[i], sources[j]) :
                           _dgreen3d_dn(k, points[i], normals[i], sources[j])
        end
    end
    return matrix, rhs
end

"""
    mfs(surface::Mesh, boundary::Union{Rigid,PressureRelease}, k;
        offset, source_mesh=surface, check_mesh=nothing,
        incidence_angle=π/2, incidence_azimuth=0, condition_limit=512)

Full three-dimensional point-source MFS on a closed surface. Place one source at each
quadrature point of `source_mesh`, displaced inward by `offset` [m]. Enforce the boundary
condition at `surface`'s quadrature points with a dense least-squares solve. Both meshes
must describe the same body. Use a coarser, low-quadrature-order `source_mesh` to keep the
number of sources below the number of collocation points.

Source placement is suitable for smooth bodies when all sources remain strictly inside.
Sharp edges and strongly concave surfaces can make normal offsets unreliable. Convergence
requires separate source-spacing, offset and collocation checks. An optional `check_mesh`
on the same body evaluates the boundary residual away from collocation points.
`diagnostics` reports this residual, the fitted residual and optional matrix conditioning.

Incidence uses the full-BEM convention, `(cos(β), sin(β)cos(α), sin(β)sin(α))`.
`scattering_amplitude(solution; direction)` evaluates the outgoing sources' far-field
amplitude [m] directly; the default direction is monostatic backscatter.
"""
function mfs(
        surface::Mesh{<:Inti.Quadrature}, boundary::Union{Rigid, PressureRelease}, k::Real;
        offset::Real, source_mesh::Mesh{<:Inti.Quadrature} = surface,
        check_mesh::Union{Nothing, Mesh{<:Inti.Quadrature}} = nothing,
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        condition_limit::Integer = 512)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    isfinite(offset) && offset > 0 ||
        throw(ArgumentError("offset must be finite and positive"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    length(source_mesh.data) <= length(surface.data) || throw(ArgumentError(
        "source_mesh must not have more points than the collocation surface"))
    incident = Tuple(_bem3d_incidence_direction(incidence_angle, incidence_azimuth))
    sources = [Tuple(q.coords - offset * q.normal) for q in source_mesh.data]
    matrix, rhs = _surface_mfs_system(surface.data, sources, boundary, k, incident)
    coefficients = matrix \ rhs
    boundary_residual = if check_mesh === nothing
        nothing
    else
        check_matrix, check_rhs = _surface_mfs_system(
            check_mesh.data, sources, boundary, k, incident)
        _linear_residual(check_matrix, coefficients, check_rhs)
    end
    report = (; method = size(matrix, 1) == size(matrix, 2) ? :direct : :least_squares,
        discretization = :full,
        converged = nothing, iterations = nothing, residual_history = nothing,
        _linear_residual(matrix, coefficients, rhs)...,
        _mfs_matrix_diagnostics(matrix; condition_limit)...,
        source_count = length(sources), collocation_count = length(surface.data),
        unknown_count = length(sources), equation_count = length(surface.data),
        check_count = check_mesh === nothing ? 0 : length(check_mesh.data), boundary_residual,
        solver_options = (; offset, incidence_angle, incidence_azimuth))
    data = _FullMFSSurfaceData(sources, coefficients, Float64(incidence_angle),
        Float64(incidence_azimuth), report)
    return MFSSolution(surface.body, boundary, Float64(k), data)
end

function scattering_amplitude(sol::MFSSolution{_FullMFSSurfaceData};
        direction::Union{Nothing, AbstractVector} = nothing)
    d = sol.data
    q = direction === nothing ?
        -_bem3d_incidence_direction(d.incidence_angle, d.incidence_azimuth) : direction
    length(q) == 3 && all(isfinite, q) && norm(q) > 0 || throw(ArgumentError(
        "direction must be a finite nonzero three-vector"))
    qhat = Tuple(q / norm(q))
    return sum(c * cis(-sol.k * _dot3(qhat, y))
    for (c, y) in zip(d.coefficients, d.sources)) / (4π)
end

function target_strength(sol::MFSSolution{_FullMFSSurfaceData}; kwargs...)
    return target_strength(scattering_amplitude(sol; kwargs...))
end
