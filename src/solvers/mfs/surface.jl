struct _FullMFSSurfaceData
    quad::Inti.Quadrature
    sources::Vector{NTuple{3, Float64}}
    coefficients::Vector{ComplexF64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    pinc::Any
    diagnostics::NamedTuple
    incident::Union{Nothing, IncidentField}
    interior_sources::Union{Nothing, Vector{NTuple{3, Float64}}}
    interior_coefficients::Union{Nothing, Vector{ComplexF64}}
end

function _FullMFSSurfaceData(quad, sources, coefficients, beta, alpha, pinc, report)
    _FullMFSSurfaceData(quad, sources, coefficients, beta, alpha, pinc, report,
        nothing, nothing, nothing)
end

function _FullMFSSurfaceData(
        quad, sources, coefficients, beta, alpha, pinc, report, incident)
    _FullMFSSurfaceData(quad, sources, coefficients, beta, alpha, pinc, report,
        incident, nothing, nothing)
end

_incident_pressure(sol::MFSSolution{_FullMFSSurfaceData}, point) = sol.data.pinc(point)

function _surface_mfs_system(quad, sources, boundary, k, pinc, gradinc)
    points = [Tuple(q.coords) for q in quad]
    normals = [Tuple(q.normal) for q in quad]
    matrix = Matrix{ComplexF64}(undef, length(quad), length(sources))
    rhs = Vector{ComplexF64}(undef, length(quad))
    Threads.@threads for i in eachindex(points)
        rhs[i] = boundary isa PressureRelease ? -pinc(points[i]) :
                 -_dot3(gradinc(points[i]), normals[i])
        for j in eachindex(sources)
            matrix[i, j] = boundary isa PressureRelease ?
                           _green3d(k, points[i], sources[j]) :
                           _dgreen3d_dn(k, points[i], normals[i], sources[j])
        end
    end
    return matrix, rhs
end

function _surface_fluid_mfs_system(quad, exterior, interior, boundary, k, pinc, gradinc)
    n, ne, ni = length(quad), length(exterior), length(interior)
    matrix = Matrix{ComplexF64}(undef, 2n, ne + ni)
    rhs = Vector{ComplexF64}(undef, 2n)
    return _fluid_mfs_interface!(matrix, rhs, quad, exterior, interior, boundary, k,
        pinc, gradinc)
end

function _surface_fluid_mfs_sources(surface, source_mesh, offset_ext, offset_int)
    exterior = [Tuple(q.coords - offset_ext * q.normal) for q in source_mesh.data]
    interior = [Tuple(q.coords + offset_int * q.normal) for q in source_mesh.data]
    patches = _region_patches(surface.data, 0)
    all(point -> _surface_location(patches, point) === :inside, exterior) ||
        throw(ArgumentError("exterior MFS sources must lie strictly inside the closed surface; reduce offset_ext or refine source_mesh"))
    all(point -> _surface_location(patches, point) === :outside, interior) ||
        throw(ArgumentError("interior MFS sources must lie strictly outside the closed surface; reduce offset_int or refine source_mesh"))
    return exterior, interior
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
amplitude [m] directly; the default direction is monostatic backscatter for a plane wave.
"""
function mfs(
        surface::Mesh{<:Inti.Quadrature}, boundary::Union{Rigid, PressureRelease}, k::Real;
        offset::Real, source_mesh::Mesh{<:Inti.Quadrature} = surface,
        check_mesh::Union{Nothing, Mesh{<:Inti.Quadrature}} = nothing,
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        incident = nothing,
        condition_limit::Integer = 512)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    isfinite(offset) && offset > 0 ||
        throw(ArgumentError("offset must be finite and positive"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    length(source_mesh.data) <= length(surface.data) || throw(ArgumentError(
        "source_mesh must not have more points than the collocation surface"))
    field, pinc, gradinc = _incident_callbacks(k, incidence_angle, incidence_azimuth; incident)
    sources = [Tuple(q.coords - offset * q.normal) for q in source_mesh.data]
    matrix, rhs = _surface_mfs_system(surface.data, sources, boundary, k, pinc, gradinc)
    coefficients = matrix \ rhs
    boundary_residual = if check_mesh === nothing
        nothing
    else
        check_matrix, check_rhs = _surface_mfs_system(
            check_mesh.data, sources, boundary, k, pinc, gradinc)
        _linear_residual(check_matrix, coefficients, check_rhs)
    end
    report = (; method = size(matrix, 1) == size(matrix, 2) ? :direct : :least_squares,
        illumination = field === nothing ? :plane_wave : :prescribed,
        discretization = :full,
        converged = nothing, iterations = nothing, residual_history = nothing,
        _linear_residual(matrix, coefficients, rhs)...,
        _mfs_matrix_diagnostics(matrix; condition_limit)...,
        source_count = length(sources), collocation_count = length(surface.data),
        unknown_count = length(sources), equation_count = length(surface.data),
        check_count = check_mesh === nothing ? 0 : length(check_mesh.data), boundary_residual,
        solver_options = (; offset, incidence_angle, incidence_azimuth))
    data = _FullMFSSurfaceData(
        surface.data, sources, coefficients, Float64(incidence_angle),
        Float64(incidence_azimuth), pinc, report, field)
    return MFSSolution(surface.body, boundary, Float64(k), data)
end

"""
    mfs(surface::Mesh, boundary::FluidFilled, k;
        offset_ext, offset_int=offset_ext, source_mesh=surface, check_mesh=nothing,
        incidence_angle=π/2, incidence_azimuth=0, condition_limit=512)

Full-3D two-domain MFS on a closed surface. Exterior scattered sources are
offset inward; interior transmitted sources are offset outward. Both source
sets are checked against the curved surface before solving pressure and
density-scaled normal-derivative continuity. Use separate, coarser
`source_mesh` and independent `check_mesh` to control source spacing and
held-out residuals. Curved, genuinely capped cylinders are supported when
the source-side checks pass. `diagnostics` retains separate held-out pressure
and velocity residuals when `check_mesh` is provided.
"""
function mfs(surface::Mesh{<:Inti.Quadrature}, boundary::FluidFilled, k::Real;
        offset_ext::Real, offset_int::Real = offset_ext,
        source_mesh::Mesh{<:Inti.Quadrature} = surface,
        check_mesh::Union{Nothing, Mesh{<:Inti.Quadrature}} = nothing,
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        incident = nothing, condition_limit::Integer = 512)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    isfinite(offset_ext) && offset_ext > 0 ||
        throw(ArgumentError("offset_ext must be finite and positive"))
    isfinite(offset_int) && offset_int > 0 ||
        throw(ArgumentError("offset_int must be finite and positive"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    length(source_mesh.data) <= length(surface.data) || throw(ArgumentError(
        "source_mesh must not have more points than the collocation surface"))
    field, pinc, gradinc = _incident_callbacks(k, incidence_angle, incidence_azimuth; incident)
    exterior, interior = _surface_fluid_mfs_sources(
        surface, source_mesh, offset_ext, offset_int)
    matrix, rhs = _surface_fluid_mfs_system(
        surface.data, exterior, interior, boundary, k, pinc, gradinc)
    coefficients = matrix \ rhs
    ne = length(exterior)
    exterior_coefficients = coefficients[1:ne]
    interior_coefficients = coefficients[(ne + 1):end]
    boundary_residual = pressure_residual = velocity_residual = nothing
    if check_mesh !== nothing
        check_matrix, check_rhs = _surface_fluid_mfs_system(
            check_mesh.data, exterior, interior, boundary, k, pinc, gradinc)
        nc = length(check_mesh.data)
        boundary_residual = _linear_residual(check_matrix, coefficients, check_rhs)
        pressure_residual = _linear_residual(
            view(check_matrix, 1:nc, :), coefficients, view(check_rhs, 1:nc))
        velocity_residual = _linear_residual(
            view(check_matrix, (nc + 1):2nc, :), coefficients,
            view(check_rhs, (nc + 1):2nc))
    end
    report = (; method = size(matrix, 1) == size(matrix, 2) ? :direct : :least_squares,
        illumination = field === nothing ? :plane_wave : :prescribed,
        discretization = :full, converged = nothing, iterations = nothing,
        residual_history = nothing, _linear_residual(matrix, coefficients, rhs)...,
        _mfs_matrix_diagnostics(matrix; condition_limit)...,
        source_count = length(exterior) + length(interior),
        collocation_count = length(surface.data), unknown_count = length(coefficients),
        equation_count = size(matrix, 1),
        check_count = check_mesh === nothing ? 0 : length(check_mesh.data),
        boundary_residual, pressure_residual, velocity_residual,
        solver_options = (; offset_ext, offset_int, incidence_angle, incidence_azimuth))
    data = _FullMFSSurfaceData(surface.data, exterior, exterior_coefficients,
        Float64(incidence_angle), Float64(incidence_azimuth), pinc, report,
        field, interior, interior_coefficients)
    return MFSSolution(surface.body, boundary, Float64(k), data)
end
