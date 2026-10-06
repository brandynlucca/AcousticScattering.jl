# Pérez-Arjona's pressure-release swimbladder and scalar-fluid backbone.
# The backbone model has no shear or elastic displacement field.
struct _BladderBackboneBoundary <: AbstractBoundaryCondition
    backbone::FluidFilled
end

struct _BladderBackboneMFSData
    exterior_sources::Vector{NTuple{3, Float64}}
    exterior_coefficients::Vector{ComplexF64}
    interior_sources::Vector{NTuple{3, Float64}}
    interior_coefficients::Vector{ComplexF64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    pinc::Any
    diagnostics::NamedTuple
end

function _bladder_backbone_system(bladder_quad, backbone_quad, exterior_sources,
        interior_sources, backbone::FluidFilled, k, pinc, gradinc)
    nbl = length(bladder_quad)
    nb = length(backbone_quad)
    ne = length(exterior_sources)
    ni = length(interior_sources)
    matrix = Matrix{ComplexF64}(undef, nbl + 2nb, ne + ni)
    rhs = Vector{ComplexF64}(undef, nbl + 2nb)
    # Pressure release on the bladder involves only the exterior source field.
    Threads.@threads for i in eachindex(bladder_quad)
        point = Tuple(bladder_quad[i].coords)
        rhs[i] = -pinc(point)
        for j in eachindex(exterior_sources)
            matrix[i, j] = _green3d(k, point, exterior_sources[j])
        end
        for j in eachindex(interior_sources)
            matrix[i, ne + j] = 0
        end
    end
    _fluid_mfs_interface!(matrix, rhs, backbone_quad, exterior_sources, interior_sources,
        backbone, k, pinc, gradinc; pressure_offset = nbl, velocity_offset = nbl + nb)
    return matrix, rhs
end

function _bladder_backbone_residuals(matrix, coefficients, rhs, nbl, nb)
    residual = matrix * coefficients - rhs
    relative(rows) = norm(view(residual, rows)) /
                     max(norm(view(rhs, rows)), eps(Float64))
    return (;
        bladder_pressure = relative(1:nbl),
        backbone_pressure = relative((nbl + 1):(nbl + nb)),
        backbone_velocity = relative((nbl + nb + 1):(nbl + 2nb)))
end

"""
    mfs(swimbladder::Mesh, backbone::Mesh, backbone_material::FluidFilled, k;
        bladder_offset, backbone_offset_ext, backbone_offset_int,
        bladder_source_mesh=swimbladder, backbone_source_mesh=backbone,
        check_bladder=nothing, check_backbone=nothing,
        incidence_angle=π/2, incidence_azimuth=0, condition_limit=512)

Coupled full-3D MFS for a pressure-release swimbladder and a disjoint penetrable
backbone, following Pérez-Arjona et al. (2020), DOI 10.1093/icesjms/fsaa160.
The package uses `exp(-iωt)` and outgoing `exp(ikr)/r`; the paper prints the
opposite Green-function phase convention. Both arguments must be closed,
outward-oriented full-3D meshes. `backbone_material` is a **scalar fluid** with
density and sound-speed contrasts relative to water; it has no shear waves.

Place exterior sources inward from both surfaces and backbone-interior sources
outward from the backbone. The three offsets are in meters and must keep each
source set outside its represented domain. Coarser source meshes provide
oversampling. Separate, finer `check_bladder` and `check_backbone` meshes
measure held-out pressure/velocity boundary residuals for each condition.
The result supports
`scattering_amplitude(sol; direction)` and `target_strength(sol; direction)`.
"""
function mfs(swimbladder::Mesh{<:Inti.Quadrature},
        backbone::Mesh{<:Inti.Quadrature}, backbone_material::FluidFilled, k::Real;
        bladder_offset::Real, backbone_offset_ext::Real,
        backbone_offset_int::Real = backbone_offset_ext,
        bladder_source_mesh::Mesh{<:Inti.Quadrature} = swimbladder,
        backbone_source_mesh::Mesh{<:Inti.Quadrature} = backbone,
        check_bladder::Union{Nothing, Mesh{<:Inti.Quadrature}} = nothing,
        check_backbone::Union{Nothing, Mesh{<:Inti.Quadrature}} = nothing,
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        incident = nothing, condition_limit::Integer = 512,
        validation::NamedTuple = (;))
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    all(isfinite, (incidence_angle, incidence_azimuth)) ||
        throw(ArgumentError("incidence angles must be finite"))
    all(o -> isfinite(o) && o > 0,
        (bladder_offset, backbone_offset_ext, backbone_offset_int)) ||
        throw(ArgumentError("source offsets must be finite and positive"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    all(m -> m.method === :full,
        (swimbladder, backbone, bladder_source_mesh, backbone_source_mesh)) ||
        throw(ArgumentError("all component meshes must be full-3D surfaces"))
    validation_report = _validate_regions([swimbladder, backbone], [0, 0]; validation...)
    field, pinc, gradinc = _incident_callbacks(k, incidence_angle, incidence_azimuth; incident)
    bladder_sources = NTuple{3, Float64}[Tuple(q.coords - bladder_offset * q.normal)
                                         for q in bladder_source_mesh.data]
    backbone_ext = NTuple{3, Float64}[Tuple(q.coords - backbone_offset_ext * q.normal)
                                      for q in backbone_source_mesh.data]
    backbone_int = NTuple{3, Float64}[Tuple(q.coords + backbone_offset_int * q.normal)
                                      for q in backbone_source_mesh.data]
    exterior_sources = vcat(bladder_sources, backbone_ext)
    nunknown = length(exterior_sources) + length(backbone_int)
    nequation = length(swimbladder.data) + 2length(backbone.data)
    nunknown <= nequation || throw(ArgumentError(
        "source count exceeds collocation equation count; refine the collocation meshes"))
    matrix, rhs = _bladder_backbone_system(swimbladder.data, backbone.data,
        exterior_sources, backbone_int, backbone_material, k, pinc, gradinc)
    coefficients = matrix \ rhs
    held_out = if check_bladder === nothing && check_backbone === nothing
        nothing
    else
        bladder_check = something(check_bladder, swimbladder)
        backbone_check = something(check_backbone, backbone)
        check_matrix, check_rhs = _bladder_backbone_system(
            bladder_check.data, backbone_check.data,
            exterior_sources, backbone_int, backbone_material, k, pinc, gradinc)
        (;
            _linear_residual(check_matrix, coefficients, check_rhs)...,
            _bladder_backbone_residuals(check_matrix, coefficients, check_rhs,
                length(bladder_check.data), length(backbone_check.data))...)
    end
    report = (; method = nequation == nunknown ? :direct : :least_squares,
        illumination = field === nothing ? :plane_wave : :prescribed,
        discretization = :full, converged = nothing, iterations = nothing,
        residual_history = nothing,
        _linear_residual(matrix, coefficients, rhs)...,
        component_residuals = _bladder_backbone_residuals(matrix, coefficients, rhs,
            length(swimbladder.data), length(backbone.data)),
        _mfs_matrix_diagnostics(matrix; condition_limit)...,
        source_count = nunknown,
        collocation_count = length(swimbladder.data) + length(backbone.data),
        unknown_count = nunknown, equation_count = nequation,
        bladder_oversampling = length(swimbladder.data) / length(bladder_sources),
        backbone_oversampling = length(backbone.data) / length(backbone_ext),
        check_count = check_bladder === nothing && check_backbone === nothing ?
                      0 :
                      length(something(check_bladder, swimbladder).data) +
                      length(something(check_backbone, backbone).data),
        boundary_residual = held_out,
        validation = validation_report,
        solver_options = (; bladder_offset, backbone_offset_ext, backbone_offset_int,
            incidence_angle, incidence_azimuth))
    ne = length(exterior_sources)
    data = _BladderBackboneMFSData(exterior_sources, coefficients[1:ne],
        backbone_int, coefficients[(ne + 1):end], Float64(incidence_angle),
        Float64(incidence_azimuth), pinc, report)
    return MFSSolution(_RegionGeometry(Mesh[swimbladder, backbone], [0, 0]),
        _BladderBackboneBoundary(backbone_material), Float64(k), data)
end
