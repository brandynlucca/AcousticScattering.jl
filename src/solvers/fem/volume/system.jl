struct _VolumeSystem
    grid::Any
    labels::Vector{Int}
    ids::Vector{Int}
    facets::Any
    n_nodes::Int
    k::Float64
    model::Any
    closure::Symbol
    L::Int
    R::Float64
    rotation::Matrix{Float64}
    free::Vector{Int}
    fixed::Vector{Int}
    soft::BitVector
    fixed_matrix::SparseMatrixCSC{ComplexF64, <:Integer}
    matrix::SparseMatrixCSC{ComplexF64, <:Integer}
    solver::_VolumeLinearSolver
    extraction::Any
    diagnostics::NamedTuple
end

# Scattered-pressure solution for a plane wave incident from the polar and azimuthal angles, in the frame of the bodies.
function _volume_solution(system::_VolumeSystem, incidence_angle::Real, incidence_azimuth::Real)
    direction = _volume_direction(incidence_angle, incidence_azimuth)
    pinc, gradinc = _plane_wave_incident(system.k, direction)
    return _volume_solution(system, pinc, gradinc, incidence_angle, incidence_azimuth;
        illumination = :plane_wave)
end

# General incident field. Angle args are only kept as default-observation metadata.
function _volume_solution(system::_VolumeSystem, pinc, gradinc,
        incidence_angle::Real, incidence_azimuth::Real; illumination::Symbol = :prescribed)
    n_nodes = system.n_nodes
    load = _volume_load(system.grid, system.labels, system.ids, system.facets,
        system.model, n_nodes, pinc, gradinc)
    values = zeros(ComplexF64, length(system.fixed))
    for (i, dof) in enumerate(system.fixed)
        system.soft[i] && (values[i] = -pinc(system.grid.nodes[dof].x))
    end
    rhs = load[system.free]
    isempty(values) || (rhs -= system.fixed_matrix * values)
    x, info = _solve!(system.solver, rhs)
    solution = zeros(ComplexF64, 4n_nodes)
    solution[system.fixed] = values
    solution[system.free] = x
    residual = norm(_apply(system.solver, x) - rhs) / max(norm(rhs), eps())
    positive, negative, shell = if system.closure === :dtn || system.extraction === nothing
        list = system.closure === :dtn ? system.facets[:outer] :
               system.facets[:pml_interface]
        (_volume_coefficients(solution, system.grid, list, n_nodes, system.L)..., nothing)
    else
        (ComplexF64[], ComplexF64[], _volume_extraction(system.extraction, solution))
    end
    diagnostics = merge(system.diagnostics, (;
        residual, info.solver, info.iterations, illumination))
    return _VolumeFEMData(positive, negative, system.R, system.L, shell, system.rotation,
        Float64(incidence_angle), Float64(incidence_azimuth), pinc, diagnostics, system, solution)
end
