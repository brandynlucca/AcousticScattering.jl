# Full 3D (non-axisymmetric) BEM surface-field data, `bem(...; method=:full)` only.
struct _FullBEMSurfaceData
    quad::Any
    p_scat::Vector{ComplexF64}
    dpdn_scat::Vector{ComplexF64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    diagnostics::NamedTuple
    single_layer_density::Union{Nothing, Vector{ComplexF64}}
    incident::Union{Nothing, IncidentField}
end

function _FullBEMSurfaceData(quad, p, dp, beta, alpha, report, density)
    _FullBEMSurfaceData(quad, p, dp, beta, alpha, report, density, nothing)
end

function _FullBEMSurfaceData(quad, p, dp, beta, alpha, report)
    _FullBEMSurfaceData(quad, p, dp, beta, alpha, report, nothing)
end

"""
    BEMSolution

Result of [`bem`](@ref): axisymmetric (`method=:axisymmetric`, `Sphere`/`Spheroid`/straight
`Cylinder`) or full 3D (`method=:full`) boundary-element surface data, including supplied
closed surfaces and coupled fluid regions.
Post-process with [`target_strength`](@ref)`(sol; angle, azimuth)` (axisymmetric) or
[`target_strength`](@ref)`(sol; direction)` (full 3D).
"""
struct BEMSolution{D} <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    method::Symbol
    data::D
end
