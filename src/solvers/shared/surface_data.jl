# Reusable axisymmetric surface-field data (bem/mfs axisymmetric solves), an internal component
# type shared by `BEMSolution`/`MFSSolution`, never exposed as its own public type.
struct _AxisymmetricSurfaceData
    mesh::MeridianMesh
    p_scat_modes::Vector{Vector{ComplexF64}}
    dpdn_scat_modes::Vector{Vector{ComplexF64}}
    p_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    dpdn_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    incidence_angle::Float64
    diagnostics::NamedTuple
    source_modes::Union{Nothing, Vector{NamedTuple}}
end

function _AxisymmetricSurfaceData(mesh, p, dp, pi, dpi, beta, report)
    _AxisymmetricSurfaceData(mesh, p, dp, pi, dpi, beta, report, nothing)
end
