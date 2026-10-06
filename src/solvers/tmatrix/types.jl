"""
    TMatrixSolution

Result of [`tmatrix`](@ref): the complex scattering amplitude `f` [m] of an elastic spheroid or
shell at the incident and observation directions `tmatrix` was called with. Post-process with
[`target_strength`](@ref) or [`scattering_amplitude`](@ref). The `method=:farfield`
path retains its angular transition internally and supports repeated direction queries
through those same functions; [`diagnostics`](@ref) reports its construction checks.
"""
struct TMatrixSolution{D} <: AbstractSolution
    body::Spheroid
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
    data::D
end

TMatrixSolution(body, boundary, k, f) = TMatrixSolution(body, boundary, k, f, nothing)

function TMatrixSolution(body::Spheroid, boundary::AbstractBoundaryCondition,
        k::Real, f::Number, data::D) where {D}
    TMatrixSolution{D}(body, boundary, Float64(k), ComplexF64(f), data)
end
