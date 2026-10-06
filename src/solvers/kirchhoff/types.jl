"""
    KirchhoffSolution

Result of [`kirchhoff`](@ref): the high-frequency (physical-optics) complex scattering amplitude
`f` [m] at the angle(s) `kirchhoff` was called with. Post-process with [`target_strength`](@ref) or
[`scattering_amplitude`](@ref).
"""
struct KirchhoffSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
end
