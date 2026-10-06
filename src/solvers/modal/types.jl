struct _SphereModalData
    coefficients::Vector{ComplexF64}
    interior_coefficients::Union{Nothing, Vector{ComplexF64}}
    shell_coefficients::Union{Nothing, Vector{NTuple{2, ComplexF64}},
        Vector{Vector{NTuple{2, ComplexF64}}},
        Vector{Vector{Vector{ComplexF64}}}}
    incident::IncidentField
end

"""
    ModalSolution

Result of [`modal`](@ref): the modal-series complex scattering amplitude `f` [m] at the
angle(s) `modal` was called with. Post-process with [`target_strength`](@ref) or
[`scattering_amplitude`](@ref). Supported spheres also retain coefficients for [`pressure`](@ref)
evaluation with [`PlaneWave`](@ref), [`SphericalWave`](@ref) or [`BesselBeam`](@ref),
including their Cartesian direction and complex amplitude. Finite-cylinder and
bent-cylinder paths include approximations.
"""
struct ModalSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
    data::Union{Nothing, _SphereModalData}
end

ModalSolution(body, boundary, k, f) = ModalSolution(body, boundary, k, f, nothing)
