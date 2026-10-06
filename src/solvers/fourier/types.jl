# --- `fourier`: conformal-mapping semi-analytic method for bodies of revolution -----

"""
    FMSolution

Result of [`fourier`](@ref). Post-process with [`target_strength`](@ref)`(sol; angle,
azimuth)` or [`scattering_amplitude`](@ref)`(sol; angle, azimuth)` (backscatter by default, same
convention as axisymmetric [`bem`](@ref)/[`mfs`](@ref)). [`diagnostics`](@ref) reports the
mapping's admissibility, the truncation orders and a truncation-consistency check. `b_check` holds
the reduced-truncation coefficients.
"""
struct FMSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    mapping::ConformalMapping
    b::Matrix{ComplexF64}
    incidence_angle::Float64
    b_check::Union{Nothing, Matrix{ComplexF64}}
end
