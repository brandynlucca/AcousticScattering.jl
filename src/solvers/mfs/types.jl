# Non-axisymmetric bent-cylinder MFS surface-field data, `mfs` on a bent `Cylinder` only.
struct _BentMFSSurfaceData
    p_scat::Vector{ComplexF64}
    dpdn_scat::Vector{ComplexF64}
    points::Vector{NTuple{3, Float64}}
    normals::Vector{NTuple{3, Float64}}
    areas::Vector{Float64}
    incidence_angle::Float64
    diagnostics::NamedTuple
end

"""
    MFSSolution

Result of [`mfs`](@ref). Axisymmetric body solves use
[`target_strength`](@ref)`(sol; angle, azimuth)`. Full closed-surface solves use
`target_strength(sol; direction)`, defaulting to backscatter. The lateral-only bent-body
overload returns a monostatic result, evaluated with `target_strength(sol)`.
"""
struct MFSSolution{D} <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    data::D
end
