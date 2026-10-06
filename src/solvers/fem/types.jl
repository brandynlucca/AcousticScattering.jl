# FEM paths that retain only target strength.
struct _ScalarFEMData
    ts::Float64
    diagnostics::NamedTuple
end

# The scattered nodal field on the sphere meridian mesh and its outgoing
# spherical-harmonic continuation beyond the artificial DtN boundary.
struct _SphereMeridianFEMData
    radii::Vector{Float64}
    angles::Vector{Float64}
    pressure::Matrix{ComplexF64}
    outgoing::Vector{ComplexF64}
    diagnostics::NamedTuple
end

"""
    _RadialFEMData

Per-degree spherical radial fields for unit incident pressure. `coefficient`
multiplies the outgoing Hankel function and Legendre polynomial, including the
plane-wave factor `(2l+1)i^l`. Fluid shell nodal pressures and regular interior
pressure coefficients describe total pressure. Interior coefficients multiply
`j_l(k_in*r)`. Exterior coefficients describe scattered pressure.

Elastic nodal `longitudinal` and `shear` values are the displacement potentials
scaled by `rho_ext*omega^2/p_inc`, including the same plane-wave factor. Both are
dimensionless. With their Legendre factors restored, the scaled displacement is
`grad(longitudinal) + curl(curl(r*shear*e_r))`. The monopole has no shear potential.
Radii are in meters; material contrasts and the exterior wavenumber are stored
in the enclosing solution. [`pressure`](@ref) evaluates acoustic pressure in fluid
regions; elastic stress and displacement evaluation is unavailable.
"""
struct _RadialFEMData
    modes::Vector{NamedTuple}
    diagnostics::NamedTuple
end

# A `fem` result on a `Shell` body: exterior (and, for a fluid-filled interior, interior) surface
# pressure/normal-derivative data plus the shell's own mechanical state, mirroring
# `solve_shell_fluid_coupled`/`solve_shell_fluid_filled_coupled`/
# `solve_general_shell_fluid_filled_coupled`'s return tuples. `p_int_modes`/`ps_int`/
# `dpdn_int_modes` are `nothing` for a vacuum/air-backed shell (no interior fluid coupling).
struct _ShellFEMSurfaceData
    ps_ext::Vector{Panel}
    p_ext_modes::Vector{Vector{ComplexF64}}
    dpdn_ext_modes::Vector{Vector{ComplexF64}}
    ps_int::Union{Nothing, Vector{Panel}}
    p_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    dpdn_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    k_interior::Union{Nothing, Float64}
    shell_state::Any
    incidence_angle::Float64
    diagnostics::NamedTuple
end

# Scattered nodal pressure on one exterior meridian mesh and optionally its
# coupled interior mesh. The outgoing coefficients continue it beyond r=R.
struct _MeridianFEMModeField
    angles::Vector{Float64}
    radial_exterior::Vector{Float64}
    inner_radius::Vector{Float64}
    outer_radius::Float64
    exterior::Matrix{ComplexF64}
    radial_interior::Union{Nothing, Vector{Float64}}
    interior::Union{Nothing, Matrix{ComplexF64}}
    outgoing::Vector{ComplexF64}
end

# Cylinder meridian FEM (rigid/pressure-release/fluid-filled): panels and per-Fourier-mode
# complex surface traces at r=R, mirroring `_ShellFEMSurfaceData`'s exterior fields. Reduces to
# `far_field`/`target_strength(ps, p_modes, dpdn_modes, k, angle, azimuth)` at any observation
# direction, not only the backscatter this path's own `_ScalarFEMData` predecessor retained.
struct _CylinderMeridianFEMData
    ps::Vector{Panel}
    p_modes::Vector{Vector{ComplexF64}}
    dpdn_modes::Vector{Vector{ComplexF64}}
    fields::Vector{_MeridianFEMModeField}
    incidence_angle::Float64
    diagnostics::NamedTuple
end

# Spheroid meridian FEM retains the same exterior traces on its enclosing
# spherical DtN boundary. The traces support complex far-field observations.
struct _SpheroidMeridianFEMData
    ps::Vector{Panel}
    p_modes::Vector{Vector{ComplexF64}}
    dpdn_modes::Vector{Vector{ComplexF64}}
    fields::Vector{_MeridianFEMModeField}
    incidence_angle::Float64
    diagnostics::NamedTuple
end

# Elastic cylinder radial FEM: raw per-azimuthal-mode coefficients from `_raw_bn_radial_fem`,
# and the axial geometry (`length`, `aspect_angle`) needed to reconstruct the backscatter
# amplitude via `_elastic_cylinder_radial_fem_amplitude`. Backscatter only, unlike the
# meridian path above: the Fraunhofer axial envelope this reduction uses has no general
# bistatic form.
struct _CylinderRadialFEMData
    modes::Vector{ComplexF64}
    length::Float64
    aspect_angle::Float64
    diagnostics::NamedTuple
end

"""
    FEMSolution

Result of [`fem`](@ref). Supported radial spheres retain complex backscatter
coefficients and radial pressure or elastic potential fields.
Post-process with [`scattering_amplitude`](@ref) or [`target_strength`](@ref).
For supported radial spheres, [`pressure`](@ref) samples acoustic pressure in the
exterior, fluid shells and fluid interiors at Cartesian points.
Structural `Shell` results retain surface traces and support observation `angle`/`azimuth`.
Cylinder radial and meridian paths retain target strength only. Observation keywords
are rejected outside the structural `Shell` case.
"""
struct FEMSolution{D} <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    method::Symbol
    data::D
end

struct _VolumeRegionGeometry <: AbstractBody
    bodies::Vector{AbstractBody}
    centers::Vector{Vector{Float64}}
    orientations::Vector{Vector{Float64}}
    parents::Vector{Int}
end

const _VolumeRegionMaterial = Union{
    FluidFilled, SpatialFluid, SolidElastic, ViscoelasticSolid, ViscousLayer}

struct _VolumeRegionMaterials <: AbstractBoundaryCondition
    materials::Vector{Any}
end
