"""
    FreeSurfaceSolution

Result of [`free_surface`](@ref): the volume FEM solutions of a body driven by the direct incident
wave and by its interface-reflected image, held separately with the interface's reflection sign.
Post-process with [`target_strength`](@ref), [`scattering_amplitude`](@ref) or [`pressure`](@ref).
"""
struct FreeSurfaceSolution <: AbstractSolution
    direct::FEMSolution
    reflected::FEMSolution
    sign::Int
    incidence_angle::Float64
    incidence_azimuth::Float64
end

"""
    free_surface(body::Union{Sphere,Spheroid}, boundary, k, depth; condition=:pressure_release,
        incidence_angle=0, incidence_azimuth=0, orientation=[0,0,1], kwargs...)

Volume FEM scattering by `body`, centered on the normal of an infinite planar interface and
submerged a `depth` [m] below it. `condition = :pressure_release` (default) is an air-water free
surface, `condition = :rigid` an idealized seafloor. `incidence_angle = 0` and `incidence_azimuth`
give the direct wave's direction along the interface's outward normal, in the convention of
`fem(...; method = :volume)`. `boundary` is a region material (`FluidFilled`, `SolidElastic`,
`ViscoelasticSolid` or `ViscousLayer`).

Solved exactly by the method of images: `body` and a mirror image across the interface form a
coupled two-body volume FEM in an unbounded fluid, solved once for the direct wave and once for its
specular reflection, and added with the interface's reflection coefficient (`-1` for
`:pressure_release`, `+1` for `:rigid`). Accepts the keywords of `fem(...; method = :volume)` for
coupled regions. Returns a [`FreeSurfaceSolution`](@ref).
"""
function free_surface(
        body::Union{Sphere, Spheroid}, boundary::_VolumeRegionMaterial, k::Real,
        depth::Real; condition::Symbol = :pressure_release, incidence_angle::Real = 0.0,
        incidence_azimuth::Real = 0.0, orientation = [0.0, 0.0, 1.0], kwargs...)
    boundary isa SpatialFluid && throw(ArgumentError(
        "free_surface does not yet reflect SpatialFluid profiles; use explicit volume regions"))
    condition in (:pressure_release, :rigid) || throw(ArgumentError(
        "condition must be :pressure_release or :rigid, got $condition"))
    isfinite(depth) && depth > 0 ||
        throw(ArgumentError("depth must be finite and positive"))
    sign = condition === :pressure_release ? -1 : 1
    centers = [[0.0, 0.0, -depth], [0.0, 0.0, depth]]
    orientations = [Float64.(orientation), [1.0, 1.0, -1.0] .* Float64.(orientation)]
    system = _volume_system_regions([body, body], [boundary, boundary], k;
        parents = [0, 0], centers, orientations, kwargs...)
    geometry = _VolumeRegionGeometry([body, body], centers, orientations, [0, 0])
    materials = _VolumeRegionMaterials([boundary, boundary])
    solve(angle) = FEMSolution(geometry, materials, Float64(k), :volume,
        _volume_solution(system, angle, incidence_azimuth))
    return FreeSurfaceSolution(solve(incidence_angle), solve(π - incidence_angle), sign,
        Float64(incidence_angle), Float64(incidence_azimuth))
end
