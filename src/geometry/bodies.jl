# Canonical bodies.

"""
    Sphere(radius)

A sphere of positive `radius` [m]. Monostatic scattering is independent of orientation.
Sphere `modal` and `kirchhoff` calls do not take `incidence_angle`. Numerical BEM/MFS
solvers accept it to set the incident direction for directional field queries.
"""
struct Sphere <: AbstractBody
    radius::Float64
    function Sphere(radius::Real)
        (radius > 0 || throw(ArgumentError("Sphere radius must be positive"));
            new(Float64(radius)))
    end
end

"""
    Cylinder(radius, length; radius_curvature=Inf, endcap_depth=0.0)

A finite circular cylinder of the given `radius` and `length` [m].

- `radius_curvature` [m]: `Inf` (default) is straight. A finite value bends the cylinder in the
  `xy` plane, with midpoint at the origin and midpoint tangent along `+x`. `modal` applies a
  near-broadside correction. `kirchhoff` and `mfs(body, ...)` describe the lateral surface only.
  Use `bem(...; method=:full)` or `mfs(mesh(...; method=:full), ...)` for closed surfaces.
- `endcap_depth` [m]: `0.0` (default) uses flat end caps. A positive value caps the cylinder
  with half-spheroid domes of that depth instead. Full BEM and straight-cylinder MFS honor
  this field. Axisymmetric BEM and FEM always use flat ends.

Full meshing requires `radius_curvature > radius`, an arc shorter than a full circle, and
nonintersecting ends. See [Closed bent cylinders](@ref bent-cylinder-tutorial) for the
bent-geometry conventions.
"""
struct Cylinder <: AbstractBody
    radius::Float64
    length::Float64
    radius_curvature::Float64
    endcap_depth::Float64
    function Cylinder(radius::Real, length::Real; radius_curvature::Real = Inf, endcap_depth::Real = 0.0)
        radius > 0 && length > 0 ||
            throw(ArgumentError("Cylinder radius/length must be positive"))
        radius_curvature > 0 ||
            throw(ArgumentError("radius_curvature must be positive (Inf for straight)"))
        endcap_depth >= 0 || throw(ArgumentError("endcap_depth must be nonnegative"))
        return new(Float64(radius), Float64(length), Float64(radius_curvature), Float64(endcap_depth))
    end
end

_isbent(body::Cylinder) = !isinf(body.radius_curvature)
_iscapped(body::Cylinder) = body.endcap_depth > 0

"""
    Shell(body, thickness)

A structural shell of positive `thickness` [m] inside the outer surface of a
[`Sphere`](@ref) or [`Spheroid`](@ref). For a prolate spheroid, thickness is measured
at the equator and the inner surface is confocal. Solve with [`fem`](@ref) and a
structural [`Shelled`](@ref) material. The thickness must be smaller than the body's radius.
"""
struct Shell <: AbstractBody
    body::AbstractBody
    thickness::Float64
    function Shell(body::AbstractBody, thickness::Real)
        body isa Union{Sphere, Spheroid} ||
            throw(ArgumentError("Shell only supports a Sphere or Spheroid base body"))
        thickness > 0 || throw(ArgumentError("Shell thickness must be positive"))
        thickness < _characteristic_radius(body) || throw(ArgumentError(
            "Shell thickness ($thickness) must be less than the base body's characteristic " *
            "radius ($(_characteristic_radius(body))) — the shell FEM theory assumes a thin " *
            "shell over a solid base body, not a thickness comparable to or larger than the body itself"))
        return new(body, Float64(thickness))
    end
end
