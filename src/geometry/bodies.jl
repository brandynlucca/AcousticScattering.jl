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

# Bodies whose full surfaces are constructed through the validated mesh interface.
abstract type _GeneratedBody <: AbstractBody end

"""
    RevolvedBody(x, radius; closed=false)
    RevolvedBody(profile; n=64, closed=false)

Revolve a simple piecewise-linear meridian about the Cartesian x axis. Coordinates
are in meters. For `closed=false`, the first and last radii must be zero and every
interior radius positive. For `closed=true`, the profile is a closed loop strictly
away from the axis, as for a toroid. The final vertex may repeat the first.
Either traversal direction is accepted and converted to outward orientation.
Crossing, touching and degenerate profiles are rejected.

The function form samples `profile(t) -> (x, radius)` on `0 <= t <= 1` using
`n` segments. Increase `n` to converge the sampled geometry. Subsequent meshing
subdivides these segments without smoothing or changing the supplied profile.
Use [`bem`](@ref) for axisymmetric or full-3D scattering, or pass a full mesh to
[`mfs`](@ref) or [`kirchhoff`](@ref).
"""
struct RevolvedBody <: _GeneratedBody
    x::Vector{Float64}
    radius::Vector{Float64}
    closed::Bool
    function RevolvedBody(x::AbstractVector, radius::AbstractVector; closed::Bool = false)
        xx, rr = _validate_revolved_profile(x, radius, closed)
        return new(xx, rr, closed)
    end
end

function RevolvedBody(profile; n::Integer = 64, closed::Bool = false)
    n >= 3 || throw(ArgumentError("n must be at least 3"))
    points = [profile(t) for t in range(0.0, 1.0; length = n + 1)]
    all(p -> length(p) == 2, points) ||
        throw(ArgumentError("profile(t) must return (x, radius)"))
    return RevolvedBody(first.(points), last.(points); closed)
end

"""
    LoftedBody(centers, semiaxes; directions=nothing, twists=nothing)

Closed ruled loft through elliptical sections. `centers` is a 3 by N matrix in
meters and `semiaxes` a 2 by N matrix of positive lengths in meters. Ends are
closed with planar elliptical caps. Adjacent sections are connected without
internal caps. Full surface BEM, MFS and Kirchhoff are available.

Optional `directions` is a 3 by N matrix of section normals pointing along the
chain. Otherwise normals follow the centerline, bisecting interior bends.
The first section's first axis is projected from +y (or +z if parallel to the
normal). Subsequent frames use parallel transport. `twists` supplies N rotations
in radians about the section normals. Section order and frames determine the
loft. Intersecting or folded surfaces are rejected during meshing.
"""
struct LoftedBody <: _GeneratedBody
    centers::Matrix{Float64}
    semiaxes::Matrix{Float64}
    frames::Vector{SMatrix{3, 3, Float64, 9}}
    function LoftedBody(centers::AbstractMatrix, semiaxes::AbstractMatrix;
            directions = nothing, twists = nothing)
        cc, aa, frames = _loft_frames(centers, semiaxes, directions, twists)
        return new(cc, aa, frames)
    end
end

"""
    Ellipsoid(a, b, c)
    Ellipsoid(semiaxes)

Triaxial ellipsoid centered at the origin, with positive Cartesian x, y, z
semi-axes in meters. Use full surface [`bem`](@ref), [`mfs`](@ref) or
[`kirchhoff`](@ref). For specialized axisymmetric solvers use [`Spheroid`](@ref).
"""
struct Ellipsoid <: _GeneratedBody
    semiaxes::NTuple{3, Float64}
    function Ellipsoid(a::Real, b::Real, c::Real)
        all(v -> isfinite(v) && v > 0, (a, b, c)) ||
            throw(ArgumentError("ellipsoid semi-axes must be finite and positive"))
        return new((Float64(a), Float64(b), Float64(c)))
    end
end
Ellipsoid(semiaxes::Tuple{Real, Real, Real}) = Ellipsoid(semiaxes...)

"""
    Superellipsoid(a, b, c; axial_exponent=1, equatorial_exponent=1)

Superellipsoid centered at the origin with semi-axes in meters. For polar angle
`theta` from +x and azimuth `phi` from +y toward +z, its surface is
`x=a*spow(cos(theta),e1)`, `y=b*sin(theta)^e1*spow(cos(phi),e2)`,
`z=c*sin(theta)^e1*spow(sin(phi),e2)`, where `spow(u,e)=sign(u)*abs(u)^e`,
`e1=axial_exponent` and `e2=equatorial_exponent`. Both exponents are positive.
Unit exponents give an ellipsoid. Small exponents produce boxlike shapes and
exponents above two can produce concave shapes with cusps.

Use full surface BEM, MFS or Kirchhoff. Geometry and quadrature must be refined
independently. Strong curvature or cusps can require linear elements and a finer
mesh to pass surface validation. No elastic-solid or volume-FEM dispatch is implied.
"""
struct Superellipsoid <: _GeneratedBody
    semiaxes::NTuple{3, Float64}
    axial_exponent::Float64
    equatorial_exponent::Float64
    function Superellipsoid(a::Real, b::Real, c::Real;
            axial_exponent::Real = 1, equatorial_exponent::Real = 1)
        axes = Ellipsoid(a, b, c).semiaxes
        all(v -> isfinite(v) && v > 0, (axial_exponent, equatorial_exponent)) ||
            throw(ArgumentError("superellipsoid exponents must be finite and positive"))
        return new(axes, Float64(axial_exponent), Float64(equatorial_exponent))
    end
end

"""
    Torus(major_radius, minor_radius)
    Torus(major_radius, (axial_radius, radial_radius))

Ring torus about the x axis, centered at the origin, with circular or elliptical
tube cross-section. All lengths are in meters. `major_radius > radial_radius`
is required, excluding singular horn tori and self-intersecting spindle tori.
Supports axisymmetric/full BEM and full surface MFS/Kirchhoff. Axisymmetric BEM
meshes the closed meridian loop. Full meshing uses the smooth toroidal surface.
"""
struct Torus <: _GeneratedBody
    major_radius::Float64
    axial_radius::Float64
    radial_radius::Float64
    function Torus(major_radius::Real, radii::Tuple{Real, Real})
        a, b = radii
        all(v -> isfinite(v) && v > 0, (major_radius, a, b)) && major_radius > b ||
            throw(ArgumentError("a ring torus needs positive finite radii and major_radius > radial_radius"))
        return new(Float64(major_radius), Float64(a), Float64(b))
    end
end
function Torus(major_radius::Real, minor_radius::Real)
    Torus(major_radius, (minor_radius, minor_radius))
end

const _RevolutionBody = Union{RevolvedBody, Torus}

"""
    Frustum(radius_left, radius_right, length)
    Frustum((b_left, c_left), (b_right, c_right), length)

Closed circular or elliptical frustum along x, centered at the origin. Lengths
are in meters. Circular end radii may be zero at one end, giving a cone.
Elliptical semi-axes must be positive. Returns a [`RevolvedBody`](@ref) or
[`LoftedBody`](@ref), respectively.
"""
function Frustum(left::Real, right::Real, length::Real)
    isfinite(length) && length > 0 ||
        throw(ArgumentError("length must be finite and positive"))
    return FrustumChain([-length / 2, length / 2], [left, right])
end
function Frustum(left::Tuple{Real, Real}, right::Tuple{Real, Real}, length::Real)
    isfinite(length) && length > 0 ||
        throw(ArgumentError("length must be finite and positive"))
    return FrustumChain([-length / 2, length / 2], hcat(collect(left), collect(right)))
end

"""
    Cone(radius, length)

Closed circular cone, with base at `x=-length/2` and tip at `x=length/2`.
Lengths are in meters. Returns a [`RevolvedBody`](@ref).
"""
Cone(radius::Real, length::Real) = Frustum(radius, zero(radius), length)

"""
    Bicone(radius, length; waist=0.5)

Two circular cones joined base-to-base with tips at `x=+/-length/2`. The common
base has `radius` in meters and lies a fraction `waist` along the total length
from the negative tip. Requires `0 < waist < 1`. Returns a [`RevolvedBody`](@ref).
"""
function Bicone(radius::Real, length::Real; waist::Real = 0.5)
    all(v -> isfinite(v) && v > 0, (radius, length)) && 0 < waist < 1 ||
        throw(ArgumentError("radius/length must be positive and 0 < waist < 1"))
    return FrustumChain([-length / 2, length * (waist - 0.5), length / 2], [
        0.0, radius, 0.0])
end

"""
    FrustumChain(x, radii)
    FrustumChain(x, semiaxes::AbstractMatrix)

Closed chain of circular or elliptical frustums with strictly increasing axial
stations `x` in meters. `radii` is a vector, or `semiaxes` has two rows for the
y and z semi-axes. Circular radii may vanish only at endpoints. Caps are added
only at the two ends. For displaced or tilted sections use [`LoftedBody`](@ref).
"""
function FrustumChain(x::AbstractVector, radii::AbstractVector)
    length(x) == length(radii) >= 2 && all(isfinite, x) && all(>(0), diff(x)) ||
        throw(ArgumentError("supply matching radii and at least two increasing finite x stations"))
    all(r -> isfinite(r) && r >= 0, radii) && any(>(0), radii) ||
        throw(ArgumentError("radii must be finite, nonnegative and not all zero"))
    xx, rr = Float64.(x), Float64.(radii)
    if !iszero(first(rr))
        pushfirst!(xx, first(xx))
        pushfirst!(rr, 0.0)
    end
    if !iszero(last(rr))
        push!(xx, last(xx))
        push!(rr, 0.0)
    end
    return RevolvedBody(xx, rr)
end
function FrustumChain(x::AbstractVector, semiaxes::AbstractMatrix)
    length(x) >= 2 && all(isfinite, x) && all(>(0), diff(x)) ||
        throw(ArgumentError("x stations must be finite and strictly increasing"))
    centers = zeros(3, length(x))
    centers[1, :] .= x
    return LoftedBody(centers, semiaxes)
end

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
