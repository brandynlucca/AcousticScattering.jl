# Planar tank-wall images of a transducer, first order (one reflection per wall, no
# wall-wall corner images, no target-wall multiple interaction). A single rigid/soft
# plane is exact in the no-scatterer limit; multiple walls and targets require more paths (see tests and
# docs/TANK_MEASUREMENT_MODELING.md's "Target and wall multiple interaction" note).

"""
    Wall(point, normal, reflection=1.0)

An infinite planar tank wall through `point` [m] with outward `normal` (need not be unit
length, pointing away from the wall into the region holding the transducers and target).
`reflection` is a constant pressure reflection coefficient or a callable
`R(k, cosine)`, where `k` is wavenumber [rad/m] and `cosine` is the absolute cosine
of the central ray's angle to the wall normal. `1` is rigid, `-1` is pressure-release (matching
[`free_surface`](@ref)'s `condition=:rigid`/`:pressure_release`), general passive complex
`abs(reflection) <= 1` is required. A finite-impedance coefficient uses the
central-ray approximation described in [`TankTransducer`](@ref).
"""
struct Wall{R}
    point::NTuple{3, Float64}
    normal::NTuple{3, Float64}
    reflection::R
    function Wall(point, normal, reflection = 1.0)
        all(isfinite, point) && length(point) == 3 ||
            throw(ArgumentError("point must be a finite three-vector"))
        length(normal)==3 || throw(ArgumentError("normal must have three components"))
        n = norm(collect(Float64.(normal)))
        isfinite(n) && n > 0 ||
            throw(ArgumentError("normal must be a finite nonzero vector"))
        if reflection isa Number
            isfinite(reflection) && abs(reflection)<=1 ||
                throw(ArgumentError("reflection must be finite with magnitude at most 1"))
            reflection=ComplexF64(reflection)
        else
            applicable(reflection, 1.0, 0.5) ||
                throw(ArgumentError("reflection must be a number or a callable R(k,cosine)"))
        end
        return new{typeof(reflection)}(Tuple(Float64.(point)), Tuple(Float64.(normal) ./ n), reflection)
    end
end

function _mirror_point(w::Wall, x)
    Tuple(collect(x) .-
          2dot(collect(x) .- collect(w.point),
        collect(w.normal)) .* collect(w.normal))
end
function _mirror_direction(w::Wall, v)
    Tuple(collect(v) .- 2dot(collect(v), collect(w.normal)) .*
                        collect(w.normal))
end
function _image(w::Wall, t::Transducer)
    Transducer(
        _mirror_point(w, t.position), _mirror_direction(w, t.axis), t.radius)
end

abstract type AbstractTank end

"""
    Tank(xlimits, ylimits, zlimits; reflections=(xmin=1, xmax=1, ymin=1,
        ymax=1, bottom=1, surface=-1))

Rectangular water volume in Cartesian coordinates [m], with `z` increasing upward.
Each limit is an increasing pair. The water surface is at `zlimits[2]` and the
bottom at `zlimits[1]`; the target mesh and transducers use this same frame.
`reflections` supplies all six constant or callable [`Wall`](@ref) coefficients.
The default walls/bottom are rigid and the surface is pressure-release.

Use `TankTransducer(source, tank)` to construct first-order wall illumination and
[`tank_field`](@ref) to sample the physical tank. Tank bounds alone do not change
an existing scattering solution or add multiple wall interactions.

    Tank(radius, zlimits; center=(0,0), reflections=(bottom=1,surface=-1))

Vertical cylindrical water volume [m]. The circular footprint is used for sampling
and containment. Only first-order bottom/surface images are supported: this does
not solve scattering from the curved sidewall or cylindrical cavity resonances.
Transmitter aperture containment uses its enclosing horizontal disk conservatively.
"""
struct Tank <: AbstractTank
    bounds::NamedTuple{(:x, :y, :z), NTuple{3, NTuple{2, Float64}}}
    walls::Vector{Wall}
    radius::Union{Nothing, Float64}
    function Tank(xlimits, ylimits, zlimits;
            reflections = (
                xmin = 1, xmax = 1, ymin = 1, ymax = 1, bottom = 1, surface = -1))
        limits=map((xlimits, ylimits, zlimits)) do pair
            length(pair)==2 && all(isfinite, pair) && pair[1]<pair[2] ||
                throw(ArgumentError("tank limits must be finite increasing pairs in meters"))
            (Float64(pair[1]), Float64(pair[2]))
        end
        names=(:xmin, :xmax, :ymin, :ymax, :bottom, :surface)
        Set(keys(reflections))==Set(names) || throw(ArgumentError(
            "reflections must specify xmin, xmax, ymin, ymax, bottom and surface"))
        x, y, z=limits
        center=ntuple(i->sum(limits[i])/2, 3)
        points=((x[1], center[2], center[3]), (x[2], center[2], center[3]),
            (center[1], y[1], center[3]), (center[1], y[2], center[3]),
            (center[1], center[2], z[1]), (center[1], center[2], z[2]))
        normals=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        walls=Wall[Wall(points[i], normals[i], getproperty(reflections, names[i]))
                   for i in 1:6]
        new((; x, y, z), walls, nothing)
    end
    function Tank(radius::Real, zlimits; center = (0.0, 0.0), reflections = (
            bottom = 1, surface = -1))
        isfinite(radius) && radius>0 ||
            throw(ArgumentError("tank radius must be finite and positive"))
        length(center)==2 && all(isfinite, center) ||
            throw(ArgumentError("center must have two finite coordinates"))
        Set(keys(reflections))==Set((:bottom, :surface)) || throw(ArgumentError(
            "cylindrical tank images support bottom and surface only; curved sidewalls are not solved"))
        x, y=Float64.(center)
        box=Tank((x-radius, x+radius), (y-radius, y+radius), zlimits;
            reflections = (xmin = 0, xmax = 0, ymin = 0, ymax = 0,
                bottom = reflections.bottom, surface = reflections.surface))
        new(box.bounds, box.walls[5:6], Float64(radius))
    end
end

function _inside_tank(tank::Tank, point)
    all(d->tank.bounds[d][1]<=point[d]<=tank.bounds[d][2], 1:3) || return false
    tank.radius===nothing && return true
    cx, cy=sum(tank.bounds.x)/2, sum(tank.bounds.y)/2
    return hypot(point[1]-cx, point[2]-cy)<=tank.radius
end

"""
    TankTransducer(source::Transducer, walls; reference_point=nothing)

`source` plus its first-order image across each `Wall` in `walls` (a single `Wall` or a
vector), each weighted by that wall's reflection coefficient. Callable wall coefficients
require an explicit `reference_point` [m], usually the target center, on the source's
side of every wall. The incidence angle follows the ray from the mirrored source
center to this point. Its coefficient is fixed across the image field, preserving
the Helmholtz equation; angular variation across the aperture and target is neglected.
This is not an exact finite-impedance wall solution. There are no wall-wall images
or target-wall repeated scattering. Use with `transducer=...` in supported solvers,
[`pressure`](@ref) for source maps, or [`received_signal`](@ref) for receiver images.
"""
struct TankTransducer <: AbstractTransducer
    source::Transducer
    walls::Vector{Wall}
    reference_point::Union{Nothing, NTuple{3, Float64}}
end
function TankTransducer(source::Transducer, walls; reference_point = nothing)
    entries=walls isa Wall ? Wall[walls] : Wall[wall for wall in walls]
    point=if reference_point===nothing
        all(w->w.reflection isa Number, entries) || throw(ArgumentError(
            "angle-dependent walls require an explicit reference_point in meters"))
        nothing
    else
        length(reference_point)==3 && all(isfinite, reference_point) ||
            throw(ArgumentError("reference_point must be a finite three-vector"))
        Tuple(Float64.(reference_point))
    end
    return TankTransducer(source, entries, point)
end

function TankTransducer(source::Transducer, tank::Tank; reference_point = nothing)
    for d in 1:3
        span=source.radius*sqrt(max(0.0, 1-source.axis[d]^2))
        lo, hi=tank.bounds[d]
        lo<=source.position[d]-span && source.position[d]+span<=hi ||
            throw(ArgumentError("the transmitter aperture must lie within the tank bounds"))
    end
    if tank.radius!==nothing
        cx, cy=sum(tank.bounds.x)/2, sum(tank.bounds.y)/2
        hypot(source.position[1]-cx, source.position[2]-cy)+source.radius<=tank.radius ||
            throw(ArgumentError("the transmitter aperture's enclosing horizontal disk must fit inside the cylindrical tank"))
    end
    return TankTransducer(source, tank.walls; reference_point)
end

function _wall_reflection(wall, source, k, point)
    wall.reflection isa Number && return wall.reflection
    source_side=dot(SVector(source.position)-SVector(wall.point), SVector(wall.normal))
    target_side=dot(SVector(point)-SVector(wall.point), SVector(wall.normal))
    source_side*target_side>0 || throw(ArgumentError(
        "source and reference_point must be on the same side of an angle-dependent wall"))
    ray=SVector(point)-SVector(_mirror_point(wall, source.position))
    norm(ray)>0 || throw(ArgumentError("reference_point coincides with an image source"))
    cosine=clamp(abs(dot(ray, SVector(wall.normal)))/norm(ray), 0.0, 1.0)
    value=wall.reflection(k, cosine)
    value isa Number && isfinite(value) && abs(value)<=1 ||
        throw(ArgumentError("reflection R(k,cosine) must be finite with magnitude at most 1"))
    return ComplexF64(value)
end

"""
    _transducer_field(t::TankTransducer, k; rtol=1e-6)

Pointwise `pinc(x)`/`gradinc(x)` for `t`, its `t.source`'s own field plus its reflection-weighted
image across each of `t.walls`, each image field from [`_transducer_field`](@ref)`(::Transducer,
...)` on the mirrored transducer.
"""
function _transducer_field(t::TankTransducer, k::Real; rtol::Real = 1e-6,
        approximation::Symbol = :rayleigh)
    direct = _transducer_field(t.source, k; rtol, approximation)
    images = map(t.walls) do wall
        coefficient=_wall_reflection(wall, t.source, k, t.reference_point)
        field=iszero(coefficient) ? nothing :
              _transducer_field(_image(wall, t.source), k; rtol, approximation)
        coefficient, field
    end
    pinc = x -> begin
        total = direct[1](x)
        for (r, field) in images
            field===nothing && continue
            total += r * field[1](x)
        end
        total
    end
    gradinc = x -> begin
        total = direct[2](x)
        for (r, field) in images
            field===nothing && continue
            total += r * field[2](x)
        end
        total
    end
    return pinc, gradinc
end
