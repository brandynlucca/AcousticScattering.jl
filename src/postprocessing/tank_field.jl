"""
    TankField

Complex pressure sampled on a physical tank plane. `horizontal` and `vertical`
are coordinate vectors [m]; `pressure[i,j]` belongs to their Cartesian product.
`regions` identifies exterior water (`0`), target regions (positive) and excluded
source apertures (`-1`), outside cylindrical water (`-2`), or excluded scattering
quadrature geometry (`-3`). Undefined pressures are complex NaNs. `plane` and `at`
specify the slice, and `field` identifies total, incident or scattered pressure.
`plot(result)` with Makie defaults to a side-on pressure-magnitude heatmap.
"""
struct TankField
    tank::AbstractTank
    horizontal::Vector{Float64}
    vertical::Vector{Float64}
    pressure::Matrix{ComplexF64}
    regions::Matrix{Int}
    plane::Symbol
    at::Float64
    k::Float64
    field::Symbol
    transducers::Vector{Pair{String, Transducer}}
    wall_images::Bool
end

function _tank_axes(plane)
    plane===:xz && return (1, 3, 2)
    plane===:yz && return (2, 3, 1)
    plane===:xy && return (1, 2, 3)
    throw(ArgumentError("plane must be :xz (side), :yz (end), or :xy (top)"))
end

function _tank_annotations(transducers)
    entries=Pair{String, Transducer}[]
    for (name, source) in pairs(transducers)
        direct=_transducer_aperture(source)
        direct isa Transducer ||
            throw(ArgumentError("transducer annotations must be Transducer or TankTransducer objects"))
        push!(entries, string(name)=>direct)
    end
    return entries
end

function _tank_aperture(point, source::Transducer)
    delta=SVector(point)-SVector(source.position)
    axial=dot(delta, SVector(source.axis))
    tolerance=64eps(Float64)*max(norm(point), norm(SVector(source.position)), source.radius)
    return abs(axial)<=tolerance && norm(delta-axial*SVector(source.axis))<=source.radius
end

const _TankSurfaceSolution=Union{BEMSolution{_FullBEMSurfaceData},
    BEMSolution{_RegionBEMData}, MFSSolution{_FullMFSSurfaceData}}

function _tank_quads(sol::Union{
        BEMSolution{_FullBEMSurfaceData}, MFSSolution{_FullMFSSurfaceData}})
    [sol.data.quad]
end
function _tank_quads(sol::BEMSolution{_RegionBEMData})
    [part.surface.data for part in sol.data.interfaces]
end

function _tank_region_locator(sol, tank)
    sol===nothing && return point->0
    quads=_tank_quads(sol)
    patches=[_region_patches(quad, 0) for quad in quads]
    boxes=map(patches) do surface
        ntuple(
            d->(minimum(p[d].lo for patch in surface for p in patch.net),
                maximum(p[d].hi for patch in surface for p in patch.net)),
            3)
    end
    for quad in quads, node in quad

        _inside_tank(tank, node.coords) || throw(ArgumentError(
            "tank bounds must enclose every target interface in the same Cartesian frame"))
    end
    return point->begin
        region=0
        for i in eachindex(patches)
            all(d->boxes[i][d][1]<=point[d]<=boxes[i][d][2], 1:3) || continue
            _surface_location(patches[i], point)===:outside || (region=i)
        end
        region
    end
end

"""
    tank_field(solution, tank::Tank; plane=:xz, at=nothing, field=:total,
        resolution=nothing, points_per_wavelength=8, transducers=nothing, batch_size=2048,
        wall_images=true, reflection_weights=nothing, scattering_fields=())

Sample the physical water volume from wall to wall and bottom to surface. The
default is a side view in the `x-z` plane through the tank's central `y`; `at` sets
that coordinate explicitly. `:yz` is an end view and `:xy` a top view. Coordinates
are in meters, with positive `z` upward. Returns a [`TankField`](@ref) whose default
Makie plot is a 2D pressure-magnitude heatmap, not a far-field scattering pattern.

Total pressure includes the solve's retained incident beam/images and scattered
field and first-order wall images of the scattered field, evaluated by its near-field representation everywhere, including distant
points. It does not switch to a far-field approximation. Supports full-3D BEM,
coupled fluid BEM and closed-surface MFS. Volume FEM is not accepted here because
its pressure evaluator is limited to its computational mesh, not the whole tank.
Rigid/soft interiors and scattered-field interior samples are masked.

The default grid has at least 201 by 101 points and resolves the exterior wavelength
with `points_per_wavelength` samples. Set `resolution=(nx,nz)` explicitly to control
cost and refine for the aperture, target geometry and shorter interior wavelengths.
Sampling is batched; large physical tanks at high frequency can require large grids.

The original transmitting transducer is annotated and its aperture masked by default
when retained by the solution. `transducers=(transmitter=tx, receiver=rx)` overrides
the annotations and adds aperture masks; the original source aperture remains masked.
These annotations do not change the solved field;
construct illumination with `TankTransducer(tx,tank)` before solving. Markers in the
side view are projections when their centers lie off the slice. Tank geometry alone
does not change the solved target response to illumination. A tether affects pressure only if included in the
scattering model; no uncoupled decorative tether is added automatically.

`wall_images=true` adds `R * p_scattered(mirror(point))` for each tank wall in
exterior water. It represents reflected outgoing target fields, as receiver images
do for a received signal. This still neglects repeated target-wall and wall-wall
scattering; it is not an exact closed-tank cavity solution. `wall_images=false`
samples the original solution without these outgoing image paths. Constant wall
coefficients are used directly. For callable walls, supply explicit constant
`reflection_weights`, frozen over the map, in xmin/xmax/ymin/ymax/bottom/surface
order for a box or bottom/surface order for a cylinder. Cylinder maps exclude the
curved-sidewall response unless explicitly included in the illumination and
`scattering_fields`. The latter adds outgoing `ScatteringField` objects in exterior
water for total/scattered maps; it does not change the target solve. Excluded
filament interiors and exact wall surfaces are masked. The fields are not
automatically re-reflected across planar walls. Match these paths to the receiver
definition. No point-dependent multiplier is introduced into a Helmholtz field.
"""
function tank_field(sol::_TankSurfaceSolution, tank::AbstractTank; kwargs...)
    return _tank_field(sol, tank, sol.k; kwargs...)
end

function tank_field(sol::AbstractSolution, tank::AbstractTank; kwargs...)
    throw(ArgumentError("whole-tank pressure maps require full-3D BEM or closed-surface MFS; volume FEM pressure is currently limited to its computational mesh"))
end

"""
    tank_field(transducer::AbstractTransducer, k, tank::Tank; kwargs...)

Map source illumination without a target, using the Rayleigh field and any images
already included in `transducer`. This overload returns `field=:incident` and labels
the source automatically. It has the same plane/grid controls as the solution overload.
"""
function tank_field(source::AbstractTransducer, k::Real, tank::AbstractTank;
        transducers = (transmitter = source,), kwargs...)
    return _tank_field(nothing, tank, k; source, transducers, field = :incident, kwargs...)
end

function _tank_field(
        sol, tank, k; plane::Symbol = :xz, at = nothing, field::Symbol = :total,
        resolution = nothing, points_per_wavelength::Real = 8, transducers = nothing,
        batch_size::Integer = 2048, source = nothing, wall_images::Bool = true,
        reflection_weights = nothing, scattering_fields = ())
    isfinite(k) && k>0 || throw(ArgumentError("k must be finite and positive"))
    field in (:total, :incident, :scattered) ||
        throw(ArgumentError("field must be :total, :incident or :scattered"))
    isfinite(points_per_wavelength) && points_per_wavelength>0 ||
        throw(ArgumentError("points_per_wavelength must be finite and positive"))
    batch_size>0 || throw(ArgumentError("batch_size must be positive"))
    weights=if wall_images && sol!==nothing && field!==:incident
        raw=reflection_weights===nothing ? [wall.reflection for wall in tank.walls] :
            collect(reflection_weights)
        length(raw)==length(tank.walls) &&
        all(r->r isa Number && isfinite(r) && abs(r)<=1, raw) ||
            throw(ArgumentError("outgoing wall images require one constant reflection_weight per modeled wall; callable walls need explicit frozen weights"))
        ComplexF64.(raw)
    else
        zeros(ComplexF64, length(tank.walls))
    end
    h, v, n=_tank_axes(plane)
    coordinate=at===nothing ? sum(tank.bounds[n])/2 : Float64(at)
    isfinite(coordinate) && tank.bounds[n][1]<=coordinate<=tank.bounds[n][2] ||
        throw(ArgumentError("slice position must lie inside the tank"))
    counts=if resolution===nothing
        ntuple(
            i->max((201, 101)[i],
                ceil(Int,
                    (tank.bounds[(h, v)[i]][2] -
                     tank.bounds[(h, v)[i]][1])*k*points_per_wavelength/(2pi))+1),
            2)
    else
        length(resolution)==2 && all(x->x isa Integer && x>=2, resolution) ||
            throw(ArgumentError("resolution must be a pair of integers at least two"))
        Tuple(Int.(resolution))
    end
    horizontal=collect(range(tank.bounds[h]...; length = counts[1]))
    vertical=collect(range(tank.bounds[v]...; length = counts[2]))
    original_source=if source!==nothing
        source
    elseif sol!==nothing && sol.data.incident isa _PointIncidentField
        sol.data.incident.source
    else
        nothing
    end
    displayed=transducers===nothing ?
              (original_source===nothing ? (;) :
               (; transmitter = original_source)) : transducers
    annotations=_tank_annotations(displayed)
    apertures=Transducer[last(entry) for entry in annotations]
    if original_source!==nothing
        push!(apertures, _transducer_aperture(original_source))
    end
    extras=collect(scattering_fields)
    all(f->f isa ScatteringField && f.k==k, extras) ||
        throw(ArgumentError("scattering_fields must contain fields at the map wavenumber"))
    excluded_fields=original_source isa ScatteringTransducer ?
                    [extras; collect(original_source.fields)] : extras
    locate=_tank_region_locator(sol, tank)
    values=fill(ComplexF64(NaN, NaN), counts)
    regions=zeros(Int, counts)
    indices=CartesianIndices(values)
    source_pressure=source===nothing ? nothing : _transducer_field(source, k)[1]
    opaque=sol isa
           Union{BEMSolution{_FullBEMSurfaceData}, MFSSolution{_FullMFSSurfaceData}} &&
           sol.boundary isa Union{Rigid, PressureRelease}
    for first_index in 1:batch_size:length(values)
        selected=Int[]
        points=NTuple{3, Float64}[]
        for index in first_index:min(first_index + batch_size - 1, length(values))
            i, j=Tuple(indices[index])
            point=ntuple(d->d==h ? horizontal[i] : d==v ? vertical[j] : coordinate, 3)
            if !_inside_tank(tank, point)
                regions[index]=-2
                continue
            end
            if any(f->_field_excluded(f.geometry, SVector(point)), excluded_fields)
                regions[index]=-3
                continue
            end
            region=locate(point)
            aperture=any(aperture_source->_tank_aperture(point, aperture_source), apertures)
            regions[index]=aperture ? -1 : region
            aperture && continue
            field!==:incident && region>0 && (opaque || field===:scattered) && continue
            push!(selected, index)
            push!(points, point)
        end
        isempty(selected) && continue
        values[selected]=sol===nothing ? source_pressure.(points) :
                         pressure(sol, points; field)
        exterior=findall(i->regions[selected[i]]==0, eachindex(selected))
        if field!==:incident
            for extra in extras
                values[selected[exterior]] .+= pressure(extra, points[exterior])
            end
        end
        if sol!==nothing && !isempty(exterior)
            for (wall, weight) in zip(tank.walls, weights)
                iszero(weight) && continue
                reflected=[_mirror_point(wall, points[i]) for i in exterior]
                values[selected[exterior]] .+= weight .*
                                               pressure(sol, reflected; field = :scattered)
            end
        end
    end
    return TankField(tank, horizontal, vertical, values, regions, plane, coordinate,
        Float64(k), field, annotations, wall_images && sol!==nothing && field!==:incident)
end
