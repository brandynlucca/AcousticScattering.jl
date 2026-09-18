"""
    pressure(solution, point; field=:total, region=nothing)
    pressure(solution, points; field=:total, region=nothing)

Complex acoustic pressure divided by incident pressure amplitude, under the
`exp(-iωt)` convention. Supported solutions are modal and radial FEM spheres
with `Rigid`, `PressureRelease`, `FluidFilled`, `SolidElastic`, fluid shells
with fluid/vacuum interiors, or elastic shells with fluid interiors. Axisymmetric/full
BEM and axisymmetric MFS support spheres and straight cylinders with rigid,
pressure-release and fluid-filled boundaries; full MFS supports their rigid and
pressure-release boundaries. Full BEM/MFS also support closed bent cylinders and supplied
meshes with those respective boundary conditions. Coupled fluid BEM supports nested,
branched and disconnected regions. Coordinates are in meters. Spheres
are centered at the origin and straight cylinders align with x; supplied meshes retain
their coordinates. Modal/FEM incidence is along +x;
BEM/MFS incidence follows the solve's `incidence_angle` and, for full solves,
`incidence_azimuth`.

`point` is a three-coordinate tuple or vector. A collection of such points
returns an array of the same shape. A real `3×N` matrix stores points in columns
and returns a vector of length `N`.

- `field=:total` returns incident plus scattered pressure outside the body,
  or total transmitted pressure in a fluid interior or shell.
- `field=:scattered` is defined on and outside the body only.
- `field=:incident` returns the unperturbed plane wave at any point.
- `field=:interior` returns total pressure in a homogeneous fluid body, a shell's fluid
  cavity, or a bounded coupled fluid region, including the inner interface trace.
- `field=:shell` returns total pressure in a fluid shell, including both of its
  surface traces. It is not defined in elastic material.

For spherical shells, at the outer surface, `:total` uses the exterior trace. At an inner interface it
selects the fluid cavity, or the fluid shell's trace for a vacuum cavity. Points within
eight floating-point spacings of either radius are treated as interface points.
Rigid, pressure-release, elastic and vacuum regions have no acoustic pressure field.
For elastic shells, `interior_coupling=:identical_fluid` uses exterior-fluid properties
in the cavity. `modal(...; angle)` selects a far-field
observation and does not rotate this incident wave. Evaluation retains the solve's
modal cutoff; refine it and the FEM radial mesh when approaching a surface.

For coupled fluid BEM, `region=nothing` selects the containing fluid automatically.
`region=0` selects the unbounded exterior; `region=i` selects the fluid immediately
inside interface `i`, excluding its children. Points must belong to that region.
At an interface, either adjacent region may be requested; without `region`, `:total`
selects the parent side and `:interior` selects the child side. `:interior` is valid
throughout any bounded fluid region, `:scattered` only in the unbounded exterior,
and `:shell` is unavailable. Region selection follows the solved polynomial meshes.
`:incident` remains the unperturbed exterior-medium plane wave; an explicit `region`
also checks its query points. The `region` keyword is exclusive to coupled fluid BEM.

Radial FEM uses its linear/quadratic nodal fields inside the exterior annulus
and linear fields in a fluid interior or shell. Elastic and layered FEM spheres use
retained spherical-wave coefficients in the exterior and cavity; acoustic FEM uses
outgoing coefficients beyond its truncation radius. This function does not evaluate
elastic stress/displacement, viscous layers, spheroids,
or pressure in nonfluid regions. Full BEM uses density-interpolation quadrature;
axisymmetric BEM subtracts the constant Laplace double layer before adaptive integration.
MFS evaluates its solved outgoing source fields. Refine the surface mesh, quadrature
order and MFS source spacing/offset to check convergence. Axisymmetric BEM retains
piecewise-constant traces: near-surface accuracy depends on meridian position as well
as panel count. Spheres and straight cylinders use analytic body dimensions for region
selection. Bent and supplied surfaces use the solved polynomial mesh, with adaptive
ray crossings; unresolved locations raise `ArgumentError`. Surface points use the
exterior trace for `:total` and the fluid-side trace for `:interior`. The geometric
stopping tolerance is 512 floating-point spacings at the largest absolute Bernstein
coordinate. Axisymmetric cylinder BEM uses flat ends; full BEM and MFS honor
`endcap_depth`. Lateral-only bent-cylinder MFS does not support this query.

# Example
```julia
solution = modal(Sphere(0.01), FluidFilled(1.2, 1.1), 100.0)
pressure(solution, (0.02, 0.0, 0.0); field=:scattered)
pressure(solution, [(0.0, 0.0, 0.0), (0.02, 0.0, 0.0)])
```
"""
function pressure(solution::AbstractSolution, points; field::Symbol = :total, region = nothing)
    _check_pressure_solution(solution)
    field in (:total, :scattered, :incident, :interior, :shell) ||
        throw(ArgumentError("field must be :total, :scattered, :incident, :interior or :shell"))
    if region !== nothing
        solution isa BEMSolution{_RegionBEMData} ||
            throw(ArgumentError("region selection requires a coupled fluid BEM solution"))
        region isa Integer && 0 <= region <= length(solution.data.interfaces) ||
            throw(ArgumentError("region must be zero or an interface index"))
    end
    return _pressure_points(solution, points, field, region)
end

function _check_pressure_solution(solution)
    spherical = solution.body isa Sphere &&
                ((solution isa ModalSolution && solution.data isa _SphereModalData) ||
                 solution isa FEMSolution{_RadialFEMData})
    full = solution isa
           Union{BEMSolution{_FullBEMSurfaceData}, MFSSolution{_FullMFSSurfaceData}}
    boundary_geometry = solution.body isa Sphere ||
                        (solution.body isa Cylinder && (full || !_isbent(solution.body))) ||
                        (full && solution.body isa _SurfaceGeometry)
    boundary_data = solution isa Union{BEMSolution{_FullBEMSurfaceData},
        BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_FullMFSSurfaceData}} ||
                    (solution isa MFSSolution{_AxisymmetricSurfaceData} &&
                     solution.data.source_modes !== nothing)
    supported = spherical || solution isa BEMSolution{_RegionBEMData} ||
                (boundary_geometry && boundary_data &&
                 solution.boundary isa Union{Rigid, PressureRelease, FluidFilled})
    supported || throw(ArgumentError(
        "pressure requires a supported acoustic solution"))
    isfinite(solution.k) && solution.k > 0 ||
        throw(ArgumentError("pressure requires a finite positive wavenumber"))
    return nothing
end

function _pressure_points(solution, point::Tuple{Real, Real, Real}, field, region = nothing)
    only(_pressure_values(solution, [point], field, region))
end

function _pressure_points(solution, point::AbstractVector{<:Real}, field, region = nothing)
    length(point) == 3 || throw(ArgumentError("a point must have three coordinates"))
    return only(_pressure_values(solution, [Tuple(point)], field, region))
end

function _pressure_points(solution, points::AbstractMatrix{<:Real}, field, region = nothing)
    size(points, 1) == 3 || throw(ArgumentError("point matrices must have three rows"))
    return _pressure_values(solution, [Tuple(point) for point in eachcol(points)], field, region)
end

function _pressure_points(solution, points::AbstractArray, field, region = nothing)
    coordinates = map(_pressure_coordinates, points)
    return reshape(_pressure_values(solution, vec(coordinates), field, region), size(points))
end

function _pressure_coordinates(point)
    point isa Union{Tuple{Real, Real, Real}, AbstractVector{<:Real}} &&
    length(point) == 3 ||
        throw(ArgumentError("each point must have three real coordinates"))
    return Tuple(point)
end

function _pressure_values(solution, points, field)
    [_pressure_point(solution, point, field) for point in points]
end

function _pressure_values(solution, points, field, region)
    _pressure_values(solution, points, field)
end

_incident_pressure(solution, point) = cis(solution.k * point[1])

function _incident_pressure(solution::Union{BEMSolution, MFSSolution}, point)
    data = solution.data
    azimuth = data isa _AxisymmetricSurfaceData ? 0.0 : data.incidence_azimuth
    direction = _bem3d_incidence_direction(data.incidence_angle, azimuth)
    return cis(solution.k * dot(direction, point))
end

function _pressure_radius(solution, point)
    all(isfinite, point) || throw(ArgumentError("point coordinates must be finite"))
    a = solution.body.radius
    r = norm(point)
    abs(r - a) <= 8eps(a) && (r = a)
    if solution.boundary isa Shelled
        b = a * solution.boundary.radius_ratio
        abs(r - b) <= 8eps(b) && (r = b)
    end
    return r
end

function _pressure_points(solution, points, field, region = nothing)
    throw(ArgumentError("provide a Cartesian point, an array of points, or a 3×N matrix"))
end

function _pressure_point(solution, point, field)
    r = _pressure_radius(solution, point)
    incident = _incident_pressure(solution, point)
    field === :incident && return incident
    region = _sphere_pressure_region(solution.boundary, solution.body.radius, r, field)
    mu = iszero(r) ? 0.0 : clamp(point[1] / r, -1.0, 1.0)
    value = _sphere_pressure(solution, r, mu, region)
    return region === :exterior && field === :total ? incident + value : value
end

function _pressure_point(solution::MFSSolution, point, field)
    all(isfinite, point) || throw(ArgumentError("point coordinates must be finite"))
    incident = _incident_pressure(solution, point)
    field === :incident && return incident
    region = _boundary_pressure_region(solution, point, field)
    value = _mfs_pressure(solution, point, region)
    return region === :exterior && field === :total ? incident + value : value
end

function _mfs_pressure(solution::MFSSolution{_FullMFSSurfaceData}, point, region)
    data = solution.data
    target = Tuple(SVector{3, Float64}(point))
    return sum(c * _green3d(solution.k, target, source)
    for (c, source) in zip(data.coefficients, data.sources))
end

function _pressure_values(solution::MFSSolution{_FullMFSSurfaceData}, points, field)
    all(p -> all(isfinite, p), points) ||
        throw(ArgumentError("point coordinates must be finite"))
    incident = ComplexF64[_incident_pressure(solution, point) for point in points]
    field === :incident && return incident
    regions = _boundary_pressure_regions(solution, points, field)
    values = ComplexF64[_mfs_pressure(solution, point, region)
                        for (point, region) in zip(points, regions)]
    return field === :total ? values + incident : values
end

function _mfs_pressure(solution::MFSSolution{_AxisymmetricSurfaceData}, point, region)
    inside = region === :interior
    k = inside ? solution.k / solution.boundary.soundspeed_contrast : solution.k
    rho, z, phi = hypot(point[2], point[3]), point[1], atan(point[3], point[2])
    value = zero(ComplexF64)
    for (i, mode) in enumerate(solution.data.source_modes)
        m = i - 1
        sources = inside ? mode.interior : mode.exterior
        for j in eachindex(sources.coefficients)
            value += sources.coefficients[j] * cos(m * phi) *
                     _azimuthal_G(
                         k, rho, z, sources.rho[j], sources.z[j]; m, rtol = mode.rtol)
        end
    end
    return value
end

function _pressure_values(solution::BEMSolution{_FullBEMSurfaceData}, points, field)
    all(p -> all(isfinite, p), points) ||
        throw(ArgumentError("point coordinates must be finite"))
    incident = ComplexF64[_incident_pressure(solution, point) for point in points]
    field === :incident && return incident
    regions = _boundary_pressure_regions(solution, points, field)
    values = zeros(ComplexF64, length(points))
    data = solution.data
    if _uses_edge_quadrature(data)
        values = solution.boundary isa FluidFilled ?
                 _edge_fluid_pressure(
            data, solution.k, points, regions, solution.boundary) :
                 _edge_pressure(data, solution.k, points)
        if field === :total
            for i in eachindex(values)
                regions[i] === :exterior && (values[i] += incident[i])
            end
        end
        return values
    end
    for region in (:exterior, :interior)
        indices = findall(==(region), regions)
        isempty(indices) && continue
        inside = region === :interior
        k = inside ? solution.k / solution.boundary.soundspeed_contrast : solution.k
        p, dp = data.p_scat, data.dpdn_scat
        if inside
            direction = _bem3d_incidence_direction(data.incidence_angle, data.incidence_azimuth)
            pinc = [_incident_pressure(solution, q.coords) for q in data.quad]
            dpinc = [im * solution.k * dot(direction, q.normal) * pinc[i]
                     for (i, q) in enumerate(data.quad)]
            p = p + pinc
            dp = solution.boundary.density_contrast .* (dp + dpinc)
        end
        values[indices] = _pressure_layer_values(
            data.quad, k, points[indices], inside, p, dp)
    end
    if field === :total
        for i in eachindex(values)
            regions[i] === :exterior && (values[i] += incident[i])
        end
    end
    return values
end

function _pressure_layer_values(quad, k, points, inside, p, dp)
    values = zeros(ComplexF64, length(points))
    bounds = _fluid_quadrature_size(quad)
    correction = (; method = :dim, target_location = inside ? :inside : :outside,
        maxdist = 2bounds.radius)
    op = Inti.Helmholtz(; k, dim = 3)
    indices = eachindex(points)
    partitions = _fluid_regular_range(k, bounds) ?
                 [[i
                   for i in indices
                   if (k*norm(SVector{3, Float64}(points[i])-bounds.center) <= 2) == near]
                  for near in (true, false)] : [indices]
    for partition in partitions
        for first in 1:256:length(partition)
            rows = partition[first:min(first + 255, length(partition))]
            targets = [(; coords = SVector{3, Float64}(points[i]),
                           normal = zero(SVector{3, Float64})) for i in rows]
            S, D = _fluid_layer_operators(
                op, targets, quad, correction; exclude_nearest = true)
            values[rows] = inside ? S * dp - D * p : D * p - S * dp
        end
    end
    return values
end

function _boundary_pressure_regions(solution, points, field)
    body = solution.body
    if body isa _SurfaceGeometry || (body isa Cylinder && _isbent(body))
        patches = _region_patches(solution.data.quad, 0)
        return [_surface_pressure_region(solution.boundary,
                    _surface_location(patches, point), field) for point in points]
    end
    return [_boundary_pressure_region(solution, point, field) for point in points]
end

function _surface_pressure_region(boundary, location, field)
    field in (:total, :scattered) && location in (:outside, :on) && return :exterior
    boundary isa FluidFilled && field in (:total, :interior) &&
        location in (:inside, :on) && return :interior
    throw(ArgumentError("the selected field has no acoustic pressure at this point"))
end

function _boundary_pressure_region(solution, point, field)
    if solution.body isa Sphere
        return _sphere_pressure_region(solution.boundary, solution.body.radius,
            _pressure_radius(solution, point), field)
    end
    body = solution.body
    rho, axial = hypot(point[2], point[3]), abs(point[1])
    depth = solution isa BEMSolution{_AxisymmetricSurfaceData} ? 0.0 : body.endcap_depth
    level = if iszero(depth)
        max(rho/body.radius, axial/(body.length/2))
    else
        hypot(rho/body.radius, max(0, axial-body.length/2)/depth)
    end
    abs(level-1) <= 8eps(Float64) && (level = 1.0)
    field in (:total, :scattered) && level >= 1 && return :exterior
    solution.boundary isa FluidFilled && field in (:total, :interior) && level <= 1 &&
        return :interior
    throw(ArgumentError("the selected field has no acoustic pressure at this point"))
end

function _pressure_point(solution::BEMSolution{_AxisymmetricSurfaceData}, point, field)
    all(isfinite, point) || throw(ArgumentError("point coordinates must be finite"))
    incident = _incident_pressure(solution, point)
    field === :incident && return incident
    region = _boundary_pressure_region(solution, point, field)
    value = _axisymmetric_pressure(solution, point, region)
    return region === :exterior && field === :total ? incident+value : value
end

function _sphere_pressure_region(boundary, a, r, field)
    field in (:total, :scattered) && r >= a && return :exterior
    field === :scattered &&
        throw(ArgumentError("field=:scattered requires exterior points"))
    if boundary isa FluidFilled && field in (:total, :interior) && r <= a
        return :interior
    elseif boundary isa Shelled
        b = a * boundary.radius_ratio
        if field in (:total, :interior) && boundary.interior isa FluidInterior && r <= b
            return :interior
        elseif field in (:total, :shell) && boundary.material isa FluidLayer && b <= r <= a
            return :shell
        end
    end
    throw(ArgumentError("the selected field has no acoustic pressure at this point"))
end

_sphere_interior_speed(boundary::FluidFilled) = boundary.soundspeed_contrast

function _sphere_interior_speed(boundary::Shelled)
    identical = boundary.material isa ElasticLayer &&
                boundary.material.interior_coupling === :identical_fluid
    return identical ? 1.0 : boundary.interior.soundspeed_contrast
end

function _sphere_pressure(solution::ModalSolution, r, mu, region)
    value = zero(ComplexF64)
    if region === :shell
        kr = solution.k * r / solution.boundary.material.soundspeed_contrast
        for (i, (regular, singular)) in enumerate(solution.data.shell_coefficients)
            l = i - 1
            value += (regular * js(l, kr) + singular * ys(l, kr)) * legendre_p(l, mu)
        end
        return value
    end
    inside = region === :interior
    coefficients = inside ? solution.data.interior_coefficients : solution.data.coefficients
    k = inside ? solution.k / _sphere_interior_speed(solution.boundary) : solution.k
    radial = inside ? js : hs
    for (i, coefficient) in enumerate(coefficients)
        l = i - 1
        value += coefficient * radial(l, k * r) * legendre_p(l, mu)
    end
    return value
end

function _sphere_pressure(solution::FEMSolution{_RadialFEMData}, r, mu, region)
    value = zero(ComplexF64)
    for (i, mode) in enumerate(solution.data.modes)
        l = i - 1
        if region === :interior
            radial = solution.boundary isa FluidFilled ?
                     _radial_pressure_value(mode.interior, r, 1) :
                     mode.interior.coefficient * js(l, mode.interior.wavenumber * r)
        elseif region === :shell
            radial = _radial_pressure_value(mode.shell, r, 1)
        elseif hasproperty(mode, :exterior) && r <= last(mode.exterior.radii)
            radial = _radial_pressure_value(mode.exterior, r, mode.order)
        else
            radial = mode.coefficient * hs(l, solution.k * r)
        end
        value += radial * legendre_p(l, mu)
    end
    return value
end

function _radial_pressure_value(domain, r, order)
    radii, values = domain.radii, domain.pressure
    j = clamp(searchsortedlast(radii, r), 1, length(radii) - 1)
    if order == 1
        t = (r - radii[j]) / (radii[j + 1] - radii[j])
        return (1 - t) * values[j] + t * values[j + 1]
    end
    left = 2 * ((j - 1) ÷ 2) + 1
    t = 2 * (r - radii[left]) / (radii[left + 2] - radii[left]) - 1
    return t * (t - 1) / 2 * values[left] + (1 - t^2) * values[left + 1] +
           t * (t + 1) / 2 * values[left + 2]
end
