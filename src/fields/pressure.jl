"""
    pressure(solution, point; field=:total, region=nothing)
    pressure(solution, points; field=:total, region=nothing)

Complex acoustic pressure under the `exp(-iωt)` convention. Plane-wave solutions use
unit incident amplitude; prescribed fields retain their supplied normalization.
Coordinates are in meters.

`point` is a three-coordinate tuple or vector. A collection of points returns an array of the
same shape. A real `3×N` matrix stores points in columns and returns a vector of length `N`.

`field` selects:
- `:total`: incident plus scattered pressure outside the body, or total transmitted pressure
  inside a fluid interior or shell.
- `:scattered`: defined on and outside the body only.
- `:incident`: the retained unperturbed incident field (a plane wave by default).
- `:interior`: total pressure inside a homogeneous fluid body, a shell's fluid cavity, or a
  bounded coupled fluid region.
- `:shell`: total pressure in a fluid shell, including both surface traces.

`region` selects a specific fluid region for coupled fluid BEM (`region=0` is the unbounded
exterior, `region=i` the fluid inside interface `i`). Other solutions reject this keyword.

Not every solver/geometry/boundary combination supports every `field`/`region` option. See
[BEM and MFS](@ref boundary-theory) and [FEM and shell coupling](@ref fem-theory) for solver
support and numerical caveats.

# Example
```julia
solution = modal(Sphere(0.01), FluidFilled(1.2, 1.1), 100.0)
pressure(solution, (0.02, 0.0, 0.0); field=:scattered)
pressure(solution, [(0.0, 0.0, 0.0), (0.02, 0.0, 0.0)])
```
"""
function pressure(solution::FreeSurfaceSolution, points; field::Symbol = :total, region = nothing)
    region === nothing ||
        throw(ArgumentError("region selection is not supported for a FreeSurfaceSolution"))
    return pressure(solution.direct, points; field) .+
           solution.sign .* pressure(solution.reflected, points; field)
end

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
    return _field_points(p -> _pressure_values(solution, p, field, region), points)
end

function _check_pressure_solution(solution)
    spherical = solution.body isa Sphere &&
                ((solution isa ModalSolution && solution.data isa _SphereModalData) ||
                 solution isa FEMSolution{_RadialFEMData} ||
                 solution isa FEMSolution{_SphereMeridianFEMData})
    full = solution isa
           Union{BEMSolution{_FullBEMSurfaceData}, MFSSolution{_FullMFSSurfaceData}}
    boundary_geometry = solution.body isa Union{Sphere, Spheroid} ||
                        (solution.body isa Cylinder && (full || !_isbent(solution.body))) ||
                        (full && solution.body isa _SurfaceGeometry)
    boundary_data = solution isa Union{BEMSolution{_FullBEMSurfaceData},
        BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_FullMFSSurfaceData}} ||
                    (solution isa MFSSolution{_AxisymmetricSurfaceData} &&
                     solution.data.source_modes !== nothing)
    supported = spherical ||
                solution isa
                Union{BEMSolution{_RegionBEMData}, FEMSolution{_VolumeFEMData},
                    FEMSolution{_CylinderMeridianFEMData},
                    FEMSolution{_SpheroidMeridianFEMData},
                    FEMSolution{_ShellFEMSurfaceData}} ||
                (boundary_geometry && boundary_data &&
                 (solution.boundary isa Union{Rigid, PressureRelease, FluidFilled} ||
                  (solution isa MFSSolution{_AxisymmetricSurfaceData} &&
                   solution.boundary isa Shelled{FluidLayer, FluidInterior})))
    supported || throw(ArgumentError(
        "pressure requires a supported acoustic solution"))
    isfinite(solution.k) && solution.k > 0 ||
        throw(ArgumentError("pressure requires a finite positive wavenumber"))
    return nothing
end

function _pressure_values(solution, points, field)
    [_pressure_point(solution, point, field) for point in points]
end

function _pressure_values(solution, points, field, region)
    _pressure_values(solution, points, field)
end

_incident_pressure(solution, point) = cis(solution.k * point[1])

function _incident_pressure(solution::ModalSolution, point)
    incident_pressure(solution.data.incident, solution.k, point)
end

_pressure_axis(solution) = SVector(1.0, 0.0, 0.0)
_pressure_axis(solution::ModalSolution) = _incident_axis(solution.data.incident)

function _incident_pressure(sol::BEMSolution{_FullBEMSurfaceData}, point)
    _data_incident_pressure(sol.data, sol.k, point)
end
function _incident_pressure(sol::BEMSolution{_RegionBEMData}, point)
    _data_incident_pressure(sol.data, sol.k, point)
end

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

function _pressure_point(solution, point, field)
    r = _pressure_radius(solution, point)
    field === :incident && return _incident_pressure(solution, point)
    region = _sphere_pressure_region(solution.boundary, solution.body.radius, r, field)
    mu = iszero(r) ? 0.0 :
         clamp(dot(_pressure_axis(solution), SVector{3}(point)) / r, -1.0, 1.0)
    value = _sphere_pressure(solution, r, mu, region)
    return region === :exterior && field === :total ?
           _incident_pressure(solution, point) + value : value
end

function _pressure_point(solution::FEMSolution{_SphereMeridianFEMData}, point, field)
    r = _pressure_radius(solution, point)
    incident = _incident_pressure(solution, point)
    field === :incident && return incident
    _sphere_pressure_region(solution.boundary, solution.body.radius, r, field)
    μ = clamp(point[1] / r, -1.0, 1.0)
    scattered = if r <= last(solution.data.radii)
        _meridian_rect_value(solution.data.radii, solution.data.angles,
            solution.data.pressure, r, acos(μ))
    else
        sum(Bl * hs(l-1, solution.k*r) * legendre_p(l-1, μ)
        for (l, Bl) in enumerate(solution.data.outgoing))
    end
    return field === :total ? incident + scattered : scattered
end

function _meridian_rect_value(radii, angles, values, r, θ)
    i = clamp(searchsortedlast(radii, r), 1, length(radii)-1)
    j = clamp(searchsortedlast(angles, θ), 1, length(angles)-1)
    u = clamp((r-radii[i])/(radii[i + 1]-radii[i]), 0.0, 1.0)
    v = clamp((θ-angles[j])/(angles[j + 1]-angles[j]), 0.0, 1.0)
    return (1-u)*(1-v)*values[i, j] + u*(1-v)*values[i + 1, j] +
           u*v*values[i + 1, j + 1] + (1-u)*v*values[i, j + 1]
end

function _pressure_point(
        solution::FEMSolution{T}, point,
        field) where {
        T <: Union{_CylinderMeridianFEMData, _SpheroidMeridianFEMData}}
    all(isfinite, point) || throw(ArgumentError("point coordinates must be finite"))
    β = solution.data.incidence_angle
    incident = cis(solution.k * (point[1]*cos(β) + point[2]*sin(β)))
    field === :incident && return incident
    region = _boundary_pressure_region(solution, point, field)
    r = norm(point)
    μ = iszero(r) ? 1.0 : clamp(point[1]/r, -1.0, 1.0)
    θ = acos(μ)
    φ = atan(point[3], point[2])
    scattered = zero(ComplexF64)
    for (mindex, mode) in enumerate(solution.data.fields)
        m = mindex-1
        iszero(r) && m > 0 && continue
        value = if region === :interior
            s = iszero(r) ? 0.0 : r/_meridian_inner_radius(mode, θ)
            _meridian_rect_value(mode.radial_interior, mode.angles,
                mode.interior, s, θ)
        elseif r <= mode.outer_radius
            inner = _meridian_inner_radius(mode, θ)
            s = (r-inner)/(mode.outer_radius-inner)
            _meridian_rect_value(mode.radial_exterior, mode.angles,
                mode.exterior, s, θ)
        else
            sum(Bl*hs(l, solution.k*r)*legendre_p(l, m, μ)
            for (offset, Bl) in enumerate(mode.outgoing)
            for l in (m+offset-1,))
        end
        scattered += value*cos(m*φ)
    end
    return field in (:total, :interior) ? incident + scattered : scattered
end

function _meridian_inner_radius(mode::_MeridianFEMModeField, θ)
    j = clamp(searchsortedlast(mode.angles, θ), 1, length(mode.angles)-1)
    t = (θ-mode.angles[j])/(mode.angles[j + 1]-mode.angles[j])
    return (1-t)*mode.inner_radius[j] + t*mode.inner_radius[j + 1]
end

function _pressure_point(solution::FEMSolution{_ShellFEMSurfaceData}, point, field)
    all(isfinite, point) || throw(ArgumentError("point coordinates must be finite"))
    β = solution.data.incidence_angle
    incident = cis(solution.k * (point[1]*cos(β) + point[2]*sin(β)))
    field === :incident && return incident
    body = solution.body.body
    r = norm(point)
    ρ = hypot(point[2], point[3])
    outer_level = body isa Sphere ? r/body.radius :
                  hypot(point[1]/body.a, ρ/body.b)
    if field in (:total, :scattered) && outer_level >= 1-8eps(Float64)
        scattered = if abs(outer_level-1) <= 1e-12
            _shell_surface_trace(solution.data.ps_ext,
                solution.data.p_ext_modes, point)
        else
            _axisymmetric_pressure_traces(solution.data.ps_ext,
                solution.data.p_ext_modes, solution.data.dpdn_ext_modes,
                solution.k, point)
        end
        return field === :total ? incident + scattered : scattered
    end
    data = solution.data
    if data.ps_int !== nothing && field in (:total, :interior)
        inner_level = if body isa Sphere
            r/(body.radius-solution.body.thickness)
        else
            b_in = body.b-solution.body.thickness
            a_in = sqrt(body.a^2-body.b^2+b_in^2)
            hypot(point[1]/a_in, ρ/b_in)
        end
        if inner_level <= 1+8eps(Float64)
            return abs(inner_level-1) <= 1e-12 ?
                   _shell_surface_trace(data.ps_int, data.p_int_modes, point) :
                   _axisymmetric_pressure_traces(data.ps_int,
                data.p_int_modes, data.dpdn_int_modes,
                data.k_interior, point; inside = true)
        end
    end
    throw(ArgumentError("the selected field has no acoustic pressure at this point"))
end

function _shell_surface_trace(ps, modes, point)
    ρ = hypot(point[2], point[3])
    z = point[1]
    φ = atan(point[3], point[2])
    distances = [begin
                     ρp, zp = _panel_point(panel, _closest_param(ρ, z, panel))
                     hypot(ρ-ρp, z-zp)
                 end
                 for panel in ps]
    nearest = findall(==(minimum(distances)), distances)
    return sum((sum(mode[j] for j in nearest)/length(nearest))*cos((m-1)*φ)
    for (m, mode) in enumerate(modes))
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
    inside = region === :interior
    sources = inside ? data.interior_sources : data.sources
    coefficients = inside ? data.interior_coefficients : data.coefficients
    wave = inside ? solution.k / solution.boundary.soundspeed_contrast : solution.k
    return sum(c * _green3d(wave, target, source)
    for (c, source) in zip(coefficients, sources))
end

function _pressure_values(solution::MFSSolution{_FullMFSSurfaceData}, points, field)
    all(p -> all(isfinite, p), points) ||
        throw(ArgumentError("point coordinates must be finite"))
    incident = field in (:total, :incident) ? _incident_pressure_values(solution, points) :
               nothing
    field === :incident && return incident
    regions = _boundary_pressure_regions(solution, points, field)
    values = ComplexF64[_mfs_pressure(solution, point, region)
                        for (point, region) in zip(points, regions)]
    if field === :total
        for i in eachindex(values)
            regions[i] === :exterior && (values[i] += incident[i])
        end
    end
    return values
end

# Evaluate each prescribed incident-field sample once, in parallel, and never
# evaluate it for a scattered-only map.
# Incident-field callbacks already have the solver's concurrent-call contract.
function _incident_pressure_values(solution, points)
    if !(solution.data.incident isa _PointIncidentField) || length(points) < 64
        return ComplexF64[_incident_pressure(solution, point) for point in points]
    end
    values = Vector{ComplexF64}(undef, length(points))
    Threads.@threads for i in eachindex(points)
        values[i] = _incident_pressure(solution, points[i])
    end
    values
end

function _mfs_pressure(solution::MFSSolution{_AxisymmetricSurfaceData}, point, region)
    if solution.boundary isa Shelled{FluidLayer, FluidInterior}
        return _nested_fluid_mfs_pressure(solution, point, region)
    end
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

function _nested_fluid_mfs_pressure(solution, point, region)
    boundary = solution.boundary
    k = region === :exterior ? solution.k :
        region === :shell ? solution.k / boundary.material.soundspeed_contrast :
        solution.k / boundary.interior.soundspeed_contrast
    rho, z, phi = hypot(point[2], point[3]), point[1], atan(point[3], point[2])
    value = zero(ComplexF64)
    for (i, mode) in enumerate(solution.data.source_modes)
        m = i - 1
        sources = region === :shell ? (mode.shell_inner, mode.shell_outer) :
                  (region === :exterior ? (mode.exterior,) : (mode.interior,))
        for source in sources, j in eachindex(source.coefficients)

            value += source.coefficients[j] * cos(m * phi) *
                     _azimuthal_G(
                         k, rho, z, source.rho[j], source.z[j]; m, rtol = mode.rtol)
        end
    end
    return value
end

function _pressure_values(solution::BEMSolution{_FullBEMSurfaceData}, points, field)
    all(p -> all(isfinite, p), points) ||
        throw(ArgumentError("point coordinates must be finite"))
    incident = field in (:total, :incident) ? _incident_pressure_values(solution, points) :
               nothing
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
            pinc, dpinc = _incident_traces(data.quad, solution.k, data.incidence_angle,
                data.incidence_azimuth, data.incident)
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
    if body isa Union{Spheroid, _SurfaceGeometry} || (body isa Cylinder && _isbent(body))
        patches = _region_patches(solution.data.quad, 0)
        # Bernstein control hulls enclose the full curved surface, including any
        # excursion beyond mesh nodes. Reject distant points once per batch before
        # allocating the detailed oriented-ray classifier's subdivision workspace.
        bounds = ntuple(
            d -> (
                minimum(p[d].lo for patch in patches for p in patch.net),
                maximum(p[d].hi for patch in patches for p in patch.net)),
            3)
        # Preserve the detailed classifier's near-surface tolerance in every
        # rotated ray frame (sqrt(3) times that tolerance bounds each coordinate).
        padding = 1024eps(maximum(max(abs(lo), abs(hi)) for (lo, hi) in bounds))
        return [_surface_pressure_region(solution.boundary,
                    all(d -> bounds[d][1]-padding <= point[d] <= bounds[d][2]+padding, 1:3) ?
                    _surface_location(patches, point) : :outside, field)
                for point in points]
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
    if solution isa MFSSolution{_AxisymmetricSurfaceData} &&
       solution.boundary isa Shelled{FluidLayer, FluidInterior}
        return _nested_fluid_pressure_region(solution, point, field)
    end
    if solution.body isa Sphere
        return _sphere_pressure_region(solution.boundary, solution.body.radius,
            _pressure_radius(solution, point), field)
    end
    body = solution.body
    rho, axial = hypot(point[2], point[3]), abs(point[1])
    level = if body isa Spheroid
        hypot(axial/body.a, rho/body.b)
    else
        depth = solution isa BEMSolution{_AxisymmetricSurfaceData} ? 0.0 :
                body.endcap_depth
        iszero(depth) ? max(rho/body.radius, axial/(body.length/2)) :
        hypot(rho/body.radius, max(0, axial-body.length/2)/depth)
    end
    abs(level-1) <= 8eps(Float64) && (level = 1.0)
    field in (:total, :scattered) && level >= 1 && return :exterior
    solution.boundary isa FluidFilled && field in (:total, :interior) && level <= 1 &&
        return :interior
    throw(ArgumentError("the selected field has no acoustic pressure at this point"))
end

function _nested_fluid_pressure_region(solution, point, field)
    body = solution.body
    inner = _nested_fluid_inner(body, solution.boundary.radius_ratio)
    rho, z = hypot(point[2], point[3]), point[1]
    outer_level = _nested_fluid_source_clearance(body, rho, z)
    inner_level = _nested_fluid_source_clearance(inner, rho, z)
    field in (:total, :scattered) && outer_level >= -8eps(Float64) && return :exterior
    field in (:total, :shell) && outer_level < -8eps(Float64) &&
        inner_level > 8eps(Float64) && return :shell
    field in (:total, :interior) && inner_level <= 8eps(Float64) && return :interior
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
        elseif field in (:total, :shell) && boundary.material isa LayeredMaterial
            layers, ratios = _layered_materials(boundary.material)
            for i in eachindex(ratios)
                lower = i == length(ratios) ? boundary.radius_ratio : ratios[i + 1]
                if lower * a <= r <= ratios[i] * a
                    layers[i] isa FluidLayer || throw(ArgumentError(
                        "acoustic pressure is undefined inside an elastic shell layer"))
                    return i
                end
            end
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
    if region === :shell || region isa Integer
        layer = region isa Integer ?
                _layered_materials(solution.boundary.material)[1][region] :
                solution.boundary.material
        kr = solution.k * r / layer.soundspeed_contrast
        for (i, mode_shell) in enumerate(solution.data.shell_coefficients)
            regular, singular = region isa Integer ? mode_shell[region] : mode_shell
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

# Complex pressure of a volume FEM solution at arbitrary Cartesian points, via Ferrite point location.
# All fluid regions carry scattered pressure relative to the common incident field;
# total pressure adds that field back. Solid regions have no scalar pressure field.
function _pressure_values(solution::FEMSolution{_VolumeFEMData}, points, field)
    all(p -> all(isfinite, p), points) ||
        throw(ArgumentError("point coordinates must be finite"))
    data = solution.data
    system = data.system
    grid = system.grid
    incident = ComplexF64[data.pinc(point) for point in points]
    field === :incident && return incident
    coords = [Ferrite.Vec{3}(Float64.(point)) for point in points]
    handler = Ferrite.PointEvalHandler(grid, coords)
    pv = Ferrite.PointValues(_VOLUME_IP, _VOLUME_GIP)
    values = Vector{ComplexF64}(undef, length(points))
    for (i, location) in enumerate(Ferrite.PointIterator(handler))
        location === nothing && throw(ArgumentError(
            "point $(points[i]) lies outside the volume FEM's computational domain"))
        cell_id = Ferrite.cellid(location)
        label = system.labels[cell_id]
        label in (_REGION_FLUID, _REGION_INTERIOR) || throw(ArgumentError(
            "pressure is only defined in fluid regions of a volume FEM solution, " *
            "point $(points[i]) is elsewhere"))
        Ferrite.reinit!(pv, location)
        cell = grid.cells[cell_id]
        raw = sum(Ferrite.shape_value(pv, 1, j) *
                  data.solution[_pressure_dof(cell.nodes[j])]
        for j in eachindex(cell.nodes))
        values[i] = if label == _REGION_FLUID
            field === :scattered ? raw :
            field === :total ? raw + incident[i] :
            throw(ArgumentError("field=:$field is not defined outside the body"))
        else
            # The interior unknown is also scattered relative to the same background incident wave.
            field in (:total, :interior) ? raw + incident[i] :
            throw(ArgumentError("field=:$field is not defined inside a fluid region"))
        end
    end
    return values
end
