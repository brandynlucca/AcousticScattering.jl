function _pressure_region(solution, patches, point, field, requested)
    containing = 0
    adjacent = nothing
    for (i, parent) in enumerate(solution.body.parents)
        parent == containing || continue
        location = _surface_location(patches[i], point)
        if location === :inside
            containing = i
        elseif location === :on
            adjacent = (parent, i)
            break
        end
    end
    region = if requested !== nothing
        valid = adjacent === nothing ? requested == containing : requested in adjacent
        valid || throw(ArgumentError("the point does not belong to the selected region"))
        requested
    elseif adjacent !== nothing
        field === :interior ? last(adjacent) : first(adjacent)
    else
        containing
    end
    field === :scattered && region != 0 &&
        throw(ArgumentError("field=:scattered requires the unbounded exterior region"))
    field === :interior && region == 0 &&
        throw(ArgumentError("field=:interior requires a bounded fluid region"))
    return region
end

function _pressure_values(solution::BEMSolution{_RegionBEMData}, points, field, requested)
    field === :shell && throw(ArgumentError("select a coupled fluid region with region=i"))
    all(p -> all(isfinite, p), points) ||
        throw(ArgumentError("point coordinates must be finite"))
    incident = ComplexF64[_incident_pressure(solution, point) for point in points]
    field === :incident && requested === nothing && return incident
    isempty(points) && return ComplexF64[]
    interfaces = solution.data.interfaces
    patches = [_region_patches(interface.surface.data, 0) for interface in interfaces]
    regions = [_pressure_region(solution, patches, point, field, requested)
               for point in points]
    field === :incident && return incident
    values = zeros(ComplexF64, length(points))
    direction = _bem3d_incidence_direction(solution.data.incidence_angle, solution.data.incidence_azimuth)
    for region in unique(regions)
        rows = findall(==(region), regions)
        targets = points[rows]
        k = region == 0 ? solution.k :
            solution.k / solution.boundary.materials[region].soundspeed_contrast
        for interface in interfaces
            inside = interface.interior == region
            (inside || interface.exterior == region) || continue
            quad = interface.surface.data
            p = interface.pressure
            dp = inside ? interface.normal_derivative_interior :
                 interface.normal_derivative_exterior
            if region == 0
                pinc = [_incident_pressure(solution, q.coords) for q in quad]
                dpinc = [im * k * dot(direction, q.normal) * pinc[i]
                         for (i, q) in enumerate(quad)]
                p = p - pinc
                dp = dp - dpinc
            end
            values[rows] .+= _pressure_layer_values(quad, k, targets, inside, p, dp)
        end
    end
    if field === :total
        for i in eachindex(values)
            regions[i] == 0 && (values[i] += incident[i])
        end
    end
    return values
end
