function _slice_units(units::Symbol)
    units === :m && return 1.0, "m"
    units === :mm && return 1000.0, "mm"
    throw(ArgumentError("units must be :m or :mm, got $units"))
end

function _slice_specification(spec)
    spec isa NamedTuple || throw(ArgumentError(
        "each slice must be a named tuple (axis=:x/:y/:z, at=value, project_to=value)"))
    all(key -> haskey(spec, key), (:axis, :at, :project_to)) || throw(ArgumentError(
        "each slice needs axis, at, and project_to fields"))
    plane = spec.axis
    plane in (:x, :y, :z) ||
        throw(ArgumentError("slice axis must be :x, :y, or :z, got $plane"))
    at, project_to = Float64(spec.at), Float64(spec.project_to)
    all(isfinite, (at, project_to)) ||
        throw(ArgumentError("slice positions must be finite"))
    return (; axis = plane, at, project_to)
end

function _slice_sample(sol, spec, grid)
    if spec.axis === :x
        points = [(spec.at, u, v) for u in grid, v in grid]
        projected = (fill(spec.project_to, length(grid), length(grid)),
            [u for u in grid, v in grid], [v for u in grid, v in grid])
    elseif spec.axis === :y
        points = [(u, spec.at, v) for u in grid, v in grid]
        projected = ([u for u in grid, v in grid],
            fill(spec.project_to, length(grid), length(grid)), [v for u in grid, v in grid])
    else
        points = [(u, v, spec.at) for u in grid, v in grid]
        projected = ([u for u in grid, v in grid], [v for u in grid, v in grid],
            fill(spec.project_to, length(grid), length(grid)))
    end
    return points, projected
end

function _slice_colorrange(values, field::Symbol)
    finite = filter(isfinite, reduce(vcat, vec.(values)))
    isempty(finite) && return (-1.0, 1.0)
    lo, hi = extrema(finite)
    if field in (:pressure_real, :pressure_imag)
        bound = max(abs(lo), abs(hi))
        return iszero(bound) ? (-1.0, 1.0) : (-bound, bound)
    end
    return lo == hi ? (lo - 0.5, hi + 0.5) : (lo, hi)
end

function _slice_body_wireframe!(ax, sol, scale)
    render = _solution_render(sol, nothing)
    if render[1] === :revolved
        wireframe!(ax, scale .* render[2], scale .* render[3], scale .* render[4];
            color = (:black, 0.3), linewidth = 0.8)
    elseif render[1] === :trimesh
        points = [Point3f(scale * p[1], scale * p[2], scale * p[3]) for p in render[2]]
        wireframe!(ax, GBMesh(points, render[3]); color = (:black, 0.3), linewidth = 0.8)
    else
        points = [Point3f(scale * p[1], scale * p[2], scale * p[3]) for p in render[2]]
        scatter!(ax, points; color = (:black, 0.35), markersize = 3)
    end
    return nothing
end

function _slice_intersection!(ax, body::AcousticScattering.Sphere, spec, scale, color,
        unit_label)
    abs(spec.at) <= body.radius || return false
    radius = sqrt(max(0.0, body.radius^2 - spec.at^2))
    angle = range(0, 2pi; length = 361)
    circle = radius .* cos.(angle), radius .* sin.(angle)
    coordinate = fill(spec.at, length(angle))
    value = round(scale * spec.at; digits = 3)
    label = "$(spec.axis) = $value $unit_label slice"
    if spec.axis === :x
        lines!(ax, scale .* coordinate, scale .* circle[1], scale .* circle[2];
            color, linewidth = 4, label)
    elseif spec.axis === :y
        lines!(ax, scale .* circle[1], scale .* coordinate, scale .* circle[2];
            color, linewidth = 4, label)
    else
        lines!(ax, scale .* circle[1], scale .* circle[2], scale .* coordinate;
            color, linewidth = 4, label)
    end
    return true
end

function _slice_incident_direction(sol)
    hasproperty(sol.data, :incidence_angle) && return _incident_direction(sol)
    return Vec3f(1, 0, 0)
end

function _plot_solution(sol::AbstractSolution, ::Val{:field_slices}; slices,
        extent::Real, resolution::Integer = 101, field::Symbol = :pressure_real,
        pressure_field::Symbol = :total, units::Symbol = :mm,
        colormap = :balance, colorrange = nothing, shading = false,
        show_body::Bool = true, mark_slices::Bool = true,
        incident_arrow::Bool = true, legend::Bool = true, colorbar::Bool = true,
        figure::NamedTuple = (size = (900, 620),), axis::NamedTuple = NamedTuple(),
        kwargs...)
    isfinite(extent) && extent > 0 ||
        throw(ArgumentError("extent must be finite and positive"))
    resolution >= 2 || throw(ArgumentError("resolution must be at least 2"))
    specs = _slice_specification.(collect(slices))
    isempty(specs) && throw(ArgumentError("slices must not be empty"))
    scale, unit_label = _slice_units(units)
    grid = range(-Float64(extent), Float64(extent); length = resolution)

    coordinates, values = Any[], Any[]
    for spec in specs
        points, projected = _slice_sample(sol, spec, grid)
        sampled = AcousticScattering.pressure(sol, points; field = pressure_field)
        push!(coordinates, projected)
        push!(values, _field_values(sampled, field))
    end
    limits = colorrange === nothing ? _slice_colorrange(values, field) : colorrange
    bound = 1.08scale *
            max(Float64(extent), maximum(abs(spec.project_to) for spec in specs))
    axis_defaults = (aspect = :data,
        xlabel = "x ($unit_label)", ylabel = "y ($unit_label)", zlabel = "z ($unit_label)",
        limits = ((-bound, bound), (-bound, bound), (-bound, bound)),
        azimuth = -0.72pi, elevation = 0.18pi)
    fig = Figure(; figure...)
    ax = Axis3(fig[1, 1]; merge(axis_defaults, axis)...)
    plots = map(zip(coordinates, values)) do (coordinate, value)
        surface!(
            ax, scale .* coordinate[1], scale .* coordinate[2], scale .* coordinate[3];
            color = value, colormap, colorrange = limits, shading, kwargs...)
    end

    show_body && _slice_body_wireframe!(ax, sol, scale)
    marked = false
    if mark_slices && sol.body isa AcousticScattering.Sphere
        colors = (:black, :goldenrod, :dodgerblue, :seagreen)
        for (i, spec) in enumerate(specs)
            marked |= _slice_intersection!(ax, sol.body, spec, scale,
                colors[mod1(i, length(colors))], unit_label)
        end
    end
    legend && marked && axislegend(ax; position = :lt)

    if incident_arrow
        direction = _slice_incident_direction(sol)
        tail = Point3f(-0.96f0 * Float32(bound) .* direction)
        vector = 0.46f0 * Float32(bound) .* direction
        arrows3d!(ax, [tail], [vector]; color = :goldenrod)
        text!(ax, [tail]; text = ["incident wave"], color = :darkgoldenrod,
            fontsize = 14, align = (:center, :top))
    end
    if colorbar
        Colorbar(fig[1, 2], first(plots);
            label = _pressure_label(field; scattered = pressure_field === :scattered))
        colgap!(fig.layout, 1, 60)
    end
    return Makie.FigureAxisPlot(fig, ax, first(plots))
end
