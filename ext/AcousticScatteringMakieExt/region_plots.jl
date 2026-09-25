const _RegionSolution = AcousticScattering.BEMSolution{AcousticScattering._RegionBEMData}

function _region_vertex_pressure(quad, pressure)
    sums = zeros(ComplexF64, length(Inti.nodes(quad.mesh)))
    counts = zeros(Int, length(sums))
    for E in Inti.element_types(quad.mesh)
        E <: SVector && continue
        connectivity = Inti.connectivity(quad.mesh, E)
        tags = Inti.etype2qtags(quad, E)
        for j in axes(connectivity, 2)
            average = sum(pressure[tags[:, j]]) / size(tags, 1)
            for vi in Inti.vertices_idxs(E)
                v = connectivity[vi, j]
                sums[v] += average
                counts[v] += 1
            end
        end
    end
    return sums ./ max.(counts, 1)
end

function _clip_region_triangle(points, values, normal, offset)
    polygon = collect(zip(points, values))
    clipped = eltype(polygon)[]
    previous = last(polygon)
    dp = sum(normal .* previous[1]) - offset
    for current in polygon
        dc = sum(normal .* current[1]) - offset
        if (dp <= 0) != (dc <= 0)
            t = dp / (dp - dc)
            0 < t < 1 && push!(clipped, ((1 - t) * previous[1] + t * current[1],
                (1 - t) * previous[2] + t * current[2]))
        end
        dc <= 0 && push!(clipped, current)
        previous, dp = current, dc
    end
    return clipped
end

function _region_plot_data(sol, field; interfaces, cutaway)
    selected = collect(interfaces)
    isempty(selected) && throw(ArgumentError("interfaces must not be empty"))
    all(i -> i isa Integer && 1 <= i <= length(sol.data.interfaces), selected) ||
        throw(ArgumentError("interfaces must contain valid interface indices"))
    length(unique(selected)) == length(selected) ||
        throw(ArgumentError("interfaces must not contain duplicates"))
    if cutaway !== nothing
        length(cutaway.normal) == 3 && all(isfinite, cutaway.normal) &&
        sum(abs2, cutaway.normal) > 0 && isfinite(cutaway.offset) ||
            throw(ArgumentError("cutaway needs a finite nonzero normal and finite offset"))
    end
    return map(selected) do i
        interface = sol.data.interfaces[i]
        quad = interface.surface.data
        points, faces = _inti_mesh_points_faces(quad)
        pressure = field === nothing ? zeros(ComplexF64, length(points)) :
                   _region_vertex_pressure(quad, interface.pressure)
        used = sort!(unique(reduce(vcat, collect.(faces))))
        indices = zeros(Int, length(points))
        indices[used] = eachindex(used)
        points, pressure = points[used], pressure[used]
        faces = [TriangleFace(indices[collect(face)]...) for face in faces]
        if cutaway !== nothing && interface.exterior == 0
            clipped_points, clipped_faces, clipped_pressure = Point3f[], TriangleFace[],
            ComplexF64[]
            for face in faces
                ids = collect(face)
                polygon = _clip_region_triangle(points[ids], pressure[ids],
                    cutaway.normal, cutaway.offset)
                start = length(clipped_points)
                append!(clipped_points, first.(polygon))
                append!(clipped_pressure, last.(polygon))
                for j in 2:(length(polygon) - 1)
                    push!(clipped_faces, TriangleFace(start + 1, start + j, start + j + 1))
                end
            end
            points, faces, pressure = clipped_points, clipped_faces, clipped_pressure
        end
        values = field === nothing ? nothing : _field_values(pressure, field)
        (; index = i, exterior = interface.exterior, points, faces, values)
    end
end

@recipe(RegionSurfacePlot, pieces) do scene
    Attributes(colormap = _MAGNITUDE_COLORMAP, colorrange = nothing,
        interface_colors = [:steelblue, :orange, :seagreen, :orchid],
        interface_alpha = Float64[], solid_interfaces = Int[],
        wireframe_interfaces = Int[], show_edges = false)
end

Makie.preferred_axis_type(::RegionSurfacePlot) = Axis3
function Makie.preferred_axis_attributes(::Type{Axis3}, ::RegionSurfacePlot)
    (aspect = :data, xticks = LinearTicks(2), yticks = LinearTicks(2),
        xticklabelsize = 14, yticklabelsize = 14, xlabeloffset = 35, ylabeloffset = 55,
        zlabeloffset = 60)
end

function Makie.plot!(plot::RegionSurfacePlot)
    pieces = plot.pieces[]
    colored = first(pieces).values !== nothing
    solid = plot.solid_interfaces[]
    alphas = plot.interface_alpha[]
    alpha_of(piece) = isempty(alphas) ? 1.0 : Float64(alphas[mod1(piece.index, length(alphas))])
    values = colored ?
             reduce(vcat, [piece.values for piece in pieces if !(piece.index in solid)];
        init = Float64[]) : Float64[]
    finite_values = filter(isfinite, values)
    lo, hi = isempty(finite_values) ? (-1.0, 1.0) : extrema(finite_values)
    limits = plot.colorrange[] === nothing ? (lo == hi ? (lo - 0.5, hi + 0.5) : (lo, hi)) :
             plot.colorrange[]
    colors = plot.interface_colors[]
    isempty(colors) && throw(ArgumentError("interface_colors must not be empty"))
    wireframes = plot.wireframe_interfaces[]
    all(i -> i in getproperty.(pieces, :index), wireframes) ||
        throw(ArgumentError("wireframe_interfaces must select displayed interfaces"))
    colored && !isempty(wireframes) &&
        throw(ArgumentError("wireframe_interfaces requires kind=:mesh"))
    for piece in sort(collect(pieces); by = p -> -alpha_of(p), alg = Base.Sort.MergeSort)
        isempty(piece.faces) && continue
        geometry = GBMesh(piece.points, piece.faces)
        alpha = alpha_of(piece)
        flat = !colored || piece.index in solid
        color = flat ? colors[mod1(piece.index, length(colors))] : piece.values
        if piece.index in wireframes
            wireframe!(plot, geometry; color)
        elseif flat
            mesh!(plot, geometry; color = alpha < 1 && !(color isa Tuple) ? (color, alpha) : color)
        else
            mesh!(plot, geometry; color, colorrange = limits, transparency = alpha < 1,
                colormap = alpha < 1 ? (plot.colormap[], alpha) : plot.colormap)
            plot.show_edges[] && wireframe!(plot, geometry; color = (:black, 0.25))
        end
    end
    return plot
end

for (kind, default_field) in ((:mesh, nothing), (:surface_field, :pressure_magnitude))
    @eval function _plot_solution(sol::_RegionSolution, ::Val{$(QuoteNode(kind))};
            field = $(QuoteNode(default_field)), interfaces = eachindex(sol.data.interfaces),
            cutaway = nothing, interface_labels = nothing,
            legend::Bool = true, colorbar::Bool = true, incident_arrow::Bool = false,
            kwargs...)
        pieces = _region_plot_data(sol, field; interfaces, cutaway)
        names = interface_labels === nothing ?
                ["Region $(p.index) / region $(p.exterior)" for p in pieces] :
                String.(collect(interface_labels))
        length(names) == length(pieces) ||
            throw(ArgumentError("supply one interface label per displayed surface"))
        result = regionsurfaceplot(pieces;
            figure = (size = (760, 550), figure_padding = (70, 20, 55, 20)),
            axis = (xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)"), kwargs...)
        if field === nothing && legend
            colors = result.plot.interface_colors[]
            elements = [p.index in result.plot.wireframe_interfaces[] ?
                        LineElement(; color = colors[mod1(p.index, length(colors))]) :
                        PolyElement(; color = colors[mod1(p.index, length(colors))])
                        for p in pieces]
            Legend(result.figure[1, 2], elements, names)
        elseif field !== nothing && colorbar
            field_mesh = first(filter(p -> p isa Makie.Mesh && p.color[] isa AbstractVector{<:Real},
                result.plot.plots))
            Colorbar(result.figure[1, 2]; colormap = result.plot.colormap[],
                limits = field_mesh.colorrange[], label = _pressure_label(field))
            colgap!(result.figure.layout, 1, 70)
        end
        incident_arrow && _add_incident_arrow!(result.axis, sol,
            reduce(vcat, [first(_inti_mesh_points_faces(interface.surface.data))
                          for interface in sol.data.interfaces]))
        return result
    end
    @eval function _plot_solution!(ax, sol::_RegionSolution, ::Val{$(QuoteNode(kind))};
            field = $(QuoteNode(default_field)), interfaces = eachindex(sol.data.interfaces),
            cutaway = nothing, kwargs...)
        pieces = _region_plot_data(sol, field; interfaces, cutaway)
        return regionsurfaceplot!(ax, pieces; kwargs...)
    end
end
