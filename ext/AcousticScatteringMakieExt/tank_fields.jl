function _tank_beam_halfangle(source, k)
    # First solution of (2J1(u)/u)^2 = 1/2 for a uniform circular piston.
    u=1.616339948310703
    ka=k*source.radius
    return ka>u ? asin(u/ka) : nothing
end

function _tank_beam_guides!(ax, source, k, h, v, height, color, distance)
    halfangle=_tank_beam_halfangle(source, k)
    halfangle===nothing && return
    projected=hypot(source.axis[h], source.axis[v])
    projected>sin(halfangle) || return
    # Orthogonal projection of the far-field cone, not a slice through that cone.
    spread=asin(sin(halfangle)/projected)
    center=atan(source.axis[v], source.axis[h])
    start=Point2d(source.position[h], height(source.position[v]))
    for angle in (center-spread, center+spread)
        stop=Point2d(source.position[h]+distance*cos(angle),
            height(source.position[v]+distance*sin(angle)))
        lines!(ax, [start, stop]; color = (color, 0.7), linestyle = :dash, linewidth = 1.2)
    end
end

function _tank_aperture_projection(source, h, v, height)
    rotation=AcousticScattering._axis_rotation(collect(source.axis))
    [Point2d(
         source.position[h]+source.radius*(rotation[h, 1]*cos(t)+rotation[h, 2]*sin(t)),
         height(source.position[v]+source.radius*(rotation[v, 1]*cos(t)+rotation[v, 2]*sin(t))))
     for t in range(0, 2pi; length = 97)]
end

function _tank_source_glyph!(ax, source, h, v, height, color, length_scale; label = nothing)
    face=_tank_aperture_projection(source, h, v, height)
    lines!(ax, face; color, linewidth = 4, label)
    start=Point2d(source.position[h], height(source.position[v]))
    stop=Point2d(source.position[h]+length_scale*source.axis[h],
        height(source.position[v]+length_scale*source.axis[v]))
    delta=stop-start
    distance=hypot(delta...)
    if distance>1e-12*length_scale
        lines!(ax, [start, stop]; color, linewidth = 2)
        tangent=delta/distance
        normal=Point2d(-tangent[2], tangent[1])
        head=min(0.25distance, 0.15length_scale)
        poly!(
            ax, [stop, stop-head*tangent+0.45head*normal,
                stop-head*tangent-0.45head*normal]; color)
    else
        scatter!(ax, [start]; color, marker = :circle, markersize = 8)
    end
    return start
end

"""
    plot(field::TankField; field=:pressure_magnitude, depth=true, phase=0,
        show_targets=true, show_transducers=true, kwargs...)

Side-on 2D view of a physical tank pressure plane. The default shows pressure
magnitude from wall to wall and from the water surface to the bottom, with depth
increasing downward. `depth=false` shows the original upward height coordinate.
Use `field=:pressure_real` and `phase` [rad] for a harmonic snapshot, or
`:pressure_imag`/`:pressure_phase`. Plotting does not recompute or renormalize pressure.
Transducer annotations are projections when they lie off the slice.
Colored aperture outlines show actual size and orientation; arrows show face normals,
not beam width. For multiple transducers, `transducer_view=true` adds a geometry-only
close-up in the complementary projection so coincident side-view markers can be resolved.
Gray interiors denote excluded pressure, not zero pressure.
`show_beam=true` draws the projected far-field half-power cone when the piston has
a half-power crossing in its front hemisphere. These dashed guides are computed
from aperture radius and wavenumber; they do not bound the near field or change it.
`field=:pressure_db` displays `20log10(abs(p)/reference)`, with a default reference
equal to the largest finite sampled magnitude. `pressure_reference` sets a positive
explicit reference; `dynamic_range` controls the default displayed range in dB.
This is a relative pressure display, not calibrated sound pressure level.
"""
function Makie.plot(sample::AcousticScattering.TankField;
        field::Symbol = :pressure_magnitude, depth::Bool = true, phase::Real = 0.0,
        show_targets::Bool = true, show_transducers::Bool = true,
        transducer_view::Bool = length(sample.transducers)>1,
        show_beam::Bool = true,
        pressure_reference = nothing, dynamic_range::Real = 50,
        colormap = nothing, colorrange = nothing, colorbar::Bool = true, legend::Bool = true,
        figure::NamedTuple = (size = (1000, transducer_view ? 760 : 500),), axis::NamedTuple = (;), kwargs...)
    isfinite(phase) || throw(ArgumentError("phase must be finite"))
    h, v, n=AcousticScattering._tank_axes(sample.plane)
    depth_view=depth && v==3
    horizontal=sample.horizontal
    vertical=depth_view ? reverse(sample.tank.bounds.z[2] .- sample.vertical) :
             sample.vertical
    pressure_values=depth_view ? reverse(sample.pressure; dims = 2) : sample.pressure
    regions=depth_view ? reverse(sample.regions; dims = 2) : sample.regions
    reference=1.0
    values=if field===:pressure_db
        isfinite(dynamic_range) && dynamic_range>0 ||
            throw(ArgumentError("dynamic_range must be finite and positive"))
        finite=filter(isfinite, vec(abs.(pressure_values)))
        reference=pressure_reference===nothing ?
                  (isempty(finite) || maximum(finite)==0 ? 1.0 : maximum(finite)) :
                  pressure_reference
        reference isa Real && isfinite(reference) && reference>0 ||
            throw(ArgumentError("pressure_reference must be finite and positive"))
        20 .* log10.(abs.(pressure_values) ./ reference)
    else
        _field_values(pressure_values .* cis(-phase), field)
    end
    map_color=colormap===nothing ?
              (field===:pressure_phase ? _PHASE_COLORMAP :
               field in (:pressure_real, :pressure_imag) ? :balance : _MAGNITUDE_COLORMAP) :
              colormap
    limits=colorrange===nothing ?
           (field===:pressure_db ? (-Float64(dynamic_range), 0.0) :
            _slice_colorrange([values], field)) : colorrange
    names=("x", "y", "z")
    defaults=(; aspect = DataAspect(), xlabel = "$(names[h]) (m)",
        xgridvisible = false, ygridvisible = false,
        ylabel = depth_view ? "Depth below water surface (m)" : "$(names[v]) (m)",
        yreversed = depth_view,
        title = "Tank $(sample.plane===:xz ? "side" : sample.plane===:yz ? "end" : "top") view, $(names[n]) = $(sample.at) m",
        limits = (first(horizontal), last(horizontal), first(vertical), last(vertical)))
    fig=Figure(; figure...)
    ax=Axis(fig[1, 1]; merge(defaults, axis)...)
    heat=heatmap!(ax, horizontal, vertical, values;
        colormap = map_color, colorrange = limits, nan_color = :gray80, kwargs...)
    if show_targets
        for region in sort(unique(regions))
            region>0 || continue
            contour!(ax, horizontal, vertical, Float64.(regions .== region);
                levels = [0.5], color = :black, linewidth = 1.5)
            cells=findall(==(region), regions)
            center=Point2d(sum(horizontal[i[1]] for i in cells)/length(cells),
                sum(vertical[i[2]] for i in cells)/length(cells))
            text!(ax, [center]; text = ["Target $region"], align = (:center, :center),
                fontsize = 12, color = :black)
        end
    end
    lo, hi=extrema(horizontal)
    bottom, top=extrema(vertical)
    if sample.tank isa ProfiledTank
        contour!(ax, horizontal, vertical, Float64.(regions .!= -2);
            levels = [0.5], color = :black, linewidth = 2)
    elseif sample.tank.radius!==nothing && sample.plane===:xy
        center=(sum(sample.tank.bounds.x)/2, sum(sample.tank.bounds.y)/2)
        outline=[Point2d(center[1]+sample.tank.radius*cos(t), center[2]+sample.tank.radius*sin(t))
                 for t in range(0, 2pi; length = 257)]
        lines!(ax, outline; color = :black, linewidth = 2)
    else
        lines!(
            ax, [lo, lo, hi, hi], [top, bottom, bottom, top]; color = :black, linewidth = 2)
    end
    if v==3
        surface=depth_view ? bottom : top
        floor=depth_view ? top : bottom
        lines!(ax, [lo, hi], [surface, surface]; color = :dodgerblue, linewidth = 3)
        sample.tank isa ProfiledTank ||
            lines!(ax, [lo, hi], [floor, floor]; color = :black, linewidth = 2)
    elseif sample.tank.radius===nothing
        lines!(ax, [lo, hi], [top, top]; color = :black, linewidth = 2)
    end
    if show_transducers
        colors=(:orangered, :deepskyblue, :limegreen, :magenta)
        height=depth_view ? z->sample.tank.bounds.z[2]-z : identity
        for (i, (name, source)) in enumerate(sample.transducers)
            projected=!isapprox(source.position[n], sample.at; atol = 1e-10, rtol = 0)
            label=projected ? "$name (projected)" : name
            halfangle=_tank_beam_halfangle(source, sample.k)
            if show_beam && halfangle!==nothing
                label*="; $(round(rad2deg(2halfangle);digits=1))° full half-power beam"
                _tank_beam_guides!(ax, source, sample.k, h, v, height,
                    colors[mod1(i, length(colors))], hypot(hi-lo, top-bottom))
            end
            _tank_source_glyph!(ax, source, h, v, height, colors[mod1(i, length(colors))],
                0.08*(hi-lo); label)
        end
        legend && !isempty(sample.transducers) && axislegend(ax; position = :rt)
        if transducer_view && !isempty(sample.transducers)
            # The omitted coordinate becomes horizontal, separating side-by-side faces.
            detail=Axis(fig[2, 1]; aspect = DataAspect(), xlabel = "$(names[n]) (m)",
                ylabel = "$(names[h]) (m)", xgridvisible = false, ygridvisible = false,
                title = "Transducer close-up ($(names[n])-$(names[h]) projection; geometry only)")
            radius=maximum(last(entry).radius for entry in sample.transducers)
            for (i, (name, source)) in enumerate(sample.transducers)
                color=colors[mod1(i, length(colors))]
                point=_tank_source_glyph!(detail, source, n, h, identity, color, 3radius)
                text!(detail, [point]; text = [name],
                    offset = (0, -18), align = (:center, :top),
                    color, fontsize = 13)
            end
            xlo=minimum(s.position[n]-s.radius for (_, s) in sample.transducers)
            xhi=maximum(s.position[n]+s.radius for (_, s) in sample.transducers)
            ylo=minimum(s.position[h]-s.radius for (_, s) in sample.transducers)
            yhi=maximum(s.position[h]+s.radius for (_, s) in sample.transducers)
            limits!(detail, xlo-2radius, xhi+2radius, ylo-2radius, yhi+4radius)
            rowsize!(fig.layout, 2, 210)
        end
    end
    label=field===:pressure_db ?
          (pressure_reference===nothing ? "Pressure magnitude (dB re map maximum)" :
           "Pressure magnitude (dB re $reference supplied units)") :
          _pressure_label(field; scattered = sample.field===:scattered, prescribed = true)
    colorbar && Colorbar(fig[1, 2], heat; label)
    return Makie.FigureAxisPlot(fig, ax, heat)
end

function _plot_solution(sol::AcousticScattering._TankSurfaceSolution, ::Val{:tank};
        tank::AcousticScattering.AbstractTank, plane::Symbol = :xz, at = nothing,
        pressure_field::Symbol = :total, resolution = nothing, points_per_wavelength::Real = 8,
        transducers = nothing, batch_size::Integer = 2048, wall_images::Bool = true,
        reflection_weights = nothing, scattering_fields = (), kwargs...)
    sample=AcousticScattering.tank_field(sol, tank; plane, at, field = pressure_field,
        resolution, points_per_wavelength, transducers, batch_size, wall_images, reflection_weights,
        scattering_fields)
    return Makie.plot(sample; kwargs...)
end
