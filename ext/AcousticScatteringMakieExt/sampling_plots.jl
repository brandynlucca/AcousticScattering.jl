@recipe(FrequencySweepPlot, sweep) do scene
    Attributes(
        color = :dodgerblue,
        linewidth = 2,
        xscale = :log10
    )
end
function Makie.plot!(plot::FrequencySweepPlot)
    sweep = plot.sweep[]
    lines!(plot, sweep.frequencies, sweep.target_strength;
        color = plot.color, linewidth = plot.linewidth)
    return plot
end

function Makie.preferred_axis_attributes(::Type{Axis}, plot::FrequencySweepPlot)
    return (
        xscale = plot.xscale[] === :log10 ? log10 : identity,
        xlabel = "Frequency (Hz)",
        ylabel = "Target strength (dB re 1 m²)"
    )
end

@recipe(IncidenceAngleSweepPlot, sweep) do scene
    Attributes(
        color = :dodgerblue,
        linewidth = 2,
        angle_units = :deg
    )
end
function Makie.plot!(plot::IncidenceAngleSweepPlot)
    sweep = plot.sweep[]
    angles = _convert_angle(sweep.angles, plot.angle_units[])
    lines!(
        plot, angles, sweep.target_strength; color = plot.color, linewidth = plot.linewidth)
    return plot
end

function Makie.preferred_axis_attributes(::Type{Axis}, plot::IncidenceAngleSweepPlot)
    return (
        xlabel = _angle_label(plot.angle_units[]),
        ylabel = "Target strength (dB re 1 m²)"
    )
end

@recipe(BistaticSweepPlot, sweep) do scene
    Attributes(
        color = :dodgerblue,
        linewidth = 2,
        kind = :polar,
        angle_units = :deg,
        show_incidence = true
    )
end
# theta=0 is forward scatter (same direction the incident wave was traveling), theta=pi is
# backscatter (looking back toward the source), matching far_field's own documented convention.
function Makie.plot!(plot::BistaticSweepPlot)
    sweep = plot.sweep[]
    kind = plot.kind[]
    if kind === :polar
        # PolarAxis requires a non-negative radius; target_strength is dB (typically negative),
        # so the radial coordinate here is target_strength shifted to start at the pattern's own
        # minimum, not the raw dB value (see preferred_axis_attributes' rticklabelsvisible note).
        r = sweep.target_strength .- minimum(sweep.target_strength)
        lines!(plot, sweep.angles, r; color = plot.color, linewidth = plot.linewidth)
        if plot.show_incidence[]
            rmax = maximum(r)
            lines!(plot, [0.0, 0.0], [0.0, rmax]; color = :gray, linestyle = :dash)
            lines!(plot, [pi, pi], [0.0, rmax]; color = :gray, linestyle = :dash)
            text!(plot, [0.0, pi], [0.5 * rmax, 0.5 * rmax];
                text = ["Forward", "Backscatter"],
                color = :gray, align = (:center, :bottom))
        end
    elseif kind === :cartesian
        angles = _convert_angle(sweep.angles, plot.angle_units[])
        lines!(plot, angles, sweep.target_strength;
            color = plot.color, linewidth = plot.linewidth)
        if plot.show_incidence[]
            fwd, back = _convert_angle([0.0, pi], plot.angle_units[])
            vlines!(plot, [fwd, back]; color = :gray, linestyle = :dash)
            ymax = maximum(sweep.target_strength)
            text!(plot, [fwd, back], [ymax, ymax]; text = ["Forward", "Backscatter"],
                color = :gray, align = (:left, :bottom))
        end
    else
        throw(ArgumentError("kind must be :polar or :cartesian, got $kind"))
    end
    return plot
end

function Makie.preferred_axis_type(plot::BistaticSweepPlot)
    return plot.kind[] === :polar ? PolarAxis : Axis
end

function Makie.preferred_axis_attributes(::Type{PolarAxis}, plot::BistaticSweepPlot)
    ts_min = minimum(plot.sweep[].target_strength)
    return (rticklabelsvisible = false,
        title = "Target strength, relative to $(round(ts_min, digits = 1)) dB re 1 m² minimum")
end

function Makie.preferred_axis_attributes(::Type{Axis}, plot::BistaticSweepPlot)
    return (
        xlabel = _angle_label(plot.angle_units[]),
        ylabel = "Target strength (dB re 1 m²)"
    )
end

@recipe(BistaticMapPlot, bistatic_map) do scene
    Attributes(
        colormap = _MAGNITUDE_COLORMAP,
        colorrange = nothing,
        angle_units = :deg
    )
end
function Makie.plot!(plot::BistaticMapPlot)
    m = plot.bistatic_map[]
    thetas = _convert_angle(m.thetas, plot.angle_units[])
    phis = _convert_angle(m.phis, plot.angle_units[])
    colorrange = plot.colorrange[] === nothing ? _default_colorrange(m.target_strength) :
                 plot.colorrange[]
    heatmap!(plot, thetas, phis, m.target_strength;
        colormap = plot.colormap, colorrange = colorrange)
    return plot
end

function Makie.preferred_axis_attributes(::Type{Axis}, plot::BistaticMapPlot)
    return (
        xlabel = "Observation polar angle, $(_angle_label(plot.angle_units[]))",
        ylabel = "Observation azimuth, $(_angle_label(plot.angle_units[]))"
    )
end

# --- Single public plot(...)/plot!(...) entry points: kind= selects the output dataviz type. ---
# Dispatched on AcousticScattering's own types (AbstractBody/AbstractBoundaryCondition for a
# re-solve sweep, AbstractSolution for a post-solve view), never on bare Function/AbstractVector,
# since extending Makie.plot for argument types this package doesn't own would be type piracy.

function _solver_dispatcher(solver::Symbol)
    solver === :modal && return AcousticScattering.modal
    solver === :kirchhoff && return AcousticScattering.kirchhoff
    solver === :bem && return AcousticScattering.bem
    solver === :mfs && return AcousticScattering.mfs
    solver === :fem && return AcousticScattering.fem
    throw(ArgumentError("solver must be :modal, :kirchhoff, :bem, :mfs, or :fem, got $solver"))
end

function _compute_sweep(body::AbstractBody, boundary::AcousticScattering.AbstractBoundaryCondition,
        xs::AbstractVector{<:Real}, ::Val{:frequency}; sound_speed::Real,
        solver::Symbol = :modal, solver_kwargs::NamedTuple = NamedTuple())
    dispatcher = _solver_dispatcher(solver)
    return AcousticScattering.frequency_sweep(
        k -> dispatcher(body, boundary, k; solver_kwargs...), xs, sound_speed)
end

function _compute_sweep(body::AbstractBody, boundary::AcousticScattering.AbstractBoundaryCondition,
        xs::AbstractVector{<:Real}, ::Val{:incidence_angle}; k::Real,
        solver::Symbol = :modal, solver_kwargs::NamedTuple = NamedTuple())
    dispatcher = _solver_dispatcher(solver)
    return AcousticScattering.incidence_angle_sweep(
        angle -> dispatcher(body, boundary, k; incidence_angle = angle, solver_kwargs...), xs)
end

function _compute_sweep(::AbstractBody, ::AcousticScattering.AbstractBoundaryCondition,
        ::AbstractVector{<:Real}, ::Val{K}; kwargs...) where {K}
    throw(ArgumentError("kind=$K not supported for (body, boundary, values); use :frequency or :incidence_angle"))
end

"""
    plot(body::AbstractBody, boundary::AbstractBoundaryCondition, xs::AbstractVector{<:Real};
         kind::Symbol, solver::Symbol=:modal, solver_kwargs::NamedTuple=NamedTuple(), kwargs...)

Sample and plot target strength across `xs`, re-solving once per point.
`kind=:frequency` treats `xs` as frequencies [Hz] and requires `sound_speed`
[m/s]; `kind=:incidence_angle` treats `xs` as incidence angles [rad] at a
fixed wavenumber and requires `k` [rad/m]. `solver` selects which of
`modal`/`kirchhoff`/`bem`/`mfs`/`fem` re-solves at each point, with
`solver_kwargs` forwarded to it. Remaining `kwargs` are rendering options
(e.g. `xscale`, `angle_units`, `color`).

# Examples
```julia
plot(Sphere(0.01), Rigid(), 10e3:1e3:100e3; kind=:frequency, sound_speed=1477.4)
plot(Spheroid(0.05, 0.02), Rigid(), 0:0.05:pi/2;
    kind=:incidence_angle, k=2pi*38000/1477.4)
```
"""
function Makie.plot(body::AbstractBody, boundary::AcousticScattering.AbstractBoundaryCondition,
        xs::AbstractVector{<:Real}; kind::Symbol, kwargs...)
    solver_keys = (:sound_speed, :k, :solver, :solver_kwargs)
    solver_kwargs_nt = NamedTuple(k => v for (k, v) in kwargs if k in solver_keys)
    plot_kwargs = NamedTuple(k => v for (k, v) in kwargs if !(k in solver_keys))
    sweep = _compute_sweep(body, boundary, xs, Val(kind); solver_kwargs_nt...)
    sweep isa AcousticScattering.FrequencySweep && return frequencysweepplot(sweep; plot_kwargs...)
    return incidenceanglesweepplot(sweep; plot_kwargs...)
end

function Makie.plot!(ax, body::AbstractBody, boundary::AcousticScattering.AbstractBoundaryCondition,
        xs::AbstractVector{<:Real}; kind::Symbol, kwargs...)
    solver_keys = (:sound_speed, :k, :solver, :solver_kwargs)
    solver_kwargs_nt = NamedTuple(k => v for (k, v) in kwargs if k in solver_keys)
    plot_kwargs = NamedTuple(k => v for (k, v) in kwargs if !(k in solver_keys))
    sweep = _compute_sweep(body, boundary, xs, Val(kind); solver_kwargs_nt...)
    sweep isa AcousticScattering.FrequencySweep && return frequencysweepplot!(ax, sweep; plot_kwargs...)
    return incidenceanglesweepplot!(ax, sweep; plot_kwargs...)
end

function _plot_solution(sol::AbstractSolution, ::Val{:bistatic_polar}; angles, azimuth::Real = 0.0, kwargs...)
    sweep = AcousticScattering.bistatic_sweep(sol, angles; azimuth = azimuth)
    return bistaticsweepplot(sweep; kind = :polar, kwargs...)
end
function _plot_solution(
        sol::AbstractSolution, ::Val{:bistatic_cartesian}; angles, azimuth::Real = 0.0, kwargs...)
    sweep = AcousticScattering.bistatic_sweep(sol, angles; azimuth = azimuth)
    return bistaticsweepplot(sweep; kind = :cartesian, kwargs...)
end
function _plot_solution(sol::AbstractSolution, ::Val{:bistatic_map}; thetas, phis, kwargs...)
    m = AcousticScattering.bistatic_map(sol, thetas, phis)
    return bistaticmapplot(m; kwargs...)
end
function _plot_solution(sol::AbstractSolution, ::Val{:mesh}; kwargs...)
    render = _solution_render(sol, nothing)
    return _render_solution_plot(render[1], render[2:end]...; kwargs...)
end
function _plot_solution(
        sol::AbstractSolution, ::Val{:surface_field}; field::Symbol = :pressure_magnitude, kwargs...)
    render = _solution_render(sol, field)
    return _render_solution_plot(render[1], render[2:end]...; kwargs...)
end
function _plot_solution(::AbstractSolution, ::Val{K}; kwargs...) where {K}
    throw(ArgumentError(
        "kind=$K not supported for a solution; use :bistatic_polar, :bistatic_cartesian, :bistatic_map, :mesh, or :surface_field"))
end

function _plot_solution!(
        ax, sol::AbstractSolution, ::Val{:bistatic_polar}; angles, azimuth::Real = 0.0, kwargs...)
    sweep = AcousticScattering.bistatic_sweep(sol, angles; azimuth = azimuth)
    return bistaticsweepplot!(ax, sweep; kind = :polar, kwargs...)
end
function _plot_solution!(
        ax, sol::AbstractSolution, ::Val{:bistatic_cartesian}; angles, azimuth::Real = 0.0, kwargs...)
    sweep = AcousticScattering.bistatic_sweep(sol, angles; azimuth = azimuth)
    return bistaticsweepplot!(ax, sweep; kind = :cartesian, kwargs...)
end
function _plot_solution!(ax, sol::AbstractSolution, ::Val{:bistatic_map}; thetas, phis, kwargs...)
    m = AcousticScattering.bistatic_map(sol, thetas, phis)
    return bistaticmapplot!(ax, m; kwargs...)
end
function _plot_solution!(ax, sol::AbstractSolution, ::Val{:mesh}; kwargs...)
    render = _solution_render(sol, nothing)
    return _render_solution_plot!(ax, render[1], render[2:end]...; kwargs...)
end
function _plot_solution!(
        ax, sol::AbstractSolution, ::Val{:surface_field}; field::Symbol = :pressure_magnitude, kwargs...)
    render = _solution_render(sol, field)
    return _render_solution_plot!(ax, render[1], render[2:end]...; kwargs...)
end
function _plot_solution!(ax, ::AbstractSolution, ::Val{K}; kwargs...) where {K}
    throw(ArgumentError(
        "kind=$K not supported for a solution; use :bistatic_polar, :bistatic_cartesian, :bistatic_map, :mesh, or :surface_field"))
end

"""
    plot(sol::AbstractSolution; kind::Symbol, kwargs...)

Plot a view of an already-solved `sol`. `kind=:bistatic_polar`/`:bistatic_cartesian` need
`angles` [rad] (and optional `azimuth`, default `0.0`); `kind=:bistatic_map` needs `thetas`/`phis`
[rad] — only solutions with reusable per-mode surface data support these (axisymmetric
[`bem`](@ref)/[`mfs`](@ref), shell [`fem`](@ref), and, for `kind=:bistatic_map` only, full 3D
[`bem`](@ref)). `kind=:mesh` shows the solved body's geometry (falling back to `mesh(sol.body;
k=sol.k)` when the solution itself carries no reusable surface mesh); `kind=:surface_field` colors
that geometry by `field` (`:pressure_magnitude`, `:pressure_phase`, `:pressure_real`, or
`:pressure_imag`, default `:pressure_magnitude`) where the solution has real surface field data,
and errors naming the actual gap otherwise (e.g. a radial/meridian `fem(...)` result never
computed a surface field, only a scalar target strength).

# Examples
```julia
sol = bem(Sphere(0.01), Rigid(), k; n=40)
plot(sol; kind=:bistatic_polar, angles=0:0.02:2pi)
plot(sol; kind=:bistatic_map, thetas=0:0.05:pi, phis=0:0.05:2pi)
plot(sol; kind=:mesh)
plot(sol; kind=:surface_field, field=:pressure_phase)
```
"""
Makie.plot(sol::AbstractSolution; kind::Symbol, kwargs...) = _plot_solution(sol, Val(kind); kwargs...)

function Makie.plot!(ax, sol::AbstractSolution; kind::Symbol, kwargs...)
    return _plot_solution!(ax, sol, Val(kind); kwargs...)
end
