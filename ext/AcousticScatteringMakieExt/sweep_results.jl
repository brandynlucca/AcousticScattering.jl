const _SweepResult = Union{AcousticScattering.FrequencySweep,
    AcousticScattering.IncidenceAngleSweep, AcousticScattering.BistaticSweep}

function _sweep_values(sweep, quantity)
    quantity === :target_strength && return sweep.target_strength
    quantity in (:phase, :magnitude, :real, :imag) ||
        throw(ArgumentError("quantity must be :target_strength, :phase, :magnitude, :real or :imag"))
    sweep.amplitudes === nothing &&
        throw(ArgumentError("this sweep has no complex amplitudes"))
    transform = quantity === :phase ? angle :
                quantity === :magnitude ? abs :
                quantity === :real ? real : imag
    return transform.(sweep.amplitudes)
end

function _sweep_coordinates(sweep, angle_units)
    sweep isa AcousticScattering.FrequencySweep &&
        return sweep.frequencies, "Frequency (Hz)"
    prefix = sweep isa AcousticScattering.IncidenceAngleSweep ? "Incidence polar angle" :
             "Observation polar angle"
    return _convert_angle(sweep.angles, angle_units), "$prefix ($(String(angle_units)))"
end

function _sweep_ylabel(quantity)
    quantity === :target_strength && return "Target strength (dB re 1 m²)"
    quantity === :phase && return "Phase (rad)"
    quantity === :magnitude && return "Amplitude magnitude (m)"
    quantity === :real && return "Real amplitude (m)"
    quantity === :imag && return "Imaginary amplitude (m)"
    throw(ArgumentError("unsupported quantity: $quantity"))
end

@recipe(SampledResponsePlot, sweep) do scene
    Attributes(quantity = :target_strength, angle_units = :deg, linewidth = 2,
        colors = [:black, :steelblue, :orange, :purple, :seagreen, :orchid],
        markers = false)
end

function Makie.plot!(plot::SampledResponsePlot)
    sweep = plot.sweep[]
    x, _ = _sweep_coordinates(sweep, plot.angle_units[])
    values = _sweep_values(sweep, plot.quantity[])
    isempty(plot.colors[]) && throw(ArgumentError("colors must not be empty"))
    series = values isa AbstractVector ? reshape(values, :, 1) : values
    for j in axes(series, 2)
        color = plot.colors[][mod1(j, length(plot.colors[]))]
        lines!(plot, x, series[:, j]; color,
            linewidth = plot.linewidth, label = sweep.labels[j])
        plot.markers[] && scatter!(plot, x, series[:, j]; color)
    end
    return plot
end

"""
    plot(sweep, sweeps...; quantity=:both, angle_units=:deg, legend=true,
        figure=(;), axis=(;), kwargs...)

Plot saved frequency, incidence or bistatic sweeps without solving again. One column
per sweep; `quantity=:both` shows target strength above wrapped phase. Scalar-only
results require `quantity=:target_strength`. Other quantities are `:magnitude`, `:real`
and `:imag`. Component labels and colours are shared across panels. Returns a `Figure`.
`plot!(axis, sweep; quantity=:target_strength, kwargs...)` overlays one quantity.
"""
function Makie.plot(sweep::_SweepResult, sweeps::_SweepResult...;
        quantity::Symbol = :both, angle_units::Symbol = :deg, legend::Bool = true,
        figure::NamedTuple = (;), axis::NamedTuple = (;), kwargs...)
    results = (sweep, sweeps...)
    all(s -> s.labels == sweep.labels, results) ||
        throw(ArgumentError("shared panels require matching response labels"))
    quantities = quantity === :both ? (:target_strength, :phase) : (quantity,)
    for result in results, q in quantities

        _sweep_values(result, q)
    end
    fig = Figure(; size = (440length(results), 290length(quantities) + 60), figure...)
    first_plot = nothing
    for (column, result) in enumerate(results), (row, q) in enumerate(quantities)

        _, xlabel = _sweep_coordinates(result, angle_units)
        ax = Axis(fig[row, column]; merge((; xlabel, ylabel = _sweep_ylabel(q)), axis)...)
        rendered = sampledresponseplot!(ax, result; quantity = q, angle_units, kwargs...)
        first_plot === nothing && (first_plot = rendered)
    end
    if legend
        colors = first_plot.colors[]
        elements = [LineElement(; color = colors[mod1(i, length(colors))],
                        linewidth = first_plot.linewidth[])
                    for i in eachindex(sweep.labels)]
        Legend(fig[length(quantities) + 1, 1:length(results)], elements, sweep.labels;
            orientation = :horizontal)
    end
    return fig
end

function Makie.plot!(ax, sweep::_SweepResult; kwargs...)
    return sampledresponseplot!(ax, sweep; kwargs...)
end
