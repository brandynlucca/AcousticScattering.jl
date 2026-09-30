struct ComponentComparison
    coupled::BEMSolution{_RegionBEMData}
    isolated::Vector{BEMSolution}
    labels::Vector{String}
end

function _component_labels(count, labels)
    names = labels === nothing ? ["Interface $i" for i in 1:count] :
            String.(collect(labels))
    length(names) == count || throw(ArgumentError("supply one label per interface"))
    return ["Coupled"; "Isolated " .* names; "Coherent sum"]
end

"""
    components(solution; labels=nothing, solver_kwargs=(;))

Compare a coupled fluid-region BEM `solution` with each interface solved in isolation
in the exterior medium. Preserve each surface's position, orientation, material contrast,
incident direction and phase origin. The coupled solution is reused; isolated solves
inherit its formulation, quadrature correction, equilibration, compression and iterative
controls, with overrides in `solver_kwargs`.

Returns a comparison retaining `coupled`, `isolated` solutions and `labels` for the
coupled response, isolated responses and their coherent sum. The coherent sum adds
complex amplitudes, omitting mutual interactions. All contrasts refer to the original
exterior medium, including for an interface originally enclosed by another region.

Pass this result to [`bistatic_sweep`](@ref), or return it from the solver callback of
[`frequency_sweep`](@ref) or [`incidence_angle_sweep`](@ref). `labels` optionally supplies
one name per interface; no further solves are needed for observation-angle sweeps.
"""
function components(sol::BEMSolution{_RegionBEMData}; labels = nothing,
        solver_kwargs::NamedTuple = (;))
    count = length(sol.body.surfaces)
    response_labels = _component_labels(count, labels)
    report = diagnostics(sol)
    compression = get(solver_kwargs, :compression, report.compression)
    gmres_kwargs = report.method === :gmres && compression.method !== :none ?
                   Base.structdiff(report.solver_options, NamedTuple{(
        :incidence_angle, :incidence_azimuth)}) : (;)
    options = merge(
        (; correction = report.correction, formulation = report.formulation,
            incident = haskey(solver_kwargs, :transducer) ? nothing : sol.data.incident,
            condition_limit = report.condition_limit,
            equilibrate = report.equilibrate, compression, gmres_kwargs),
        solver_kwargs)
    isolated = BEMSolution[bem(surface, material, sol.k;
                               incidence_angle = sol.data.incidence_angle,
                               incidence_azimuth = sol.data.incidence_azimuth, options...)
                           for (surface, material) in zip(sol.body.surfaces, sol.boundary.materials)]
    return ComponentComparison(sol, isolated, response_labels)
end

function _comparison_amplitudes(result::ComponentComparison; kwargs...)
    isolated = [scattering_amplitude(sol; kwargs...) for sol in result.isolated]
    return [scattering_amplitude(result.coupled; kwargs...); isolated; sum(isolated)]
end
