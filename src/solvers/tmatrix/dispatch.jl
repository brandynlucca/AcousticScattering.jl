function modal(::Spheroid, ::_ElasticSpheroidBoundary, ::Real; kwargs...)
    throw(ArgumentError("elastic spheroids and shells are solved by `tmatrix`, not `modal`"))
end

# --- `tmatrix`: transition-matrix solution for elastic spheroids ------------

"""
    tmatrix(body::Spheroid, boundary, k; incidence_angle=π/2, incidence_azimuth=0,
        scatter_angle=π-incidence_angle, scatter_azimuth=incidence_azimuth+π, m_max, n_max, check=true)

Transition-matrix scattering by a prolate or oblate spheroid, returns a [`TMatrixSolution`](@ref).
`boundary` is a `SolidElastic` solid or `Shelled(ElasticLayer(...), FluidInterior(...) or VacuumInterior(),
radius_ratio)`, an elastic shell whose confocal inner surface has an equatorial semi-axis
`radius_ratio` times the outer one. A `Shelled(LayeredMaterial(...), ...)` containing both
fluid and generalized elastic layers is also supported for prolate spheroids with aspect
ratio at most 1.25 and `ka ≤ 1.5` through the default projected method. For a fluid
core, `method=:farfield` builds a volume-FEM-generated angular transition through
aspect ratio 1.5 and `ka ≤ 1.8`. The returned `TMatrixSolution` retains this transition:
reuse it with `scattering_amplitude(solution; incidence_angle, angle, azimuth)`
or `target_strength` to evaluate other directions without rebuilding FEM.
This path accepts `polar_order=6`, `m_max=4`, `points_per_wavelength=6`,
`h`, `h_body`, `domain_radius` and `dtn_order`. The default polar/azimuthal grid
requires three incidence solves on one assembly; `check=true` adds a broadside
solve and rejects a held-out angular error above 1%. Increase `polar_order` and
`m_max` together to check angular convergence; refine the volume/interface meshes
and exterior closure separately. Angles are in radians from the axis of symmetry, and
the defaults give backscatter. `m_max` and `n_max` truncate the azimuthal orders and degrees.
Single elastic shells warn when the reduced-order amplitude changes by more than 1%; mixed
layers reject a change above 3%. Pass `check = false` to skip that check. Post-process with
[`target_strength`](@ref) or
[`scattering_amplitude`](@ref).

Single elastic spheroids/shells accept `incident=field`, projecting incident pressure
and gradient into their regular spheroidal basis. `incident_n_eta=max(32,2n_max+12)`
and `incident_n_phi=max(32,4m_max+4)` control incident quadrature separately.
The transition is assembled once per azimuthal order for both incident Fourier sectors.
This option is not available for mixed-layer or `method=:farfield` transitions.

See [Transition-matrix solutions](@ref tmatrix-theory).
"""
function tmatrix(body::Spheroid, boundary::_ElasticSpheroidBoundary, k::Real;
        incidence_angle::Real = π / 2, incident = nothing, kwargs...)
    legacy = incident === nothing || (incident isa PlaneWave &&
              incident.direction === nothing && incident.amplitude == 1)
    f = if legacy && !haskey(kwargs, :incident_n_eta) && !haskey(kwargs, :incident_n_phi)
        form_function(boundary, k, body; incidence_angle, kwargs...)
    else
        _elastic_spheroid_incident_amplitude(
            boundary, k, body; incidence_angle,
            incident = something(incident, PlaneWave()), kwargs...)
    end
    return TMatrixSolution(body, boundary, k, f)
end

function _elastic_spheroid_incident_amplitude(boundary, k, body;
        m_max::Integer = _default_elastic_orders(boundary, k, body),
        n_max::Integer = _default_elastic_orders(boundary, k, body),
        check::Bool = true, kwargs...)
    amplitude(m, n) = _spheroid_incident_amplitude(boundary, k, body; m_max = m, n_max = n,
        transition = _elastic_transition_function(boundary, k, body, n), kwargs...)
    result = amplitude(m_max, n_max)
    if check && boundary isa Shelled && n_max > 2 && m_max > 2
        _warn_elastic_truncation(abs(result - amplitude(m_max - 2, n_max - 2)) /
                                 abs(result))
    end
    return result
end

function tmatrix(body::Spheroid, boundary::Shelled{<:LayeredMaterial},
        k::Real; incidence_angle::Real = π / 2,
        incidence_azimuth::Real = 0.0,
        scatter_angle::Real = π - incidence_angle,
        scatter_azimuth::Real = incidence_azimuth + π,
        method::Symbol = :projected, kwargs...)
    f = if method === :projected
        form_function(boundary, k, body; incidence_angle, incidence_azimuth,
            scatter_angle, scatter_azimuth, kwargs...)
    elseif method === :farfield
        return _farfield_tmatrix(body, boundary, k; incidence_angle, incidence_azimuth,
            scatter_angle, scatter_azimuth, kwargs...)
    else
        throw(ArgumentError("mixed spheroid tmatrix method must be :projected or :farfield"))
    end
    return TMatrixSolution(body, boundary, k, f)
end

function modal(::Spheroid, ::Shelled{<:LayeredMaterial}, ::Real; kwargs...)
    throw(ArgumentError("mixed layered spheroids are solved by `tmatrix` or `fem`, not `modal`"))
end
