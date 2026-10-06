# --- `modal`: exact modal series --------------------------------------------

"""
    modal(body::AbstractBody, boundary::AbstractBoundaryCondition, k; incidence_angle=π/2, kwargs...)

Modal-series result, returns a [`ModalSolution`](@ref). Dispatches on `body`'s concrete type.
`Sphere` backscatter is angle-independent, with no `incidence_angle` keyword, but accepts
`incident::IncidentField=PlaneWave()` (see [`SphericalWave`](@ref) and [`BesselBeam`](@ref));
`angle` is then measured from the incident field's own axis. A bent `Cylinder` automatically
applies the Fresnel bend-coherence correction (see [`Cylinder`](@ref)).
Acoustic spheroids accept `incident=field`, including pointwise callbacks, by surface
projection. `incident_n_eta=max(32,2n_max+12)` and
`incident_n_phi=max(32,4m_max+4)` control incident quadrature independently of modal
truncation. Observation angles remain body-frame `scatter_angle`/`scatter_azimuth`.
Prescribed spheroid fields currently provide far-field amplitudes only.
Post-process with [`target_strength`](@ref)`(sol)` in dB re 1 m² or [`scattering_amplitude`](@ref)`(sol)`,
the complex scattering amplitude in m.
"""
function modal(body::Sphere, boundary::AbstractBoundaryCondition, k::Real; kwargs...)
    f = form_function(boundary, k, body.radius; kwargs...)
    return ModalSolution(body, boundary, k, f)
end

function modal(body::Sphere, boundary::_VESMShell, k::Real; kwargs...)
    return ModalSolution(body, boundary, k,
        form_function(boundary, k, body.radius; kwargs...))
end

function modal(body::Sphere,
        boundary::Union{Rigid, PressureRelease, Impedance, FluidFilled,
            SolidElastic, Shelled{ElasticLayer, FluidInterior},
            Shelled{FluidLayer, FluidInterior}, Shelled{FluidLayer, VacuumInterior},
            Shelled{<:LayeredMaterial}},
        k::Real;
        angle::Real = π, m_max::Integer = _default_mode_count(k * body.radius),
        incident::IncidentField = PlaneWave())
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    m_max >= 0 || throw(ArgumentError("m_max must be nonnegative"))
    incident isa _AxisIncidentField || throw(ArgumentError(
        "sphere modal requires PlaneWave, SphericalWave or zeroth-order BesselBeam"))
    incident isa SphericalWave && incident.range <= body.radius &&
        throw(ArgumentError("spherical source range must exceed the sphere radius"))
    layered_materials = boundary isa Shelled{<:LayeredMaterial} ?
                        _layered_materials(boundary.material)[1] : nothing
    if layered_materials !== nothing
        all(layer -> layer isa Union{FluidLayer, ElasticLayer}, layered_materials) ||
            throw(ArgumentError("spherical layered modal scattering supports FluidLayer and ElasticLayer only"))
    end
    coefficients = ComplexF64[]
    has_interior = boundary isa FluidFilled ||
                   (boundary isa Shelled && boundary.interior isa FluidInterior)
    interior = has_interior ? ComplexF64[] : nothing
    shell = if boundary isa Shelled{FluidLayer}
        NTuple{2, ComplexF64}[]
    elseif boundary isa Shelled{<:LayeredMaterial}
        all(layer -> layer isa FluidLayer, layered_materials) ?
        Vector{NTuple{2, ComplexF64}}[] : Vector{Vector{ComplexF64}}[]
    else
        nothing
    end
    total = zero(ComplexF64)
    for l in 0:m_max
        prefactor = (2l + 1) * incident_coefficient(incident, l, k)
        if boundary isa FluidFilled
            mode = _sphere_fluid_coefficients(boundary, l, k, body.radius)
            coefficient = mode.scattered
            push!(interior, prefactor * mode.interior)
        elseif boundary isa Shelled
            mode = _sphere_shell_coefficients(boundary, l, k, body.radius)
            coefficient = mode.scattered
            interior === nothing || push!(interior, prefactor * mode.interior)
            if shell !== nothing
                shell_mode = mode.shell isa Tuple ? prefactor .* mode.shell :
                             [prefactor .* pair for pair in mode.shell]
                push!(shell, shell_mode)
            end
        else
            coefficient = _modal_coefficient(boundary, l, k, body.radius)
        end
        push!(coefficients, prefactor * coefficient)
        total += prefactor * (-im)^(l + 1) * legendre_p(l, cos(angle)) * coefficient
    end
    return ModalSolution(body, boundary, k, total / k,
        _SphereModalData(coefficients, interior, shell, incident))
end

function modal(body::Spheroid, boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, incident = nothing, kwargs...)
    legacy = incident === nothing || (incident isa PlaneWave &&
              incident.direction === nothing && incident.amplitude == 1)
    f = if legacy && !haskey(kwargs, :incident_n_eta) && !haskey(kwargs, :incident_n_phi)
        form_function(boundary, k, body; incidence_angle, kwargs...)
    else
        _spheroid_incident_amplitude(
            boundary, k, body; incidence_angle,
            incident = something(incident, PlaneWave()), kwargs...)
    end
    return ModalSolution(body, boundary, k, f)
end

function modal(body::Cylinder,
        boundary::Union{Rigid, PressureRelease, FluidFilled,
            Shelled{ElasticLayer, FluidInterior}, SolidElastic},
        k::Real;
        incidence_angle::Real = π / 2,
        m_max::Integer = _default_mode_count(k * sin(incidence_angle) * body.radius), kwargs...)
    f_straight = form_function(boundary, k, body.radius, body.length;
        aspect_angle = incidence_angle, m_max = m_max, kwargs...)
    f = if _isbent(body)
        # Mirrors `bcms_target_strength`'s Fresnel bend-coherence correction (bent_cylinder.jl).
        Lebc = equivalent_length_fresnel(k, body.length, body.radius_curvature)
        Lebc * f_straight / body.length
    else
        f_straight
    end
    return ModalSolution(body, boundary, k, f)
end
