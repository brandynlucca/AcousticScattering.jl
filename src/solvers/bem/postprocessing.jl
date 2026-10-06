function scattering_amplitude(
        sol::BEMSolution{_FullBEMSurfaceData}; direction::Union{Nothing, AbstractVector} = nothing)
    d = sol.data
    xhat = direction === nothing ?
           .-_bem3d_incidence_direction(d.incidence_angle, d.incidence_azimuth) :
           direction
    if _uses_edge_quadrature(d)
        return sol.boundary isa FluidFilled ?
               _edge_fluid_far_field(d, sol.k, xhat, sol.boundary) :
               _edge_far_field(d, sol.k, xhat)
    end
    return far_field(d.quad, xhat, sol.k, d.p_scat, d.dpdn_scat)
end

function scattering_amplitude(sol::BEMSolution{_RegionBEMData};
        direction::Union{Nothing, AbstractVector} = nothing)
    incident = _bem3d_incidence_direction(sol.data.incidence_angle, sol.data.incidence_azimuth)
    observation = direction === nothing ? -incident : direction
    length(observation) == 3 && all(isfinite, observation) &&
    isapprox(norm(observation), 1) ||
        throw(ArgumentError("direction must be a finite unit vector with three components"))
    amplitude = zero(ComplexF64)
    for interface in sol.data.interfaces
        interface.exterior == 0 || continue
        quad = interface.surface.data
        p_inc, q_inc = _incident_traces(quad, sol.k, sol.data.incidence_angle,
            sol.data.incidence_azimuth, sol.data.incident)
        amplitude += far_field(quad, observation, sol.k, interface.pressure - p_inc,
            interface.normal_derivative_exterior - q_inc)
    end
    return amplitude
end
