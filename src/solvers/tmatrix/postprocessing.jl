function scattering_amplitude(sol::TMatrixSolution{Nothing}; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "scattering_amplitude(::TMatrixSolution) takes no keywords: the directions were already " *
        "fixed when tmatrix(...) was called. Call tmatrix(...) again with different angles."))
    return sol.f
end

function scattering_amplitude(sol::TMatrixSolution{_FarFieldTMatrixData};
        incidence_angle::Union{Nothing, Real} = nothing,
        incidence_azimuth::Union{Nothing, Real} = nothing,
        angle::Union{Nothing, Real} = nothing,
        azimuth::Union{Nothing, Real} = nothing)
    data = sol.data
    changed_incidence = incidence_angle !== nothing || incidence_azimuth !== nothing
    beta = something(incidence_angle, data.incidence_angle)
    alpha = something(incidence_azimuth, data.incidence_azimuth)
    theta = something(angle, changed_incidence ? π - beta : data.scatter_angle)
    phi = something(azimuth, changed_incidence ? alpha + π : data.scatter_azimuth)
    return _farfield_transition_amplitude(data.blocks, data.nodes,
        beta, alpha, theta, phi)
end

diagnostics(sol::TMatrixSolution{_FarFieldTMatrixData}) = sol.data.report
