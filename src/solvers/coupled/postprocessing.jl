function scattering_amplitude(sol::FreeSurfaceSolution;
        angle::Real = π - sol.incidence_angle, azimuth::Real = sol.incidence_azimuth + π)
    return scattering_amplitude(sol.direct; angle, azimuth) +
           sol.sign * scattering_amplitude(sol.reflected; angle, azimuth)
end

function scattering_amplitude(sol::CoupledSolution{_SpheroidFEMBEMData};
        angle::Real = pi-sol.data.incidence_angle, azimuth::Real = pi)
    isfinite(angle) && isfinite(azimuth) ||
        throw(ArgumentError("observation angles must be finite"))
    return sum(_spheroid_curved_amplitude(field, sol.k, i-1, angle, azimuth;
                   quadrature_extra = sol.data.diagnostics.solver_options.quadrature_extra)
    for (i, field) in enumerate(sol.data.modes))
end
