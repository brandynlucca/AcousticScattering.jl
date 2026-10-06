function scattering_amplitude(sol::FreeSurfaceSolution;
        angle::Real = π - sol.incidence_angle, azimuth::Real = sol.incidence_azimuth + π)
    return scattering_amplitude(sol.direct; angle, azimuth) +
           sol.sign * scattering_amplitude(sol.reflected; angle, azimuth)
end
