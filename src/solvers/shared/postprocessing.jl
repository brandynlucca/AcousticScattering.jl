function _axisymmetric_amplitude(k::Real, d::_AxisymmetricSurfaceData; angle::Real, azimuth::Real)
    d.source_modes === nothing ||
        return _mfs_source_amplitude(k, d.source_modes, angle, azimuth)
    ps = panels(d.mesh)
    length(d.p_scat_modes) == 1 &&
        return far_field(ps, d.p_scat_modes[1], d.dpdn_scat_modes[1], k, angle)
    return far_field(ps, d.p_scat_modes, d.dpdn_scat_modes, k, angle, azimuth)
end

function scattering_amplitude(
        sol::Union{
            BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData}};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π)
    _axisymmetric_amplitude(sol.k, sol.data; angle = angle, azimuth = azimuth)
end

diagnostics(sol::Union{BEMSolution, MFSSolution, FEMSolution}) = sol.data.diagnostics
