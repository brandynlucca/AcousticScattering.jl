
function scattering_amplitude(
        sol::FMSolution; angle::Real = π - sol.incidence_angle, azimuth::Real = π)
    return fourier_matching_amplitude(sol.b, sol.k, angle, azimuth)
end

function diagnostics(sol::FMSolution)
    convergence = _fm_convergence(sol.b, sol.b_check, sol.k)
    (; admissible = is_admissible(sol.mapping),
        m_max = size(sol.b, 2) - 1, n_max = size(sol.b, 1) - 1, convergence,
        convergence_tolerance = _FM_CONVERGENCE_TOLERANCE,
        converged = isnan(convergence) ? nothing : convergence <= _FM_CONVERGENCE_TOLERANCE)
end
