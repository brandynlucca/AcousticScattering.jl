function scattering_amplitude(sol::FEMSolution{_RadialFEMData}; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "Radial sphere FEM supports backscatter without observation keywords."))
    return _radial_fem_amplitude(sol.data.modes, sol.k)
end

function scattering_amplitude(sol::FEMSolution{_SphereMeridianFEMData}; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "Sphere meridian FEM supports backscatter without observation keywords."))
    total = sum(Bl * (-im)^(l-1) * legendre_p(l-1, -1.0)
    for (l, Bl) in enumerate(sol.data.outgoing))
    return -im / sol.k * total
end

function target_strength(sol::FEMSolution{_ScalarFEMData}; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "target_strength(::FEMSolution) from a :radial/:meridian solve takes no keywords: the " *
        "observation angle was already fixed when fem(...) was called. Call fem(...) again with " *
        "a different `incidence_angle` to get a different result. (A `Shell`-body fem(...) result " *
        "does support `angle`/`azimuth` here — this method is for the scalar-only :radial/:meridian solvers.)"))
    return sol.data.ts
end
function scattering_amplitude(sol::FEMSolution{_ScalarFEMData}; kwargs...)
    throw(ArgumentError(
        "scattering_amplitude is not available for this FEMSolution: this FEM path " *
        "retains only target strength. Use target_strength(sol) " *
        "instead, or use modal(...)/kirchhoff(...) for this body/boundary combination if you need " *
        "the complex amplitude."))
end

function scattering_amplitude(sol::FEMSolution{_VolumeFEMData};
        angle::Real = π - sol.data.incidence_angle,
        azimuth::Real = sol.data.incidence_azimuth + π)
    return _volume_amplitude(sol.data, sol.k; angle, azimuth)
end

function scattering_amplitude(sol::FEMSolution{_ShellFEMSurfaceData};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π)
    d = sol.data
    length(d.p_ext_modes) == 1 &&
        return far_field(d.ps_ext, d.p_ext_modes[1], d.dpdn_ext_modes[1], sol.k, angle)
    return far_field(d.ps_ext, d.p_ext_modes, d.dpdn_ext_modes, sol.k, angle, azimuth)
end

function scattering_amplitude(sol::FEMSolution{T};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π) where {
        T <: Union{_CylinderMeridianFEMData, _SpheroidMeridianFEMData}}
    d = sol.data
    length(d.p_modes) == 1 &&
        return far_field(d.ps, d.p_modes[1], d.dpdn_modes[1], sol.k, angle)
    return far_field(d.ps, d.p_modes, d.dpdn_modes, sol.k, angle, azimuth)
end

function scattering_amplitude(sol::FEMSolution{_CylinderRadialFEMData})
    d = sol.data
    return _elastic_cylinder_radial_fem_amplitude(d.modes, sol.k, d.length, d.aspect_angle)
end
