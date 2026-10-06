function scattering_amplitude(sol::MFSSolution{_BentMFSSurfaceData})
    d = sol.data
    β = d.incidence_angle
    q̂ = (-cos(β), -sin(β), 0.0)
    f = zero(ComplexF64)
    for i in eachindex(d.points)
        f += (im * sol.k * _dot3(q̂, d.normals[i]) * d.p_scat[i] + d.dpdn_scat[i]) *
             cis(-sol.k * _dot3(q̂, d.points[i])) * d.areas[i]
    end
    return f / (4π)
end

function scattering_amplitude(sol::MFSSolution{_FullMFSSurfaceData};
        direction::Union{Nothing, AbstractVector} = nothing)
    d = sol.data
    return _point_source_amplitude(sol.k, d.coefficients, d.sources,
        d.incidence_angle, d.incidence_azimuth, direction)
end

function scattering_amplitude(sol::MFSSolution{_BladderBackboneMFSData};
        direction::Union{Nothing, AbstractVector} = nothing)
    d = sol.data
    return _point_source_amplitude(sol.k, d.exterior_coefficients, d.exterior_sources,
        d.incidence_angle, d.incidence_azimuth, direction)
end
