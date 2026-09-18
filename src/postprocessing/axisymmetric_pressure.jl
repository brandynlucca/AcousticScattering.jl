function _pressure_integral(f, breaks; atol = 1e-11)
    value, error = quadgk(f, breaks...; rtol = 1e-7, atol, maxevals = 20000)
    error <= max(atol, 1e-7abs(value)) ||
        throw(ArgumentError("pressure quadrature did not converge; refine the meridian mesh"))
    return value
end

function _pressure_breakpoints(center, scale, lower, upper)
    values = [lower, upper, center]
    if scale > 0
        step = scale
        while step < upper-lower
            lower < center-step < upper && push!(values, center-step)
            lower < center+step < upper && push!(values, center+step)
            step *= 8
        end
    end
    return sort!(unique!(values))
end

function _pressure_kernel_remainder(u)
    z = im*u
    abs(u) >= 0.01 && return cis(u)*(1-z)-1
    term, value = z*z/2, -z*z/2
    for n in 3:8
        term *= z/n
        value -= (n-1)*term
    end
    return value
end

function _pressure_ring(k, rho, axial, panel, s, m, p, dp, local_pressure)
    source_rho, source_axial = _panel_point(panel, s)
    separation = hypot(rho-source_rho, axial-source_axial)
    scale = iszero(rho*source_rho) ? pi : separation/sqrt(rho*source_rho)
    breaks = _pressure_breakpoints(0.0, scale, 0.0, Float64(pi))
    span = iszero(rho*source_rho) ? 0.0 :
           4rho*source_rho /
           (hypot(rho+source_rho, axial-source_axial)+separation)
    intervals = max(1, 2m, ceil(Int, 4k*span/pi))
    append!(breaks, range(0.0, pi; length = intervals+1))
    sort!(unique!(breaks))
    projection = panel.nrho*(rho-panel.rhom)+panel.nz*(axial-panel.zm)
    value = _pressure_integral(breaks; atol = 1e-12) do phi
        radius = _ring_distance(rho, axial, source_rho, source_axial, phi)
        normal = projection-2panel.nrho*rho*sin(phi/2)^2
        cosine = cos(m*phi)
        difference = (p-local_pressure)+p*(_pressure_kernel_remainder(k*radius)*cosine-2sin(m*phi/2)^2)
        difference*normal/(4pi*radius^3)-dp*cis(k*radius)*cosine/(4pi*radius)
    end
    return 2value*source_rho*panel.L
end

"""
    _axisymmetric_pressure(solution, point, region)

Evaluate the retained Fourier Cauchy traces on the revolved meridian. Subtract the
nearest panel's constant pressure from the Laplace double layer and restore its
closed-surface identity analytically. This removes the local double-layer peak;
distance-scaled integration intervals resolve the remaining weak singularity.
At a shared endpoint, average the equally near panel traces to select the angular
bisector limit of the piecewise-constant density. The selected region specifies
the one-sided trace at the represented boundary.
"""
function _axisymmetric_pressure(solution, point, region)
    data = solution.data
    ps = panels(data.mesh)
    rho, axial, phi = hypot(point[2], point[3]), point[1], atan(point[3], point[2])
    parameters = [_closest_param(rho, axial, panel) for panel in ps]
    distances = [hypot(rho-_panel_point(panel, s)[1], axial-_panel_point(panel, s)[2])
                 for (panel, s) in zip(ps, parameters)]
    nearest = findall(==(minimum(distances)), distances)
    inside = region === :interior
    k = inside ? solution.k/solution.boundary.soundspeed_contrast : solution.k
    value = zero(ComplexF64)
    for i in eachindex(data.p_scat_modes)
        m = i-1
        iszero(rho) && m > 0 && continue
        p, dp = data.p_scat_modes[i], data.dpdn_scat_modes[i]
        if inside
            p = p + [_p_inc_mode(m, solution.k, data.incidence_angle, panel.rhom, panel.zm)
                 for panel in ps]
            dp = solution.boundary.density_contrast .* (dp +
                  [_dpdn_inc_mode(m, solution.k, data.incidence_angle,
                       panel.rhom, panel.zm, panel.nrho, panel.nz) for panel in ps])
        end
        local_pressure = sum(p[j] for j in nearest)/length(nearest)
        mode = inside ? -local_pressure : zero(ComplexF64)
        for (j, panel) in enumerate(ps)
            breaks = _pressure_breakpoints(parameters[j], distances[j]/panel.L, 0.0, 1.0)
            mode += _pressure_integral(breaks) do s
                _pressure_ring(k, rho, axial, panel, s, m, p[j], dp[j], local_pressure)
            end
        end
        value += (inside ? -mode : mode)*cos(m*phi)
    end
    return value
end
