# --- `fem` on a `Shell` body: thin and general shell-fluid coupling -----

function _prolate_shell_geometry(s::Shell)
    s.body isa Spheroid ||
        throw(ArgumentError("this shell method requires a Spheroid-based Shell (got $(typeof(s.body)))"))
    return ProlateShellGeometry(s.body.a, s.body.b, s.thickness)
end

"""
    fem(shell::Shell, boundary::Shelled, ext_density, ext_soundspeed, int_density, int_soundspeed, k;
        method=:general, incidence_angle=π/2, kwargs...)

Elastic-shell/fluid coupling, returns a [`FEMSolution`](@ref). Post-process with
[`target_strength`](@ref)`(sol; angle, azimuth)`. `boundary` is built via
[`Shelled`](@ref)`(poisson, density, youngs_modulus)`. `ext_density`/`ext_soundspeed` and
`int_density`/`int_soundspeed` are the absolute density [kg/m³] and sound speed in m/s of the
exterior and interior fluids, not contrasts.

`method=:thin` is an axisymmetric reduction, `shell.body isa Spheroid` only, axial incidence
only (`incidence_angle = 0.0`). Pass `int_density=0` for a vacuum-backed shell.

`method=:general` (default) is a full through-thickness 2D shell FEM, general incidence,
`shell.body isa Union{Sphere,Spheroid}`, always fluid-filled. Use a small `int_density`/
`int_soundspeed` for a near-vacuum limit.

See [FEM and shell coupling](@ref fem-theory) for the shell theory.
"""
function fem(s::Shell, boundary::Shelled{ElasticFEMLayer, Nothing},
        ext_density::Real, ext_soundspeed::Real,
        int_density::Real, int_soundspeed::Real, k::Real;
        method::Symbol = :general, incidence_angle::Real = π / 2,
        m_max::Integer = _default_mode_count(k * _characteristic_radius(s.body)),
        n_eta::Integer = 65, n_t::Integer = 3,
        pole_offset::Real = method === :thin ? 1e-4 : 1e-3, kwargs...)
    material = boundary.material
    freq_hz = k * ext_soundspeed / (2π)
    reports = _SolveReports()
    options = (; incidence_angle, m_max = method === :thin ? 0 : m_max,
        n_eta, n_t, pole_offset, kwargs...)
    if method === :thin
        iszero(incidence_angle) || throw(ArgumentError(
            "fem(::Shell, ...; method=:thin) only supports axial incidence (the axisymmetric " *
            "reduction retains m=0), use method=:general for oblique incidence"))
        geometry = _prolate_shell_geometry(s)
        if iszero(int_density)
            p_scat, dpdn_scat, ps, shell_state, _ = solve_shell_fluid_coupled(
                geometry, material, ext_density, ext_soundspeed,
                freq_hz; n_eta, pole_offset, solve_reports = reports, kwargs...)
            report = _summarize_solves(reports; method, solver_options = options)
            data = _ShellFEMSurfaceData(
                ps, [p_scat], [dpdn_scat], nothing, nothing,
                nothing, nothing, shell_state, incidence_angle, report)
            return FEMSolution(s, boundary, k, method, data)
        end
        p_ext, dpdn_ext, ps_ext, p_int, dpdn_int, ps_int,
        shell_state, _ = solve_shell_fluid_filled_coupled(
            geometry, material, ext_density, ext_soundspeed, int_density,
            int_soundspeed, freq_hz; n_eta, pole_offset, solve_reports = reports, kwargs...)
        report = _summarize_solves(reports; method, solver_options = options)
        data = _ShellFEMSurfaceData(
            ps_ext, [p_ext], [dpdn_ext], ps_int, [p_int],
            [dpdn_int], Float64(k * ext_soundspeed / int_soundspeed),
            shell_state, incidence_angle, report)
        return FEMSolution(s, boundary, k, method, data)
    end
    method === :general ||
        throw(ArgumentError("fem(::Shell, ...) supports method=:thin or :general, got $method"))
    int_density > 0 || throw(ArgumentError(
        "fem(::Shell, ...; method=:general) requires positive interior density"))
    p_scat_modes, dpdn_scat_modes, ps, p_int_modes, dpdn_int_modes, ps_int = if s.body isa
                                                                                Spheroid
        solve_general_shell_fluid_filled_coupled(
            _prolate_shell_geometry(s), material.density,
            material.youngs_modulus, material.poisson,
            ext_density, ext_soundspeed, int_density, int_soundspeed, freq_hz,
            incidence_angle; m_max = m_max, n_eta = n_eta, n_t = n_t,
            pole_offset = pole_offset, solve_reports = reports,
            retain_interior = true, kwargs...)
    else
        mesh = build_structured_spherical_shell(s.body.radius, s.thickness, n_eta, n_t;
            pole_offset = pole_offset)
        solve_general_shell_fluid_filled_coupled(
            mesh, material.density, material.youngs_modulus, material.poisson,
            ext_density, ext_soundspeed, int_density, int_soundspeed,
            freq_hz, incidence_angle; m_max = m_max, solve_reports = reports,
            retain_interior = true, kwargs...)
    end
    report = _summarize_solves(reports; method, solver_options = options)
    data = _ShellFEMSurfaceData(
        ps, p_scat_modes, dpdn_scat_modes, ps_int, p_int_modes,
        dpdn_int_modes, Float64(k * ext_soundspeed / int_soundspeed),
        nothing, incidence_angle, report)
    return FEMSolution(s, boundary, k, method, data)
end
