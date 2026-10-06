# --- `bem`: axisymmetric and full 3D boundary element method ---------------

"""
    bem(body::AbstractBody, boundary::AbstractBoundaryCondition, k; method=:axisymmetric, incidence_angle=π/2, kwargs...)

Boundary-element solve, returns a [`BEMSolution`](@ref). `method=:axisymmetric` supports
`Sphere`/`Spheroid`/straight `Cylinder`. `method=:full` supports a generated
`Sphere`/`Spheroid`/`Cylinder` or a supplied `Mesh`, and is required for a bent `Cylinder`.
Post-process with [`target_strength`](@ref)`(sol; angle, azimuth)` (axisymmetric) or
[`target_strength`](@ref)`(sol; direction)` (full 3D).

Full BEM accepts `meshsize` in m, geometry `mesh_order` (1, 2 or 3, default 2), quadrature
`qorder` (default 4), `correction`, `compression` and `gmres_kwargs`. Rigid/soft full BEM
defaults to `formulation=:burton_miller` (`:cbie` selects the conventional equation). Fluid
full BEM defaults to `formulation=:muller` (`:cbie` selects the pressure-only system
with interior traces eliminated).
`equilibrate=true` scales the matrix before factorization. `condition_limit=512` bounds the
optional SVD condition-number calculation.
Single-interface fluid Müller also accepts `compression=(method=:hmatrix, tol=1e-8)`
with density interpolation, using block-preconditioned GMRES. It scales from local blocks,
skips SVD condition estimates and accepts `gmres_kwargs`; dense LU remains the default.

See [BEM and MFS](@ref boundary-theory) for the underlying formulations and
[`diagnostics`](@ref) for convergence and residual checks.
"""
function bem(body::Union{Sphere, Spheroid, Cylinder},
        boundary::AbstractBoundaryCondition, k::Real;
        method::Symbol = :axisymmetric, incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)), kwargs...)
    if method === :full
        return _bem_full(body, boundary, k; incidence_angle = incidence_angle,
            incidence_azimuth = incidence_azimuth, kwargs...)
    end
    method === :axisymmetric ||
        throw(ArgumentError("bem(...) supports method=:axisymmetric or :full, got $method"))
    body isa Cylinder && _isbent(body) &&
        throw(ArgumentError(
            "a bent Cylinder requires bem(...; method=:full)"))
    mesh = _axisymmetric_mesh(body, n)
    iszero(incidence_angle) && return _bem_axial(body, boundary, k, mesh; kwargs...)
    return _bem_oblique(body, boundary, k, mesh, incidence_angle; m_max = m_max, kwargs...)
end

"""
    bem(body::Sphere, boundary::Shelled{FluidLayer}, k; n=default, kwargs...)

Two-surface (outer + inner mesh) fluid-shell BEM, `boundary`'s own
`radius_ratio` field gives the inner surface's radius, `body.radius *
boundary.radius_ratio`. This two-surface formulation supports axial incidence only.
"""
function bem(body::Sphere,
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}}, k::Real;
        n::Integer = _axisymmetric_default_panels(body, k), kwargs...)
    mesh_outer = sphere_mesh(body.radius, n)
    mesh_inner = sphere_mesh(body.radius * boundary.radius_ratio, n)
    reports = _SolveReports()
    p_scat, dpdn_scat, _ = solve_axial(boundary, k, mesh_outer, mesh_inner;
        solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric, solver_options = (;
        n, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh_outer, [p_scat], [dpdn_scat], nothing, nothing, 0.0, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_axial(body::AbstractBody, boundary::Union{Rigid, PressureRelease, Impedance},
        k::Real, mesh::MeridianMesh; kwargs...)
    reports = _SolveReports()
    p_scat, dpdn_scat, _ = solve_axial(
        boundary, k, mesh; solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(mesh), incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], nothing, nothing, 0.0, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_axial(
        body::AbstractBody, boundary::FluidFilled, k::Real, mesh::MeridianMesh; kwargs...)
    reports = _SolveReports()
    p_scat, dpdn_scat, _, p_int, dpdn_int = solve_axial(
        boundary, k, mesh; solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(mesh), incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], [p_int], [dpdn_int], 0.0, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_oblique(
        body::AbstractBody, boundary::Union{Rigid, PressureRelease, FluidFilled, Impedance},
        k::Real, mesh::MeridianMesh,
        incidence_angle::Real; m_max::Integer, kwargs...)
    reports = _SolveReports()
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique(
        boundary, k, mesh, incidence_angle; m_max, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(mesh), incidence_angle, m_max, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, p_scat_modes, dpdn_scat_modes, nothing, nothing, incidence_angle, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_full(body::Union{Sphere, Spheroid},
        boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        incident = nothing,
        meshsize::Real = bem3d_elements_per_wavelength(k), qorder::Integer = 4,
        mesh_order::Integer = 2, kwargs...)
    incident = _resolve_incident(
        k, incidence_angle, incidence_azimuth; incident)
    quad = body isa Sphere ?
           gmsh_sphere_mesh(body.radius; meshsize, qorder, mesh_order) :
           gmsh_spheroid_mesh(body.a, body.b; meshsize, qorder, mesh_order)
    density = Ref{Union{Nothing, Vector{ComplexF64}}}(nothing)
    capture = boundary isa Rigid ? (; _density = density) : (;)
    p_scat, dpdn_scat, _, diagnostics = solve_full_bem(
        boundary, k, quad; incidence_angle = incidence_angle,
        incidence_azimuth = incidence_azimuth, incident, return_diagnostics = true, capture..., kwargs...)
    diagnostics = merge(diagnostics,
        (;
            meshsize = Float64(meshsize), quadrature_order = qorder, mesh_order,
            illumination = incident === nothing ? :plane_wave : :prescribed))
    data = _FullBEMSurfaceData(quad, p_scat, dpdn_scat, incidence_angle, incidence_azimuth,
        diagnostics, density[], incident)
    return BEMSolution(body, boundary, Float64(k), :full, data)
end
function _bem_full(
        body::Cylinder, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        meshsize::Real = bem3d_elements_per_wavelength(k), qorder::Integer = 4,
        mesh_order::Integer = 2, kwargs...)
    surface = _cylinder_full_mesh(body; meshsize, qorder, mesh_order)
    solution = bem(surface, boundary, k; kwargs...)
    d = solution.data
    report = merge(d.diagnostics, (;
        meshsize = Float64(meshsize), mesh_order, quadrature_order = qorder))
    data = _FullBEMSurfaceData(d.quad, d.p_scat, d.dpdn_scat,
        d.incidence_angle, d.incidence_azimuth, report, d.single_layer_density, d.incident)
    return BEMSolution(body, boundary, Float64(k), :full, data)
end
