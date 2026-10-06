# --- `mfs`: axisymmetric and bent-cylinder method of fundamental solutions --

function _reject_rigid_flat_cylinder_mfs(body, boundary)
    if body isa Cylinder && !_iscapped(body) && boundary isa Rigid
        throw(ArgumentError(
            "mfs: axisymmetric rigid flat-cylinder rims are unsupported; " *
            "use bem(body, Rigid(), k; ...) or a physically capped Cylinder"))
    end
    return nothing
end

"""
    mfs(body::AbstractBody, boundary::AbstractBoundaryCondition, k; incidence_angle=π/2, offset=0.3*characteristic_radius, kwargs...)

Method-of-fundamental-solutions solve, returns an [`MFSSolution`](@ref). Axisymmetric for
`Sphere`/`Spheroid`/straight `Cylinder`. Lateral-only 3D point sources for a bent `Cylinder`
(omits end caps). Use `mfs(mesh(body; method=:full, ...), ...)` for closed-surface conditions
and general observation directions. `offset` is a maximum source displacement in m from the
surface, reduced near flat-cap rims.

Axisymmetric MFS does not support a rigid straight cylinder with flat caps: the
normal-offset source basis fails independent rim boundary checks. Use `bem` for
that sharp-rim geometry, or a `Cylinder` with physically justified smooth caps.

For bent cylinders, `n_s`/`n_phi` (defaults 40/32, at least 3) set the axial and azimuthal
source-grid counts. `oversampling` (integer, default 1) multiplies the collocation budget while
keeping the source grid fixed. Values above 1 give a least-squares solve.

See [BEM and MFS](@ref boundary-theory) for source placement guidance and
[`diagnostics`](@ref) for residuals, conditioning and rank (`condition_limit=512` bounds the
SVD size, 0 skips it).
"""
function mfs(body::Union{Sphere, Spheroid, Cylinder},
        boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, offset::Real = 0.3_characteristic_radius(body),
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)),
        oversampling::Integer = 1, condition_limit::Integer = 512, kwargs...)
    oversampling >= 1 || throw(ArgumentError("mfs: oversampling must be at least 1"))
    condition_limit >= 0 || throw(ArgumentError("mfs: condition_limit must be nonnegative"))
    if body isa Cylinder && _isbent(body)
        return _mfs_bent(body, boundary, k; incidence_angle = incidence_angle,
            offset = offset, oversampling = oversampling, condition_limit, kwargs...)
    end
    _reject_rigid_flat_cylinder_mfs(body, boundary)
    mesh, source_mesh = _mfs_meridian_meshes(body, n, oversampling)
    iszero(incidence_angle) &&
        return _mfs_axial(
            body, boundary, k, mesh; offset, source_mesh,
            oversampling, condition_limit, kwargs...)
    return _mfs_oblique(
        body, boundary, k, mesh, incidence_angle; offset,
        m_max, source_mesh, oversampling, condition_limit, kwargs...)
end

function _mfs_meridian_meshes(body, n, oversampling)
    source_mesh = (body isa Cylinder && _iscapped(body)) ?
                  cylinder_spheroidal_endcap_mesh(body.radius, body.length, body.endcap_depth, n) :
                  _axisymmetric_mesh(body, n)
    mesh = oversampling == 1 ? source_mesh :
           (body isa Cylinder && _iscapped(body)) ?
           cylinder_spheroidal_endcap_mesh(body.radius, body.length, body.endcap_depth, oversampling *
                                                                                        n) :
           _axisymmetric_mesh(body, oversampling * n)
    return mesh, source_mesh
end

function _mfs_axial(body::AbstractBody, boundary::Union{Rigid, PressureRelease, Impedance},
        k::Real, mesh::MeridianMesh; offset::Real, source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat, dpdn_scat, _ = solve_axial_mfs(boundary, k, mesh;
        offset, source_mesh, source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling,
            offset, incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], nothing, nothing, 0.0, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_axial(
        body::AbstractBody, boundary::FluidFilled, k::Real, mesh::MeridianMesh; offset::Real,
        offset_ext::Real = offset, offset_int::Real = offset,
        source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat, dpdn_scat, _, p_int, dpdn_int = solve_axial_mfs(
        boundary, k, mesh; offset_ext, offset_int, source_mesh,
        source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling, offset_ext, offset_int,
            incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], [p_int], [dpdn_int], 0.0, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_oblique(
        body::AbstractBody, boundary::Union{Rigid, PressureRelease, Impedance},
        k::Real, mesh::MeridianMesh, incidence_angle::Real;
        offset::Real, m_max::Integer, source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique_mfs(
        boundary, k, mesh, incidence_angle; m_max, offset,
        source_mesh, source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling,
            offset, incidence_angle, m_max, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, p_scat_modes, dpdn_scat_modes, nothing,
        nothing, incidence_angle, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_oblique(body::AbstractBody, boundary::FluidFilled, k::Real,
        mesh::MeridianMesh, incidence_angle::Real;
        offset::Real, m_max::Integer, offset_ext::Real = offset, offset_int::Real = offset,
        source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique_mfs(
        boundary, k, mesh, incidence_angle;
        m_max, offset_ext, offset_int, source_mesh,
        source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling, offset_ext, offset_int,
            incidence_angle, m_max, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, p_scat_modes, dpdn_scat_modes, nothing,
        nothing, incidence_angle, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_bent(body::Cylinder, boundary::Union{Rigid, PressureRelease}, k::Real;
        incidence_angle::Real = π / 2, offset::Real, n_s::Integer = 40,
        n_phi::Union{Nothing, Integer} = nothing, n_φ::Union{Nothing, Integer} = nothing,
        oversampling::Integer = 1, kwargs...)
    n_phi !== nothing && n_φ !== nothing &&
        throw(ArgumentError("mfs: supply only n_phi, not both n_phi and the legacy n_φ"))
    azimuth_count = something(n_phi, n_φ, 32)
    n_s >= 3 || throw(ArgumentError("mfs: n_s must be at least 3"))
    azimuth_count >= 3 || throw(ArgumentError("mfs: n_phi must be at least 3"))
    reports = _SolveReports()
    p_scat, dpdn_scat, points, normals, areas = solve_bent_cylinder_mfs(
        boundary, k, body.radius, body.length, body.radius_curvature;
        aspect_angle = incidence_angle, offset = offset, n_s = n_s, n_φ = azimuth_count,
        oversampling, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :bent,
        solver_options = (;
            n_s, n_phi = azimuth_count, offset, incidence_angle, oversampling, kwargs...))
    data = _BentMFSSurfaceData(
        p_scat, dpdn_scat, points, normals, areas, incidence_angle, report)
    return MFSSolution(body, boundary, k, data)
end
