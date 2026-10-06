# Axisymmetric two-interface fluid MFS. Each represented region has its own
# free-space Green-function basis, with all singularities outside that region.

function _nested_fluid_inner(body::Sphere, ratio)
    Sphere(body.radius * ratio)
end

function _nested_fluid_inner(body::Spheroid, ratio)
    b = body.b * ratio
    Spheroid(sqrt(body.q^2 + b^2), b)
end

function _nested_fluid_sources(outer_mesh, inner_mesh, outer_offset, inner_offset)
    exterior = mfs_source_points(outer_mesh, outer_offset)
    shell_inner = mfs_source_points(inner_mesh, inner_offset)
    shell_outer = mfs_source_points(outer_mesh, -outer_offset)
    core = mfs_source_points(inner_mesh, -inner_offset)
    return (; exterior, shell_inner, shell_outer, core)
end

function _nested_fluid_source_clearance(body::Sphere, rho, z)
    hypot(rho, z) / body.radius - 1
end

function _nested_fluid_source_clearance(body::Spheroid, rho, z)
    hypot(z / body.a, rho / body.b) - 1
end

function _check_nested_fluid_sources(outer, inner, sources)
    for (name, body, expected) in ((:exterior, outer, -1),
        (:shell_inner, inner, -1), (:shell_outer, outer, 1),
        (:core, inner, 1))
        rho, z = getproperty(sources, name)
        all(i -> expected * _nested_fluid_source_clearance(body, rho[i], z[i]) > 1e-10,
            eachindex(rho)) || throw(ArgumentError(
            "nested fluid MFS $name sources must lie outside their represented region"))
    end
    return nothing
end

function _nested_fluid_mfs_system(boundary, k, outer_mesh, inner_mesh, sources,
        incidence_angle, m, rtol)
    shell = boundary.material
    core = boundary.interior
    ks = k / shell.soundspeed_contrast
    kc = k / core.soundspeed_contrast
    op(mesh, wave, source) = assemble_mfs_operators(
        mesh, wave, source[1], source[2]; m, rtol)
    Poe, Voe, po = op(outer_mesh, k, sources.exterior)
    Posi, Vosi, _ = op(outer_mesh, ks, sources.shell_inner)
    Poso, Voso, _ = op(outer_mesh, ks, sources.shell_outer)
    Pisi, Visi, pi = op(inner_mesh, ks, sources.shell_inner)
    Piso, Viso, _ = op(inner_mesh, ks, sources.shell_outer)
    Pic, Vic, _ = op(inner_mesh, kc, sources.core)

    no, ni = length(po), length(pi)
    ne, nsi, nso, nc = map(s -> length(s[1]),
        (sources.exterior, sources.shell_inner, sources.shell_outer, sources.core))
    ie = 1:ne
    isi = (ne + 1):(ne + nsi)
    iso = (ne + nsi + 1):(ne + nsi + nso)
    ic = (ne + nsi + nso + 1):(ne + nsi + nso + nc)
    rop = 1:no
    rov = (no + 1):(2no)
    rip = (2no + 1):(2no + ni)
    riv = (2no + ni + 1):(2no + 2ni)
    A = zeros(ComplexF64, 2(no + ni), ne + nsi + nso + nc)
    gshell = shell.density_contrast
    gcore = core.density_contrast
    A[rop, ie] = Poe
    A[rop, isi] = -Posi
    A[rop, iso] = -Poso
    A[rov, ie] = Voe
    A[rov, isi] = -Vosi / gshell
    A[rov, iso] = -Voso / gshell
    A[rip, isi] = Pisi
    A[rip, iso] = Piso
    A[rip, ic] = -Pic
    A[riv, isi] = Visi / gshell
    A[riv, iso] = Viso / gshell
    A[riv, ic] = -Vic / gcore

    b = zeros(ComplexF64, size(A, 1))
    b[rop] = -ComplexF64[_p_inc_mode(m, k, incidence_angle, p.rhom, p.zm) for p in po]
    b[rov] = -ComplexF64[_dpdn_inc_mode(m, k, incidence_angle,
                             p.rhom, p.zm, p.nrho, p.nz) for p in po]
    return A, b, (; ie, isi, iso, ic, rop, rov, rip, riv, Poe, Voe)
end

"""
    mfs(body::Union{Sphere,Spheroid}, boundary::Shelled{FluidLayer,FluidInterior}, k;
        incidence_angle=0, n=40, oversampling=2, offset_outer=0.2b,
        offset_inner=0.12b, rtol=1e-6)

Axisymmetric two-interface fluid MFS with separate exterior, shell and core
source sets. `Shelled` uses the usual outer-surface-relative fluid contrasts;
the inner spheroid is confocal with equatorial semi-axis `radius_ratio*b`.
Both offsets are in metres and must be positive. `oversampling` refines the
collocation meshes while keeping the source meshes fixed. Held-out interface
pressure and density-scaled normal-derivative RMS mismatches, normalized by
unit incident pressure and by `k` respectively, are in `diagnostics`.
The current implementation accepts axial incidence only.
"""
function mfs(body::Union{Sphere, Spheroid},
        boundary::Shelled{FluidLayer, FluidInterior}, k::Real;
        incidence_angle::Real = 0.0,
        n::Integer = 40, oversampling::Integer = 2,
        offset_outer::Real = 0.2 * (body isa Sphere ? body.radius : body.b),
        offset_inner::Real = 0.12 * (body isa Sphere ? body.radius : body.b),
        rtol::Real = 1e-6,
        condition_limit::Integer = 512)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    body isa Spheroid && body.kind !== :prolate &&
        throw(ArgumentError(
            "nested fluid MFS currently supports prolate spheroids only"))
    iszero(incidence_angle) || throw(ArgumentError(
        "nested fluid MFS currently supports axial incidence only"))
    n >= 8 || throw(ArgumentError("nested fluid MFS requires n ≥ 8"))
    oversampling >= 1 || throw(ArgumentError("oversampling must be at least 1"))
    all(x -> isfinite(x) && x > 0, (offset_outer, offset_inner)) ||
        throw(ArgumentError("nested fluid MFS offsets must be finite and positive"))
    isfinite(rtol) && 0 < rtol < 1 || throw(ArgumentError("rtol must lie in (0,1)"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    inner = _nested_fluid_inner(body, boundary.radius_ratio)
    outer_source_mesh = _axisymmetric_mesh(body, n)
    inner_source_mesh = _axisymmetric_mesh(inner, n)
    outer_mesh = oversampling == 1 ? outer_source_mesh :
                 _axisymmetric_mesh(body, n * oversampling)
    inner_mesh = oversampling == 1 ? inner_source_mesh :
                 _axisymmetric_mesh(inner, n * oversampling)
    sources = _nested_fluid_sources(outer_source_mesh, inner_source_mesh,
        offset_outer, offset_inner)
    _check_nested_fluid_sources(body, inner, sources)
    A, b, blocks = _nested_fluid_mfs_system(boundary, k, outer_mesh, inner_mesh,
        sources, incidence_angle, 0, rtol)
    x = A \ b
    check_outer = _mfs_check_mesh(outer_mesh)
    check_inner = _mfs_check_mesh(inner_mesh)
    Ac, bc, check_blocks = _nested_fluid_mfs_system(boundary, k,
        check_outer, check_inner, sources, incidence_angle, 0, rtol)
    check_rms(rows, scale) = norm(view(Ac, rows, :) * x - view(bc, rows)) /
                             (scale * sqrt(length(rows)))
    interface_residuals = (;
        outer_pressure = check_rms(check_blocks.rop, 1),
        outer_velocity = check_rms(check_blocks.rov, k),
        inner_pressure = check_rms(check_blocks.rip, 1),
        inner_velocity = check_rms(check_blocks.riv, k))
    pressure_residual = max(interface_residuals.outer_pressure,
        interface_residuals.inner_pressure)
    velocity_residual = max(interface_residuals.outer_velocity,
        interface_residuals.inner_velocity)
    reports = _SolveReports()
    _record_solve!(reports, A, x, b; mode = 0, rtol,
        source_count = size(A, 2), collocation_count = length(blocks.rop) +
                                                       length(blocks.rip),
        check_count = length(check_blocks.rop) + length(check_blocks.rip),
        boundary_residual = _linear_residual(Ac, x, bc),
        pressure_residual, velocity_residual, interface_residuals,
        offset_ext = offset_outer, offset_int = offset_inner,
        _mfs_matrix_diagnostics(A; condition_limit)...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n, oversampling, offset_outer, offset_inner,
            incidence_angle, rtol, condition_limit))
    packed(source, coefficients) = (; rho = source[1], z = source[2], coefficients)
    mode = (; exterior = packed(sources.exterior, x[blocks.ie]),
        shell_inner = packed(sources.shell_inner, x[blocks.isi]),
        shell_outer = packed(sources.shell_outer, x[blocks.iso]),
        interior = packed(sources.core, x[blocks.ic]), rtol)
    data = _AxisymmetricSurfaceData(outer_mesh,
        [blocks.Poe * x[blocks.ie]], [blocks.Voe * x[blocks.ie]],
        nothing, nothing, Float64(incidence_angle), report, [mode])
    return MFSSolution(body, boundary, k, data)
end
