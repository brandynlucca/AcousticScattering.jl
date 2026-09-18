# Axisymmetric method of fundamental solutions (MFS/ESM).
# Point sources inside the body, boundary condition collocated on the true surface.

# ∂G/∂n_x at field point x, the field-point-normal counterpart of `_ring_dGdn`'s source-point-normal derivative.
function _ring_dGdn_field(
        k::Real, ρ::Real, z::Real, nρ::Real, nz::Real, ρ2::Real, z2::Real, Δφ::Real)
    r = _ring_distance(ρ, z, ρ2, z2, Δφ)
    r < _RING_DISTANCE_FLOOR && return zero(complex(k)) * zero(r)
    G = cis(k * r) / (4π * r)
    proj = nρ * ((ρ - ρ2) + 2ρ2*sin(Δφ/2)^2) + nz * (z - z2)
    return (im * k - 1 / r) * G * proj / r
end

"Fourier-mode-`m` (unnormalized) azimuthal integral over Δφ ∈ [0,2π) of the field-point-normal ring kernel; see [`_ring_dGdn_field`](@ref)."
function _azimuthal_dGdn_field(
        k::Real, ρ::Real, z::Real, nρ::Real, nz::Real, ρ2::Real, z2::Real;
        m::Integer = 0, rtol::Real = 1e-6, atol::Real = _QUAD_ATOL)
    breaks = collect(_azimuthal_half_breakpoints(m))
    scale = iszero(ρ*ρ2) ? pi : hypot(ρ-ρ2, z-z2)/sqrt(ρ*ρ2)
    while 0 < scale < pi
        push!(breaks, scale)
        scale *= 8
    end
    sort!(unique!(breaks))
    val, error = quadgk(Δφ -> _ring_dGdn_field(k, ρ, z, nρ, nz, ρ2, z2, Δφ) * cos(m * Δφ),
        breaks...; rtol, atol = atol/2, maxevals = _AZIMUTHAL_MAXEVALS)
    error <= max(atol/2, rtol*abs(val)) ||
        throw(ArgumentError("MFS normal-derivative quadrature did not converge"))
    return 2val
end

"""
    mfs_source_points(mesh::MeridianMesh, offset)

Virtual source ring positions for axisymmetric MFS: one source per panel
of `mesh` (see [`panels`](@ref)), placed at each panel's midpoint offset
`offset` [m] *inward* along that panel's own outward normal. At sharp meridian
corners, cap the offset by half the distance to the corner so the source mesh
can resolve the edge field under panel refinement. Negative offsets place
sources outward for interior representations. A corner is a meridian junction
whose adjacent unit normals have dot product below 0.25. Returns `(ρ_s, z_s)`
vectors, same length and ordering as
`panels(mesh)`.
"""
function mfs_source_points(mesh::MeridianMesh, offset::Real)
    ps = panels(mesh)
    corners = [(mesh.rho[i + 1], mesh.z[i + 1])
               for i in 1:(length(ps) - 1)
               if ps[i].nrho*ps[i + 1].nrho + ps[i].nz*ps[i + 1].nz < 0.25]
    offsets = [copysign(
                   minimum((hypot(p.rhom-rho, p.zm-z)/2 for (rho, z) in corners);
                       init = abs(offset)),
                   offset) for p in ps]
    ρ_s = [max(p.rhom - shift * p.nrho, 0.0) for (p, shift) in zip(ps, offsets)]
    z_s = [p.zm - shift * p.nz for (p, shift) in zip(ps, offsets)]
    return ρ_s, z_s
end

"""
    assemble_mfs_operators(mesh, k, ρ_s, z_s; m=0, rtol=1e-6)

Pressure and normal-velocity operators for axisymmetric MFS at Fourier
mode `m`: `P[i,j]` is the (azimuthally-integrated) free-space Green's
function from collocation point `i` (panel midpoint on the true boundary)
to source ring `j` at `(ρ_s[j], z_s[j])`; `V[i,j]` is its normal
derivative at collocation point `i`'s own outward normal (the kernel a
source amplitude vector must be multiplied by to recover the boundary
normal velocity, up to the standard `1/(iωρ)` Euler-equation factor,
folded into the boundary-condition right-hand side instead, see
[`solve_axial`](@ref)`(::Union{Rigid,PressureRelease}, ...)`'s own
`dpdn` convention, which this matches).
"""
function assemble_mfs_operators(mesh::MeridianMesh, k::Real, ρ_s::AbstractVector{<:Real},
        z_s::AbstractVector{<:Real};
        m::Integer = 0, rtol::Real = 1e-6)
    ps = panels(mesh)
    n, ns = length(ps), length(ρ_s)
    P = zeros(ComplexF64, n, ns)
    V = zeros(ComplexF64, n, ns)
    if Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            pi = ps[i]
            for j in 1:ns
                P[i, j] = _azimuthal_G(
                    k, pi.rhom, pi.zm, ρ_s[j], z_s[j]; m = m, rtol = rtol)
                V[i, j] = _azimuthal_dGdn_field(
                    k, pi.rhom, pi.zm, pi.nrho, pi.nz, ρ_s[j], z_s[j]; m = m, rtol = rtol)
            end
        end
    else
        for i in 1:n
            pi = ps[i]
            for j in 1:ns
                P[i, j] = _azimuthal_G(
                    k, pi.rhom, pi.zm, ρ_s[j], z_s[j]; m = m, rtol = rtol)
                V[i, j] = _azimuthal_dGdn_field(
                    k, pi.rhom, pi.zm, pi.nrho, pi.nz, ρ_s[j], z_s[j]; m = m, rtol = rtol)
            end
        end
    end
    return P, V, ps
end

function _mfs_check_mesh(mesh::MeridianMesh)
    ps = panels(mesh)
    rho, z = Float64[], Float64[]
    for p in ps
        push!(rho, p.rho1, p.rhom)
        push!(z, p.z1, p.zm)
    end
    push!(rho, last(ps).rho2)
    push!(z, last(ps).z2)
    return MeridianMesh(rho, z)
end

function _mfs_system(boundary, P, V, P_int, V_int, k, ps, beta, m)
    p_inc = ComplexF64[_p_inc_mode(m, k, beta, p.rhom, p.zm) for p in ps]
    dpdn_inc = ComplexF64[_dpdn_inc_mode(m, k, beta, p.rhom, p.zm, p.nrho, p.nz)
                          for p in ps]
    boundary isa PressureRelease && return P, -p_inc
    boundary isa Rigid && return V, -dpdn_inc
    return [P -P_int; V -V_int ./ boundary.density_contrast], [-p_inc; -dpdn_inc]
end

function _solve_mfs_mode(boundary, k, mesh, beta, m;
        source_mesh = mesh, offset_ext, offset_int = offset_ext, rtol,
        condition_limit::Integer = 512, solve_reports = nothing, source_modes = nothing)
    rho_ext, z_ext = mfs_source_points(source_mesh, offset_ext)
    P, V, ps = assemble_mfs_operators(mesh, k, rho_ext, z_ext; m, rtol)
    P_int = V_int = nothing
    if boundary isa FluidFilled
        rho_int, z_int = mfs_source_points(source_mesh, -offset_int)
        P_int, V_int, _ = assemble_mfs_operators(
            mesh, k / boundary.soundspeed_contrast, rho_int, z_int; m, rtol)
    end
    A, b = _mfs_system(boundary, P, V, P_int, V_int, k, ps, beta, m)
    x = A \ b
    ns = length(rho_ext)
    ext = x[1:ns]
    int = boundary isa FluidFilled ? x[(ns + 1):end] : nothing
    if source_modes !== nothing
        interior = int === nothing ? nothing :
                   (; rho = rho_int, z = z_int, coefficients = int)
        push!(source_modes, (; exterior = (; rho = rho_ext, z = z_ext, coefficients = ext),
            interior, rtol))
    end
    if solve_reports !== nothing
        check_mesh = _mfs_check_mesh(mesh)
        Pc, Vc, checks = assemble_mfs_operators(check_mesh, k, rho_ext, z_ext; m, rtol)
        Pi = Vi = nothing
        if boundary isa FluidFilled
            Pi, Vi, _ = assemble_mfs_operators(
                check_mesh, k / boundary.soundspeed_contrast, rho_int, z_int; m, rtol)
        end
        Ac, bc = _mfs_system(boundary, Pc, Vc, Pi, Vi, k, checks, beta, m)
        boundary_residual = _linear_residual(Ac, x, bc)
        pressure_residual = velocity_residual = nothing
        if boundary isa FluidFilled
            ncheck = length(checks)
            pressure_residual = _linear_residual(Ac[1:ncheck, :], x, bc[1:ncheck])
            velocity_residual = _linear_residual(Ac[(ncheck + 1):end, :], x, bc[(ncheck + 1):end])
        end
        _record_solve!(solve_reports, A, x, b; mode = m, rtol, source_count = size(A, 2),
            collocation_count = length(ps), check_count = length(checks), offset_ext,
            offset_int = boundary isa FluidFilled ? offset_int : nothing,
            boundary_residual, pressure_residual, velocity_residual,
            _mfs_matrix_diagnostics(A; condition_limit)...)
    end
    return P * ext, V * ext, ps,
    int === nothing ? nothing : P_int * int, int === nothing ? nothing : V_int * int
end

"""
    solve_axial_mfs(boundary::Union{Rigid,PressureRelease}, k, mesh; offset, rtol=1e-6)

Axisymmetric MFS solution for a unit-amplitude axial plane wave `e^{ikz}`
scattering off `boundary`, using one source per panel offset `offset` [m]
inward along that panel's own normal (see [`mfs_source_points`](@ref)).
Sources use `source_mesh` (default `mesh`); a coarser source mesh gives a
least-squares system on the collocation mesh `mesh`.
`PressureRelease` collocates `p_scat = -p_inc` directly; `Rigid` collocates
`∂p_scat/∂n = -∂p_inc/∂n`. Returns `(p_scat, dpdn_scat, ps)`, surface
values at the true boundary's panel midpoints, in the same shape
[`far_field`](@ref)/[`target_strength`](@ref) already accept for the
axisymmetric BEM, so the exact same postprocessing applies unchanged.
"""
function solve_axial_mfs(
        boundary::Union{Rigid, PressureRelease}, k::Real, mesh::MeridianMesh;
        offset::Real, rtol::Real = 1e-6, source_mesh = mesh,
        condition_limit::Integer = 512, solve_reports = nothing, source_modes = nothing)
    p_scat, dpdn_scat, ps, _, _ = _solve_mfs_mode(boundary, k, mesh, 0.0, 0;
        offset_ext = offset, rtol, source_mesh, condition_limit, solve_reports, source_modes)
    return p_scat, dpdn_scat, ps
end

"""
    solve_axial_mfs(boundary::FluidFilled, k, mesh::MeridianMesh; offset_ext, offset_int, rtol=1e-6)

MFS transmission solve, direct analogue of
[`solve_axial`](@ref)`(::FluidFilled, ...)`'s two-domain coupled system
(see that method's own docstring for the continuity-condition derivation)
but with each domain represented by fundamental solutions instead of a
CBIE: the exterior scattered field by sources offset `offset_ext` [m]
*inward* from the boundary (regular in the true exterior, exactly
[`mfs_source_points`](@ref)'s usual placement), and the interior
transmitted field by a *second* set of sources offset `offset_int` [m]
*outward* from the boundary (`mfs_source_points(mesh, -offset_int)`, regular everywhere inside the body, by the same "keep the singularity out
of the domain you're representing" logic mirrored across the interface).
The interior sources radiate at the interior wavenumber
`k_int = k / boundary.soundspeed_contrast`.

Unknowns are the two source-amplitude vectors; the same two continuity
equations as the CBIE case (pressure and Euler-relation velocity matching,
scaled by `boundary.density_contrast`) give a square system when `source_mesh=mesh`.
A coarser `source_mesh` gives an overdetermined least-squares system. No incident
field appears on the interior side.

Returns `(p_scat, dpdn_scat, ps, p_int, dpdn_int)`, matching
[`solve_axial`](@ref)`(::FluidFilled, ...)`'s return shape.
"""
function solve_axial_mfs(boundary::FluidFilled, k::Real, mesh::MeridianMesh;
        offset_ext::Real, offset_int::Real, rtol::Real = 1e-6,
        source_mesh = mesh, condition_limit::Integer = 512, solve_reports = nothing, source_modes = nothing)
    return _solve_mfs_mode(boundary, k, mesh, 0.0, 0;
        offset_ext, offset_int, rtol, source_mesh, condition_limit, solve_reports, source_modes)
end

"""
    solve_oblique_mfs(boundary::Union{Rigid,PressureRelease}, k, mesh, incidence_angle; m_max, offset, rtol=1e-6)

Axisymmetric MFS solution for a unit-amplitude plane wave arriving at
`incidence_angle` [rad] from the x-axis (`0` = axial/end-on, matching
[`solve_axial_mfs`](@ref); `π/2` = broadside), decomposed into azimuthal
Fourier modes `m = 0, …, m_max` exactly as [`solve_oblique`](@ref) does
for the axisymmetric BEM, each mode's source amplitudes are independent
(the geometry, and hence the source ring positions, don't depend on `m`,
but the pressure/velocity kernels and the incident field do), so this
reuses the *same* source points across every mode and solves each mode's
otherwise-decoupled system on its own.

Returns `(p_scat_modes, dpdn_scat_modes, ps)` in the same shape
`solve_oblique` already returns, so [`far_field`](@ref)'s bistatic method
applies unchanged.
"""
function solve_oblique_mfs(boundary::Union{Rigid, PressureRelease}, k::Real,
        mesh::MeridianMesh, incidence_angle::Real;
        m_max::Integer, offset::Real, rtol::Real = 1e-6,
        source_mesh = mesh, condition_limit::Integer = 512, solve_reports = nothing, source_modes = nothing)
    β = incidence_angle
    ps = panels(mesh)

    p_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)

    for m in 0:m_max
        p_scat_modes[m + 1], dpdn_scat_modes[m + 1], _, _, _ = _solve_mfs_mode(
            boundary, k, mesh, β, m; offset_ext = offset, rtol,
            source_mesh, condition_limit, solve_reports, source_modes)
    end

    return p_scat_modes, dpdn_scat_modes, ps
end

"""
    solve_oblique_mfs(boundary::FluidFilled, k, mesh::MeridianMesh, incidence_angle; m_max, offset_ext, offset_int, rtol=1e-6)

Direct per-Fourier-mode extension of
[`solve_axial_mfs`](@ref)`(::FluidFilled, ...)` to general oblique
incidence, mirroring how [`solve_oblique_mfs`](@ref)`(::Union{Rigid,
PressureRelease}, ...)` extends its own axial case: the same two
source sets (exterior, offset inward; interior, offset outward, see
`solve_axial_mfs(::FluidFilled, ...)`'s docstring) and the same
pressure/velocity continuity coupling are solved independently at each
mode `m = 0, …, m_max`, with only the incident-field right-hand side
(`_p_inc_mode`/`_dpdn_inc_mode`) changing between modes, the ring-kernel
azimuthal reduction already decouples modes exactly as it does for every
other oblique solver in this package. The coupled system has two rows per
collocation point and two coefficients per source-mesh panel. It is square
when `source_mesh=mesh` and uses least squares when the collocation mesh is finer.

Returns `(p_scat_modes, dpdn_scat_modes, ps)`, matching
[`solve_oblique`](@ref)`(::FluidFilled, ...)`'s shape.
"""
function solve_oblique_mfs(
        boundary::FluidFilled, k::Real, mesh::MeridianMesh, incidence_angle::Real;
        m_max::Integer, offset_ext::Real, offset_int::Real, rtol::Real = 1e-6,
        source_mesh = mesh, condition_limit::Integer = 512, solve_reports = nothing, source_modes = nothing)
    β = incidence_angle
    ps = panels(mesh)

    p_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)

    for m in 0:m_max
        p_scat_modes[m + 1], dpdn_scat_modes[m + 1], _, _, _ = _solve_mfs_mode(
            boundary, k, mesh, β, m; offset_ext, offset_int, rtol,
            source_mesh, condition_limit, solve_reports, source_modes)
    end

    return p_scat_modes, dpdn_scat_modes, ps
end

function _mfs_source_amplitude(k, modes, theta, phi)
    value = zero(ComplexF64)
    for (i, mode) in enumerate(modes)
        m = i - 1
        sources = mode.exterior
        for j in eachindex(sources.coefficients)
            value += sources.coefficients[j] * (-im)^m / 2 * cos(m * phi) *
                     besselj(m, k * sources.rho[j] * sin(theta)) *
                     cis(-k * sources.z[j] * cos(theta))
        end
    end
    return value
end
