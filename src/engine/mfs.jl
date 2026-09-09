# Axisymmetric method-of-fundamental-solutions (MFS/ESM), Pérez-Arjona, Godinho & Espinosa (2018).
# Point sources inside the body, boundary condition collocated on the true surface.

# ∂G/∂n_x at field point x, the field-point-normal counterpart of `_ring_dGdn`'s source-point-normal derivative.
function _ring_dGdn_field(
        k::Real, ρ::Real, z::Real, nρ::Real, nz::Real, ρ2::Real, z2::Real, Δφ::Real)
    r = _ring_distance(ρ, z, ρ2, z2, Δφ)
    r < _RING_DISTANCE_FLOOR && return zero(complex(k)) * zero(r)
    G = cis(k * r) / (4π * r)
    proj = nρ * (ρ - ρ2 * cos(Δφ)) + nz * (z - z2)
    return (im * k - 1 / r) * G * proj / r
end

"Fourier-mode-`m` (unnormalized) azimuthal integral over Δφ ∈ [0,2π) of the field-point-normal ring kernel; see [`_ring_dGdn_field`](@ref)."
function _azimuthal_dGdn_field(
        k::Real, ρ::Real, z::Real, nρ::Real, nz::Real, ρ2::Real, z2::Real;
        m::Integer = 0, rtol::Real = 1e-6, atol::Real = _QUAD_ATOL)
    val, _ = quadgk(Δφ -> _ring_dGdn_field(k, ρ, z, nρ, nz, ρ2, z2, Δφ) * cos(m * Δφ),
        _azimuthal_breakpoints(m)...; rtol = rtol,
        atol = atol, maxevals = _AZIMUTHAL_MAXEVALS)
    return val
end

"""
    mfs_source_points(mesh::MeridianMesh, offset)

Virtual source ring positions for axisymmetric MFS: one source per panel
of `mesh` (see [`panels`](@ref)), placed at each panel's midpoint offset
`offset` [m] *inward* along that panel's own outward normal, so every
source sits strictly inside the true boundary regardless of local
curvature. Returns `(ρ_s, z_s)` vectors, same length and ordering as
`panels(mesh)`.
"""
function mfs_source_points(mesh::MeridianMesh, offset::Real)
    ps = panels(mesh)
    ρ_s = [max(p.rhom - offset * p.nrho, 0.0) for p in ps]
    z_s = [p.zm - offset * p.nz for p in ps]
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

"""
    solve_axial_mfs(boundary::Union{Rigid,PressureRelease}, k, mesh; offset, rtol=1e-6)

Axisymmetric MFS solution for a unit-amplitude axial plane wave `e^{ikz}`
scattering off `boundary`, using one source per panel offset `offset` [m]
inward along that panel's own normal (see [`mfs_source_points`](@ref)).
`PressureRelease` collocates `p_scat = -p_inc` directly; `Rigid` collocates
`∂p_scat/∂n = -∂p_inc/∂n`. Returns `(p_scat, dpdn_scat, ps)`, surface
values at the true boundary's panel midpoints, in the same shape
[`far_field`](@ref)/[`target_strength`](@ref) already accept for the
axisymmetric BEM, so the exact same postprocessing applies unchanged.
"""
function solve_axial_mfs(
        boundary::Union{Rigid, PressureRelease}, k::Real, mesh::MeridianMesh;
        offset::Real, rtol::Real = 1e-6)
    ρ_s, z_s = mfs_source_points(mesh, offset)
    P, V, ps = assemble_mfs_operators(mesh, k, ρ_s, z_s; m = 0, rtol = rtol)

    p_inc = ComplexF64[_p_inc_mode(0, k, 0.0, p.rhom, p.zm) for p in ps]
    dpdn_inc = ComplexF64[_dpdn_inc_mode(0, k, 0.0, p.rhom, p.zm, p.nrho, p.nz) for p in ps]

    if boundary isa PressureRelease
        A = P \ (-p_inc)
    else
        A = V \ (-dpdn_inc)
    end
    p_scat = P * A
    dpdn_scat = V * A
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
scaled by `boundary.density_contrast`) give a square system, no incident
field appearing on the interior side.

Returns `(p_scat, dpdn_scat, ps, p_int, dpdn_int)`, matching
[`solve_axial`](@ref)`(::FluidFilled, ...)`'s return shape.
"""
function solve_axial_mfs(boundary::FluidFilled, k::Real, mesh::MeridianMesh;
        offset_ext::Real, offset_int::Real, rtol::Real = 1e-6)
    g = boundary.density_contrast
    k_int = k / boundary.soundspeed_contrast

    ρ_s_ext, z_s_ext = mfs_source_points(mesh, offset_ext)
    ρ_s_int, z_s_int = mfs_source_points(mesh, -offset_int)

    P_ext, V_ext, ps = assemble_mfs_operators(mesh, k, ρ_s_ext, z_s_ext; m = 0, rtol = rtol)
    P_int, V_int, _ = assemble_mfs_operators(
        mesh, k_int, ρ_s_int, z_s_int; m = 0, rtol = rtol)
    n = length(ps)

    p_inc = ComplexF64[_p_inc_mode(0, k, 0.0, p.rhom, p.zm) for p in ps]
    dpdn_inc = ComplexF64[_dpdn_inc_mode(0, k, 0.0, p.rhom, p.zm, p.nrho, p.nz) for p in ps]

    off_ext, off_int = 0, n
    A = zeros(ComplexF64, 2n, 2n)
    b = zeros(ComplexF64, 2n)

    rows = 1:n
    A[rows, (off_ext + 1):(off_ext + n)] = P_ext
    A[rows, (off_int + 1):(off_int + n)] = -P_int
    b[rows] = -p_inc

    rows = (n + 1):(2n)
    A[rows, (off_ext + 1):(off_ext + n)] = V_ext
    A[rows, (off_int + 1):(off_int + n)] = -V_int ./ g
    b[rows] = -dpdn_inc

    x = A \ b
    A_ext = x[(off_ext + 1):(off_ext + n)]
    A_int = x[(off_int + 1):(off_int + n)]

    p_scat = P_ext * A_ext
    dpdn_scat = V_ext * A_ext
    p_int = P_int * A_int
    dpdn_int = V_int * A_int
    return p_scat, dpdn_scat, ps, p_int, dpdn_int
end

"""
    solve_oblique_mfs(boundary::Union{Rigid,PressureRelease}, k, mesh, incidence_angle; m_max, offset, rtol=1e-6)

Axisymmetric MFS solution for a unit-amplitude plane wave arriving at
`incidence_angle` [rad] from the z-axis (`0` = axial/end-on, matching
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
        m_max::Integer, offset::Real, rtol::Real = 1e-6)
    β = incidence_angle
    ρ_s, z_s = mfs_source_points(mesh, offset)
    ps = panels(mesh)

    p_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)

    for m in 0:m_max
        P, V, _ = assemble_mfs_operators(mesh, k, ρ_s, z_s; m = m, rtol = rtol)
        p_inc = ComplexF64[_p_inc_mode(m, k, β, p.rhom, p.zm) for p in ps]
        dpdn_inc = ComplexF64[_dpdn_inc_mode(m, k, β, p.rhom, p.zm, p.nrho, p.nz)
                              for p in ps]

        A = boundary isa PressureRelease ? P \ (-p_inc) : V \ (-dpdn_inc)
        p_scat_modes[m + 1] = P * A
        dpdn_scat_modes[m + 1] = V * A
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
other oblique solver in this package. Unlike
[`solve_oblique`](@ref)`(::FluidFilled, ...)`, this does **not** use that
method's reduced-`2n×2n`/CHIEF/extended-precision machinery (built to
chase a hard near-unity-contrast CBIE case, see that method's own
docstring), the naive `2n×2n` block system in `[A_ext; A_int]` is used
directly, since MFS's axial `FluidFilled` case showed no sign of that
failure mode; revisit with the same tools if a similarly stuck
weakly-scattering case turns up here.

Returns `(p_scat_modes, dpdn_scat_modes, ps)`, matching
[`solve_oblique`](@ref)`(::FluidFilled, ...)`'s shape.
"""
function solve_oblique_mfs(
        boundary::FluidFilled, k::Real, mesh::MeridianMesh, incidence_angle::Real;
        m_max::Integer, offset_ext::Real, offset_int::Real, rtol::Real = 1e-6)
    β = incidence_angle
    g = boundary.density_contrast
    k_int = k / boundary.soundspeed_contrast

    ρ_s_ext, z_s_ext = mfs_source_points(mesh, offset_ext)
    ρ_s_int, z_s_int = mfs_source_points(mesh, -offset_int)
    ps = panels(mesh)
    n = length(ps)
    off_ext, off_int = 0, n

    p_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)

    for m in 0:m_max
        P_ext, V_ext, _ = assemble_mfs_operators(
            mesh, k, ρ_s_ext, z_s_ext; m = m, rtol = rtol)
        P_int, V_int, _ = assemble_mfs_operators(
            mesh, k_int, ρ_s_int, z_s_int; m = m, rtol = rtol)

        p_inc = ComplexF64[_p_inc_mode(m, k, β, p.rhom, p.zm) for p in ps]
        dpdn_inc = ComplexF64[_dpdn_inc_mode(m, k, β, p.rhom, p.zm, p.nrho, p.nz)
                              for p in ps]

        A = zeros(ComplexF64, 2n, 2n)
        b = zeros(ComplexF64, 2n)

        rows = 1:n
        A[rows, (off_ext + 1):(off_ext + n)] = P_ext
        A[rows, (off_int + 1):(off_int + n)] = -P_int
        b[rows] = -p_inc

        rows = (n + 1):(2n)
        A[rows, (off_ext + 1):(off_ext + n)] = V_ext
        A[rows, (off_int + 1):(off_int + n)] = -V_int ./ g
        b[rows] = -dpdn_inc

        x = A \ b
        A_ext = x[(off_ext + 1):(off_ext + n)]

        p_scat_modes[m + 1] = P_ext * A_ext
        dpdn_scat_modes[m + 1] = V_ext * A_ext
    end

    return p_scat_modes, dpdn_scat_modes, ps
end
