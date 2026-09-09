# General (non-axisymmetric) 3D BEM on Inti.jl (Nyström, near-singularity correction, H-matrix
# compression) and Gmsh.jl meshing, handles arbitrary/oblique geometry where axisymmetric_bem.jl's per-mode ring-kernel solve is slow or pathological. FluidFilled uses an unaccelerated dense 4n x 4n block system; Burton-Miller/CHIEF irregular-frequency regularization not yet implemented.

"""
    gmsh_sphere_mesh(radius; meshsize)

Full 3D triangulated surface quadrature (an `Inti.Quadrature`) for a sphere
of the given `radius` [m], via Gmsh's OpenCASCADE kernel. `meshsize` [m] is
the target element edge length, see [`bem3d_elements_per_wavelength`](@ref)
for a wavelength-based default.
"""
function gmsh_sphere_mesh(radius::Real; meshsize::Real, qorder::Integer = 4)
    meshsize > 0 || throw(ArgumentError("meshsize must be positive, got $meshsize"))
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("sphere")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
    return Inti.Quadrature(view(msh, Γ); qorder = qorder)
end

"""
    gmsh_spheroid_mesh(a, b; meshsize)

Full 3D triangulated surface quadrature for a prolate/oblate spheroid with
semi-axis `a` [m] along the axis of symmetry and equatorial semi-axis `b`
[m] (matching [`Spheroid`](@ref)'s convention: prolate if `a > b`, oblate
if `a < b`), via Gmsh (a unit sphere, non-uniformly scaled with OpenCASCADE's
`dilate`).
"""
function gmsh_spheroid_mesh(a::Real, b::Real; meshsize::Real, qorder::Integer = 4)
    meshsize > 0 || throw(ArgumentError("meshsize must be positive, got $meshsize"))
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("spheroid")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        tag = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, tag)], 0.0, 0.0, 0.0, b, b, a)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
    return Inti.Quadrature(view(msh, Γ); qorder = qorder)
end

"""
    bem3d_elements_per_wavelength(k; elements_per_wavelength=10)

Recommended Gmsh `meshsize` [m] for [`gmsh_sphere_mesh`](@ref)/
[`gmsh_spheroid_mesh`](@ref) at wavenumber `k` [1/m], targeting
`elements_per_wavelength` triangle edges per acoustic wavelength
(`λ = 2π/k`). Inti.jl's Nyström discretization with `qorder≥4` converges
much faster per element than `axisymmetric_bem.jl`'s piecewise-constant
panels, so a far smaller `elements_per_wavelength` than
[`bem_panel_count`](@ref)'s `30` default is already accurate, the default
here matches a rigid-sphere cross-check against the exact modal series
(`solve_full_bem`'s own validation) landing within ~0.02-0.09 dB across
`ka = 2.5-6.4`, start there, not at `30`, to avoid needlessly large
meshes.
"""
function bem3d_elements_per_wavelength(k::Real; elements_per_wavelength::Real = 10)
    (2π / k) / elements_per_wavelength
end

# Plane wave from -d̂, d̂ = (sinβ cosα, sinβ sinα, cosβ), matching axisymmetric_bem.jl's convention.
function _bem3d_incidence_direction(incidence_angle::Real, incidence_azimuth::Real)
    β, α = incidence_angle, incidence_azimuth
    return SVector(sin(β) * cos(α), sin(β) * sin(α), cos(β))
end

"""
    solve_full_bem(boundary::Union{Rigid,PressureRelease}, k, quad;
                   incidence_angle=0.0, incidence_azimuth=0.0,
                   compression=(method=:hmatrix, tol=1e-3),
                   correction=(method=:dim,),
                   gmres_kwargs=(reltol=1e-4, restart=150, maxiter=1200))

Solve the direct CBIE for a rigid or pressure-release scatterer over the
full 3D surface `quad` (from [`gmsh_sphere_mesh`](@ref)/
[`gmsh_spheroid_mesh`](@ref)) at a unit-amplitude plane wave arriving from
`incidence_angle`/`incidence_azimuth` [rad] (see [`_bem3d_incidence_direction`](@ref)),
via `Inti.jl`'s single-/double-layer operators (H-matrix-compressed, GMRES-
solved, see the module docstring's derivation and validation).

Returns `(p_scat, dpdn_scat, quad)`, the surface scattered pressure and
its normal derivative at every quadrature node, and `quad` itself (for
[`far_field`](@ref)).
"""
function solve_full_bem(boundary::Union{Rigid, PressureRelease}, k::Real, quad;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        compression = (method = :hmatrix, tol = 1e-5),
        correction = (method = :dim,),
        gmres_kwargs = (reltol = 1e-4, restart = 150, maxiter = 1200))
    d̂ = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    op = Inti.Helmholtz(; k = k, dim = 3)
    S, D = Inti.single_double_layer(;
        op, target = quad, source = quad, compression, correction)
    n = length(quad)

    p_inc = ComplexF64[cis(k * dot(d̂, q.coords)) for q in quad]
    dpdn_inc = ComplexF64[im * k * dot(d̂, q.normal) * p_inc[i]
                          for (i, q) in enumerate(quad)]

    if boundary isa Rigid
        dpdn_scat = -dpdn_inc
        A = LinearMap{ComplexF64}((y, x) -> (mul!(y, D, x); y .= 0.5 .* x .- y), n, n)
        rhs = S * dpdn_inc
        p_scat, hist = IterativeSolvers.gmres(A, rhs; log = true, gmres_kwargs...)
        hist.isconverged ||
            @warn "solve_full_bem: GMRES did not converge to the requested tolerance (Rigid CBIE), result may be inaccurate" iters = hist.iters
    else
        p_scat = -p_inc
        A = LinearMap{ComplexF64}((y, x) -> mul!(y, S, x), n, n)
        rhs = (D * p_scat) .- (0.5 .* p_scat)
        dpdn_scat, hist = IterativeSolvers.gmres(A, rhs; log = true, gmres_kwargs...)
        hist.isconverged ||
            @warn "solve_full_bem: GMRES did not converge to the requested tolerance (PressureRelease CBIE), result may be inaccurate" iters = hist.iters
    end

    return p_scat, dpdn_scat, quad
end

"""
    solve_full_bem(boundary::FluidFilled, k, quad;
                   incidence_angle=0.0, incidence_azimuth=0.0,
                   correction=(method=:dim,))

Solve the direct transmission CBIE for a fluid/gas-filled scatterer over
the full 3D surface `quad`, via a dense `4n×4n` block system (see the
module docstring's derivation and validation, *not* H-matrix/GMRES-
accelerated yet, so only practical at modest `n`; see
[`bem3d_elements_per_wavelength`](@ref) and keep `n` well under a few
thousand).

Returns `(p_scat, dpdn_scat, quad)`, the *exterior scattered* pressure and
its normal derivative, in the same form [`far_field`](@ref)/
[`target_strength`](@ref) expect (the interior trace is discarded, matching
`axisymmetric_bem.jl`'s own `solve_axial(::FluidFilled, ...)` convention).
"""
function solve_full_bem(boundary::FluidFilled, k::Real, quad;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        correction = (method = :dim,))
    d̂ = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    g = boundary.density_contrast
    k_int = k / boundary.soundspeed_contrast

    op_ext = Inti.Helmholtz(; k = k, dim = 3)
    op_int = Inti.Helmholtz(; k = k_int, dim = 3)
    S_ext, D_ext = Inti.single_double_layer(; op = op_ext, target = quad, source = quad,
        compression = (method = :none,), correction)
    S_int, D_int = Inti.single_double_layer(; op = op_int, target = quad, source = quad,
        compression = (method = :none,), correction)
    n = length(quad)

    p_inc = ComplexF64[cis(k * dot(d̂, q.coords)) for q in quad]
    dpdn_inc = ComplexF64[im * k * dot(d̂, q.normal) * p_inc[i]
                          for (i, q) in enumerate(quad)]

    Id = Matrix{ComplexF64}(I, n, n)
    off_p, off_d, off_pi, off_di = 0, n, 2n, 3n
    ntot = 4n
    A = zeros(ComplexF64, ntot, ntot)
    b = zeros(ComplexF64, ntot)

    rows = 1:n
    A[rows, (off_p + 1):(off_p + n)] = 0.5 .* Id .- D_ext
    A[rows, (off_d + 1):(off_d + n)] = S_ext

    rows = (n + 1):(2n)
    A[rows, (off_pi + 1):(off_pi + n)] = 0.5 .* Id .+ D_int
    A[rows, (off_di + 1):(off_di + n)] = -S_int

    rows = (2n + 1):(3n)
    A[rows, (off_p + 1):(off_p + n)] = Id
    A[rows, (off_pi + 1):(off_pi + n)] = -Id
    b[rows] = -p_inc

    rows = (3n + 1):(4n)
    A[rows, (off_d + 1):(off_d + n)] = Id
    A[rows, (off_di + 1):(off_di + n)] = -Id ./ g
    b[rows] = -dpdn_inc

    x = A \ b
    p_scat = x[(off_p + 1):(off_p + n)]
    dpdn_scat = x[(off_d + 1):(off_d + n)]

    return p_scat, dpdn_scat, quad
end

"""
    far_field(quad, xhat, k, p_scat, dpdn_scat)

Far-field scattering amplitude f(x̂) [m] extrapolated from a full 3D BEM
surface solution (as returned by [`solve_full_bem`](@ref)) at observation
direction `xhat` (a unit `SVector{3}`, the direction the receiver looks
*from*, so `xhat = -d̂` is monostatic backscatter for an incident direction
`d̂`), via the same Kirchhoff-Helmholtz far-field reduction
`axisymmetric_bem.jl`'s own `far_field` uses, now a genuine 3D surface
integral (no azimuthal closed form needed, since there's no assumed
symmetry to exploit).
"""
function far_field(quad, xhat::AbstractVector, k::Real, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number})
    total = zero(ComplexF64)
    for (i, q) in enumerate(quad)
        y = q.coords
        xdotn = dot(xhat, q.normal)
        xdoty = dot(xhat, y)
        total += (im * k * xdotn * p_scat[i] + dpdn_scat[i]) * cis(-k * xdoty) * q.weight
    end
    return -total / (4π)
end

"""
    target_strength(quad, xhat, k, p_scat, dpdn_scat)

Target strength [dB re 1 m²] of a full 3D BEM solution at observation
direction `xhat` (see [`far_field`](@ref)).
"""
function target_strength(
        quad, xhat::AbstractVector, k::Real, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number})
    return target_strength(far_field(quad, xhat, k, p_scat, dpdn_scat))
end
