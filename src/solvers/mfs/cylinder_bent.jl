"""
    solve_bent_cylinder_mfs(boundary::Union{Rigid,PressureRelease}, k, radius, length, radius_curvature; aspect_angle=π/2, offset, n_s=20, n_φ=16)

3D (non-axisymmetric) MFS solution for a unit-amplitude plane wave
`e^{ik k̂ᵢ·x}`, `k̂ᵢ = (cos(aspect_angle), sin(aspect_angle), 0)` (in the
bend's own plane, matching [`bent_cylinder_kirchhoff_form_function`](@ref)'s
convention), scattering off a uniformly bent finite cylinder. One source
per collocation point (see [`bent_cylinder_mfs_points`](@ref)), offset
`offset` [m] inward along that point's own normal, the same simplest
MFS scheme [`solve_axial_mfs`](@ref) uses for canonical axisymmetric
bodies, here without any azimuthal reduction since none is available.
Returns `(p_scat, dpdn_scat, points, normals, areas)`.

The grid omits end caps: this is a lateral-surface approximation, not the exterior
boundary-value problem of a closed cylinder. Neither a small fitted residual nor
agreement with a finite-length modal approximation establishes closed-body accuracy.
For a closed finite body, pass a full surface mesh to [`mfs`](@ref) and refine source
spacing, inward offset and collocation independently.
"""
function solve_bent_cylinder_mfs(
        boundary::Union{Rigid, PressureRelease}, k::Real, radius::Real, length::Real,
        radius_curvature::Real; aspect_angle::Real = π / 2, offset::Real, n_s::Integer = 20, n_φ::Integer = 16,
        oversampling::Integer = 1, condition_limit::Integer = 512, solve_reports = nothing)
    oversampling >= 1 || throw(ArgumentError("oversampling must be at least 1"))
    source_points, source_normals, _ = bent_cylinder_mfs_points(
        radius, length, radius_curvature, n_s, n_φ)
    points, normals, areas = bent_cylinder_mfs_points(
        radius, length, radius_curvature, oversampling * n_s, oversampling * n_φ)
    n = Base.length(points)
    sources = [_sub3(p, offset .* normal)
               for (p, normal) in zip(source_points, source_normals)]
    ns = Base.length(sources)

    k̂ᵢ = (cos(aspect_angle), sin(aspect_angle), 0.0)
    p_inc = [cis(k * _dot3(k̂ᵢ, points[i])) for i in 1:n]
    dpdn_inc = [im * k * _dot3(k̂ᵢ, normals[i]) * p_inc[i] for i in 1:n]

    P = [_green3d(k, points[i], sources[j]) for i in 1:n, j in 1:ns]
    V = [_dgreen3d_dn(k, points[i], normals[i], sources[j]) for i in 1:n, j in 1:ns]

    matrix = boundary isa PressureRelease ? P : V
    rhs = boundary isa PressureRelease ? -p_inc : -dpdn_inc
    coefficients = matrix \ rhs
    if solve_reports !== nothing
        checks, check_normals, _ = bent_cylinder_mfs_points(
            radius, length, radius_curvature, 2oversampling * n_s, 2oversampling * n_φ)
        pc = [cis(k * _dot3(k̂ᵢ, p)) for p in checks]
        check_rhs = boundary isa PressureRelease ? -pc :
                    [-im * k * _dot3(k̂ᵢ, normal) * p
                     for (normal, p) in zip(check_normals, pc)]
        check_values = zeros(ComplexF64, Base.length(checks))
        Threads.@threads for i in eachindex(checks)
            for j in eachindex(sources)
                kernel = boundary isa PressureRelease ? _green3d(k, checks[i], sources[j]) :
                         _dgreen3d_dn(k, checks[i], check_normals[i], sources[j])
                check_values[i] += kernel * coefficients[j]
            end
        end
        _record_solve!(solve_reports, matrix, coefficients, rhs; offset, n_s, n_phi = n_φ,
            oversampling, source_count = ns, collocation_count = n, check_count = Base.length(checks),
            boundary_residual = _residual_report(check_values - check_rhs, check_rhs),
            _mfs_matrix_diagnostics(matrix; condition_limit)...)
    end
    p_scat = P * coefficients
    dpdn_scat = V * coefficients
    return p_scat, dpdn_scat, points, normals, areas
end

"""
    bent_cylinder_mfs_target_strength(boundary, k, radius, length, radius_curvature; aspect_angle=π/2, offset, n_s=20, n_φ=16)

Monostatic backscatter target strength [dB re 1 m²] from
[`solve_bent_cylinder_mfs`](@ref), via the standard Kirchhoff-Helmholtz
far-field reduction evaluated as a direct surface sum over the same
collocation grid (consistent with the discretization already used to
solve for `p_scat`/`dpdn_scat`, rather than a separate finer quadrature):

```math
f = \\frac{1}{4\\pi}\\sum_i \\left[ik\\,(\\hat q\\cdot\\hat n_i)\\,p_i + \\partial_n p_i\\right] e^{-ik\\hat q \\cdot r_i}\\,\\Delta S_i,
\\qquad \\hat q = -\\hat k_i.
```
"""
function bent_cylinder_mfs_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real, radius::Real, length::Real,
        radius_curvature::Real; aspect_angle::Real = π / 2, offset::Real, n_s::Integer = 20, n_φ::Integer = 16)
    p_scat, dpdn_scat, points, normals, areas = solve_bent_cylinder_mfs(
        boundary, k, radius, length, radius_curvature;
        aspect_angle = aspect_angle, offset = offset, n_s = n_s, n_φ = n_φ)
    q̂ = (-cos(aspect_angle), -sin(aspect_angle), 0.0)
    f = zero(ComplexF64)
    for i in eachindex(points)
        f += (im * k * _dot3(q̂, normals[i]) * p_scat[i] + dpdn_scat[i]) *
             cis(-k * _dot3(q̂, points[i])) * areas[i]
    end
    return target_strength(f / (4π))
end
