# Radial-only elastic FEM for the finite elastic cylinder (solid and shelled), the cylindrical
# counterpart of elastic_radial_fem.jl, mirroring cylinder_elastic_modal.jl's boundary matrices.

# Assembles the tridiagonal FEM matrix for the cylindrical radial ODE on
# mesh `nodes` (linear elements), mode m.
function _elastic_cylinder_radial_mode_matrix(nodes::Vector{Float64}, mode::Integer)
    n_node = length(nodes)
    m2 = Float64(mode)^2
    dl = zeros(Float64, n_node - 1)
    d = zeros(Float64, n_node)
    for e in 1:(n_node - 1)
        x0, x1 = nodes[e], nodes[e + 1]
        h = x1 - x0
        local00 = 0.0
        local01 = 0.0
        local11 = 0.0
        for (xi, wi) in zip(_GL4_X, _GL4_W)
            x = 0.5 * (x0 + x1) + 0.5 * h * xi
            w = 0.5 * h * wi
            φ0, φ1 = (x1 - x) / h, (x - x0) / h
            dφ0, dφ1 = -1 / h, 1 / h
            weight = m2 / x - x
            local00 += w * (x * dφ0 * dφ0 + weight * φ0 * φ0)
            local01 += w * (x * dφ0 * dφ1 + weight * φ0 * φ1)
            local11 += w * (x * dφ1 * dφ1 + weight * φ1 * φ1)
        end
        d[e] += local00
        d[e + 1] += local11
        dl[e] += local01
    end
    return dl, d
end

# (stress, radial, shear) row operators from a boundary's (value, deriv), deriv is d(value)/dx.
function _cylinder_longitudinal_ops(
        mode::Integer, value::Real, deriv::Real, kL::Real, ρ::Real, ω::Real, μ::Real)
    (stress = value * (2μ * mode^2 - ρ * ω^2) - 2μ * kL * deriv,
        radial = kL * deriv,
        shear = 2μ * mode * (value - kL * deriv))
end

function _cylinder_shear_ops(mode::Integer, value::Real, deriv::Real, kT::Real, μ::Real)
    (stress = 2μ * mode * (kT * deriv - value),
        radial = mode * value,
        shear = μ * ((kT^2 - 2mode^2) * value + 2kT * deriv))
end

# Solid (regular-at-origin) cylinder: one basis per potential kind, value=1 at the outer radius.
function _regular_solid_cylinder_basis_value_deriv(mode::Integer, x_outer::Real, n_elements::Integer)
    nodes = collect(range(0.0, x_outer; length = n_elements + 1))
    dl, d = _elastic_cylinder_radial_mode_matrix(nodes, mode)
    n_node = n_elements + 1
    A = zeros(Float64, n_node, n_node)
    for i in 1:n_node
        A[i, i] = d[i]
    end
    for i in 1:(n_node - 1)
        A[i, i + 1] = dl[i]
        A[i + 1, i] = dl[i]
    end
    rhs = zeros(Float64, n_node)
    A[end, :] .= 0.0
    A[end, end] = 1.0
    rhs[end] = 1.0
    if mode > 0
        A[1, :] .= 0.0
        A[1, 1] = 1.0
        rhs[1] = 0.0
    end
    values = A \ rhs
    return values[end], _boundary_derivative(nodes, values, :outer)
end

# Shelled cylinder: outer-anchored / inner-anchored bases, each returned at both boundaries.
function _cylinder_shell_basis_value_deriv(mode::Integer, x_inner::Real, x_outer::Real, n_elements::Integer)
    nodes = collect(range(x_inner, x_outer; length = n_elements + 1))
    dl, d = _elastic_cylinder_radial_mode_matrix(nodes, mode)
    n_node = n_elements + 1
    outer_vals = _solve_dirichlet_basis(dl, d, n_node, 0.0, 1.0)
    inner_vals = _solve_dirichlet_basis(dl, d, n_node, 1.0, 0.0)
    return (
        outer_at_outer = (outer_vals[end], _boundary_derivative(nodes, outer_vals, :outer)),
        outer_at_inner = (outer_vals[1], _boundary_derivative(nodes, outer_vals, :inner)),
        inner_at_outer = (inner_vals[end], _boundary_derivative(nodes, inner_vals, :outer)),
        inner_at_inner = (inner_vals[1], _boundary_derivative(nodes, inner_vals, :inner))
    )
end

function _raw_bn_radial_fem(bc::SolidElastic, mode::Integer, k1a::Real, n_elements::Integer)
    ω = Float64(k1a)
    ρ = bc.density_contrast
    cL, cT = bc.speed_longitudinal_contrast, bc.speed_transversal_contrast
    μ = ρ * cT^2
    kL, kT = ω / cL, ω / cT

    vL, dL = _regular_solid_cylinder_basis_value_deriv(mode, kL, n_elements)
    vT, dT = _regular_solid_cylinder_basis_value_deriv(mode, kT, n_elements)
    long = _cylinder_longitudinal_ops(mode, vL, dL, kL, ρ, ω, μ)
    shear = _cylinder_shear_ops(mode, vT, dT, kT, μ)

    M = zeros(ComplexF64, 3, 3)
    rhs = zeros(ComplexF64, 3)
    M[1, 1] = besselh(mode, 1, ω)
    M[1, 2] = long.stress
    M[1, 3] = shear.stress
    rhs[1] = -besselj(mode, ω)
    M[2, 1] = -(1 / ω) * hcd(mode, ω)
    M[2, 2] = long.radial
    M[2, 3] = shear.radial
    rhs[2] = (1 / ω) * jcd(mode, ω)
    M[3, 2] = long.shear
    M[3, 3] = shear.shear

    M_num = copy(M)
    M_num[:, 1] = rhs
    return det(M_num) / det(M)
end

function _raw_bn_radial_fem(
        bc::Shelled{ElasticLayer, FluidInterior}, mode::Integer, k1a::Real, n_elements::Integer)
    ω = Float64(k1a)
    ρ_shell = bc.material.density_contrast
    cL, cT = bc.material.speed_longitudinal_contrast, bc.material.speed_transversal_contrast
    μ = ρ_shell * cT^2
    kL, kT = ω / cL, ω / cT
    k3 = ω / bc.interior.soundspeed_contrast
    b = bc.radius_ratio
    xLb, xTb, x3b = kL * b, kT * b, k3 * b
    ρ_int, c_int = bc.interior.density_contrast, bc.interior.soundspeed_contrast

    long = _cylinder_shell_basis_value_deriv(mode, xLb, kL, n_elements)
    shear = _cylinder_shell_basis_value_deriv(mode, xTb, kT, n_elements)

    long_oo = _cylinder_longitudinal_ops(mode, long.outer_at_outer..., kL, ρ_shell, ω, μ)
    long_io = _cylinder_longitudinal_ops(mode, long.inner_at_outer..., kL, ρ_shell, ω, μ)
    long_oi = _cylinder_longitudinal_ops(mode, long.outer_at_inner..., kL, ρ_shell, ω, μ)
    long_ii = _cylinder_longitudinal_ops(mode, long.inner_at_inner..., kL, ρ_shell, ω, μ)
    shear_oo = _cylinder_shear_ops(mode, shear.outer_at_outer..., kT, μ)
    shear_io = _cylinder_shear_ops(mode, shear.inner_at_outer..., kT, μ)
    shear_oi = _cylinder_shear_ops(mode, shear.outer_at_inner..., kT, μ)
    shear_ii = _cylinder_shear_ops(mode, shear.inner_at_inner..., kT, μ)

    M = zeros(ComplexF64, 6, 6)
    rhs = zeros(ComplexF64, 6)

    M[1, 1] = besselh(mode, 1, ω)
    M[1, 2], M[1, 3], M[1, 4], M[1, 5] = long_oo.stress, long_io.stress, shear_oo.stress,
    shear_io.stress
    rhs[1] = -besselj(mode, ω)

    M[2, 1] = -(1 / ω) * hcd(mode, ω)
    M[2, 2], M[2, 3], M[2, 4], M[2, 5] = long_oo.radial, long_io.radial, shear_oo.radial,
    shear_io.radial
    rhs[2] = (1 / ω) * jcd(mode, ω)

    M[3, 2], M[3, 3], M[3, 4], M[3, 5] = long_oo.shear, long_io.shear, shear_oo.shear,
    shear_io.shear

    M[4, 2], M[4, 3], M[4, 4], M[4, 5] = long_oi.stress, long_ii.stress, shear_oi.stress,
    shear_ii.stress
    M[4, 6] = besselj(mode, x3b)

    M[5, 2], M[5, 3], M[5, 4], M[5, 5] = long_oi.radial, long_ii.radial, shear_oi.radial,
    shear_ii.radial
    M[5, 6] = -(1 / (ω * c_int * ρ_int)) * jcd(mode, x3b)

    M[6, 2], M[6, 3], M[6, 4], M[6, 5] = long_oi.shear, long_ii.shear, shear_oi.shear,
    shear_ii.shear

    M_num = copy(M)
    M_num[:, 1] = rhs
    return det(M_num) / det(M)
end

"""
    elastic_cylinder_radial_fem_target_strength(boundary::Union{SolidElastic,Shelled{ElasticLayer}}, k, radius, length; aspect_angle=π/2, n_elements=320, m_max=default)

Backscatter target strength [dB re 1 m²] of a finite solid or elastic-
shelled cylinder via the radial FEM above, the cylindrical counterpart of
[`solid_elastic_sphere_radial_fem_target_strength`](@ref)/
[`elastic_shell_sphere_radial_fem_target_strength`](@ref), combined into a
finite-length target strength the same way
[`target_strength`](@ref)`(::Union{Rigid,...}, k, radius, length)` does
(exact per-mode cross-section solution × Fraunhofer axial envelope,
no end-cap scattering). `n_elements` controls the 1D radial mesh
resolution (shared by every mode).
"""
function elastic_cylinder_radial_fem_target_strength(
        boundary::Union{SolidElastic, Shelled{ElasticLayer, FluidInterior}}, k::Real, radius::Real, length::Real;
        aspect_angle::Real = π / 2,
        n_elements::Integer = 320,
        m_max::Integer = _default_mode_count(k * sin(aspect_angle) * radius))
    k1a = k * sin(aspect_angle) * radius
    k1L = k * length
    x = k1L * cos(aspect_angle)
    length_term = iszero(x) ? one(x) : sin(x) / x

    total = zero(ComplexF64)
    for m in 0:m_max
        raw = _raw_bn_radial_fem(boundary, m, k1a, n_elements)
        total += -neumann_factor(m) * (-1)^m * raw
    end

    f_bs = im * (length / π) * length_term * total
    return target_strength(f_bs)
end
