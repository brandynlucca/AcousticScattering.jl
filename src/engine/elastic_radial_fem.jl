# Radial-only elastic FEM for a solid elastic sphere (SolidElastic): angular dependence handled
# exactly via spherical-harmonic modes, only the radial displacement potential is solved via 1D FEM.

# 4-point Gauss-Legendre nodes/weights on [-1,1].
const _GL4_X = (
    -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
const _GL4_W = (
    0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538)

# Assembles the tridiagonal FEM matrix for the radial ODE above on a 1D
# mesh `nodes` (linear elements), mode ℓ = mode*(mode+1).
function _elastic_radial_mode_matrix(nodes::Vector{Float64}, mode::Integer)
    n_node = length(nodes)
    ℓ = Float64(mode * (mode + 1))
    dl = zeros(Float64, n_node - 1)   # sub/super-diagonal
    d = zeros(Float64, n_node)        # diagonal
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
            local00 += w * (x^2 * dφ0 * dφ0 + (ℓ - x^2) * φ0 * φ0)
            local01 += w * (x^2 * dφ0 * dφ1 + (ℓ - x^2) * φ0 * φ1)
            local11 += w * (x^2 * dφ1 * dφ1 + (ℓ - x^2) * φ1 * φ1)
        end
        d[e] += local00
        d[e + 1] += local11
        dl[e] += local01
    end
    return dl, d
end

# Solves the ODE with Dirichlet data value_inner/value_outer at the two ends.
function _solve_dirichlet_basis(dl::Vector{Float64}, d::Vector{Float64}, n_node::Integer,
        value_inner::Real, value_outer::Real)
    y = zeros(Float64, n_node)
    y[1] = value_inner
    y[end] = value_outer
    n_node <= 2 && return y
    ni = n_node - 2
    A = zeros(Float64, ni, ni)
    for i in 1:ni
        A[i, i] = d[i + 1]
        if i > 1
            A[i, i - 1] = dl[i]
        end
        if i < ni
            A[i, i + 1] = dl[i + 1]
        end
    end
    rhs = zeros(Float64, ni)
    rhs[1] -= dl[1] * value_inner
    rhs[end] -= dl[end] * value_outer
    interior = A \ rhs
    y[2:(end - 1)] .= interior
    return y
end

# One-sided finite-difference derivative at a boundary node, using the
# (generally non-uniform) mesh spacing directly.
function _boundary_derivative(nodes::Vector{Float64}, values::Vector{Float64}, side::Symbol)
    n = length(nodes)
    if n < 3
        return side == :inner ? (values[2] - values[1]) / (nodes[2] - nodes[1]) :
               (values[end] - values[end - 1]) / (nodes[end] - nodes[end - 1])
    end
    if side == :inner
        h0, h1 = nodes[2] - nodes[1], nodes[3] - nodes[2]
        if abs(h0 - h1) < 1e-14 * max(1.0, abs(h0), abs(h1))
            h = 0.5 * (h0 + h1)
            return (-3values[1] + 4values[2] - values[3]) / (2h)
        end
        return (values[2] - values[1]) / h0
    else
        hm0, hm1 = nodes[end] - nodes[end - 1], nodes[end - 1] - nodes[end - 2]
        if abs(hm0 - hm1) < 1e-14 * max(1.0, abs(hm0), abs(hm1))
            h = 0.5 * (hm0 + hm1)
            return (3values[end] - 4values[end - 1] + values[end - 2]) / (2h)
        end
        return (values[end] - values[end - 1]) / hm0
    end
end

function _ode_second_derivative(mode::Integer, x::Real, value::Real, deriv::Real)
    -(2 / x) * deriv - (1 - mode * (mode + 1) / x^2) * value
end

# Stress/radial/shear operators at a boundary, from the FEM-solved radial
# profile there (value, derivative), for the longitudinal (P) potential.
function _longitudinal_ops(
        mode::Integer, x::Real, value::Real, deriv::Real, λ::Real, μ::Real)
    ydd = _ode_second_derivative(mode, x, value, deriv)
    stress = (λ * value - 2μ * ydd) / (λ + 2μ)
    radial = x * deriv
    shear = 2 * (x * deriv - value)
    return (stress = stress, radial = radial, shear = shear)
end

# Same, for the shear (SV) potential.
function _shear_ops(mode::Integer, x::Real, value::Real, deriv::Real)
    ℓ = mode * (mode + 1)
    ydd = _ode_second_derivative(mode, x, value, deriv)
    stress = -2ℓ / x^2 * (x * deriv - value)
    radial = ℓ * value
    shear = x^2 * ydd + (mode + 2) * (mode - 1) * value
    return (stress = stress, radial = radial, shear = shear)
end

# For a solid (regular-at-origin) body: one basis function per potential kind, value=1 at the
# outer radius, with the origin condition that keeps the solution regular there.
function _regular_solid_basis_ops(
        mode::Integer, x_outer::Real, n_elements::Integer, kind::Symbol, λ::Real, μ::Real)
    nodes = collect(range(0.0, x_outer; length = n_elements + 1))
    dl, d = _elastic_radial_mode_matrix(nodes, mode)
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
    deriv_outer = _boundary_derivative(nodes, values, :outer)
    ops = kind === :longitudinal ?
          _longitudinal_ops(mode, x_outer, values[end], deriv_outer, λ, μ) :
          _shear_ops(mode, x_outer, values[end], deriv_outer)
    return ops
end

# Raw modal scattering coefficient bₙ for a solid elastic sphere, mode `mode`, via the radial FEM above.
function _solid_elastic_radial_fem_mode(mode::Integer, k::Real, a::Real,
        density_contrast::Real, cL::Real, cT::Real, n_elements::Integer)
    μ = density_contrast * cT^2
    λ = density_contrast * cL^2 - 2μ
    kL, kT = k / cL, k / cT
    x1a, xLa, xTa = k * a, kL * a, kT * a

    ρ_ratio = 1 / density_contrast
    a1 = ρ_ratio * js(mode, x1a)
    a2 = x1a * jsd(mode, x1a)
    α11 = ρ_ratio * hs(mode, x1a)
    α21 = x1a * hsd(mode, x1a)

    long_ops = _regular_solid_basis_ops(mode, xLa, n_elements, :longitudinal, λ, μ)

    if mode == 0
        M = ComplexF64[α11 long_ops.stress; α21 long_ops.radial]
        r = ComplexF64[a1, a2]
        return (M \ r)[1]
    end

    shear_ops = _regular_solid_basis_ops(mode, xTa, n_elements, :shear, λ, μ)
    M = ComplexF64[α11 long_ops.stress shear_ops.stress
                   α21 long_ops.radial shear_ops.radial
                   0.0 long_ops.shear shear_ops.shear]
    r = ComplexF64[a1, a2, 0.0]
    return (M \ r)[1]
end

# Same idea as _regular_solid_basis_ops, but for a finite-thickness shell: needs TWO independent
# basis functions (outer-anchored and inner-anchored), returned at both boundaries.
function _shell_basis_ops(mode::Integer, x_inner::Real, x_outer::Real, n_elements::Integer,
        kind::Symbol, λ::Real, μ::Real)
    nodes = collect(range(x_inner, x_outer; length = n_elements + 1))
    dl, d = _elastic_radial_mode_matrix(nodes, mode)
    n_node = n_elements + 1
    outer_vals = _solve_dirichlet_basis(dl, d, n_node, 0.0, 1.0)
    inner_vals = _solve_dirichlet_basis(dl, d, n_node, 1.0, 0.0)
    d_outer_at_outer = _boundary_derivative(nodes, outer_vals, :outer)
    d_outer_at_inner = _boundary_derivative(nodes, outer_vals, :inner)
    d_inner_at_outer = _boundary_derivative(nodes, inner_vals, :outer)
    d_inner_at_inner = _boundary_derivative(nodes, inner_vals, :inner)
    opsfn = kind === :longitudinal ? (x, v, dv) -> _longitudinal_ops(mode, x, v, dv, λ, μ) :
            (x, v, dv) -> _shear_ops(mode, x, v, dv)
    return (
        outer_at_outer = opsfn(x_outer, outer_vals[end], d_outer_at_outer),
        outer_at_inner = opsfn(x_inner, outer_vals[1], d_outer_at_inner),
        inner_at_outer = opsfn(x_outer, inner_vals[end], d_inner_at_outer),
        inner_at_inner = opsfn(x_inner, inner_vals[1], d_inner_at_inner)
    )
end

# Raw modal scattering coefficient bₙ for an elastic-shelled sphere via the shell radial FEM
# above. Only `interior_coupling=:generalized` is implemented.
function _shelled_elastic_radial_fem_mode(mode::Integer, k::Real, a::Real, b::Real,
        density_shell_contrast::Real, cL::Real, cT::Real,
        density_interior_contrast::Real, c_interior::Real,
        n_elements::Integer)
    μ = density_shell_contrast * cT^2
    λ = density_shell_contrast * cL^2 - 2μ
    kL, kT = k / cL, k / cT
    x1a = k * a
    xLa, xLb = kL * a, kL * b
    xTa, xTb = kT * a, kT * b
    k3b = (k / c_interior) * b

    inv_rho_shell = 1 / density_shell_contrast
    ρ_int_over_shell = density_interior_contrast / density_shell_contrast

    a1 = inv_rho_shell * js(mode, x1a)
    a2 = x1a * jsd(mode, x1a)
    α11 = inv_rho_shell * hs(mode, x1a)
    α21 = x1a * hsd(mode, x1a)
    a46 = ρ_int_over_shell * js(mode, k3b)
    a56 = k3b * jsd(mode, k3b)

    long = _shell_basis_ops(mode, xLb, xLa, n_elements, :longitudinal, λ, μ)

    if mode == 0
        M = ComplexF64[α11 long.outer_at_outer.stress long.inner_at_outer.stress 0.0
                       α21 long.outer_at_outer.radial long.inner_at_outer.radial 0.0
                       0.0 long.outer_at_inner.stress long.inner_at_inner.stress a46
                       0.0 long.outer_at_inner.radial long.inner_at_inner.radial a56]
        r = ComplexF64[a1, a2, 0.0, 0.0]
        return (M \ r)[1]
    end

    shear = _shell_basis_ops(mode, xTb, xTa, n_elements, :shear, λ, μ)
    M = ComplexF64[α11 long.outer_at_outer.stress shear.outer_at_outer.stress long.inner_at_outer.stress shear.inner_at_outer.stress 0.0
                   α21 long.outer_at_outer.radial shear.outer_at_outer.radial long.inner_at_outer.radial shear.inner_at_outer.radial 0.0
                   0.0 long.outer_at_outer.shear shear.outer_at_outer.shear long.inner_at_outer.shear shear.inner_at_outer.shear 0.0
                   0.0 long.outer_at_inner.stress shear.outer_at_inner.stress long.inner_at_inner.stress shear.inner_at_inner.stress a46
                   0.0 long.outer_at_inner.radial shear.outer_at_inner.radial long.inner_at_inner.radial shear.inner_at_inner.radial a56
                   0.0 long.outer_at_inner.shear shear.outer_at_inner.shear long.inner_at_inner.shear shear.inner_at_inner.shear 0.0]
    r = ComplexF64[a1, a2, 0.0, 0.0, 0.0, 0.0]
    return (M \ r)[1]
end

"""
    elastic_shell_sphere_radial_fem_target_strength(k, a; density_shell_contrast, speed_longitudinal_contrast, speed_transversal_contrast, radius_ratio, density_interior_contrast, soundspeed_interior_contrast, n_elements=320, m_max=default)

Backscatter target strength [dB re 1 m²] of an elastic-shelled sphere
(interior fluid + elastic shell + exterior fluid) via the shell radial
FEM above, the shelled-sphere counterpart of
[`solid_elastic_sphere_radial_fem_target_strength`](@ref), matching
[`ElasticLayer`](@ref)'s `:generalized` interior coupling. `n_elements`
controls the 1D radial mesh resolution within the shell (shared by every
mode).
"""
function elastic_shell_sphere_radial_fem_target_strength(k::Real, a::Real;
        density_shell_contrast::Real, speed_longitudinal_contrast::Real,
        speed_transversal_contrast::Real, radius_ratio::Real,
        density_interior_contrast::Real, soundspeed_interior_contrast::Real,
        n_elements::Integer = 320,
        m_max::Integer = _default_mode_count(k * a))
    b = radius_ratio * a
    total = zero(ComplexF64)
    for m in 0:m_max
        bm = _shelled_elastic_radial_fem_mode(
            m, k, a, b, density_shell_contrast, speed_longitudinal_contrast,
            speed_transversal_contrast, density_interior_contrast, soundspeed_interior_contrast, n_elements)
        total += (2m + 1) * legendre_p(m, -1.0) * bm
    end
    f = -im / k * total
    return target_strength(f)
end

"""
    solid_elastic_sphere_radial_fem_target_strength(k, a; density_contrast, speed_longitudinal_contrast, speed_transversal_contrast, n_elements=320, m_max=default)

Backscatter target strength [dB re 1 m²] of a solid elastic sphere via a
radial-only FEM: the angular dependence is handled exactly (same
spherical-harmonic mode decomposition as [`SolidElastic`](@ref)'s modal
series), and only each mode's radial potential profile is solved
numerically (1D Galerkin FEM on its governing ODE) rather than read off
closed-form spherical Bessel functions, see the module docstring for why
this, rather than a genuine 2D meridian FEM, is used here. `n_elements`
controls the 1D radial mesh resolution (shared by every mode).
"""
function solid_elastic_sphere_radial_fem_target_strength(k::Real, a::Real;
        density_contrast::Real, speed_longitudinal_contrast::Real,
        speed_transversal_contrast::Real,
        n_elements::Integer = 320,
        m_max::Integer = _default_mode_count(k * a))
    total = zero(ComplexF64)
    for m in 0:m_max
        bm = _solid_elastic_radial_fem_mode(
            m, k, a, density_contrast, speed_longitudinal_contrast,
            speed_transversal_contrast, n_elements)
        total += (2m + 1) * legendre_p(m, -1.0) * bm
    end
    f = -im / k * total
    return target_strength(f)
end
