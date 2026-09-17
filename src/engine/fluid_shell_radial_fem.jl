# Radial-only acoustic FEM for a fluid spherical shell (Shelled{FluidLayer}), the fluid
# counterpart of elastic_radial_fem.jl, strictly simpler since a fluid supports no shear.

# (value, deriv) at both boundaries for the outer-anchored (=1 at outer, =0 at inner) and
# inner-anchored (=0 at outer, =1 at inner) shell bases.
function _fluid_shell_basis_value_deriv(
        mode::Integer, x_inner::Real, x_outer::Real, n_elements::Integer;
        solve_reports = nothing)
    nodes = collect(range(x_inner, x_outer; length = n_elements + 1))
    dl, d = _elastic_radial_mode_matrix(nodes, mode)
    n_node = n_elements + 1
    outer_vals = _solve_dirichlet_basis(dl, d, n_node, 0.0, 1.0; solve_reports, mode)
    inner_vals = _solve_dirichlet_basis(dl, d, n_node, 1.0, 0.0; solve_reports, mode)
    return (;
        outer_at_outer = (outer_vals[end], _boundary_derivative(nodes, outer_vals, :outer)),
        outer_at_inner = (outer_vals[1], _boundary_derivative(nodes, outer_vals, :inner)),
        inner_at_outer = (inner_vals[end], _boundary_derivative(nodes, inner_vals, :outer)),
        inner_at_inner = (inner_vals[1], _boundary_derivative(nodes, inner_vals, :inner)),
        nodes, outer_vals, inner_vals
    )
end

# FluidLayer/VacuumInterior: p(b)=0 forces the inner-anchored basis's coefficient to exactly zero,
# leaving a single shell unknown (the outer-anchored basis), a 2-unknown (bₙ, shell coefficient) system.
function _fluid_shell_radial_fem_mode(
        bc::Shelled{FluidLayer, VacuumInterior}, mode::Integer, k::Real, a::Real, n_elements::Integer;
        solve_reports = nothing)
    x1a = k * a
    k2 = k / bc.material.soundspeed_contrast
    x2a, x2b = k2 * a, k2 * a * bc.radius_ratio
    gh_shell = bc.material.density_contrast * bc.material.soundspeed_contrast

    basis = _fluid_shell_basis_value_deriv(mode, x2b, x2a, n_elements; solve_reports)
    v_out, d_out = basis.outer_at_outer

    M = ComplexF64[hs(mode, x1a) -v_out
                   hsd(mode, x1a) -d_out/gh_shell]
    rhs = ComplexF64[-js(mode, x1a), -jsd(mode, x1a)]
    x = _solve_reported(M, rhs, solve_reports; mode, component = :interface)
    prefactor = (2mode + 1) * im^mode
    return (; coefficient = prefactor * x[1], order = 1,
        shell = (;
            radii = basis.nodes ./ k2, pressure = prefactor * x[2] .* basis.outer_vals),
        interior = nothing)
end

# FluidLayer/FluidInterior: shell needs both basis coefficients, a 4-unknown, 4-equation system per mode.
function _fluid_shell_radial_fem_mode(
        bc::Shelled{FluidLayer, FluidInterior}, mode::Integer, k::Real, a::Real, n_elements::Integer;
        solve_reports = nothing)
    x1a = k * a
    k2 = k / bc.material.soundspeed_contrast
    x2a, x2b = k2 * a, k2 * a * bc.radius_ratio
    k3 = k / bc.interior.soundspeed_contrast
    x3b = k3 * a * bc.radius_ratio
    gh_shell = bc.material.density_contrast * bc.material.soundspeed_contrast
    gh_int = bc.interior.density_contrast * bc.interior.soundspeed_contrast

    basis = _fluid_shell_basis_value_deriv(mode, x2b, x2a, n_elements; solve_reports)
    v_int, d_int = js(mode, x3b), jsd(mode, x3b)

    M = zeros(ComplexF64, 4, 4)
    rhs = zeros(ComplexF64, 4)
    M[1, 1] = hs(mode, x1a)
    M[1, 2], M[1, 3] = -basis.outer_at_outer[1], -basis.inner_at_outer[1]
    rhs[1] = -js(mode, x1a)

    M[2, 1] = hsd(mode, x1a)
    M[2, 2], M[2, 3] = -basis.outer_at_outer[2] / gh_shell,
    -basis.inner_at_outer[2] / gh_shell
    rhs[2] = -jsd(mode, x1a)

    M[3, 2], M[3, 3], M[3, 4] = basis.outer_at_inner[1], basis.inner_at_inner[1], -v_int

    M[4, 2], M[4, 3] = basis.outer_at_inner[2] / gh_shell,
    basis.inner_at_inner[2] / gh_shell
    M[4, 4] = -d_int / gh_int

    x = _solve_reported(M, rhs, solve_reports; mode, component = :interface)
    prefactor = (2mode + 1) * im^mode
    return (; coefficient = prefactor * x[1], order = 1,
        shell = (; radii = basis.nodes ./ k2,
            pressure = prefactor .* (x[2] .* basis.outer_vals .+ x[3] .* basis.inner_vals)),
        interior = (; coefficient = prefactor * x[4], wavenumber = k3,
            radius = a * bc.radius_ratio))
end

"""
    fluid_shell_sphere_radial_fem_target_strength(boundary::Shelled{FluidLayer}, k, a; n_elements=320, m_max=default)

Backscatter target strength [dB re 1 m²] of a fluid-shelled sphere
(`Shelled{FluidLayer,VacuumInterior}` or `Shelled{FluidLayer,FluidInterior}`) via a radial-only FEM, the
fluid-shell counterpart of
[`elastic_shell_sphere_radial_fem_target_strength`](@ref); see
`fluid_shell_radial_fem.jl`'s module docstring for the physics and why
this architecture is used. `n_elements` controls the 1D mesh resolution
within the shell (shared by every mode).
"""
function fluid_shell_sphere_radial_fem_target_strength(
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}},
        k::Real, a::Real; kwargs...)
    return target_strength(_radial_fem_amplitude(
        _fluid_shell_sphere_radial_fem_modes(boundary, k, a; kwargs...), k))
end

function _fluid_shell_sphere_radial_fem_modes(
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}}, k::Real, a::Real;
        n_elements::Integer = 320,
        m_max::Integer = _default_mode_count(k * a), solve_reports = nothing)
    return NamedTuple[_fluid_shell_radial_fem_mode(boundary, m, k, a, n_elements;
                          solve_reports) for m in 0:m_max]
end
