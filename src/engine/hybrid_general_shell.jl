# Oblique/broadside coupled shell-fluid scattering using the general (Fourier-mode m >= 0) shell
# FEM from shell_fem_general.jl, unlike shell_fem.jl's axisymmetric-only (m = 0) operator.

"""
    solve_general_shell_fluid_filled_coupled(geometry, rho_shell, youngs_modulus, poisson,
                                              exterior_density, exterior_soundspeed,
                                              interior_density, interior_soundspeed,
                                              frequency_hz, incidence_angle;
                                              m_max, n_eta=65, n_t=3, pole_offset=1e-3, rtol=1e-5)

Solve the coupled general-shell/fluid scattering problem (fluid-filled
interior cavity) for a unit-amplitude plane wave arriving at
`incidence_angle` [rad] from the z-axis (`0` = axial/end-on, `π/2` =
broadside), decomposed into azimuthal Fourier modes `m = 0, …, m_max`
exactly as [`solve_oblique`](@ref) does for the rigid/pressure-release
cases, this is what extends fluid-filled elastic-shell scattering beyond
axial incidence (see the module preamble; [`solve_shell_fluid_filled_coupled`](@ref)
in hybrid.jl is axial-only, tied to the nontorsional thin-shell theory).

Returns `(p_scat_modes, dpdn_scat_modes, ps)` for the *exterior* surface,
in the same shape [`solve_oblique`](@ref) returns, use directly with
[`far_field`](@ref)'s bistatic method.

This method builds the shell mesh from a `ProlateShellGeometry` (its
confocal-spheroidal-coordinate mesh generator, [`build_structured_shell_strip`](@ref))
and delegates to the [`GeneralShellMesh`](@ref)-taking method below, which
does the actual coupled solve, geometry-agnostic once the mesh exists (see
that method for e.g. [`build_structured_spherical_shell`](@ref) reusing it
for a literal sphere, where the confocal parametrization is a genuine
coordinate singularity).
"""
function solve_general_shell_fluid_filled_coupled(geometry::ProlateShellGeometry,
        rho_shell::Real, youngs_modulus::Real, poisson::Real,
        exterior_density::Real, exterior_soundspeed::Real,
        interior_density::Real, interior_soundspeed::Real,
        frequency_hz::Real, incidence_angle::Real;
        m_max::Integer, n_eta::Integer = 65, n_t::Integer = 3,
        pole_offset::Real = 1e-3, rtol::Real = 1e-5)
    shell_mesh = build_structured_shell_strip(geometry, n_eta, n_t; pole_offset = pole_offset)
    return solve_general_shell_fluid_filled_coupled(
        shell_mesh, rho_shell, youngs_modulus, poisson,
        exterior_density, exterior_soundspeed, interior_density, interior_soundspeed,
        frequency_hz, incidence_angle; m_max = m_max, rtol = rtol)
end

"""
    solve_general_shell_fluid_filled_coupled(shell_mesh::GeneralShellMesh, rho_shell, youngs_modulus, poisson,
                                              exterior_density, exterior_soundspeed,
                                              interior_density, interior_soundspeed,
                                              frequency_hz, incidence_angle; m_max, rtol=1e-5)

Same coupled shell/fluid solve as the `ProlateShellGeometry` method above,
taking an already-built [`GeneralShellMesh`](@ref) directly, the part of
this solver that's actually geometry-specific is only mesh construction;
everything from here on (Fourier-mode CBIE assembly, shell FEM assembly,
the coupled linear system) only needs the mesh's outer/inner curves and
normals, not how they were generated.
"""
function solve_general_shell_fluid_filled_coupled(shell_mesh::GeneralShellMesh,
        rho_shell::Real, youngs_modulus::Real, poisson::Real,
        exterior_density::Real, exterior_soundspeed::Real,
        interior_density::Real, interior_soundspeed::Real,
        frequency_hz::Real, incidence_angle::Real;
        m_max::Integer, rtol::Real = 1e-5)
    omega = 2pi * frequency_hz
    k_ext = omega / exterior_soundspeed
    k_int = omega / interior_soundspeed
    β = incidence_angle
    n_eta = length(shell_mesh.eta)

    outer_mesh = MeridianMesh(reverse(shell_mesh.outer_ρ), reverse(shell_mesh.outer_z))
    inner_mesh = MeridianMesh(reverse(shell_mesh.inner_ρ), reverse(shell_mesh.inner_z))
    n_bem = n_eta - 1
    L_s2b = shell_to_bem_interpolation(n_eta)
    L_b2s = bem_to_shell_interpolation(n_eta)
    I_bem = Matrix{ComplexF64}(I, n_bem, n_bem)

    p_out_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_out_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    local ps_out

    for m in 0:m_max
        K_out, V_out, ps_out_m = assemble_cbie_operators(outer_mesh, k_ext; m = m, rtol = rtol)
        K_in, V_in, _ = assemble_cbie_operators(inner_mesh, k_int; m = m, rtol = rtol)
        ps_out = ps_out_m
        ops = assemble_shell_modal_operators(
            shell_mesh, m, omega, rho_shell, youngs_modulus, poisson)
        n_keep = length(ops.keep)

        dpdn_inc_out = ComplexF64[_dpdn_inc_mode(m, k_ext, β, p.rhom, p.zm, p.nrho, p.nz)
                                  for p in ps_out]
        p_inc_shell_out = ComplexF64[_p_inc_mode(m, k_ext, β, shell_mesh.outer_ρ[j],
                                         shell_mesh.outer_z[j]) for j in 1:n_eta]

        off_pout, off_dout, off_pin, off_din, off_state = 0, n_bem, 2n_bem, 3n_bem, 4n_bem
        n_total = 4n_bem + n_keep
        A = zeros(ComplexF64, n_total, n_total)
        b = zeros(ComplexF64, n_total)

        rows = 1:n_bem
        A[rows, (off_pout + 1):(off_pout + n_bem)] = 0.5I - K_out
        A[rows, (off_dout + 1):(off_dout + n_bem)] = V_out

        rows = (n_bem + 1):(2n_bem)
        A[rows, (off_dout + 1):(off_dout + n_bem)] = I_bem
        A[rows, (off_state + 1):(off_state + n_keep)] = -exterior_density * omega^2 .*
                                                        (L_s2b * ops.B_out)
        b[rows] = -dpdn_inc_out

        rows = (2n_bem + 1):(3n_bem)
        A[rows, (off_pin + 1):(off_pin + n_bem)] = 0.5I + K_in
        A[rows, (off_din + 1):(off_din + n_bem)] = -V_in

        rows = (3n_bem + 1):(4n_bem)
        A[rows, (off_din + 1):(off_din + n_bem)] = I_bem
        A[rows, (off_state + 1):(off_state + n_keep)] = -interior_density * omega^2 .*
                                                        (L_s2b * ops.B_in)

        rows = (4n_bem + 1):(4n_bem + n_keep)
        A[rows, (off_pout + 1):(off_pout + n_bem)] = -ops.Q_out * L_b2s
        A[rows, (off_pin + 1):(off_pin + n_bem)] = ops.Q_in * L_b2s
        A[rows, (off_state + 1):(off_state + n_keep)] = ops.dynamic_matrix
        b[rows] = ops.Q_out * p_inc_shell_out

        x = A \ b
        p_out_modes[m + 1] = x[(off_pout + 1):(off_pout + n_bem)]
        dpdn_out_modes[m + 1] = x[(off_dout + 1):(off_dout + n_bem)]
    end

    return p_out_modes, dpdn_out_modes, ps_out
end
