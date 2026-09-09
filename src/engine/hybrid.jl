# Coupled shell/fluid axisymmetric scattering, links the axisymmetric BEM and the Hayek & Boisvert
# shell operator for axial (m = 0) incidence. Interior CBIE has the opposite sign from the exterior.

"""
    solve_shell_fluid_coupled(geometry, material, exterior_density, exterior_soundspeed, frequency_hz;
                               n_eta=65, pole_offset=1e-4, rtol=1e-5)

Solve the coupled axisymmetric shell/exterior-fluid scattering problem for
an axial (end-on) unit-amplitude incident plane wave on a confocal prolate
spheroidal elastic shell (vacuum/air-backed, no interior fluid).

Returns `(p_scat, dpdn_scat, ps, shell_state, shell)`: the BEM surface
solution and panel geometry (for [`far_field`](@ref), exactly as
[`solve_axial`](@ref) returns them) plus the shell displacement state
`[u; w; β]` and the assembled [`ShellSystem`](@ref).
"""
function solve_shell_fluid_coupled(
        geometry::ProlateShellGeometry, material::ElasticFEMLayer,
        exterior_density::Real, exterior_soundspeed::Real, frequency_hz::Real;
        n_eta::Integer = 65, pole_offset::Real = 1e-4, rtol::Real = 1e-5)
    shell = assemble_shell_system(
        geometry, material, frequency_hz; n_eta = n_eta, pole_offset = pole_offset)
    eta = shell.eta
    n_shell = length(eta)

    omega = 2pi * frequency_hz
    k = omega / exterior_soundspeed

    outer_mesh = prolate_confocal_mesh(geometry, eta; surface = :outer)
    K, V, ps = assemble_cbie_operators(outer_mesh, k; rtol = rtol)
    n_bem = length(ps)

    L_s2b = shell_to_bem_interpolation(n_shell)
    L_b2s = bem_to_shell_interpolation(n_shell)

    dpdn_inc = ComplexF64[im * k * p.nz * cis(k * p.zm) for p in ps]
    z_shell = geometry.focal_radius * geometry.a_shape .* eta
    p_inc_shell = cis.(k .* z_shell)

    n_state = 3 * n_shell
    n_total = 2 * n_bem + n_state
    A = zeros(ComplexF64, n_total, n_total)
    b = zeros(ComplexF64, n_total)

    w_rows = (2n_bem + n_shell + 1):(2n_bem + 2n_shell)

    A[1:n_bem, 1:n_bem] = 0.5I - K
    A[1:n_bem, (n_bem + 1):(2n_bem)] = V

    A[(n_bem + 1):(2n_bem), (n_bem + 1):(2n_bem)] = Matrix{ComplexF64}(I, n_bem, n_bem)
    A[(n_bem + 1):(2n_bem), w_rows] = -exterior_density * omega^2 .* L_s2b
    b[(n_bem + 1):(2n_bem)] = -dpdn_inc

    A[w_rows, 1:n_bem] = -Diagonal(shell.load_scale_q) * L_b2s
    A[(2n_bem + 1):(2n_bem + n_state), (2n_bem + 1):(2n_bem + n_state)] = shell.dynamic_matrix
    b[w_rows] = shell.load_scale_q .* p_inc_shell

    x = A \ b
    p_scat = x[1:n_bem]
    dpdn_scat = x[(n_bem + 1):(2n_bem)]
    shell_state = x[(2n_bem + 1):end]

    return p_scat, dpdn_scat, ps, shell_state, shell
end

"""
    solve_shell_fluid_filled_coupled(geometry, material,
                                      exterior_density, exterior_soundspeed,
                                      interior_density, interior_soundspeed, frequency_hz;
                                      n_eta=65, pole_offset=1e-4, rtol=1e-5)

Solve the coupled axisymmetric shell/fluid scattering problem for an
axial (end-on) unit-amplitude incident plane wave on a confocal prolate
spheroidal elastic shell with a fluid-filled interior cavity.

Returns `(p_ext, dpdn_ext, ps_ext, p_int, dpdn_int, ps_int, shell_state, shell)`:
exterior and interior BEM surface solutions and panel geometry (the
exterior pair usable with [`far_field`](@ref) exactly as
[`solve_axial`](@ref)'s are) plus the shell displacement state
`[u; w; β]` and the assembled [`ShellSystem`](@ref).
"""
function solve_shell_fluid_filled_coupled(
        geometry::ProlateShellGeometry, material::ElasticFEMLayer,
        exterior_density::Real, exterior_soundspeed::Real,
        interior_density::Real, interior_soundspeed::Real, frequency_hz::Real;
        n_eta::Integer = 65, pole_offset::Real = 1e-4, rtol::Real = 1e-5)
    shell = assemble_shell_system(
        geometry, material, frequency_hz; n_eta = n_eta, pole_offset = pole_offset)
    eta = shell.eta
    n_shell = length(eta)

    omega = 2pi * frequency_hz
    k_ext = omega / exterior_soundspeed
    k_int = omega / interior_soundspeed

    outer_mesh = prolate_confocal_mesh(geometry, eta; surface = :outer)
    inner_mesh = prolate_confocal_mesh(geometry, eta; surface = :inner)
    K_ext, V_ext, ps_ext = assemble_cbie_operators(outer_mesh, k_ext; rtol = rtol)
    K_int, V_int, ps_int = assemble_cbie_operators(inner_mesh, k_int; rtol = rtol)
    n_bem = length(ps_ext)

    L_s2b = shell_to_bem_interpolation(n_shell)
    L_b2s = bem_to_shell_interpolation(n_shell)

    dpdn_inc = ComplexF64[im * k_ext * p.nz * cis(k_ext * p.zm) for p in ps_ext]
    z_shell = geometry.focal_radius * geometry.a_shape .* eta
    p_inc_shell = cis.(k_ext .* z_shell)

    n_state = 3 * n_shell
    off_pext, off_dext, off_pint, off_dint, off_state = 0, n_bem, 2n_bem, 3n_bem, 4n_bem
    n_total = 4n_bem + n_state
    A = zeros(ComplexF64, n_total, n_total)
    b = zeros(ComplexF64, n_total)

    w_rows = (off_state + n_shell + 1):(off_state + 2n_shell)
    I_bem = Matrix{ComplexF64}(I, n_bem, n_bem)

    # Exterior CBIE
    rows = 1:n_bem
    A[rows, (off_pext + 1):(off_pext + n_bem)] = 0.5I - K_ext
    A[rows, (off_dext + 1):(off_dext + n_bem)] = V_ext

    # Exterior fluid-structure coupling
    rows = (n_bem + 1):(2n_bem)
    A[rows, (off_dext + 1):(off_dext + n_bem)] = I_bem
    A[rows, w_rows] = -exterior_density * omega^2 .* L_s2b
    b[rows] = -dpdn_inc

    # Interior CBIE (opposite sign, see file header derivation)
    rows = (2n_bem + 1):(3n_bem)
    A[rows, (off_pint + 1):(off_pint + n_bem)] = 0.5I + K_int
    A[rows, (off_dint + 1):(off_dint + n_bem)] = -V_int

    # Interior fluid-structure coupling (no incident field inside)
    rows = (3n_bem + 1):(4n_bem)
    A[rows, (off_dint + 1):(off_dint + n_bem)] = I_bem
    A[rows, w_rows] = -interior_density * omega^2 .* L_s2b

    # Shell: net load = outer total pressure minus inner pressure
    A[w_rows, (off_pext + 1):(off_pext + n_bem)] = -Diagonal(shell.load_scale_q) * L_b2s
    A[w_rows, (off_pint + 1):(off_pint + n_bem)] = Diagonal(shell.load_scale_q) * L_b2s
    A[(off_state + 1):(off_state + n_state), (off_state + 1):(off_state + n_state)] = shell.dynamic_matrix
    b[w_rows] = shell.load_scale_q .* p_inc_shell

    x = A \ b
    p_ext = x[(off_pext + 1):(off_pext + n_bem)]
    dpdn_ext = x[(off_dext + 1):(off_dext + n_bem)]
    p_int = x[(off_pint + 1):(off_pint + n_bem)]
    dpdn_int = x[(off_dint + 1):(off_dint + n_bem)]
    shell_state = x[(off_state + 1):end]

    return p_ext, dpdn_ext, ps_ext, p_int, dpdn_int, ps_int, shell_state, shell
end
