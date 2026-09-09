# Axisymmetric BEM for a fluid spherical shell, extending the single-interface coupled-domain
# CBIE to two boundary surfaces bounding three domains.

"""
    _assemble_cbie_cross_operators(mesh_field::MeridianMesh, mesh_src::MeridianMesh, k::Real; m=0, rtol=1e-6)

Cross-mesh generalization of [`assemble_cbie_operators`](@ref): `K[i,j]`,
`V[i,j]` for field collocation point `i` on `mesh_field` against source
panel `j` on `mesh_src` (a different surface, no self-interaction
singularity, since the two surfaces never coincide).
"""
function _assemble_cbie_cross_operators(
        mesh_field::MeridianMesh, mesh_src::MeridianMesh, k::Real;
        m::Integer = 0, rtol::Real = 1e-6)
    ps_field = panels(mesh_field)
    ps_src = panels(mesh_src)
    nf, ns = length(ps_field), length(ps_src)
    K = zeros(ComplexF64, nf, ns)
    V = zeros(ComplexF64, nf, ns)
    for i in 1:nf
        xρ, xz = ps_field[i].rhom, ps_field[i].zm
        for j in 1:ns
            pj = ps_src[j]
            K[i, j] = _pair_K(k, xρ, xz, pj, false, rtol; m = m)
            V[i, j] = _pair_V(k, xρ, xz, pj, false, rtol; m = m)
        end
    end
    return K, V
end

"""
    solve_axial(boundary::Shelled{FluidLayer,FluidInterior}, k, mesh_outer::MeridianMesh, mesh_inner::MeridianMesh; rtol=1e-6)

Solve the coupled axisymmetric CBIE for a fluid-shelled sphere (three
domains: exterior, fluid shell, interior fluid) under axial incidence.
`mesh_outer`/`mesh_inner` are separate meshes at the shell's outer/inner
radii (e.g. `sphere_mesh(a, n)`, `sphere_mesh(b, n)`). See this file's
module docstring for the two-surface representation-formula sign
convention. Returns `(p_scat, dpdn_scat, ps_outer)`, the exterior
scattered field and outer-mesh panel geometry, usable with
[`far_field`](@ref) exactly as [`solve_axial`](@ref)'s other methods.
"""
function solve_axial(boundary::Shelled{FluidLayer, FluidInterior}, k::Real,
        mesh_outer::MeridianMesh, mesh_inner::MeridianMesh;
        rtol::Real = 1e-6)
    k2 = k / boundary.material.soundspeed_contrast
    k3 = k / boundary.interior.soundspeed_contrast
    g_shell = boundary.material.density_contrast
    g_int = boundary.interior.density_contrast

    K_oo1, V_oo1, ps_o = assemble_cbie_operators(mesh_outer, k; rtol = rtol)
    K_oo2, V_oo2, _ = assemble_cbie_operators(mesh_outer, k2; rtol = rtol)
    K_ii2, V_ii2, ps_i = assemble_cbie_operators(mesh_inner, k2; rtol = rtol)
    K_ii3, V_ii3, _ = assemble_cbie_operators(mesh_inner, k3; rtol = rtol)
    K_oi2, V_oi2 = _assemble_cbie_cross_operators(mesh_outer, mesh_inner, k2; rtol = rtol)
    K_io2, V_io2 = _assemble_cbie_cross_operators(mesh_inner, mesh_outer, k2; rtol = rtol)

    no, ni = length(ps_o), length(ps_i)
    off_pe, off_de = 0, no
    off_pso, off_dso = 2no, 3no
    off_psi, off_dsi = 4no, 4no + ni
    off_pint, off_dint = 4no + 2ni, 4no + 3ni
    n_total = 4no + 4ni
    A = zeros(ComplexF64, n_total, n_total)
    b = zeros(ComplexF64, n_total)
    Io = Matrix{ComplexF64}(I, no, no)
    Ii = Matrix{ComplexF64}(I, ni, ni)

    dpdn_inc = ComplexF64[im * k * p.nz * cis(k * p.zm) for p in ps_o]
    p_inc = ComplexF64[cis(k * p.zm) for p in ps_o]

    # Eq(A): exterior, outer mesh, unchanged single-interface pattern.
    rows = 1:no
    A[rows, (off_pe + 1):(off_pe + no)] = 0.5Io - K_oo1
    A[rows, (off_de + 1):(off_de + no)] = V_oo1

    # Eq(B): shell, collocated on outer mesh (outer piece: away-from-shell
    # → +K,-V; inner piece: into-shell → -K,+V, cross-mesh).
    rows = (no + 1):(2no)
    A[rows, (off_pso + 1):(off_pso + no)] = 0.5Io + K_oo2
    A[rows, (off_dso + 1):(off_dso + no)] = -V_oo2
    A[rows, (off_psi + 1):(off_psi + ni)] = -K_oi2
    A[rows, (off_dsi + 1):(off_dsi + ni)] = V_oi2

    # Eq(C): shell, collocated on inner mesh (outer piece: away-from-shell
    # → +K,-V, cross-mesh; inner piece: into-shell → -K,+V).
    rows = (2no + 1):(2no + ni)
    A[rows, (off_pso + 1):(off_pso + no)] = K_io2
    A[rows, (off_dso + 1):(off_dso + no)] = -V_io2
    A[rows, (off_psi + 1):(off_psi + ni)] = 0.5Ii - K_ii2
    A[rows, (off_dsi + 1):(off_dsi + ni)] = V_ii2

    # Eq(D): interior domain, inner mesh, same pattern as the single-interface FluidFilled case.
    rows = (2no + ni + 1):(2no + 2ni)
    A[rows, (off_pint + 1):(off_pint + ni)] = 0.5Ii + K_ii3
    A[rows, (off_dint + 1):(off_dint + ni)] = -V_ii3

    # Outer-interface continuity (pressure, then density-weighted velocity).
    rows = (2no + 2ni + 1):(3no + 2ni)
    A[rows, (off_pe + 1):(off_pe + no)] = Io
    A[rows, (off_pso + 1):(off_pso + no)] = -Io
    b[rows] = -p_inc

    rows = (3no + 2ni + 1):(4no + 2ni)
    A[rows, (off_de + 1):(off_de + no)] = Io
    A[rows, (off_dso + 1):(off_dso + no)] = -Io ./ g_shell
    b[rows] = -dpdn_inc

    # Inner-interface continuity.
    rows = (4no + 2ni + 1):(4no + 3ni)
    A[rows, (off_psi + 1):(off_psi + ni)] = Ii
    A[rows, (off_pint + 1):(off_pint + ni)] = -Ii

    rows = (4no + 3ni + 1):(4no + 4ni)
    A[rows, (off_dsi + 1):(off_dsi + ni)] = Ii ./ g_shell
    A[rows, (off_dint + 1):(off_dint + ni)] = -Ii ./ g_int

    x = A \ b
    p_scat = x[(off_pe + 1):(off_pe + no)]
    dpdn_scat = x[(off_de + 1):(off_de + no)]
    return p_scat, dpdn_scat, ps_o
end

"""
    solve_axial(boundary::Shelled{FluidLayer,VacuumInterior}, k, mesh_outer::MeridianMesh, mesh_inner::MeridianMesh; rtol=1e-6)

Solve the coupled axisymmetric CBIE for a fluid-shelled sphere with a
pressure-release (void) interior cavity, same architecture as
`solve_axial(::Shelled{FluidLayer,FluidInterior},...)` but with no interior domain: `p=0`
on the shell's inner surface is imposed directly (an essential/Dirichlet
condition, not a fourth coupled domain), so `mesh_inner`'s only remaining
unknown there is the shell's own inner-surface `∂p/∂n`.
"""
function solve_axial(
        boundary::Shelled{FluidLayer, VacuumInterior}, k::Real, mesh_outer::MeridianMesh, mesh_inner::MeridianMesh;
        rtol::Real = 1e-6)
    k2 = k / boundary.material.soundspeed_contrast
    g_shell = boundary.material.density_contrast

    K_oo1, V_oo1, ps_o = assemble_cbie_operators(mesh_outer, k; rtol = rtol)
    K_oo2, V_oo2, _ = assemble_cbie_operators(mesh_outer, k2; rtol = rtol)
    K_ii2, V_ii2, ps_i = assemble_cbie_operators(mesh_inner, k2; rtol = rtol)
    K_oi2, V_oi2 = _assemble_cbie_cross_operators(mesh_outer, mesh_inner, k2; rtol = rtol)
    K_io2, V_io2 = _assemble_cbie_cross_operators(mesh_inner, mesh_outer, k2; rtol = rtol)

    no, ni = length(ps_o), length(ps_i)
    off_pe, off_de = 0, no
    off_pso, off_dso = 2no, 3no
    off_dsi = 4no
    n_total = 4no + ni
    A = zeros(ComplexF64, n_total, n_total)
    b = zeros(ComplexF64, n_total)
    Io = Matrix{ComplexF64}(I, no, no)
    Ii = Matrix{ComplexF64}(I, ni, ni)

    dpdn_inc = ComplexF64[im * k * p.nz * cis(k * p.zm) for p in ps_o]
    p_inc = ComplexF64[cis(k * p.zm) for p in ps_o]

    rows = 1:no
    A[rows, (off_pe + 1):(off_pe + no)] = 0.5Io - K_oo1
    A[rows, (off_de + 1):(off_de + no)] = V_oo1

    # Eq(B): shell at outer mesh, with p_sh_i ≡ 0 already substituted, only the V-term survives.
    rows = (no + 1):(2no)
    A[rows, (off_pso + 1):(off_pso + no)] = 0.5Io + K_oo2
    A[rows, (off_dso + 1):(off_dso + no)] = -V_oo2
    A[rows, (off_dsi + 1):(off_dsi + ni)] = V_oi2

    # Eq(C): shell at inner mesh, same substitution (p_sh_i=0 drops the
    # 0.5I-K_ii2 term's own unknown entirely, leaving only its V_ii2 term).
    rows = (2no + 1):(2no + ni)
    A[rows, (off_pso + 1):(off_pso + no)] = K_io2
    A[rows, (off_dso + 1):(off_dso + no)] = -V_io2
    A[rows, (off_dsi + 1):(off_dsi + ni)] = V_ii2

    rows = (2no + ni + 1):(3no + ni)
    A[rows, (off_pe + 1):(off_pe + no)] = Io
    A[rows, (off_pso + 1):(off_pso + no)] = -Io
    b[rows] = -p_inc

    rows = (3no + ni + 1):(4no + ni)
    A[rows, (off_de + 1):(off_de + no)] = Io
    A[rows, (off_dso + 1):(off_dso + no)] = -Io ./ g_shell
    b[rows] = -dpdn_inc

    x = A \ b
    p_scat = x[(off_pe + 1):(off_pe + no)]
    dpdn_scat = x[(off_de + 1):(off_de + no)]
    return p_scat, dpdn_scat, ps_o
end
