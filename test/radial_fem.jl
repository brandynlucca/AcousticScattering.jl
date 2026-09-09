using AcousticScattering
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "radial FEM (sphere)" begin
    c_water = 1477.4
    a = 0.01
    freq = 38000.0
    k = 2pi * freq / c_water

    sphere = AS.Sphere(a)
    ts_modal_rigid = AS.target_strength(AS.modal(sphere, AS.Rigid(), k))
    ts_fem_rigid = AS.target_strength(AS.fem(
        sphere, AS.Rigid(), k; R = 3a, n_elements = 200))
    @test ts_fem_rigid ≈ ts_modal_rigid atol = 1e-3

    ts_modal_pr = AS.target_strength(AS.modal(sphere, AS.PressureRelease(), k))
    ts_fem_pr = AS.target_strength(AS.fem(
        sphere, AS.PressureRelease(), k; R = 3a, n_elements = 200))
    @test ts_fem_pr ≈ ts_modal_pr atol = 1e-3

    ts_fem_R2 = AS.target_strength(AS.fem(
        sphere, AS.Rigid(), k; R = 5a, n_elements = 200))
    @test ts_fem_rigid ≈ ts_fem_R2 atol = 1e-3

    ts_fem_quad = AS.target_strength(AS.fem(
        sphere, AS.Rigid(), k; R = 3a, n_elements = 100, order = 2))
    @test ts_fem_quad ≈ ts_modal_rigid atol = 1e-3

    c_francis = 1477.3
    range_cases_khz = (12.0, 100.0, 200.0, 300.0, 386.0)
    for freq_khz in range_cases_khz
        kk = 2pi * freq_khz * 1000 / c_francis
        ts_modal = AS.target_strength(AS.modal(sphere, AS.Rigid(), kk))
        ts_fem_adaptive = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), kk; R = 3a, adaptive = true, target_tol = 0.005))
        @test ts_fem_adaptive ≈ ts_modal atol = 0.01
    end

    range_cases_pr_khz = (12.0, 100.0, 200.0, 300.0, 386.0)
    for freq_khz in range_cases_pr_khz
        kk = 2pi * freq_khz * 1000 / c_francis
        ts_modal_pr2 = AS.target_strength(AS.modal(sphere, AS.PressureRelease(), kk))
        ts_fem_pr_adaptive = AS.target_strength(AS.fem(
            sphere, AS.PressureRelease(), kk; R = 3a,
            adaptive = true, target_tol = 0.005))
        @test ts_fem_pr_adaptive ≈ ts_modal_pr2 atol = 0.01
    end
end

@testset "elastic sphere/cylinder radial FEM (vs modal series)" begin
    freq = 38000.0
    c_ext = 1477.4
    rho_ext = 1026.8
    k = 2pi * freq / c_ext
    rho_shell, G, lambda = 2700.0, 2.6e10, 5.3e10
    cL_shell = sqrt((lambda + 2G) / rho_shell)
    cT_shell = sqrt(G / rho_shell)
    radius_shell, radius_fluid = 0.05, 0.048

    solid = AS.SolidElastic(14900.0 / rho_ext, 6853.0 / c_ext, 4171.0 / c_ext)
    sphere1 = AS.Sphere(0.019)
    @test AS.target_strength(AS.fem(sphere1, solid, k; n_elements = 200)) ≈
          AS.target_strength(AS.modal(sphere1, solid, k; m_max = 13)) atol = 0.05

    shell_water = AS.Shelled(
        AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
        AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), radius_fluid / radius_shell)
    sphere_shell = AS.Sphere(radius_shell)
    @test AS.target_strength(AS.fem(
        sphere_shell, shell_water, k; n_elements = 200, m_max = 20)) ≈
          AS.target_strength(AS.modal(sphere_shell, shell_water, k; m_max = 20)) atol = 0.001

    radius, length = 0.05, 0.5
    rho0, c1, c2 = 7810.0, 5973.0, 3193.0
    solid_cyl = AS.SolidElastic(rho0 / rho_ext, c1 / c_ext, c2 / c_ext)
    cyl = AS.Cylinder(radius, length)
    @test AS.target_strength(AS.fem(cyl, solid_cyl, k; n_elements = 200)) ≈
          AS.target_strength(AS.modal(cyl, solid_cyl, k)) atol = 0.01

    shell_cyl = AS.Shelled(
        AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
        AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), radius_fluid / radius_shell)
    @test AS.target_strength(AS.fem(cyl, shell_cyl, k; n_elements = 200)) ≈
          AS.target_strength(AS.modal(cyl, shell_cyl, k)) atol = 0.001

    for angle_deg in (90.0, 70.0, 50.0, 30.0)
        ang = deg2rad(angle_deg)
        @test AS.target_strength(AS.fem(
            cyl, solid_cyl, k; incidence_angle = ang, n_elements = 200)) ≈
              AS.target_strength(AS.modal(cyl, solid_cyl, k; incidence_angle = ang)) atol = 0.01
        @test AS.target_strength(AS.fem(
            cyl, shell_cyl, k; incidence_angle = ang, n_elements = 200)) ≈
              AS.target_strength(AS.modal(cyl, shell_cyl, k; incidence_angle = ang)) atol = 0.001
    end

    ts_solid_modal = AS.target_strength(AS.modal(sphere_shell, solid, k))
    shell_limit_diffs = Float64[]
    for rr in (0.1, 0.02, 0.005)
        bc = AS.Shelled(
            AS.ElasticLayer(solid.density_contrast, solid.speed_longitudinal_contrast,
                solid.speed_transversal_contrast),
            AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), rr)
        ts_fem = AS.target_strength(AS.fem(
            sphere_shell, bc, k; n_elements = 200, m_max = 20))
        push!(shell_limit_diffs, abs(ts_fem - ts_solid_modal))
    end
    @test shell_limit_diffs[1] > 10 * maximum(shell_limit_diffs[2:end])
    @test maximum(shell_limit_diffs[2:end]) < 0.05
end

@testset "fluid shell sphere radial FEM (vs modal series)" begin
    freq = 38000.0
    c_ext = 1477.4
    rho_ext = 1026.8
    k = 2pi * freq / c_ext
    a = 0.05
    rr = 0.9

    sphere = AS.Sphere(a)
    bc_pr = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), rr)
    @test AS.target_strength(AS.fem(sphere, bc_pr, k; n_elements = 200, m_max = 20)) ≈
          AS.target_strength(AS.modal(sphere, bc_pr, k; m_max = 20)) atol = 1e-4

    bc_g = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext),
        AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), rr)
    @test AS.target_strength(AS.fem(sphere, bc_g, k; n_elements = 200, m_max = 20)) ≈
          AS.target_strength(AS.modal(sphere, bc_g, k; m_max = 20)) atol = 1e-4

    bc_plain_fluid = AS.FluidFilled(1.24 / rho_ext, 345.0 / c_ext)
    ts_plain = AS.target_strength(AS.modal(sphere, bc_plain_fluid, k; m_max = 20))
    bc_g2 = AS.Shelled(AS.FluidLayer(1.24 / rho_ext, 345.0 / c_ext),
        AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), 0.999)
    @test AS.target_strength(AS.fem(sphere, bc_g2, k; n_elements = 400, m_max = 20)) ≈
          ts_plain atol = 1e-3

    bc_g3 = AS.Shelled(AS.FluidLayer(1.24 / rho_ext, 345.0 / c_ext),
        AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), 0.999999)
    @test AS.reflection_coefficient(bc_g3, k, a) ≈
          AS.reflection_coefficient(AS.FluidFilled(1.24 / rho_ext, 345.0 / c_ext)) atol = 1e-6
    bc_pr3 = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), 0.999999)
    @test AS.reflection_coefficient(bc_pr3, k, a) ≈
          AS.reflection_coefficient(AS.PressureRelease()) atol = 1e-4

    for f in (5e3, 38e3, 120e3, 250e3)
        kk = 2pi * f / c_ext
        @test abs(AS.reflection_coefficient(bc_g, kk, a)) <= 1.0 + 1e-10
    end

    ts_ka = AS.target_strength(AS.kirchhoff(sphere, bc_g, 2pi * 200e3 / c_ext))
    ts_modal_hika = AS.target_strength(AS.modal(sphere, bc_g, 2pi * 200e3 / c_ext; m_max = 40))
    @test ts_ka ≈ ts_modal_hika atol = 0.1
end

@testset "ShellFEM (Hayek & Boisvert axisymmetric shell operator)" begin
    geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)

    @testset "geometry" begin
        @test geometry.focal_radius ≈ 0.0342928563989645
        @test geometry.semimajor_inner ≈ 0.03490343822605447
        @test geometry.semiminor_inner ≈ 0.006500000000000001
        @test geometry.xi_outer ≈ 1.0206207261596576
        @test geometry.xi_inner ≈ 1.0178049276498415
        @test geometry.xi_mid ≈ 1.0192128269047496
        @test geometry.semimajor_mid ≈ 0.03495171911302723
        @test geometry.semiminor_mid ≈ 0.006754455489227317
        @test geometry.a_shape ≈ 1.0192128269047496
        @test geometry.aspect_ratio ≈ 5.174616839028955
        @test geometry.bending_epsilon ≈ 1.705382018748745e-5

        @test_throws ArgumentError AS.ProlateShellGeometry(0.007, 0.035, 0.0005)
        @test_throws ArgumentError AS.ProlateShellGeometry(0.035, 0.007, 0.007)
        @test_throws ArgumentError AS.Shelled(0.6, 2565.0, 70e9)
    end

    material = AS.Shelled(0.32, 2565.0, 70e9)

    @testset "assembled system, freq=$freq" for freq in (12000.0, 38000.0)
        sys = AS.assemble_shell_system(geometry, material.material, freq; n_eta = 9)

        omega_hat_golden = freq == 12000.0 ? 0.4779315959914645 : 1.5134500539729712
        @test sys.nondimensional_frequency ≈ omega_hat_golden

        fro_golden = Dict(
            :K_uu => 7.0917050782e+03, :K_uw => 1.0232814400e+01, :K_u_beta =>
                4.1159927894e-01,
            :K_wu => 8.2081258324e+02, :K_ww => 1.0624527234e+02, :K_w_beta =>
                5.4102101877e+00,
            :K_beta_u => 4.1185967896e-01, :K_beta_w => 3.5052072706e-01, :K_beta_beta =>
                1.6278895367e-01
        )
        for (name, golden) in fro_golden
            @test norm(sys.structural_blocks[name]) ≈ golden rtol = 1e-9
        end

        mass_fro_golden = Dict(
            :M_uu => 3.0081981566e+00, :M_u_beta => 6.7653941525e-05,
            :M_ww => 4.4878125468e-01, :M_beta_u => 6.7653941525e-05, :M_beta_beta =>
                7.6518924142e-06
        )
        for (name, golden) in mass_fro_golden
            @test norm(sys.mass_blocks[name]) ≈ golden rtol = 1e-9
        end

        dynamic_fro_golden = freq == 12000.0 ? 7139.522101805418 : 7136.5777497886775
        @test norm(sys.dynamic_matrix) ≈ dynamic_fro_golden rtol = 1e-9
        @test norm(sys.load_scale_q) ≈ 1.4057207588996422e-11 rtol = 1e-9
        @test norm(sys.load_scale_m) ≈ 1.150699634692459e-08 rtol = 1e-9

        @test sys.structural_blocks[:K_uu][1, 1:5] ≈
              [-5014.550258592734, 15.900070807275364,
            -3.966239524917636, -0.003192064327710995, 0.0]
        dynamic_row1_golden = freq == 12000.0 ? -5014.319075589088 : -5012.232006806174
        @test sys.dynamic_matrix[1, 1:5] ≈
              [dynamic_row1_golden, 15.900070807275364,
            -3.966239524917636, -0.003192064327710995, 0.0]
    end
end

@testset "Interior CBIE (closed-form spherical-cavity cross-check)" begin
    a, k_test = 0.01, 37.5
    mesh = AS.sphere_mesh(a, 20)
    K, V, ps = AS.assemble_cbie_operators(mesh, k_test; rtol = 1e-6)
    n = length(ps)

    p_bc = ones(ComplexF64, n)
    dpdn = V \ ((0.5I + K) * p_bc)

    j0ka = AS.js(0, k_test * a)
    j0dka = AS.jsd(0, k_test * a)
    exact_dpdn = k_test * j0dka / j0ka

    @test all(d -> abs(d - exact_dpdn) / abs(exact_dpdn) < 0.01, dpdn)
    @test maximum(abs, dpdn) - minimum(abs, dpdn) < 1e-3 * abs(exact_dpdn)
end

@testset "Coupled shell/fluid axisymmetric scattering" begin
    geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
    shell_body = AS.Shell(AS.Spheroid(0.035, 0.007), 0.0005)
    rho_ext, c_ext = 1026.8, 1477.3
    rho_int, c_int = 1077.3, 1575.0
    freq = 12000.0
    k = 2pi * freq / c_ext

    @testset "geometry/interpolation building blocks" begin
        eta = AS.uniform_eta_grid(9; pole_offset = 0.001)
        mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
        ps = AS.panels(mesh)
        @test length(ps) == 8

        a, b = geometry.semimajor_length, geometry.semiminor_length
        for p in ps
            g = (p.rhom / b^2, p.zm / a^2)
            gn = hypot(g...)
            @test (p.nrho * g[1] + p.nz * g[2]) / gn ≈ 1.0 atol = 1e-9
        end

        L_s2b = AS.shell_to_bem_interpolation(9)
        L_b2s = AS.bem_to_shell_interpolation(9)
        f = geometry.focal_radius
        interp_eta = L_s2b * eta
        actual_eta = [p.zm / (f * geometry.xi_outer) for p in ps]
        @test interp_eta ≈ actual_eta atol = 1e-12

        roundtrip = L_b2s * interp_eta
        @test roundtrip[2:8] ≈ eta[2:8] atol = 1e-12
    end

    @testset "rigid limit (independent cross-check against solve_axial)" begin
        n_eta = 9
        eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
        outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
        p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
        ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

        diffs = Float64[]
        for E in (1e12, 1e14, 1e16)
            stiff_material = AS.Shelled(0.32, 2565.0, E)
            sol = AS.fem(shell_body, stiff_material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            ts = AS.target_strength(sol)
            push!(diffs, abs(ts - ts_rigid))
        end
        @test diffs[1] > diffs[2]
        @test diffs[3] < 0.1
    end

    @testset "realistic material: finite, sane, and resolution-stable at high n_eta" begin
        material = AS.Shelled(0.32, 2565.0, 70e9)
        results = Float64[]
        for n_eta in (97, 129)
            sol = AS.fem(shell_body, material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            @test all(isfinite, sol.data.p_ext_modes[1])
            @test all(isfinite, sol.data.dpdn_ext_modes[1])
            @test all(isfinite, sol.data.shell_state)
            push!(results, AS.target_strength(sol))
        end
        @test all(ts -> -150.0 < ts < 0.0, results)
        @test abs(results[1] - results[2]) < 2.0
    end

    @testset "fluid-filled: rigid limit (independent cross-check)" begin
        n_eta = 9
        eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
        outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
        p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
        ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

        diffs = Float64[]
        for E in (1e12, 1e14, 1e16)
            stiff_material = AS.Shelled(0.32, 2565.0, E)
            sol = AS.fem(
                shell_body, stiff_material, rho_ext, c_ext, rho_int, c_int, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            ts = AS.target_strength(sol)
            push!(diffs, abs(ts - ts_rigid))
            @test maximum(abs, sol.data.p_int_modes[1]) < 1.0
        end
        @test diffs[1] > diffs[2]
        @test diffs[3] < 0.1
    end

    @testset "fluid-filled: zero-interior-density limit recovers the vacuum-backed solver" begin
        n_eta = 9
        material = AS.Shelled(0.32, 2565.0, 70e9)
        sol_vac = AS.fem(shell_body, material, rho_ext, c_ext, 0.0, 1.0, k;
            method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
        ts_vac = AS.target_strength(sol_vac)

        diffs = Float64[]
        for rho_int_test in (1.0, 1e-3, 1e-6)
            sol = AS.fem(
                shell_body, material, rho_ext, c_ext, rho_int_test, c_int, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            ts = AS.target_strength(sol)
            push!(diffs, abs(ts - ts_vac))
        end
        @test diffs[1] > diffs[2] > diffs[3]
        @test diffs[3] < 1e-4
    end
end

@testset "General (Fourier-mode) solid shell FEM and oblique coupling" begin
    python_geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
    geometry = AS.ProlateShellGeometry(0.02, 0.005, 0.0005)
    c_water = 1477.4
    freq_hz = 12000.0
    k = 2pi * freq_hz / c_water
    n_eta = 13
    n_t = 3

    @testset "mesh matches the Python reference geometry exactly" begin
        mesh = AS.build_structured_shell_strip(python_geometry, 9, 3; pole_offset = 0.001)
        @test AS.Ferrite.getnnodes(mesh.grid) == 27
        @test AS.Ferrite.getncells(mesh.grid) == 32
        @test mesh.outer_ρ[1] ≈ 0.00036756472001377 atol = 1e-12
        @test mesh.outer_z[end] ≈ 0.035208966293679235 atol = 1e-12
        @test mesh.inner_ρ[5] ≈ 0.00675 atol = 1e-12
    end

    @testset "shell operators: eigenvalues match the Python reference (m=0,1,2)" begin
        mesh = AS.build_structured_shell_strip(python_geometry, 9, 3; pole_offset = 0.001)
        omega = 2pi * 12000.0
        expected_smallest = Dict(
            0 => 632372.37319513,
            1 => 372895.73609484,
            2 => 434667.49165402
        )
        for m in (0, 1, 2)
            ops = AS.assemble_shell_modal_operators(mesh, m, omega, 2565.0, 70e9, 0.32)
            evals_abs = sort(abs.(eigvals(ops.dynamic_matrix)))
            @test evals_abs[1] ≈ expected_smallest[m] rtol = 1e-6
            @test tr(ops.B_out * ops.B_out') ≈ 9.0 atol = 1e-10
            @test tr(ops.Q_out' * ops.Q_out) ≈ 4.16258628395464e-07 rtol = 1e-6
        end
    end

    @testset "coupled oblique rigid limit (independent cross-check against solve_oblique)" begin
        beta = deg2rad(45.0)
        m_max = 3
        shell_mesh = AS.build_structured_shell_strip(geometry, n_eta, n_t; pole_offset = 0.001)
        outer_mesh = AS.MeridianMesh(reverse(shell_mesh.outer_ρ), reverse(shell_mesh.outer_z))
        p_rigid, dpdn_rigid, ps_rigid = AS.solve_oblique(
            AS.Rigid(), k, outer_mesh, beta; m_max = m_max, rtol = 1e-4)
        ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi - beta, pi)

        shell_body2 = AS.Shell(AS.Spheroid(0.02, 0.005), 0.0005)
        material_stiff = AS.Shelled(0.3, 1e6, 1e14)
        sol = AS.fem(
            shell_body2, material_stiff, 1026.8, c_water, 1026.8, c_water, k;
            method = :general, incidence_angle = beta, m_max = m_max, n_eta = n_eta,
            n_t = n_t, pole_offset = 0.001, rtol = 1e-4)
        ts_stiff = AS.target_strength(sol; angle = pi - beta, azimuth = pi)

        @test ts_stiff ≈ ts_rigid atol = 0.1
    end
end
