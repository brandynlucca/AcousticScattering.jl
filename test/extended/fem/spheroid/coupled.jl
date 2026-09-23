using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "ShellFEM (axisymmetric shell operator)" @testset "ShellFEM (axisymmetric shell operator)" begin
        geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)

        @time "geometry" @testset "geometry" begin
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

        @time @testset "assembled system, freq=$freq" for freq in (12000.0, 38000.0)
            sys = AS.assemble_shell_system(geometry, material.material, freq; n_eta = 9)

            omega_hat_golden = freq == 12000.0 ? 0.4779315959914645 : 1.5134500539729712
            @test sys.nondimensional_frequency ≈ omega_hat_golden

            @test size(sys.dynamic_matrix) == (27, 27)
            @test all(isfinite, sys.dynamic_matrix)
            @test norm(sys.load_scale_q) ≈ 1.4057207588996422e-11 rtol = 1e-9
        end
    end

    @time "Independent prolate shell frequencies" @testset "Independent prolate shell frequencies" begin
        geometry = AS.ProlateShellGeometry(1.0097040331600713, 0.7207583814259635, 0.02745)
        material = Shelled(0.3, 2700.0, 70e9).material
        frequency = AS.extensional_plate_speed(material) / (2pi * geometry.semimajor_mid)
        static = AS.assemble_shell_system(geometry, material, 0.0; n_eta = 257)
        dynamic = AS.assemble_shell_system(geometry, material, frequency; n_eta = 257)
        squared_frequencies = eigvals(-static.dynamic_matrix,
            dynamic.dynamic_matrix - static.dynamic_matrix)
        frequencies = sort(sqrt.(real.(filter(
            value -> abs(imag(value)) < 1e-6 && real(value) > 0.1, squared_frequencies))))
        # First three flexural modes; reference frequencies are rounded to 0.001.
        for (computed, expected) in zip(frequencies[1:3], (1.043, 1.285, 1.372))
            @test computed≈expected atol=0.0005 rtol=0
        end
    end

    @time "Coupled shell/fluid axisymmetric scattering" @testset "Coupled shell/fluid axisymmetric scattering" begin
        geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
        shell_body = AS.Shell(AS.Spheroid(0.035, 0.007), 0.0005)
        rho_ext, c_ext = 1026.8, 1477.3
        rho_int, c_int = 1077.3, 1575.0
        freq = 12000.0
        k = 2pi * freq / c_ext

        @time "geometry/interpolation building blocks" @testset "geometry/interpolation building blocks" begin
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

        @time "rigid limit (independent cross-check against solve_axial)" @testset "rigid limit (independent cross-check against solve_axial)" begin
            n_eta = 33
            eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
            outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

            # Scale density too, so rigid-body motion also vanishes.
            stiff_material = AS.Shelled(0.32, 2565.0 * 1e16 / 70e9, 1e16)
            sol = AS.fem(shell_body, stiff_material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            @test abs(AS.target_strength(sol) - ts_rigid) < 0.1
        end

        @time "thin-shell backscatter: mesh and pole refinement below 0.1 dB" @testset "thin-shell backscatter: mesh and pole refinement below 0.1 dB" begin
            # NOTE: the pole-offset refinement comparison (n_eta=257 at two offsets) is covered
            # at full fidelity in perf/radial_fem.jl; here a single configuration checks sanity.
            material = AS.Shelled(0.32, 2565.0, 70e9)
            sol = AS.fem(shell_body, material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = 257,
                pole_offset = 0.001, rtol = 1e-4)
            @test all(isfinite, sol.data.p_ext_modes[1])
            @test all(isfinite, sol.data.dpdn_ext_modes[1])
            @test all(isfinite, sol.data.shell_state)
            ts = AS.target_strength(sol)
            @test -150.0 < ts < 0.0
        end

        @time "fluid-filled: rigid limit (independent cross-check)" @testset "fluid-filled: rigid limit (independent cross-check)" begin
            n_eta = 33
            eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
            outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

            stiff_material = AS.Shelled(0.32, 2565.0 * 1e16 / 70e9, 1e16)
            sol = AS.fem(shell_body, stiff_material, rho_ext, c_ext, rho_int, c_int, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            @test maximum(abs, sol.data.p_int_modes[1]) < 1.0
            @test abs(AS.target_strength(sol) - ts_rigid) < 0.1
        end

        @time "fluid-filled: zero-interior-density limit recovers the vacuum-backed solver" @testset "fluid-filled: zero-interior-density limit recovers the vacuum-backed solver" begin
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

    @time "General (Fourier-mode) solid shell FEM and oblique coupling" @testset "General (Fourier-mode) solid shell FEM and oblique coupling" begin
        reference_geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
        geometry = AS.ProlateShellGeometry(0.02, 0.005, 0.0005)
        c_water = 1477.4
        freq_hz = 12000.0
        k = 2pi * freq_hz / c_water
        n_eta = 13
        n_t = 3

        @time "outer dimensions and confocal cavity" @testset "outer dimensions and confocal cavity" begin
            mesh = AS.build_structured_shell_strip(reference_geometry, 9, 3; pole_offset = 0.001)
            @test AS.Ferrite.getnnodes(mesh.grid) == 27
            @test AS.Ferrite.getncells(mesh.grid) == 32
            @test mesh.outer_ρ[1] ≈ 0.000312971244685512 atol = 1e-12
            @test mesh.outer_z[end] ≈ 0.034965 atol = 1e-12
            @test mesh.inner_ρ[5] ≈ 0.0065 atol = 1e-12
        end

        @time "elastic reciprocity and unit-normal projections (m=0,1,2)" @testset "elastic reciprocity and unit-normal projections (m=0,1,2)" begin
            mesh = AS.build_structured_shell_strip(reference_geometry, 9, 3; pole_offset = 0.001)
            omega = 2pi * 12000.0
            for m in (0, 1, 2)
                ops = AS.assemble_shell_modal_operators(mesh, m, omega, 2565.0, 70e9, 0.32)
                @test ops.dynamic_matrix ≈ ops.dynamic_matrix' rtol = 1e-12
                @test tr(ops.B_out * ops.B_out') ≈ 9.0 atol = 1e-10
                @test tr(ops.B_in * ops.B_in') ≈ 9.0 atol = 1e-10
            end
        end

        @time "coupled oblique rigid limit (independent cross-check against solve_oblique)" @testset "coupled oblique rigid limit (independent cross-check against solve_oblique)" begin
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
            @test AS.target_strength(sol) == ts_stiff
        end
    end

    @time "Confocal elastic shell against independent outputs" @testset "Confocal elastic shell against independent outputs" begin
        solution = fem(Shell(Spheroid(1.5, 1.0), 0.2), Shelled(0.33, 2700.0, 70e9),
            1000.0, 1500.0, 1000.0, 1500.0, 1.0;
            method = :general, incidence_angle = pi / 3, n_eta = 161, n_t = 17,
            m_max = 6, pole_offset = 1e-4, rtol = 1e-5)
        for (angle, azimuth, reference) in (
            (2pi / 3, pi, -0.106487925705214 + 0.0747097439019188im),
            (pi / 3, 0.0, 0.0814480252840919 + 0.0804910782181885im),
            (pi / 2, pi / 2, -0.467543104429619 + 0.0525831637378128im))
            @test abs(target_strength(solution; angle, azimuth) - 20log10(abs(reference))) <
                  0.1
            @test scattering_amplitude(solution; angle, azimuth) ≈ reference rtol = 0.01
        end
    end
end
