using AcousticScattering
using Test
using SparseArrays, LinearAlgebra

@testset "Cylinder" begin
    body = Cylinder(0.01, 0.02)
    k = 100.0

    @testset "Meridian FEM: rigid, pressure-release, and fluid-filled" begin
        for (name, boundary) in (("rigid", Rigid()),
            ("pressure-release", PressureRelease()),
            ("fluid-filled", FluidFilled(1.05, 1.02)))
            @testset "$name" begin
                solution = fem(body, boundary, k; R = 0.018,
                    incidence_angle = pi / 4, m_max = 1,
                    n_r = 4, n_theta = 12, l_max = 4)
                @test solution isa FEMSolution
                @test solution.method == :meridian
                @test target_strength(solution) isa Float64
                @test isfinite(target_strength(solution))
                @test -300.0 < target_strength(solution) < 50.0
                @test scattering_amplitude(solution) isa ComplexF64
                @test target_strength(solution) ≈
                      20log10(abs(scattering_amplitude(solution)))
                # Off-backscatter observation direction exercises the retained per-mode traces,
                # not just the solve-time default.
                @test scattering_amplitude(solution; angle = pi / 3, azimuth = 0.5) isa
                      ComplexF64
            end
        end
    end

    @testset "Radial FEM: solid elastic and elastic shell" begin
        solid = SolidElastic(7810.0 / 1026.8,
            5973.0 / 1477.4, 3193.0 / 1477.4)
        shell = Shelled(ElasticLayer(2700.0 / 1026.8,
                5700.0 / 1477.4, 3100.0 / 1477.4),
            FluidInterior(1.0, 1.0), 0.8)
        for (name, boundary) in (("solid elastic", solid),
            ("elastic shell", shell))
            @testset "$name" begin
                solution = fem(body, boundary, k; n_elements = 20, m_max = 2)
                reference = modal(body, boundary, k; m_max = 2)
                @test solution isa FEMSolution
                @test solution.method == :radial
                @test target_strength(solution) isa Float64
                @test isfinite(target_strength(solution))
                @test abs(target_strength(solution) - target_strength(reference)) < 3.0
                @test scattering_amplitude(solution) isa ComplexF64
                @test target_strength(solution) ≈
                      20log10(abs(scattering_amplitude(solution)))
            end
        end
    end

    @testset "Optional nodal field retention" begin
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
            options = (; R = 0.018, incidence_angle = pi / 4,
                m_max = 2, n_r = 6, n_theta = 16, l_max = 6)
            retained = fem(body, boundary, k; options...)
            compact = fem(body, boundary, k; options..., retain_fields = false)
            @test scattering_amplitude(compact) ≈ scattering_amplitude(retained) rtol = 1e-12
            @test scattering_amplitude(compact; angle = 0.7, azimuth = 0.4) ≈
                  scattering_amplitude(retained; angle = 0.7, azimuth = 0.4) rtol = 1e-12
            @test isempty(compact.data.fields)
            @test !isempty(retained.data.fields)
            @test diagnostics(retained).solver_options.retain_fields
            @test !diagnostics(compact).solver_options.retain_fields
            @test Base.summarysize(compact) < Base.summarysize(retained)
            @test_throws ArgumentError pressure(compact, (0.03, 0.0, 0.0))
            @test pressure(compact, (0.03, 0.0, 0.0); field = :incident) ≈
                  pressure(retained, (0.03, 0.0, 0.0); field = :incident)
        end
    end

    @testset "Shared bulk assembly agrees with direct quadrature" begin
        AS = AcousticScattering
        nr, nt, degree, radius_outer = 6, 18, 8, 0.018
        angles = AS._graded_theta_grid(nt, atan(body.radius / (body.length / 2)))
        radii = [AS._cylinder_r_inner(t, body.radius, body.length) for t in angles]
        hankel, derivative = AS._spherical_hankel_cache(degree, k * radius_outer)
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02),
            GasFilled(0.00129, 0.23))
            setup = AS._meridian_fem_mode_setup(boundary, k, angles, radii,
                radius_outer, nr, nt)
            original = copy(setup.matrix)
            for m in (0, 1, 4)
                args = (boundary, m, k, pi / 3, angles, radii, radius_outer,
                    nr, nt, degree, hankel, derivative)
                reference = AS._meridian_fem_mode_trace(args...)
                shared = AS._meridian_fem_mode_trace(args...; setup)
                @test shared[2]≈reference[2] rtol=3e-10 atol=1e-12
                @test shared[3]≈reference[3] rtol=3e-10 atol=1e-10
                @test shared[4].exterior≈reference[4].exterior rtol=3e-10 atol=1e-12
                if boundary isa FluidFilled
                    @test shared[4].interior≈reference[4].interior rtol=3e-10 atol=1e-12
                end
            end
            @test setup.matrix == original
        end
    end

    @testset "Bounded concurrent modes" begin
        options = (; R = 0.018, incidence_angle = pi / 3,
            m_max = 3, n_r = 8, n_theta = 24, l_max = 8)
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
            serial = fem(body, boundary, k; options...)
            threaded = fem(body, boundary, k; options..., mode_threads = 2)
            @test scattering_amplitude(threaded) ≈ scattering_amplitude(serial) rtol = 1e-12
            @test scattering_amplitude(threaded; angle = 0.7, azimuth = 0.4) ≈
                  scattering_amplitude(serial; angle = 0.7, azimuth = 0.4) rtol = 1e-12
            @test pressure(threaded, (0.014, 0.005, 0.002)) ≈
                  pressure(serial, (0.014, 0.005, 0.002)) rtol = 1e-12
            if boundary isa FluidFilled
                @test pressure(threaded, (0.002, 0.003, 0.001); field = :interior) ≈
                      pressure(serial, (0.002, 0.003, 0.001); field = :interior) rtol = 1e-12
            end
            report = diagnostics(threaded)
            @test [r.mode for r in report.systems] == 0:3
            @test all(r.mode_workers == min(2, Threads.nthreads()) for r in report.systems)
            @test diagnostics(serial).solver_options.mode_threads == 1
        end
        @test AcousticScattering._cylinder_fem_mode_workers(500_000, 26, 8) == 1
        @test_throws ArgumentError fem(body, Rigid(), k; options..., mode_threads = 0)
    end

    @testset "Higher-order mapped cylinder" begin
        AS = AcousticScattering
        wave = 40.0
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02),
            GasFilled(0.00129, 0.23))
            options = (; R = 0.018, order = 3, n_r = 3, n_theta = 12,
                m_max = 3, l_max = 8, incidence_angle = pi / 3)
            retained = fem(body, boundary, wave; options...)
            compact = fem(body, boundary, wave; options..., retain_fields = false,
                mode_threads = 2)
            @test scattering_amplitude(compact) ≈ scattering_amplitude(retained) rtol = 1e-12
            @test scattering_amplitude(compact; angle = 0.7, azimuth = 0.4) ≈
                  scattering_amplitude(retained; angle = 0.7, azimuth = 0.4) rtol = 1e-12
            @test isfinite(pressure(retained, (0.014, 0.005, 0.002)))
            @test_throws ArgumentError pressure(compact, (0.014, 0.005, 0.002))
            if boundary isa FluidFilled
                # Continuity is imposed for the retained Fourier expansion.
                # The public total field adds the untruncated plane wave.
                surface = (0.0, body.radius, 0.0)
                incident_modes = sum(AS._p_inc_mode(m, wave, pi / 3, body.radius, 0.0)
                for m in 0:3)
                @test pressure(retained, surface; field = :scattered) + incident_modes ≈
                      pressure(retained, (0.0, body.radius, 0.0); field = :interior) rtol = 1e-10
            end
            refined = fem(body, boundary, wave; options..., adaptive = true,
                target_tol = 0.01, max_refinements = 3, retain_fields = false)
            @test diagnostics(refined).refinement.converged
            @test diagnostics(refined).refinement.change_db <= 0.01
            @test abs(target_strength(refined) - target_strength(retained)) < 0.02
        end
        mesh = AS._meridian_hp_geometry(body, Rigid(), 0.018, 3, 12, 3; half = false)
        cap = atan(body.radius / (body.length / 2))
        @test cap in mesh.angular_edges
        @test pi - cap in mesh.angular_edges
        @test AS._cylinder_hp_profile(0.0, body.radius, body.length) ==
              (body.length / 2, 0.0)
        flat = Cylinder(0.02, 0.002)
        flat_cap = atan(flat.radius / (flat.length / 2))
        for nt in (3, 4, 8)
            edges = AS._graded_theta_grid(nt, flat_cap)
            @test length(edges) == nt + 1
            @test flat_cap in edges && pi - flat_cap in edges
            @test all(>(0), diff(edges))
        end
        flat_solution = fem(flat, Rigid(), wave; R = 0.025,
            order = 3, n_r = 3, n_theta = 4, m_max = 3, l_max = 8)
        @test isfinite(scattering_amplitude(flat_solution))
        for order in (2, 4)
            transparent = fem(body, FluidFilled(1.0, 1.0), wave;
                R = 0.018, order, n_r = 8, n_theta = 32, m_max = 5,
                l_max = 10, incidence_angle = pi / 3)
            @test abs(scattering_amplitude(transparent)) < 2e-8
            @test pressure(transparent, (0.002, 0.003, 0.001); field = :interior) ≈
                  pressure(transparent, (0.002, 0.003, 0.001); field = :incident) atol = 3e-6
        end
        for options in ((; order = 0), (; order = 9), (; order = 3, R = 0.01),
            (; order = 3, n_theta = 2), (; order = 3, max_refinements = 0))
            @test_throws ArgumentError fem(body, Rigid(), wave; options...)
        end
    end

    @testset "Augmented modal DtN" begin
        AS = AcousticScattering
        for order in (1, 3),
            boundary in (Rigid(), PressureRelease(),
                FluidFilled(1.05, 1.02), GasFilled(0.00129, 0.23))

            options = (; R = 0.018, order, n_r = 4, n_theta = 12,
                m_max = 3, l_max = 8, incidence_angle = pi / 3)
            assembled = fem(body, boundary, 40.0; options...)
            augmented = fem(body, boundary, 40.0; options..., dtn_form = :augmented)
            @test scattering_amplitude(augmented) ≈ scattering_amplitude(assembled) rtol = 3e-9
            @test scattering_amplitude(augmented; angle = 0.7, azimuth = 0.4) ≈
                  scattering_amplitude(assembled; angle = 0.7, azimuth = 0.4) rtol = 3e-9
            @test pressure(augmented, (0.014, 0.005, 0.002)) ≈
                  pressure(assembled, (0.014, 0.005, 0.002)) rtol = 3e-9
            if boundary isa FluidFilled
                @test pressure(augmented, (0.002, 0.003, 0.001); field = :interior) ≈
                      pressure(assembled, (0.002, 0.003, 0.001); field = :interior) rtol = 3e-9
            end
            @test diagnostics(augmented).solver_options.dtn_form === :augmented
            @test diagnostics(augmented).relative_residual < 1e-9
        end
        # Independent Schur complement, including a nonzero prescribed value
        # on the artificial boundary, checks both coupling signs and lifting.
        K = sparse(ComplexF64[4 1 0 0; 1 5 1 0; 0 1 6 1; 0 0 1 7])
        rhs = ComplexF64[1, 2, 3, 4]
        Q = [0.2 0.7; 0.6 -0.3]
        d = ComplexF64[-2 + 0.3im, -3 + 0.1im]
        outer, radius = [3, 4], 0.8
        direct = copy(K)
        direct[outer, outer] -= radius^2 * Q' * (d .* Q)
        direct_rhs = copy(rhs)
        augmented, augmented_rhs = AS._meridian_fem_boundary_system(
            K, rhs, outer, Q, d, radius, :augmented)
        for (matrix, load) in ((direct, direct_rhs), (augmented, augmented_rhs))
            AS._meridian_fem_dirichlet!(matrix, load, [1, 4], ComplexF64[0.5, 0.2im])
        end
        @test (augmented \ augmented_rhs)[1:4] ≈ direct \ direct_rhs rtol = 1e-13
        @test_throws ArgumentError fem(body, Rigid(), k; dtn_form = :invalid)
    end

    bent = Cylinder(0.01, 0.02; radius_curvature = 0.1)
    @test_throws ArgumentError fem(bent, Rigid(), k)
    @test_throws ArgumentError fem(bent, SolidElastic(2.0, 2.0, 1.0), k)
end

@testset "Meridian FEM complex amplitude matches modal" begin
    # modal(Cylinder, ...) is the DCM/FCMS reduction, not exact, and needs a long, slender
    # cylinder to be accurate (see the cylinder envelope tests); PressureRelease's DCM
    # approximation degrades faster than Rigid's at short aspect ratios (confirmed against
    # full 3D BEM directly: at aspect 2, FEM matches full BEM to ~1% for both boundaries while
    # matching modal to only ~4%/54%, so this uses a long enough cylinder for modal itself to
    # be trustworthy as the reference).
    radius = 0.01
    body = Cylinder(radius, 8 * 2radius)
    k = 100.0
    for boundary in (Rigid(), PressureRelease())
        solution = fem(body, boundary, k; R = 1.2 * (body.length / 2),
            incidence_angle = pi / 2, m_max = 8, n_r = 24, n_theta = 80, l_max = 30)
        reference = modal(body, boundary, k; incidence_angle = pi / 2, m_max = 20)
        @test scattering_amplitude(solution) ≈
              scattering_amplitude(reference) rtol = 0.08
    end
end

@testset "Axial meridian cylinder FEM" begin
    ts = AcousticScattering.cylinder_meridian_fem_target_strength(
        Rigid(), 0.5, 0.05, 0.1, 0.12; n_r = 6, n_theta = 12)
    @test isfinite(ts)
    ts_soft = AcousticScattering.cylinder_meridian_fem_target_strength(
        PressureRelease(), 0.5, 0.05, 0.1, 0.12; n_r = 6, n_theta = 12, l_max = 5)
    @test isfinite(ts_soft)
    @test ts != ts_soft
end
