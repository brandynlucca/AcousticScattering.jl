using AcousticScattering
using Test

@testset "Spheroid" begin
    k = 2pi * 12000.0 / 1477.4
    boundaries = (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.05, 1.02)))

    @testset "Meridian FEM: prolate and oblate" begin
        for (shape, body) in ((:prolate, Spheroid(0.02, 0.015)),
            (:oblate, Spheroid(0.015, 0.02)))
            @testset "$shape" begin
                @test body.kind == shape
                for (name, boundary) in boundaries
                    @testset "$name" begin
                        solution = fem(body, boundary, k; incidence_angle = pi / 4,
                            m_max = 1, n_r = 4, n_theta = 10, l_max = 4)
                        @test solution isa FEMSolution
                        @test solution.method == :meridian
                        @test target_strength(solution) isa Float64
                        @test isfinite(target_strength(solution))
                        @test -300.0 < target_strength(solution) < 50.0
                    end
                end
            end
        end
        @test_throws ArgumentError fem(Spheroid(0.02, 0.015), Rigid(), k;
            method = :radial)
    end

    @testset "Meridian mode memory controls" begin
        body = Spheroid(0.07, 0.01)
        for boundary in (PressureRelease(), GasFilled(0.00129, 0.23))
            options = (; n_r = 8, n_theta = 24, m_max = 3, l_max = 12)
            retained = fem(body, boundary, k; options...)
            compact = fem(body, boundary, k; options..., retain_fields = false,
                mode_threads = 2)
            @test scattering_amplitude(compact) ≈ scattering_amplitude(retained) rtol = 1e-12
            @test scattering_amplitude(compact; angle = 0.6, azimuth = 0.4) ≈
                  scattering_amplitude(retained; angle = 0.6, azimuth = 0.4) rtol = 1e-12
            @test isempty(compact.data.fields)
            @test !isempty(retained.data.fields)
            @test diagnostics(retained).solver_options.mode_threads == 1
            @test length(diagnostics(compact).systems) == 4
            @test_throws ArgumentError pressure(compact, (0.1, 0.0, 0.0))
            @test pressure(compact, (0.1, 0.0, 0.0); field = :incident) ≈
                  pressure(retained, (0.1, 0.0, 0.0); field = :incident)
        end
        for options in ((; mode_threads = 0), (; l_max = 1, m_max = 2),
            (; n_r = 0), (; n_theta = 1), (; R = 0.07), (; m_max = -1))
            @test_throws ArgumentError fem(body, PressureRelease(), k; options...)
        end
        # Coarse smoke check of finite arithmetic at the upper benchmark frequency.
        # This mesh does not establish spatial convergence at 200 kHz.
        for boundary in (PressureRelease(), GasFilled(0.00129, 0.23))
            high = fem(body, boundary, 2pi * 200000 / 1477.4;
                n_r = 4, n_theta = 12)
            @test isfinite(scattering_amplitude(high))
            @test isfinite(pressure(high, (0.12, 0.02, 0.01)))
            legacy = fem(body, boundary, high.k; n_r = 4, n_theta = 12,
                m_max = AcousticScattering._default_mode_count(high.k * body.a))
            @test scattering_amplitude(high)≈scattering_amplitude(legacy) rtol=1e-9 atol=1e-12
            @test pressure(high, (0.12, 0.02, 0.01))≈
            pressure(legacy, (0.12, 0.02, 0.01)) rtol=1e-9 atol=1e-12
        end
        @test_throws ArgumentError fem(body, PressureRelease(), k;
            method = :volume, retain_fields = false)
        @test_throws ArgumentError fem(body, PressureRelease(), k;
            method = :volume, mode_threads = 2)
    end

    @testset "Spheroid azimuthal cutoff and axial symmetry" begin
        body = Spheroid(0.07, 0.01)
        k = 2pi * 38000 / 1477.4
        legacy = AcousticScattering._default_mode_count(k * body.a)
        for boundary in (Rigid(), PressureRelease(), GasFilled(0.00129, 0.23))
            options = (; n_r = 12, n_theta = 48, incidence_angle = pi / 3)
            selected = fem(body, boundary, k; options...)
            reference = fem(body, boundary, k; options..., m_max = legacy)
            @test length(diagnostics(selected).systems) <
                  length(diagnostics(reference).systems) / 2
            @test diagnostics(selected).systems[1].l_max ==
                  diagnostics(reference).systems[1].l_max
            for angle in (pi / 3, 2pi / 3, pi)
                @test scattering_amplitude(selected; angle, azimuth = 0.4)≈
                scattering_amplitude(reference; angle, azimuth = 0.4) rtol=1e-9 atol=1e-12
            end
            @test pressure(selected, (0.10, 0.02, 0.01))≈
            pressure(reference, (0.10, 0.02, 0.01)) rtol=1e-9 atol=1e-12
            if boundary isa FluidFilled
                @test pressure(selected, (0.02, 0.002, 0.0); field = :interior)≈
                pressure(reference, (0.02, 0.002, 0.0); field = :interior) rtol=1e-9 atol=1e-12
            end
            for incidence_angle in (0.0, pi, Float64(pi))
                axial_options = (; n_r = 8, n_theta = 24, incidence_angle)
                axial = fem(body, boundary, k; axial_options...)
                reference_axial = fem(body, boundary, k; axial_options..., m_max = 4)
                @test length(diagnostics(axial).systems) == 1
                @test scattering_amplitude(axial) ≈ scattering_amplitude(reference_axial) rtol = 1e-10
                @test scattering_amplitude(axial; angle = 0.7, azimuth = 0.0) ≈
                      scattering_amplitude(axial; angle = 0.7, azimuth = 1.2) rtol = 1e-12
            end
        end
        # The gas default keeps interior angular orders that exterior-only
        # scaling would omit, without exceeding the previous default budget.
        k_high = 2pi * 200000 / 1477.4
        gas = GasFilled(0.00129, 0.23)
        gas_order = AcousticScattering._spheroid_meridian_default_mmax(body, gas, k_high, pi /
                                                                                          2)
        @test gas_order > k_high * body.b / gas.soundspeed_contrast
        @test gas_order < AcousticScattering._default_mode_count(k_high * body.a) / 2
    end

    @testset "Gas resonance azimuthal convergence" begin
        body = Spheroid(0.07, 0.01)
        boundary = GasFilled(0.00129, 0.23)
        # Discrete m=1 resonance on this mesh, not a mesh-converged frequency.
        peak_frequency = 10323.64408764604
        responses = Float64[]
        for frequency in peak_frequency .+ (-1.0, 0.0, 1.0)
            k = 2pi * frequency / 1477.4
            options = (; n_r = 40, n_theta = 160, retain_fields = false)
            selected = fem(body, boundary, k; options...)
            reference = fem(body, boundary, k; options...,
                m_max = AcousticScattering._default_mode_count(k * body.a))
            for (angle, azimuth) in ((pi / 2, pi), (pi / 2, 0.0), (0.7, 0.4))
                @test scattering_amplitude(selected; angle, azimuth)≈
                scattering_amplitude(reference; angle, azimuth) rtol=1e-9 atol=1e-12
            end
            push!(responses, sqrt(sum(abs2, selected.data.p_modes[2])))
        end
        @test responses[2] > 1.5max(responses[1], responses[3])
    end

    @testset "Higher-order spheroid public controls" begin
        body = Spheroid(0.07, 0.01)
        k = 2pi * 1000 / 1477.4
        for boundary in (PressureRelease(), GasFilled(0.00129, 0.23))
            options = (; order = 6, n_r = 4, n_theta = 16, m_max = 5, l_max = 12)
            retained = fem(body, boundary, k; options...)
            compact = fem(body, boundary, k; options..., retain_fields = false,
                mode_threads = 2)
            @test scattering_amplitude(retained) ≈ scattering_amplitude(compact) rtol = 1e-12
            @test scattering_amplitude(compact; angle = -0.7, azimuth = 0.4) ≈
                  scattering_amplitude(compact; angle = 0.7, azimuth = 0.4 + pi) rtol = 1e-12
            @test diagnostics(compact).equatorial_symmetry
            @test isempty(compact.data.fields)
            @test_throws ArgumentError pressure(compact, (0.09, 0.0, 0.0))
            @test pressure(compact, (0.09, 0.0, 0.0); field = :incident) ≈ 1
            angles = [0.3, pi / 2, 2.7]
            phis = [0.0, 0.7, pi]
            pattern = bistatic_map(compact, angles, phis)
            @test pattern.target_strength ≈ [target_strength(retained; angle, azimuth)
                   for angle in angles, azimuth in phis]
            @test pressure(retained, (0.04, 0.011, 0.002)) ≈
                  pressure(retained, (-0.04, 0.011, 0.002)) rtol = 1e-12
            if boundary isa FluidFilled
                # Nodal incident lifting must preserve total-pressure continuity.
                surface = (0.0, body.b, 0.0)
                @test pressure(retained, surface; field = :total) ≈
                      pressure(retained, surface; field = :interior) rtol = 1e-10
            end
            adaptive = fem(body, boundary, k; order = 6, m_max = 5, l_max = 12,
                adaptive = true, target_tol = 0.005, retain_fields = false)
            report = diagnostics(adaptive)
            @test report.refinement.converged
            @test report.refinement.change_db <= 0.005
            @test report.solver_options.n_r > 4
            @test abs(target_strength(adaptive) - target_strength(retained)) < 0.01
        end
        for order in (2, 4, 8)
            sol = fem(Spheroid(0.01, 0.02), PressureRelease(), k;
                order, n_r = 3, n_theta = 12, m_max = 2, l_max = 8,
                incidence_angle = pi / 3)
            @test isfinite(target_strength(sol))
            @test !diagnostics(sol).equatorial_symmetry
        end
        for options in ((; order = 0), (; order = 9), (; order = 6, n_r = 0),
            (; order = 6, n_theta = 1), (; order = 6, max_refinements = 0),
            (; order = 6, target_tol = 0.0), (; order = 6, mode_threads = 0),
            (; order = 6, m_max = 2, l_max = 1), (; order = 6, R = 0.07))
            @test_throws ArgumentError fem(body, PressureRelease(), k; options...)
        end
        unresolved = @test_logs (:warn, r"did not reach target_tol") fem(
            body, PressureRelease(), k; order = 2, n_r = 2, n_theta = 8,
            m_max = 2, l_max = 8, adaptive = true, target_tol = 1e-12,
            max_refinements = 1, retain_fields = false)
        @test !diagnostics(unresolved).refinement.converged
        @test diagnostics(unresolved).refinement.change_db > 1e-12

        # Exhausted significant modes must not force negligible modes through
        # every remaining mesh when exhausted modes dominate the estimate.
        budget_limited = @test_logs (:warn, r"did not reach target_tol") fem(
            body, GasFilled(0.00129, 0.23), k; order = 3, n_r = 1, n_theta = 4,
            m_max = 8, l_max = 12, adaptive = true, target_tol = 1e-8,
            max_refinements = 2, retain_fields = false)
        budget_report = diagnostics(budget_limited)
        @test !budget_report.refinement.converged
        @test minimum(budget_report.refinement.mode_refinements) == 1
        @test maximum(budget_report.refinement.mode_refinements) == 2
        @test length(budget_report.systems) < 9 * 3

        gas=GasFilled(0.00129, 0.23)
        mixed=fem(body, gas, k; order = 3, n_r = 1, n_theta = 4, m_max = 4, l_max = 12,
            adaptive = true, target_tol = 0.005)
        report=diagnostics(mixed)
        @test report.refinement.converged
        @test minimum(report.refinement.mode_n_r)<maximum(report.refinement.mode_n_r)
        uniform=fem(body, gas, k; order = 3, n_r = report.solver_options.n_r,
            n_theta = report.solver_options.n_theta, m_max = 4, l_max = 12)
        @test abs(target_strength(mixed)-target_strength(uniform))<0.01
        for point in ((0.04, 0.011, 0.002), (0.01, 0.002, 0.0), (0.0, -0.1, 0.0))
            @test pressure(mixed, point)≈pressure(uniform, point) rtol=0.002 atol=1e-6
        end
    end

    @testset "Structural spheroidal shell FEM" begin
        shell = Shell(Spheroid(0.035, 0.007), 0.0005)
        boundary = Shelled(0.32, 2565.0, 70e9)
        k_shell = 2pi * 12000.0 / 1477.3

        vacuum = fem(shell, boundary, 1026.8, 1477.3, 0.0, 1.0, k_shell;
            method = :thin, incidence_angle = 0.0, n_eta = 9)
        filled = fem(shell, boundary, 1026.8, 1477.3, 1077.3, 1575.0, k_shell;
            method = :thin, incidence_angle = 0.0, n_eta = 9)
        general = fem(shell, boundary, 1026.8, 1477.3, 1077.3, 1575.0, k_shell;
            method = :general, incidence_angle = 0.0, m_max = 0,
            n_eta = 7, n_t = 2)

        for (method, solution) in ((:thin, vacuum), (:thin, filled),
            (:general, general))
            @test solution isa FEMSolution
            @test solution.method == method
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
        end
        @test_throws ArgumentError fem(shell, boundary,
            1026.8, 1477.3, 0.0, 1.0, k_shell;
            method = :thin, incidence_angle = pi / 4, n_eta = 9)
    end
end
