@testset "Full BEM fluid transmission" begin
    beta, alpha = pi / 3, 0.4
    incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
    observations = ((pi, -incident), (0.0, incident),
        (pi / 2, [0.0, -sin(alpha), cos(alpha)]))

    @testset "Material contrasts and irregular frequencies" begin
        for (g, h, k) in ((0.0012, 0.23, 1.0), (0.01, 0.5, 1.0),
            (1.05, 1.02, 1.0), (1000.0, 2.0, 1.0), (1e6, 2.0, 1.0),
            (0.0012, 0.23, 2.0815759778181), (1000.0, 2.0, Float64(pi)))
            boundary = FluidFilled(g, h)
            solution = bem(Sphere(1.0), boundary, k; method = :full,
                incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4)
            report = diagnostics(solution)
            @test report.formulation == :muller
            @test report.unknown_count == 2report.quadrature_nodes
            @test report.relative_residual < 1e-8
            @test report.scaled_relative_residual < 1e-11
            @test report.conditioning == :not_computed
            @test report.condition_number === nothing
            for (angle, direction) in observations
                actual = scattering_amplitude(solution; direction)
                reference = scattering_amplitude(modal(Sphere(1.0), boundary, k; angle))
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
        end
    end

    @testset "Spheroids and quadrature refinement" begin
        body = Spheroid(1.5, 1.0)
        for boundary in (FluidFilled(0.0012, 0.23), FluidFilled(1000.0, 2.0))
            solution = bem(body, boundary, 1.0; method = :full,
                incidence_angle = pi / 3, meshsize = 0.4)
            reference = scattering_amplitude(modal(body, boundary, 1.0;
                incidence_angle = pi / 3, m_max = 6, n_max = 20))
            actual = scattering_amplitude(solution)
            @test abs(target_strength(actual) - target_strength(reference)) < 0.1
            @test abs(actual - reference) / abs(reference) < 0.01
        end
        boundary = FluidFilled(0.0012, 0.23)
        coarse = bem(Sphere(1.0), boundary, 1.0; method = :full, meshsize = 0.4, qorder = 4)
        fine = bem(Sphere(1.0), boundary, 1.0; method = :full, meshsize = 0.4, qorder = 5)
        @test abs(target_strength(fine) - target_strength(coarse)) < 0.1
        @test scattering_amplitude(fine) ≈ scattering_amplitude(coarse) rtol = 0.01
    end

    @testset "Gas resonance and mesh refinement" begin
        boundary = FluidFilled(0.0012, 0.23)
        for k in (0.012, 0.0138, 0.014, 0.016)
            solution = bem(Sphere(1.0), boundary, k; method = :full,
                incidence_angle = beta, incidence_azimuth = alpha,
                meshsize = 0.35, mesh_order = 3, qorder = 5)
            @test diagnostics(solution).relative_residual < 1e-7
            @test diagnostics(solution).scaled_relative_residual < 1e-11
            for (angle, direction) in observations
                actual = scattering_amplitude(solution; direction)
                reference = scattering_amplitude(modal(Sphere(1.0), boundary, k; angle))
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
            if k == 0.0138
                coarse = bem(Sphere(1.0), boundary, k; method = :full,
                    incidence_angle = beta, incidence_azimuth = alpha,
                    meshsize = 0.4, mesh_order = 3, qorder = 5)
                @test abs(target_strength(solution) - target_strength(coarse)) < 0.1
                @test scattering_amplitude(solution) ≈ scattering_amplitude(coarse) rtol = 0.01
            end
        end
    end

    @testset "Equilibration, conventional system, and length scaling" begin
        boundary = FluidFilled(0.0012, 0.23)
        balanced = bem(Sphere(1.0), boundary, 1.0; method = :full,
            meshsize = 0.8, condition_limit = 700)
        unscaled = bem(Sphere(1.0), boundary, 1.0; method = :full,
            meshsize = 0.8, equilibrate = false, condition_limit = 0)
        report = diagnostics(balanced)
        @test report.conditioning == :svd
        @test report.scaled_condition_number < report.condition_number / 100
        @test scattering_amplitude(balanced) ≈ scattering_amplitude(unscaled) rtol = 1e-8
        @test !diagnostics(unscaled).equilibrate
        @test diagnostics(unscaled).condition_number === nothing
        @test diagnostics(unscaled).scaled_condition_number === nothing

        for form in (:muller, :cbie)
            solution = bem(Sphere(0.01), boundary, 100.0; method = :full,
                meshsize = 0.004, formulation = form)
            reference = scattering_amplitude(modal(Sphere(0.01), boundary, 100.0))
            actual = scattering_amplitude(solution)
            @test diagnostics(solution).formulation == form
            @test diagnostics(solution).unknown_count ==
                  (form == :muller ? 2 : 4) * diagnostics(solution).quadrature_nodes
            @test abs(target_strength(actual) - target_strength(reference)) < 0.1
            @test abs(actual - reference) / abs(reference) < 0.01
        end

        invisible = bem(Sphere(1.0), FluidFilled(1.0, 1.0), 1.0;
            method = :full, meshsize = 0.8)
        @test abs(scattering_amplitude(invisible)) < 1e-12
        quad = balanced.data.quad
        low_level = AS.solve_full_bem(boundary, 1.0, quad; incidence_angle = pi / 2)
        @test length(low_level) == 3
        @test low_level[1] ≈ balanced.data.p_scat
        @test low_level[2] ≈ balanced.data.dpdn_scat
        @test_throws ArgumentError AS.solve_full_bem(boundary, 1.0, quad; formulation = :burton_miller)
        @test_throws ArgumentError AS.solve_full_bem(boundary, 1.0, quad; condition_limit = -1)
        for k in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError AS.solve_full_bem(boundary, k, quad)
        end
        for invalid in (FluidFilled(0.0, 1.0), FluidFilled(1.0, -1.0),
            FluidFilled(Inf, 1.0), FluidFilled(1.0, NaN))
            @test_throws ArgumentError AS.solve_full_bem(invalid, 1.0, quad)
        end
    end

    @testset "Independent transmission amplitudes" begin
        cases = (
            (0.0012, 0.23, 0.0138, 0.35, 5, 3,
                (-0.2603525438230707 + 72.4628313282518im,
                    -0.2607320117001016 + 72.46283132858299im,
                    -0.2605422717353993 + 72.46283132841739im)),
            (0.0012, 0.23, 1.0, 0.4, 4, 2,
                (0.08627220103266989 + 0.5718263756162451im,
                    -1.169649453390533 + 0.8439868202224056im,
                    -0.4123286201575636 + 0.7056764613690221im)),
            (1000.0, 2.0, 1.0, 0.4, 4, 2,
                (-0.4684232462649632 + 0.0118836351691168im,
                    0.1742206481610343 + 0.08026589273243463im,
                    -0.2384449751929127 + 0.04496309639283315im)))
        for (g, h, k, meshsize, qorder, mesh_order, references) in cases
            solution = bem(Sphere(1.0), FluidFilled(g, h), k; method = :full,
                incidence_angle = 0.0, meshsize, qorder, mesh_order)
            for (direction, reference) in zip(
                ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]), references)
                actual = scattering_amplitude(solution; direction)
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
        end
    end
end
