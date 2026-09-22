using AcousticScattering
using Test

@time "Complex acoustic radial FEM against modal spheres" @testset "Complex acoustic radial FEM against modal spheres" begin
    body = Sphere(1.0)
    for boundary in (Rigid(), PressureRelease()), k in (0.4, 4.0), order in (1, 2)
        solution = fem(body, boundary, k; R = 3.0, n_elements = 200, order)
        reference = modal(body, boundary, k)
        amplitude = scattering_amplitude(solution)
        @test amplitude ≈ scattering_amplitude(reference) rtol = 0.001
        @test abs(target_strength(solution) - target_strength(reference)) < 0.01
        @test target_strength(solution) == target_strength(amplitude)
        mode = first(solution.data.modes)
        @test mode.order == order
        @test length(mode.exterior.pressure) == length(mode.exterior.radii) ==
              order * 200 + 1
        @test extrema(mode.exterior.radii) == (1.0, 3.0)
        @test all(isfinite, mode.exterior.pressure)
        @test mode.interior === nothing
        @test_throws ArgumentError scattering_amplitude(solution; angle = 0.3)
    end

    for (boundary, ks) in ((FluidFilled(1.04, 1.04), (0.4, 1.6)),
        (FluidFilled(10.0, 0.5), (0.4, 1.6)),
        (GasFilled(0.0012, 0.23), (0.012, 0.0138, 0.016)))
        for k in ks
            solution = fem(body, boundary, k; n_elements_int = 200, n_elements_ext = 100)
            reference = modal(body, boundary, k)
            @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.001
            @test abs(target_strength(solution) - target_strength(reference)) < 0.01
            mode = first(solution.data.modes)
            @test length(mode.interior.pressure) == length(mode.interior.radii) == 201
            @test length(mode.exterior.pressure) == length(mode.exterior.radii) == 101
            @test extrema(mode.interior.radii) == (0.0, 1.0)
            @test extrema(mode.exterior.radii) == (1.0, 1.2)
            @test all(isfinite, mode.interior.pressure)
            @test isfinite(mode.interior.interface_flux)
            @test isfinite(mode.exterior.interface_flux)
        end
    end
end

@time "Adaptive radial FEM retains the final complex solution" @testset "Adaptive radial FEM retains the final complex solution" begin
    body, k = Sphere(1.0), 1.0
    for (boundary, element_options) in ((Rigid(), (; order = 1)),
        (PressureRelease(), (; order = 1)), (FluidFilled(1.2, 1.1), (;)))
        solution = fem(body, boundary, k; adaptive = true, m_max = 3,
            n_elements_start = 16, max_n_elements = 128, target_tol = 0.001,
            element_options...)
        report = diagnostics(solution)
        @test report.refinement.converged
        n = report.refinement.n_elements
        options = boundary isa FluidFilled ?
                  (; n_elements_int = n, n_elements_ext = n) : (; n_elements = n)
        fixed = fem(body, boundary, k; m_max = 3, options..., element_options...)
        @test scattering_amplitude(solution) == scattering_amplitude(fixed)
        @test solution.data.modes == fixed.data.modes
        @test scattering_amplitude(solution) ≈
              scattering_amplitude(modal(body, boundary, k; m_max = 3)) rtol = 0.001
        @test target_strength(solution) == AcousticScattering.radial_fem_target_strength(
            boundary, k, body.radius, 1.2; m_max = 3, options..., element_options...)

        failed = @test_logs (:warn, r"did not converge") fem(body, boundary, k;
            adaptive = true, m_max = 3, n_elements_start = 4, max_n_elements = 8,
            target_tol = 0.0, element_options...)
        @test !diagnostics(failed).refinement.converged
        options = boundary isa FluidFilled ?
                  (; n_elements_int = 8, n_elements_ext = 8) : (; n_elements = 8)
        finest = fem(body, boundary, k; m_max = 3, options..., element_options...)
        @test failed.data.modes == finest.data.modes
        @test scattering_amplitude(failed) == scattering_amplitude(finest)
    end
end

@time "Radial FEM complex frequency sweeps" @testset "Radial FEM complex frequency sweeps" begin
    body, boundary = Sphere(0.01), Rigid()
    solve = k -> fem(body, boundary, k; order = 2, n_elements = 40)
    sweep = frequency_sweep(solve, [12000.0, 38000.0, 80000.0], 1500.0)
    references = scattering_amplitude.([modal(body, boundary, k) for k in sweep.k])
    @test sweep.amplitudes == scattering_amplitude.(solve.(sweep.k))
    @test sweep.amplitudes ≈ references rtol = 1e-5
    @test sweep.target_strength == target_strength.(sweep.amplitudes)
end
