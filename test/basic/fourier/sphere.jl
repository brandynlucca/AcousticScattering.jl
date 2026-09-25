using AcousticScattering
using Test

@testset "Sphere" begin
    body = Sphere(1.0)
    k = 0.5
    options = (; mapping_order = 2, continuation_steps = 1,
        m_max = 2, n_max = 2)
    for (name, boundary) in (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.05, 1.02)))
        @testset "$name" begin
            solution = fourier(body, boundary, k;
                incidence_angle = pi / 3, options...)
            reference = modal(body, boundary, k; m_max = 2)
            @test solution isa FMSolution
            @test solution.body === body
            @test diagnostics(solution).admissible
            @test scattering_amplitude(solution) isa ComplexF64
            @test scattering_amplitude(solution) ≈
                  scattering_amplitude(reference) rtol = 0.01
        end
    end
end

@testset "Truncation-consistency guard" begin
    body = Sphere(1.0)
    options = (; mapping_order = 2, continuation_steps = 1)
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
        @testset "$(nameof(typeof(boundary)))" begin
            solution = @test_logs fourier(body, boundary, 0.5;
                incidence_angle = pi / 3, m_max = 6, n_max = 6, options...)
            report = diagnostics(solution)
            @test report.converged
            @test report.convergence < report.convergence_tolerance
            @test solution.b_check isa Matrix{ComplexF64}
            @test size(solution.b_check) == (5, 5)

            @test_logs (:warn, r"not converged") fourier(body, boundary, 3.0;
                incidence_angle = pi / 3, m_max = 4, n_max = 4, options...)
            unresolved = @test_logs (:warn, r"not converged") match_mode=:any fourier(
                body, boundary, 3.0; incidence_angle = pi / 3, m_max = 4, n_max = 4,
                options...)
            @test !diagnostics(unresolved).converged
        end
    end

    tiny = fourier(body, Rigid(), 0.5; incidence_angle = pi / 3, m_max = 2, n_max = 2,
        options...)
    @test tiny.b_check === nothing
    @test isnan(diagnostics(tiny).convergence)
    @test diagnostics(tiny).converged === nothing
end
