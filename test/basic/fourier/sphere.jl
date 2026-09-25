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

@testset "Boundary-matching building blocks on a sphere" begin
    AS = AcousticScattering
    body = Irregular(1.0, Float64[], Float64[])
    mapping = AS.solve_mapping(body, 2; continuation_steps = 1)
    k, theta0 = 0.5, pi / 3
    @test AS.profile_radius(body, 0.7) ≈ 1.0
    @test AS.mapping_theta(mapping, 0.7)≈0.7 atol=1e-8

    for matrices in (AS._boundary_matrices, AS._rigid_boundary_matrices)
        R, Q = matrices(mapping, k, 1; n_max = 3)
        @test size(R) == size(Q) == (3, 3)
        @test maximum(abs, R - AS.Diagonal(AS.diag(R))) < 1e-8
    end
    S, Sp = AS._interior_boundary_matrices(mapping, k / 1.02, 1; n_max = 3)
    @test size(S) == size(Sp) == (3, 3)

    soft = AS.solve_pressure_release(mapping, k, theta0; m_max = 2, n_max = 3)
    hard = AS.solve_rigid(mapping, k, theta0; m_max = 2, n_max = 3)
    fluid = AS.solve_fluid(mapping, k, theta0, 1.05, 1.02; m_max = 2, n_max = 3)
    for b in (soft, hard, fluid)
        @test size(b) == (4, 3) && all(isfinite, b)
    end
    @test_throws ArgumentError AS.solve_pressure_release(mapping, -1.0, theta0)
    @test_throws ArgumentError AS.solve_rigid(mapping, -1.0, theta0)
    @test_throws ArgumentError AS.solve_fluid(mapping, -1.0, theta0, 1.05, 1.02)
    @test_throws ArgumentError fourier(
        Sphere(1.0), Shelled(FluidLayer(1.1, 1.02),
            VacuumInterior(), 0.9),
        k; mapping_order = 2, continuation_steps = 1)
end
