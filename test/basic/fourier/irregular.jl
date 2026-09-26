using AcousticScattering
using Test

@testset "Irregular" begin
    body = Irregular(1.0, [0.02], [0.0])
    options = (; mapping_order = 4, continuation_steps = 2,
        m_max = 2, n_max = 3)
    for (name, boundary) in (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.05, 1.02)))
        @testset "$name" begin
            solution = fourier(body, boundary, 0.5;
                incidence_angle = pi / 3, options...)
            @test solution isa FMSolution
            @test solution.body === body
            @test diagnostics(solution).admissible
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
        end
    end
    @test_throws ArgumentError fourier(body, Rigid(), -0.5; options...)
    @test_throws ArgumentError fourier(body, Rigid(), 0.0; options...)
    @test_throws ArgumentError fourier(body, Rigid(), Inf; options...)
end

@testset "Irregular profile is a body of revolution" begin
    @test_throws ArgumentError Irregular(1.0, [0.02], [0.01])
    # The profile is mirrored about the axis, so a sine-like function fits a cosine series.
    body = Irregular(theta -> 1 + 0.1 * sin(theta), 6)
    @test all(iszero, body.rs)
    @test AcousticScattering.profile_radius(body, pi / 3) ≈
          AcousticScattering.profile_radius(body, -pi / 3)
    body = Irregular(theta -> 1 + 0.1 * cos(3theta) + 0.05 * cos(2theta), 8)
    mapping = AcousticScattering.solve_mapping(body, 8; continuation_steps = 4)
    for w in (0.0, pi)
        @test abs(AcousticScattering.mapping_surface(mapping, w)[2]) < 1e-10
    end
end
