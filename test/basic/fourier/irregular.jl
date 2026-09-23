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
