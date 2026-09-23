using AcousticScattering
using Test

@testset "Spheroid" begin
    options = (; mapping_order = 8, continuation_steps = 2,
        m_max = 2, n_max = 3)
    for (shape, body) in ((:prolate, Spheroid(1.2, 1.0)),
        (:oblate, Spheroid(1.0, 1.2)))
        @testset "$shape" begin
            @test body.kind == shape
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
        end
    end
end
