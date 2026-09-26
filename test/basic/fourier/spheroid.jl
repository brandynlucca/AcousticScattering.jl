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

@testset "Spheroid defaults follow aspect ratio" begin
    defaults = AcousticScattering._spheroid_fourier_defaults
    moderate = Spheroid(2.0, 1.0)
    @test defaults(moderate, nothing, (;)) == (12, (;))
    @test defaults(moderate, 20, (;)) == (20, (;))
    for body in (Spheroid(4.0, 1.0), Spheroid(1.0, 4.0))
        order, options = defaults(body, nothing, (;))
        @test order == 80
        @test options.m_max == 12
        @test defaults(body, 60, (;)) == (60, (; m_max = 10))
        @test defaults(body, nothing, (; m_max = 7))[2] == (; m_max = 7)
        @test defaults(body, nothing, (; n_max = 7, rtol = 1e-4))[2] ==
              (; n_max = 7, rtol = 1e-4)
    end
end
