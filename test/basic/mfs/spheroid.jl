using AcousticScattering
using Test

@testset "Spheroid" begin
    for (shape, body) in ((:prolate, Spheroid(1.2, 1.0)),
        (:oblate, Spheroid(1.0, 1.2)))
        @testset "$shape" begin
            @test body.kind == shape
            for (name, boundary) in (("rigid", Rigid()),
                ("pressure-release", PressureRelease()),
                ("fluid-filled", FluidFilled(1.2, 1.1)))
                @testset "$name" begin
                    solution = mfs(body, boundary, 0.5;
                        incidence_angle = 0.0, n = 12, offset = 0.2,
                        condition_limit = 0)
                    @test solution isa MFSSolution
                    @test scattering_amplitude(solution) isa ComplexF64
                    @test isfinite(target_strength(solution))
                end
            end
        end
    end
end
