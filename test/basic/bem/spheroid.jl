using AcousticScattering
using Test

@testset "Spheroid" begin
    k = 0.5
    for (shape, body) in ((:prolate, Spheroid(1.2, 1.0)),
        (:oblate, Spheroid(1.0, 1.2)))
        @testset "$shape" begin
            @test body.kind == shape
            for (name, boundary) in (("rigid", Rigid()),
                ("pressure-release", PressureRelease()),
                ("fluid-filled", FluidFilled(1.2, 1.1)))
                @testset "$name" begin
                    oblique = (shape == :prolate && boundary isa PressureRelease) ||
                              (shape == :oblate && boundary isa FluidFilled)
                    incidence_angle = oblique ? pi / 3 : 0.0
                    solution = bem(body, boundary, k; incidence_angle, n = 8, m_max = 1)
                    @test solution isa BEMSolution
                    @test solution.method == :axisymmetric
                    @test scattering_amplitude(solution) isa ComplexF64
                    @test isfinite(target_strength(solution))
                end
            end
        end
    end

    full = bem(Spheroid(1.2, 1.0), Rigid(), k;
        method = :full, meshsize = 1.0, mesh_order = 1, qorder = 2,
        compression = (method = :none,))
    @test full isa BEMSolution
    @test full.method == :full
    @test isfinite(scattering_amplitude(full))
end
