using AcousticScattering
using Test

@testset "Spheroid" begin
    @testset "Prolate reference and supported boundaries" begin
        body = Spheroid(0.07, 0.01)
        k = 2pi * 38000.0 / 1477.3
        angle = deg2rad(28.0)
        rigid = kirchhoff(body, Rigid(), k; incidence_angle = angle)
        soft = kirchhoff(body, PressureRelease(), k; incidence_angle = angle)
        fluid = FluidFilled(1.05, 1.02)
        filled = kirchhoff(body, fluid, k; incidence_angle = angle)
        reflection = (fluid.density_contrast * fluid.soundspeed_contrast - 1) /
                     (fluid.density_contrast * fluid.soundspeed_contrast + 1)

        @test body.kind == :prolate
        for solution in (rigid, soft, filled)
            @test solution isa KirchhoffSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
        end
        @test target_strength(rigid) ≈ -62.21 atol = 0.01
        @test scattering_amplitude(soft) ≈ -scattering_amplitude(rigid)
        @test scattering_amplitude(filled) ≈ reflection * scattering_amplitude(rigid)
    end

    @testset "Oblate near-sphere limit" begin
        radius = 0.01
        body = Spheroid(radius - 0.0001, radius + 0.0001)
        k = 2pi * 12000.0 / 1477.3
        angle = pi / 3
        @test body.kind == :oblate

        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
            solution = kirchhoff(body, boundary, k; incidence_angle = angle)
            sphere_reference = kirchhoff(Sphere(radius), boundary, k)
            @test solution isa KirchhoffSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
            @test target_strength(solution) ≈ target_strength(sphere_reference) atol = 0.3
        end
    end
end
