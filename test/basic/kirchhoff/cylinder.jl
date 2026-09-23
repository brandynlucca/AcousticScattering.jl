using AcousticScattering
using Test

@testset "Cylinder" begin
    body = Cylinder(0.01, 0.07)
    k = 2pi * 38000.0 / 1477.3

    @testset "Straight-cylinder reference and supported boundaries" begin
        angle = deg2rad(88.0)
        rigid = kirchhoff(body, Rigid(), k; incidence_angle = angle)
        soft = kirchhoff(body, PressureRelease(), k; incidence_angle = angle)
        fluid = FluidFilled(1.05, 1.02)
        filled = kirchhoff(body, fluid, k; incidence_angle = angle)
        reflection = (fluid.density_contrast * fluid.soundspeed_contrast - 1) /
                     (fluid.density_contrast * fluid.soundspeed_contrast + 1)

        for solution in (rigid, soft, filled)
            @test solution isa KirchhoffSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
        end
        @test target_strength(rigid) ≈ -31.29 atol = 0.01
        @test scattering_amplitude(soft) ≈ -scattering_amplitude(rigid)
        @test scattering_amplitude(filled) ≈ reflection * scattering_amplitude(rigid)

        endon = kirchhoff(body, Rigid(), k; incidence_angle = 0.0)
        @test abs(scattering_amplitude(endon)) ≈ k * body.radius^2 / 2 atol = 1e-9
    end

    @testset "Bent-cylinder route" begin
        bent = Cylinder(0.01, 0.07; radius_curvature = 0.2)
        rigid = kirchhoff(bent, Rigid(), k)
        soft = kirchhoff(bent, PressureRelease(), k)
        fluid = FluidFilled(1.05, 1.02)
        filled = kirchhoff(bent, fluid, k)
        reflection = (fluid.density_contrast * fluid.soundspeed_contrast - 1) /
                     (fluid.density_contrast * fluid.soundspeed_contrast + 1)

        @test rigid isa KirchhoffSolution
        @test isfinite(target_strength(rigid))
        @test scattering_amplitude(soft) ≈ -scattering_amplitude(rigid)
        @test scattering_amplitude(filled) ≈ reflection * scattering_amplitude(rigid)
    end
end
