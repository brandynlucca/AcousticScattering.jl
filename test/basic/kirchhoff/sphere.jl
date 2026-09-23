using AcousticScattering
using Test

@testset "Sphere" begin
    body = Sphere(0.01)
    k = 2pi * 12000.0 / 1477.3

    @testset "Rigid, pressure-release, and fluid-filled" begin
        rigid = kirchhoff(body, Rigid(), k)
        soft = kirchhoff(body, PressureRelease(), k)
        fluid = FluidFilled(1.05, 1.02)
        filled = kirchhoff(body, fluid, k)
        reflection = (fluid.density_contrast * fluid.soundspeed_contrast - 1) /
                     (fluid.density_contrast * fluid.soundspeed_contrast + 1)

        for solution in (rigid, soft, filled)
            @test solution isa KirchhoffSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test target_strength(solution) isa Float64
        end
        @test target_strength(rigid) ≈ -52.12 atol = 0.01
        @test scattering_amplitude(soft) ≈ -scattering_amplitude(rigid)
        @test scattering_amplitude(filled) ≈ reflection * scattering_amplitude(rigid)
        @test diagnostics(rigid) === nothing
        @test_throws ArgumentError target_strength(rigid; angle = 0.0)
        @test_throws ArgumentError scattering_amplitude(rigid; angle = 0.0)
    end

    @testset "Fluid shells with vacuum and fluid interiors" begin
        rigid_amplitude = scattering_amplitude(kirchhoff(body, Rigid(), k))
        ratio = 0.8
        thickness = body.radius * (1 - ratio)
        phase = cis(2k * thickness)
        layer = FluidLayer(1.0, 1.0)

        vacuum = kirchhoff(body, Shelled(layer, VacuumInterior(), ratio), k)
        @test vacuum isa KirchhoffSolution
        @test scattering_amplitude(vacuum) isa ComplexF64
        @test scattering_amplitude(vacuum) ≈ -phase * rigid_amplitude

        interior = FluidInterior(1.05, 1.02)
        filled = kirchhoff(body, Shelled(layer, interior, ratio), k)
        impedance = interior.density_contrast * interior.soundspeed_contrast
        inner_reflection = (impedance - 1) / (impedance + 1)
        @test filled isa KirchhoffSolution
        @test scattering_amplitude(filled) isa ComplexF64
        @test scattering_amplitude(filled) ≈ inner_reflection * phase * rigid_amplitude
    end
end
