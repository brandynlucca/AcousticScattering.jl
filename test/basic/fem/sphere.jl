using AcousticScattering
using Test

@testset "Sphere" begin
    body = Sphere(0.01)
    k = 2pi * 12000.0 / 1477.4

    @testset "Radial FEM: all supported boundaries" begin
        elastic = ElasticLayer(2.0, 2.0, 1.0)
        fluid_layer = FluidLayer(1.1, 1.02)
        cases = (
            ("rigid", Rigid()),
            ("pressure-release", PressureRelease()),
            ("fluid-filled", FluidFilled(1.05, 1.02)),
            ("solid elastic", SolidElastic(2.0, 2.0, 1.0)),
            ("elastic shell", Shelled(elastic, FluidInterior(1.0, 1.0), 0.8)),
            ("fluid shell, fluid interior",
                Shelled(fluid_layer, FluidInterior(1.05, 1.02), 0.8)),
            ("fluid shell, vacuum interior",
                Shelled(fluid_layer, VacuumInterior(), 0.8)),
        )
        for (name, boundary) in cases
            @testset "$name" begin
                mesh_options = if boundary isa FluidFilled
                    (; n_elements_int = 20, n_elements_ext = 20)
                elseif boundary isa Rigid
                    (; n_elements = 12, order = 2)
                else
                    (; n_elements = 20)
                end
                solution = fem(body, boundary, k; mesh_options..., m_max = 3)
                reference = modal(body, boundary, k; m_max = 3)
                @test solution isa FEMSolution
                @test solution.method == :radial
                @test scattering_amplitude(solution) isa ComplexF64
                @test target_strength(solution) isa Float64
                @test isfinite(target_strength(solution))
                @test abs(target_strength(solution) - target_strength(reference)) < 3.0
                if boundary isa Rigid
                    @test_throws ArgumentError scattering_amplitude(solution; angle = 0.0)
                end
            end
        end
    end

    @testset "Meridian FEM" begin
        for (name, boundary) in (("rigid", Rigid()),
            ("pressure-release", PressureRelease()))
            @testset "$name" begin
                solution = fem(body, boundary, k; method = :meridian,
                    R = 0.02, n_r = 6, n_theta = 16, l_max = 5)
                @test solution isa FEMSolution
                @test solution.method == :meridian
                @test target_strength(solution) isa Float64
                @test isfinite(target_strength(solution))
                @test -200.0 < target_strength(solution) < 20.0
                if boundary isa Rigid
                    @test_throws ArgumentError target_strength(solution; angle = 0.0)
                    @test_throws ArgumentError scattering_amplitude(solution)
                end
            end
        end
        @test_throws ArgumentError fem(body, Rigid(), k; method = :unknown)
    end

    @testset "Structural spherical shell FEM" begin
        shell = Shell(body, 0.002)
        boundary = Shelled(0.33, 2700.0, 70e9)
        solution = fem(shell, boundary, 1000.0, 1500.0, 1000.0, 1500.0, 100.0;
            method = :general, incidence_angle = 0.0, m_max = 0,
            n_eta = 7, n_t = 2)
        @test solution isa FEMSolution
        @test solution.method == :general
        @test scattering_amplitude(solution) isa ComplexF64
        @test isfinite(target_strength(solution))
    end
end
