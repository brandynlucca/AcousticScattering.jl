using AcousticScattering
using Test

@testset "Sphere" begin
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = Sphere(0.01)

    @testset "Rigid, pressure-release, and fluid-filled" begin
        cases = (
            (Rigid(), -49.088291, 1e-4),
            (PressureRelease(), -44.997865, 1e-4),
            (FluidFilled(1028.9 / 1026.8, 1480.3 / c_water), -94.278687, 1e-3)
        )
        for (boundary, expected_ts, tolerance) in cases
            solution = modal(sphere, boundary, k)
            @test solution isa ModalSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test target_strength(solution) isa Float64
            @test target_strength(solution) ≈ expected_ts atol = tolerance
        end
        @test GasFilled === FluidFilled
    end

    @testset "Solid elastic and elastic shell" begin
        rho_ext = 1026.8
        rho_shell, shear, lambda = 2700.0, 2.6e10, 5.3e10
        c_longitudinal = sqrt((lambda + 2shear) / rho_shell)
        c_transverse = sqrt(shear / rho_shell)
        wall = ElasticLayer(rho_shell / rho_ext, c_longitudinal / c_water,
            c_transverse / c_water)
        cases = (
            (Sphere(0.019),
                SolidElastic(14900.0 / rho_ext,
                    6853.0 / c_water, 4171.0 / c_water),
                13, -42.3768620168836),
            (Sphere(0.05), Shelled(wall, FluidInterior(1.0, 1.0), 0.048 / 0.05),
                20, -28.236014616259297),
            (Sphere(0.05),
                Shelled(wall,
                    FluidInterior(1.24 / rho_ext, 345.0 / c_water), 0.048 / 0.05),
                20, -34.93644485774581)
        )
        for (body, boundary, m_max, expected_ts) in cases
            solution = modal(body, boundary, k; m_max)
            @test solution isa ModalSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test target_strength(solution) ≈ expected_ts atol = 1e-4
        end
    end

    @testset "Fluid shell with fluid or vacuum interior" begin
        # Matching the layer to its adjacent fluid removes that interface.
        fluid = FluidFilled(1.05, 1.02)
        filled_shell = Shelled(FluidLayer(1.05, 1.02),
            FluidInterior(1.05, 1.02), 0.9)
        filled = modal(sphere, filled_shell, k; m_max = 8)
        homogeneous = modal(sphere, fluid, k; m_max = 8)
        @test filled isa ModalSolution
        @test scattering_amplitude(filled) isa ComplexF64
        @test scattering_amplitude(filled) ≈ scattering_amplitude(homogeneous) rtol = 1e-8
        @test isfinite(pressure(filled, (0.0095, 0.0, 0.0); field = :shell))

        vacuum_shell = Shelled(FluidLayer(1.0, 1.0), VacuumInterior(), 0.9)
        vacuum = modal(sphere, vacuum_shell, k; m_max = 8)
        smaller_soft_sphere = modal(Sphere(0.009), PressureRelease(), k; m_max = 8)
        @test vacuum isa ModalSolution
        @test scattering_amplitude(vacuum) isa ComplexF64
        @test scattering_amplitude(vacuum) ≈
              scattering_amplitude(smaller_soft_sphere) rtol = 1e-8
    end

    @testset "Layered viscous-elastic shell (monopole only)" begin
        core = FluidInterior(71 / 1027, 325 / 1500)
        wall = ElasticLayer(1040 / 1027, 1520 / 1500,
            sqrt(0.2e6 / 1040) / 1500)
        flesh = ViscousLayer(1500.0, 1040 / 1027, 1510 / 1500,
            (4 / 3) / 1040, 1 / 1040)
        boundary = Shelled(LayeredMaterial(flesh, wall, 1.02 / 1.1), core, 1 / 1.1)
        body = Sphere(1.1e-3)
        k_vesm = 2pi * 23100.0 / 1500.0
        solution = modal(body, boundary, k_vesm; m_max = 0)
        @test solution isa ModalSolution
        @test scattering_amplitude(solution) isa ComplexF64
        @test target_strength(solution) ≈ -40.689 atol = 0.02
        @test_throws ArgumentError modal(body, boundary, k_vesm; m_max = 1)
    end

    @testset "Invalid modal parameters" begin
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02),
            SolidElastic(2.0, 2.0, 1.0),
            Shelled(FluidLayer(1.05, 1.02), FluidInterior(1.0, 1.0), 0.9),
            Shelled(FluidLayer(1.05, 1.02), VacuumInterior(), 0.9),
            Shelled(ElasticLayer(2.0, 2.0, 1.0), FluidInterior(1.0, 1.0), 0.9))
            @test_throws ArgumentError modal(sphere, boundary, -k; m_max = 0)
            @test_throws ArgumentError modal(sphere, boundary, k; m_max = -1)
        end
    end
end
