using AcousticScattering
using Test

@testset "Cylinder" begin
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    body = Cylinder(0.01, 0.1)

    @testset "Rigid, pressure-release, and fluid-filled" begin
        cases = (
            (Rigid(), pi / 2, -30.5254506539618),
            (PressureRelease(), pi / 2, -28.4384596592856),
            (FluidFilled(1028.9 / 1026.8, 1480.3 / c_water),
                pi / 2, -81.847628998419),
            (Rigid(), 1.2, -54.5632700811014),
        )
        for (boundary, incidence_angle, expected_ts) in cases
            solution = modal(body, boundary, k; incidence_angle, m_max = 20)
            @test solution isa ModalSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test target_strength(solution) isa Float64
            @test target_strength(solution) ≈ expected_ts atol = 1e-5
        end
    end

    @testset "Solid elastic and elastic shell" begin
        c_ref = 1477.3
        k_ref = 2pi * 38000.0 / c_ref
        solid = SolidElastic(2800.0 / 1026.8, 6398.0 / c_ref, 3122.0 / c_ref)
        solution = modal(Cylinder(0.005, 0.04), solid, k_ref; m_max = 40)
        @test solution isa ModalSolution
        @test scattering_amplitude(solution) isa ComplexF64
        @test target_strength(solution) ≈ -43.3581252581 atol = 1e-5

        # Existing ECMS regression: a vanishing cavity approaches the solid cylinder.
        elastic_body = Cylinder(0.02, 0.2)
        wall = ElasticLayer(7900.0 / 1026.8, 5610.0 / c_water, 3060.0 / c_water)
        shell = Shelled(wall, FluidInterior(1.0, 1.0), 0.001)
        shell_solution = modal(elastic_body, shell, k; m_max = 25)
        solid_solution = modal(elastic_body,
            SolidElastic(wall.density_contrast, wall.speed_longitudinal_contrast,
                wall.speed_transversal_contrast), k; m_max = 25)
        @test shell_solution isa ModalSolution
        @test scattering_amplitude(shell_solution) isa ComplexF64
        @test abs(target_strength(shell_solution) - target_strength(solid_solution)) < 0.01
    end

    @testset "Bent-cylinder coherence correction" begin
        c_ref = 1477.3
        k_ref = 2pi * 38000.0 / c_ref
        radius, length = 1e-3, 10.5e-3
        straight = Cylinder(radius, length)
        bent = Cylinder(radius, length; radius_curvature = 1.5length)
        fluid = FluidFilled(1.0357, 1.0279)

        # Golden value from the existing BCMS test, plus the API's complex-amplitude contract.
        fluid_bent = modal(bent, fluid, k_ref)
        @test target_strength(fluid_bent) ≈ -101.762375463093 atol = 1e-4

        scale = AcousticScattering.equivalent_length_fresnel(
            k_ref, length, bent.radius_curvature) / length
        shell = Shelled(ElasticLayer(2.0, 2.0, 1.0), FluidInterior(1.0, 1.0), 0.8)
        for boundary in (Rigid(), PressureRelease(), fluid,
            SolidElastic(2.0, 2.0, 1.0), shell)
            bent_solution = modal(bent, boundary, k_ref; m_max = 4)
            straight_solution = modal(straight, boundary, k_ref; m_max = 4)
            @test bent_solution isa ModalSolution
            @test scattering_amplitude(bent_solution) isa ComplexF64
            @test scattering_amplitude(bent_solution) ≈
                  scale * scattering_amplitude(straight_solution) rtol = 1e-10
        end
    end
end
