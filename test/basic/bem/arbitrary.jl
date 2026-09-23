using AcousticScattering
using Test

@testset "Arbitrary surfaces and fluid regions" begin
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    surface = mesh(nodes, triangles; qorder = 2)
    solution = bem(surface, PressureRelease(), 0.3;
        compression = (method = :none,))
    @test solution isa BEMSolution
    @test solution.method == :full
    @test isfinite(scattering_amplitude(solution))
    @test_throws ArgumentError bem(surface, PressureRelease(), -0.3;
        compression = (method = :none,))
    @test_throws ArgumentError bem(surface, PressureRelease(), 0.3;
        formulation = :unknown, compression = (method = :none,))
    @test_throws ArgumentError bem(surface, PressureRelease(), 0.3;
        correction = (method = :edge,), compression = (method = :none,))

    sphere_surface = mesh(Sphere(1.0); method = :full, resolution = 1.0,
        mesh_order = 2, qorder = 2)
    region = bem([sphere_surface], [FluidFilled(1.0, 1.0)], 0.3;
        condition_limit = 0)
    @test region isa BEMSolution
    @test diagnostics(region).interface_count == 1
    @test abs(scattering_amplitude(region)) < 1e-8
    @test pressure(region, (1.5, 0.0, 0.0)) ≈
          pressure(region, (1.5, 0.0, 0.0); field = :incident) atol = 1e-7
    @test pressure(region, (0.0, 0.0, 0.0); field = :interior, region = 1) ≈
          1.0 atol = 5e-5
end
