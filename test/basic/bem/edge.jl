using AcousticScattering
using Test

@testset "Edge quadrature on a flat-ended cylinder" begin
    surface = mesh(Cylinder(0.5, 1.0); method = :full, resolution = 0.9, mesh_order = 1,
        qorder = 2)
    edge = (formulation = :cbie, compression = (method = :none,),
        correction = (method = :edge,))
    points = [(1.0, 0.2, 0.1), (0.0, 0.0, 1.2)]
    for (boundary, rtol) in ((PressureRelease(), 0.08), (Rigid(), 0.03))
        solution = bem(surface, boundary, 0.5; incidence_angle = 0.4, edge...)
        reference = bem(surface, boundary, 0.5; incidence_angle = 0.4,
            compression = (method = :none,))
        @test scattering_amplitude(solution)≈scattering_amplitude(reference) rtol=rtol
        @test pressure(solution, points)≈pressure(reference, points) rtol=0.05
        @test pressure(solution, points; field = :scattered) ≈
              pressure(solution, points) - pressure(solution, points; field = :incident)
    end
    @test_throws ArgumentError bem(surface, Rigid(), 0.5;
        correction = (method = :edge,), compression = (method = :none,))
    @test_throws ArgumentError bem(surface, FluidFilled(1.2, 1.1), 0.5;
        correction = (method = :edge,))
    @test_throws ArgumentError mesh(Cylinder(0.5, 1.0); method = :full,
        resolution = 0.9, mesh_order = 4, qorder = 2)
end
