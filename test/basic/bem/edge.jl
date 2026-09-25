using AcousticScattering
using Test

include(joinpath(@__DIR__, "..", "..", "helpers", "structured_cylinder.jl"))

@testset "Edge quadrature on a flat-ended cylinder" begin
    nodes, triangles = structured_cylinder(16, 3, 3)
    surface = mesh(nodes, triangles; qorder = 2)
    edge = (formulation = :cbie, compression = (method = :none,),
        correction = (method = :edge,))
    far = (1.0, 0.2, 0.5)
    near = [(0.51, 0.1, 0.5), (0.0, 0.51, 0.5), (0.2, 0.0, 1.02)]
    for (boundary, rtol) in ((PressureRelease(), 0.1), (Rigid(), 0.05))
        solution = bem(surface, boundary, 4.0; incidence_angle = 0.4, edge...)
        reference = bem(surface, boundary, 4.0; incidence_angle = 0.4,
            compression = (method = :none,))
        @test scattering_amplitude(solution)≈scattering_amplitude(reference) rtol=rtol
        @test pressure(solution, far)≈pressure(reference, far) rtol=0.1
        @test all(isfinite, pressure(solution, near))
        @test pressure(solution, near; field = :scattered) ≈
              pressure(solution, near) - pressure(solution, near; field = :incident)
    end
    @test_throws ArgumentError bem(surface, Rigid(), 4.0;
        correction = (method = :edge,), compression = (method = :none,))
    @test_throws ArgumentError bem(surface, FluidFilled(1.2, 1.1), 4.0;
        correction = (method = :edge,))
end
