using AcousticScattering
using Test

@time "Fluid edge quadrature, weak-contrast identity" @testset "Fluid edge quadrature, weak-contrast identity" begin
    # A body with the exterior's density and sound speed scatters nothing and leaves the pressure equal to the incident wave on any mesh, so these checks do not depend on the mesh Gmsh returns.
    surface = mesh(Cylinder(0.5, 1.0); method = :full, resolution = 0.9, mesh_order = 1,
        qorder = 2)
    solution = bem(surface, FluidFilled(1.0, 1.0), 0.5; incidence_angle = 0.4,
        formulation = :cbie, correction = (method = :edge,))
    points = [(1.0, 0.2, 0.1), (0.1, 0.05, 0.0), (0.51, 0.1, 0.1)]
    @test abs(scattering_amplitude(solution)) < 1e-4
    @test pressure(solution, points)≈pressure(solution, points; field = :incident) atol=1e-3
end
