using AcousticScattering
using Test
using LinearAlgebra: norm

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

@time "Fluid rim singular self integral" @testset "Fluid rim singular self integral" begin
    AS = AcousticScattering
    quad = mesh(Cylinder(0.5, 1.0); method = :full,
        resolution = 0.2, mesh_order = 3, qorder = 7).data
    _, flux = AS._edge_fluid_patches(quad, 1.2)
    target = AS.SVector(0.5, -0.44, -0.237)
    index = argmin(norm(q.coords-target) for q in quad)
    patch = only(filter(p -> index in p.columns, flux))
    @test !iszero(patch.exponent)
    anchor = patch.refs[findfirst(==(index), patch.columns)]
    requested = AS._edge_options((; rtol = 1e-8, atol = 1e-11))
    actual = AS._edge_self_single_layer(0.5, patch, anchor,
        AS._edge_self_options(requested, patch))
    reference = AS._edge_self_single_layer(0.5, patch, anchor,
        AS._edge_options((; rtol = 1e-13, atol = 1e-16)))
    @test norm(actual-reference)/norm(reference) < 1e-4
end
