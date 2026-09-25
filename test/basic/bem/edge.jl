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

@testset "Fluid edge quadrature, weak-contrast identity" begin
    # A body with the exterior's density and sound speed scatters nothing and leaves the pressure equal to the incident wave, whatever the mesh.
    nodes, triangles = structured_cylinder(16, 1, 1)
    surface = mesh(nodes, triangles; qorder = 2)
    solution = bem(surface, FluidFilled(1.0, 1.0), 4.0; incidence_angle = 0.4,
        formulation = :cbie, correction = (method = :edge,))
    points = [(1.0, 0.2, 0.5), (0.51, 0.1, 0.5), (0.1, 0.05, 0.5)]
    @test abs(scattering_amplitude(solution)) < 2e-3
    @test pressure(solution, points)≈pressure(solution, points; field = :incident) atol=1e-2
end

@testset "Edge quadrature building blocks" begin
    AS = AcousticScattering
    nodes, triangles = structured_cylinder(16, 1, 1)
    quad = mesh(nodes, triangles; qorder = 2).data

    @test AS._edge_power_difference(0.0, 0.0, 0.5) ≈ -2
    @test AS._edge_power_difference(0.0, 0.3, 0.5) == 0
    @test AS._edge_power_difference(0.4, 0.2, 0.9)≈(0.4^0.9 - 0.4^0.2) / 0.7
    values = [1.0, -2.0, 0.5, 3.0]
    coefficients = AS._edge_cubic_coefficients(values)
    @test [AS._edge_polynomial(coefficients, t) for t in (0, 1 / 3, 2 / 3, 1)]≈values

    pressure_patches, flux_patches = AS._edge_fluid_patches(quad, 1.2)
    @test any(!iszero(patch.exponent) for patch in flux_patches)
    for patch in [pressure_patches; flux_patches]
        count = length(patch.refs)
        basis = hcat([patch.basis(ref) for ref in patch.refs]...)
        @test basis≈AS.Diagonal(ones(count)) atol=1e-6
    end

    patch = first(AS._edge_patches(quad))
    center = patch.el(AS.SVector(1 / 3, 1 / 3))
    normal = AS.Inti._normal(
        AS.Inti.jacobian(patch.el, AS.SVector(1 / 3, 1 / 3)), patch.orientation)
    x = center + 0.1patch.radius * normal
    @test AS._edge_nearest_reference(patch, x)≈AS.SVector(1 / 3, 1 / 3) atol=1e-8
    outside = patch.el(AS.SVector(1.0, 0.0)) + 0.05patch.radius * normal
    @test AS._edge_nearest_reference(patch, outside)≈AS.SVector(1.0, 0.0) atol=1e-8

    options = AS._edge_options((method = :edge,))
    for double_layer in (false, true)
        near = AS._edge_near_single_layer(4.0, patch, x, options; double_layer)
        regular = AS._edge_regular_single_layer(
            4.0, patch, x, options, double_layer ? :source : nothing)
        @test near≈regular rtol=1e-4
    end
    for z in (0.005, 0.0099, 0.05)
        @test AS._edge_double_remainder(z)≈(1 - im * z) * cis(z) - 1 rtol=1e-6
    end
    uv = AS.SVector(0.3, 0.6)
    @test AS._edge_sample_normal(patch, uv)≈normal atol=1e-10
end
