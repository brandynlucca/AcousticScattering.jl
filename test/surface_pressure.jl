using AcousticScattering
using LinearAlgebra: BLAS, dot, norm
using Test

BLAS.set_num_threads(4)
const AS = AcousticScattering

function compare_surface_pressure(actual, expected)
    for (got, wanted) in zip(actual, expected)
        @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        @test abs(20log10(abs(got/wanted))) < 0.01
    end
end

@testset "Curved surface point locations" begin
    for order in (1, 2, 3)
        surface = mesh(; semiaxes = (1.0, 1.0, 1.0), center = (0.3, -0.2, 0.1),
            resolution = 0.5, mesh_order = order, qorder = 4)
        patches = AS._region_patches(surface.data, 0)
        @test AS._surface_location(patches, (0.3, -0.2, 0.1)) === :inside
        @test AS._surface_location(patches, (2.0, 0.0, 0.0)) === :outside
        for i in (23, 71, 131)
            q = surface.data[i]
            @test AS._surface_location(patches, q.coords) === :on
            @test AS._surface_location(patches, q.coords + 1e-8*q.normal) === :outside
            @test AS._surface_location(patches, q.coords - 1e-8*q.normal) === :inside
        end
        for node in eachcol(surface.body.nodes[:, 1:3])
            @test AS._surface_location(patches, node) === :on
        end
    end
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    faces = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    surface = mesh(nodes, faces)
    patches = AS._region_patches(surface.data, 0)
    @test AS._surface_location(patches, (0.1, 0.1, 0.1)) === :inside
    @test AS._surface_location(patches, (0.5, 0.5, 0.5)) === :outside
    for point in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.2, 0.2, 0.0))
        @test AS._surface_location(patches, point) === :on
    end
    torus = mesh(; qorder = 4) do g
        g.model.add("closed ring")
        g.model.occ.addTorus(0, 0, 0, 1.0, 0.25)
        g.model.occ.synchronize()
        g.option.setNumber("Mesh.MeshSizeMin", 0.25)
        g.option.setNumber("Mesh.MeshSizeMax", 0.25)
        g.model.mesh.generate(2)
        g.model.mesh.setOrder(3)
    end
    patches = AS._region_patches(torus.data, 0)
    @test AS._surface_location(patches, (0.0, 0.0, 0.0)) === :outside
    @test AS._surface_location(patches, (1.0, 0.0, 0.0)) === :inside
    @test AS._surface_location(patches, (1.5, 0.0, 0.0)) === :outside
    q = torus.data[23]
    @test AS._surface_location(patches, q.coords) === :on
    @test AS._surface_location(patches, q.coords + 1e-8*q.normal) === :outside
    @test AS._surface_location(patches, q.coords - 1e-8*q.normal) === :inside
end

@testset "Supplied sphere pressure" begin
    center = [0.3, -0.2, 0.1]
    beta, alpha = pi/3, 0.4
    direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
    local_points = [(1.01, 0.0, 0.0), (0.0, 0.606, 0.808), (-0.72, 0.0, 0.96)]
    points = [Tuple(collect(p) + center) for p in local_points]
    reference_points = [(dot(direction, p), sqrt(norm(p)^2-dot(direction, p)^2), 0.0)
                        for p in local_points]
    for boundary in (Rigid(), FluidFilled(1.2, 1.1))
        k = boundary isa FluidFilled ? 0.3 : 1.0
        h = boundary isa FluidFilled ? 0.25 : 0.4
        generated = mesh(; semiaxes = (1.0, 1.0, 1.0), center = Tuple(center),
            resolution = h, mesh_order = 3, qorder = 5)
        surface = mesh(generated.body.nodes, hcat(generated.body.connectivity...); qorder = 5)
        options = boundary isa FluidFilled ? (; condition_limit = 0) :
                  (; compression = (method = :hmatrix, tol = 1e-7),
            gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 600))
        solution = bem(surface, boundary, k; incidence_angle = beta,
            incidence_azimuth = alpha, options...)
        reference = modal(Sphere(1.0), boundary, k)
        phase = cis(k*dot(direction, center))
        expected = phase .* pressure(reference, reference_points; field = :scattered)
        @testset "$(typeof(boundary)) field" begin
            compare_surface_pressure(pressure(solution, points; field = :scattered), expected)
        end
        @test diagnostics(solution).relative_residual < 1e-8
        @test pressure(solution, points) ≈
              pressure(solution, points; field = :incident) +
              pressure(solution, points; field = :scattered)
        @test_throws ArgumentError pressure(solution, center; field = :scattered)
        @test isempty(pressure(solution, NTuple{3, Float64}[]))
        if boundary isa FluidFilled
            compare_surface_pressure([pressure(solution, center)],
                [phase*pressure(reference, (0.0, 0.0, 0.0))])
            q = surface.data[23]
            traces = [Tuple(q.coords), Tuple(q.coords + 1e-8*q.normal)]
            compare_surface_pressure(pressure(solution, traces),
                [pressure(solution, Tuple(q.coords); field = :interior),
                    pressure(solution, Tuple(q.coords - 1e-8*q.normal))])
        else
            sources = mesh(surface.body.nodes, hcat(surface.body.connectivity...); qorder = 1)
            full = mfs(surface, boundary, k; source_mesh = sources, offset = 0.35,
                incidence_angle = beta, incidence_azimuth = alpha, condition_limit = 0)
            compare_surface_pressure(pressure(full, points; field = :scattered), expected)
            q = surface.data[23]
            close = [Tuple(q.coords), Tuple(q.coords + 1e-8*q.normal)]
            compare_surface_pressure(pressure(solution, close; field = :scattered),
                pressure(full, close; field = :scattered))
            @test_throws ArgumentError pressure(full, center)
            @test pressure(full, first(points)) ≈ first(pressure(full, points))
        end
    end
end

@testset "Closed bent cylinder pressure" begin
    body = Cylinder(0.5, 1.0; radius_curvature = 2.0, endcap_depth = 0.5)
    surface = mesh(body; method = :full, resolution = 0.25, mesh_order = 3, qorder = 5)
    rigid_surface = mesh(
        body; method = :full, resolution = 0.25, mesh_order = 3, qorder = 5)
    sources = mesh(body; method = :full, resolution = 0.25, mesh_order = 3, qorder = 1)
    coarse_sources = mesh(
        body; method = :full, resolution = 0.28, mesh_order = 3, qorder = 1)
    anchors = (AS.SVector(0.8, 0.1, 0.25), AS.SVector(0.0, -0.5, 0.0),
        AS.SVector(-0.5, 0.2, 0.45))
    samples = [surface.data[argmin(norm(node.coords-anchor) for node in surface.data)]
               for anchor in anchors]
    q = first(samples)
    points = [Tuple(node.coords + d*node.normal) for node in samples
              for d in (0.0, 1e-8, 1e-6, 1e-4, 0.01, 0.1)]
    append!(points, [(1.5, 0.2, 0.3), (-1.2, 0.5, 0.5)])
    for boundary in (Rigid(),)
        grid = rigid_surface
        solution = bem(
            grid, boundary, 0.5; incidence_angle = pi/3, incidence_azimuth = 0.4,
            compression = (method = :none,),
            gmres_kwargs = (; reltol = 1e-9, restart = 400, maxiter = 2400))
        reference = mfs(surface, boundary, 0.5; source_mesh = sources, offset = 0.2,
            incidence_angle = pi/3, incidence_azimuth = 0.4, condition_limit = 0)
        expected = pressure(reference, points; field = :scattered)
        @testset "$(typeof(boundary)) field" begin
            compare_surface_pressure(pressure(solution, points; field = :scattered), expected)
        end
        coarse = mfs(surface, boundary, 0.5; source_mesh = coarse_sources, offset = 0.2,
            incidence_angle = pi/3, incidence_azimuth = 0.4, condition_limit = 0)
        @testset "$(typeof(boundary)) source refinement" begin
            compare_surface_pressure(pressure(coarse, points; field = :scattered), expected)
        end
        @test diagnostics(solution).converged
        for solved in (solution, reference)
            @test_throws ArgumentError pressure(solved, (0.0, 0.0, 0.0))
            @test_throws ArgumentError pressure(solved, Tuple(q.coords - 1e-8*q.normal))
            @test pressure(solved, points) ≈
                  pressure(solved, points; field = :incident) +
                  pressure(solved, points; field = :scattered)
            direction, distance = [0.36, 0.48, 0.8], 1e6
            farpoint = Tuple(distance .* direction)
            far = pressure(solved, farpoint; field = :scattered)*distance*cis(-0.5distance)
            @test isapprox(far, scattering_amplitude(solved; direction); rtol = 1e-4, atol = 1e-12)
            @test pressure(solved, [first(points), farpoint]; field = :scattered) ≈
                  [pressure(solved, p; field = :scattered)
                   for p in (first(points), farpoint)]
        end
    end
    boundary = FluidFilled(1.2, 1.1)
    fluid_surface = mesh(body; method = :full, resolution = 0.28, mesh_order = 3, qorder = 5)
    solution = bem(fluid_surface, boundary, 0.5; incidence_angle = pi/3,
        incidence_azimuth = 0.4, condition_limit = 0)
    refined = bem(body, boundary, 0.5; method = :full, meshsize = 0.25,
        mesh_order = 3, qorder = 5, incidence_angle = pi/3,
        incidence_azimuth = 0.4, condition_limit = 0)
    exterior = [points[6], points[end - 1], points[end]]
    compare_surface_pressure(pressure(solution, exterior; field = :scattered),
        pressure(refined, exterior; field = :scattered))
    inside = [(0.0, 0.0, 0.0), (0.3, 0.1, 0.15)]
    compare_surface_pressure(pressure(solution, inside), pressure(refined, inside))
    q = fluid_surface.data[argmin(norm(node.coords-first(anchors))
    for node in fluid_surface.data)]
    compare_surface_pressure(
        pressure(solution, [Tuple(q.coords), Tuple(q.coords + 1e-8*q.normal)]),
        [pressure(solution, Tuple(q.coords); field = :interior),
            pressure(solution, Tuple(q.coords - 1e-8*q.normal))])
    @test pressure(solution, inside) ≈ pressure(solution, inside; field = :interior)
end
