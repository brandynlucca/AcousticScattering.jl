using AcousticScattering
using Test
using LinearAlgebra: dot, norm, det, I

@testset "Generated geometries" begin
    AS = AcousticScattering
    volume(surface) = sum(q.weight * dot(q.coords, q.normal) for q in surface.data) / 3
    cases = (
        (Cone(0.5, 1.0), pi / 12),
        (Bicone(0.5, 1.0; waist = 0.3), pi / 12),
        (Frustum(0.5, 0.3, 1.0), pi * (0.25 + 0.15 + 0.09) / 3),
        (Frustum((0.5, 0.3), (0.3, 0.2), 1.0), pi * (2 * 0.15 + 0.1 + 0.09 + 2 * 0.06) / 6),
        (Frustum((0.3, 0.6), (0.5, 0.2), 1.0), pi * (2 * 0.18 + 0.06 + 0.3 + 2 * 0.1) / 6),
        (Ellipsoid(1.0, 0.6, 0.4), 4pi * 0.24 / 3),
        (Superellipsoid(1.0, 0.6, 0.4), 4pi * 0.24 / 3),
        (Torus(1.0, (0.2, 0.3)), 2pi^2 * 0.2 * 0.3))
    for (body, exact_volume) in cases
        @testset "$(typeof(body))" begin
            surface = mesh(
                body; method = :full, resolution = 0.2, mesh_order = 2, qorder = 4)
            @test surface.body === body
            @test volume(surface) ≈ exact_volume rtol = 0.005
            @test norm(sum(q.weight * q.normal for q in surface.data)) < 0.005
        end
    end

    @testset "Meridian topology and corner preservation" begin
        body = FrustumChain([-1.0, 0.0, 1.0], [0.3, 0.6, 0.4])
        surface = mesh(body; resolution = 24)
        @test AS.element_count(surface) == 24
        @test all(
            i -> any(
                j -> surface.data.z[j] == body.x[i] &&
                     surface.data.rho[j] == body.radius[i],
                eachindex(surface.data.z)),
            eachindex(body.x))
        reversed = RevolvedBody(reverse(body.x), reverse(body.radius))
        @test reversed.x == body.x
        @test reversed.radius == body.radius
        @test AS._meridian_location(surface.data, (0.0, 0.0, 0.0)) == :inside
        @test AS._meridian_location(surface.data, (0.0, 1.0, 0.0)) == :outside
        @test AS._meridian_location(surface.data, (0.0, 0.6, 0.0)) == :on
        ring = RevolvedBody([-0.2, 0.2, 0.2, -0.2], [0.7, 0.7, 1.3, 1.3]; closed = true)
        ringmesh = mesh(ring; resolution = 16)
        @test AS._meridian_location(ringmesh.data, (0.0, 0.0, 0.0)) == :outside
        @test AS._meridian_location(ringmesh.data, (0.0, 1.0, 0.0)) == :inside
        full = mesh(ring; method = :full, resolution = 0.5, qorder = 4)
        @test volume(full) ≈ pi * (1.3^2 - 0.7^2) * 0.4 rtol = 0.005
        sampled = RevolvedBody(t -> (cospi(t), sinpi(t)); n = 16)
        @test AS.element_count(mesh(sampled; resolution = 32)) == 32
        @test_throws ArgumentError mesh(body; resolution = 2)
        @test_throws ArgumentError mesh(body; resolution = 24.5)
        @test_throws ArgumentError RevolvedBody([1.0, 0, -1], [0.0, 0.5, 0.1])
        @test_throws ArgumentError RevolvedBody([1.0, 0, 0, -1], [0.0, 0.5, 0.5, 0.0])
        @test_throws ArgumentError RevolvedBody([0.0, 1, 0, 1], [1.0, 2, 2, 1]; closed = true)
        @test_throws ArgumentError Torus(0.3, 0.3)
        @test_throws ArgumentError Torus(0.2, 0.3)
        @test_throws ArgumentError FrustumChain([0.0, 1, 2], [1.0, 0, 1])
        @test_throws ArgumentError FrustumChain([0.0, 0], [1.0, 1])
        @test_throws ArgumentError Ellipsoid(1, -1, 1)
        @test_throws ArgumentError Superellipsoid(1, 1, 1; axial_exponent = 0)
        @test_throws ArgumentError mesh(Ellipsoid(1, 2, 3); method = :axisymmetric, resolution = 20)
    end

    @testset "Section frames and superellipsoid surface" begin
        centers = [-1.0 0 1; 0 0.1 0.3; 0 0 0.1]
        body = LoftedBody(centers, [0.4 0.5 0.3; 0.25 0.3 0.2]; twists = [0.0, 0.1, 0.2])
        @test all(
            frame -> isapprox(frame' * frame, I; atol = 1e-14) &&
                     isapprox(det(frame), 1; atol = 1e-14),
            body.frames)
        @test volume(mesh(body; resolution = 0.4, qorder = 4)) > 0
        @test_throws ArgumentError LoftedBody(zeros(3, 2), ones(2, 2))
        @test_throws ArgumentError LoftedBody(centers, ones(2, 2))
        @test_throws ArgumentError LoftedBody(centers, ones(2, 3); directions = zeros(3, 3))
        body = Superellipsoid(
            1.0, 0.6, 0.4; axial_exponent = 0.8, equatorial_exponent = 1.2)
        surface = mesh(body; resolution = 0.3, mesh_order = 1, qorder = 2)
        for p in AS.Inti.nodes(AS.Inti.mesh(surface.data))
            level = abs(p[1])^(2 / 0.8) +
                    (abs(p[2] / 0.6)^(2 / 1.2) + abs(p[3] / 0.4)^(2 / 1.2))^(1.2 / 0.8)
            @test level ≈ 1 atol = 1e-12
        end
    end
end

@testset "Mesh generation" begin
    bodies = (
        Sphere(1.0),
        Spheroid(1.0, 0.5),
        Spheroid(0.5, 1.0),
        Cylinder(0.5, 1.0)
    )
    for body in bodies
        surface = mesh(body; method = :axisymmetric, resolution = 8)
        @test surface isa Mesh
        @test surface.method == :axisymmetric
        @test AcousticScattering.element_count(surface) == 8
        @test length(AcousticScattering.coordinates(surface)) == 8
        @test length(AcousticScattering.normals(surface)) == 8
    end

    for (name, body) in (("sphere", Sphere(1.0)),
        ("spheroid", Spheroid(1.0, 0.5)),
        ("cylinder", Cylinder(0.5, 1.0)),
        ("bent cylinder", Cylinder(0.5, 1.0; radius_curvature = 3.0)))
        @testset "$name full surface" begin
            surface = mesh(body; method = :full, resolution = 0.8,
                mesh_order = 1, qorder = 2)
            @test surface isa Mesh
            @test surface.method == :full
            @test AcousticScattering.element_count(surface) > 0
        end
    end

    ellipsoid = mesh(; semiaxes = (1.0, 0.5, 0.4), resolution = 1.5,
        mesh_order = 1, qorder = 2)
    @test ellipsoid isa Mesh
    @test ellipsoid.method == :full

    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    supplied = mesh(nodes, triangles; qorder = 2, units = :cm)
    @test supplied isa Mesh
    @test supplied.body.input_units == :cm
    @test maximum(abs, supplied.body.nodes) ≈ 0.01

    mktempdir() do directory
        path = joinpath(directory, "surface.msh")
        generated = mesh(; qorder = 2) do g
            g.model.add("small sphere")
            g.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
            g.model.occ.synchronize()
            g.option.setNumber("Mesh.MeshSizeMin", 1.5)
            g.option.setNumber("Mesh.MeshSizeMax", 1.5)
            g.model.mesh.generate(2)
            g.write(path)
        end
        @test generated isa Mesh
        loaded = mesh(path; qorder = 2)
        @test loaded isa Mesh
        @test AcousticScattering.element_count(loaded) ==
              AcousticScattering.element_count(generated)
    end

    @test_throws ArgumentError mesh(Sphere(1.0))
    @test_throws ArgumentError mesh(Sphere(1.0); resolution = 8, k = 1.0)
    @test_throws ArgumentError mesh(Sphere(1.0); resolution = 8, method = :unknown)
    @test_throws ArgumentError mesh(Cylinder(0.5, 1.0; radius_curvature = 3.0);
        resolution = 8)
    @test_throws ArgumentError mesh(; semiaxes = (1.0, -1.0, 1.0))
    @test_throws ArgumentError mesh("missing-surface.msh")

    @testset "Invalid geometry and mesh inputs" begin
        @test_throws ArgumentError Sphere(0.0)
        @test_throws ArgumentError Cylinder(-1.0, 1.0)
        @test_throws ArgumentError Cylinder(1.0, 1.0; radius_curvature = 0.0)
        @test_throws ArgumentError Cylinder(1.0, 1.0; endcap_depth = -0.1)
        @test_throws ArgumentError Shell(Cylinder(1.0, 1.0), 0.1)
        @test_throws ArgumentError Shell(Sphere(1.0), 0.0)
        @test_throws ArgumentError Shell(Sphere(1.0), 1.0)

        @test_throws ArgumentError mesh(; semiaxes = (1.0, 1.0, 1.0),
            center = (0.0, NaN, 0.0))
        @test_throws ArgumentError mesh(; semiaxes = (1.0, 1.0, 1.0),
            resolution = 0.0)
        @test_throws ArgumentError mesh(; semiaxes = (1.0, 1.0, 1.0),
            tip_ratio = 0.0)
        @test_throws ArgumentError mesh(; semiaxes = (1.0, 1.0, 1.0),
            mesh_order = 4)
        @test_throws ArgumentError mesh(; semiaxes = (1.0, 1.0, 1.0),
            rotation = (axis = (0.0, 0.0, 0.0), angle = 0.0))

        @test_throws ArgumentError mesh(nodes[1:2, :], triangles)
        @test_throws ArgumentError mesh(nodes, triangles[1:2, :])
        @test_throws ArgumentError mesh(nodes, triangles .+ 4)
        @test_throws ArgumentError mesh(nodes, triangles; labels = [1])
        @test_throws ArgumentError mesh(nodes, triangles; units = :feet)
    end
end
