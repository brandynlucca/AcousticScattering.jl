using AcousticScattering
using Test

@testset "Mesh generation" begin
    bodies = (
        Sphere(1.0),
        Spheroid(1.0, 0.5),
        Spheroid(0.5, 1.0),
        Cylinder(0.5, 1.0),
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
