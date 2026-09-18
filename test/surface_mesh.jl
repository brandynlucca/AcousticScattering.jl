@testset "Supplied closed surfaces" begin
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    surface = mesh(1000nodes, triangles; units = :mm, labels = [1, 1, 2, 2],
        provenance = "inline tetrahedron")
    @test surface isa Mesh
    @test surface.method == :full
    @test surface.body.nodes == nodes
    @test surface.body.connectivity == collect(eachcol(triangles))
    @test first.(only.(surface.body.labels)) == [1, 1, 2, 2]
    @test surface.body.units == :m
    @test surface.body.input_units == :mm
    @test surface.body.orientation == :outward
    @test surface.body.provenance == "inline tetrahedron"
    @test surface.resolution ≈ sqrt(2)
    @test AS.element_count(surface) == length(AS.coordinates(surface))
    @test all(n -> AS.norm(n) ≈ 1, AS.normals(surface))

    @testset "Invalid surface input" begin
        @test_throws ArgumentError mesh(nodes[1:2, :], triangles)
        @test_throws ArgumentError mesh(nodes, triangles[1:2, :])
        @test_throws ArgumentError mesh(nodes, triangles .+ 1)
        @test_throws ArgumentError mesh(nodes, triangles[:, 1:3])
        @test_throws ArgumentError mesh(nodes, triangles[[1, 3, 2], :])
        @test_throws ArgumentError mesh(nodes, hcat(triangles, triangles[:, 1]))
        @test_throws ArgumentError mesh(nodes, triangles; labels = [1, 2])
        @test_throws ArgumentError mesh(nodes, triangles; labels = [0, 1, 1, 1])
        @test_throws ArgumentError mesh(nodes, triangles; units = :unknown)
        @test_throws ArgumentError mesh(nodes, triangles; qorder = 0)
        @test_throws ArgumentError mesh(nodes .* NaN, triangles)
        @test_throws ArgumentError mesh(zeros(3, 4), triangles)
        reversed = copy(triangles)
        reversed[:, 1] = reversed[[1, 3, 2], 1]
        @test_throws ArgumentError mesh(nodes, reversed)
        @test_throws ArgumentError mesh(hcat(nodes, nodes .+ 3), hcat(triangles, triangles .+
                                                                                 4))
        crossing_nodes = [1.0 0 -1 0 0 2; 0 1 0 -1 0 0; 0 0 0 0 1 0.5]
        crossing_faces = [1 2 3 4 2 3 4 1; 2 3 4 1 1 2 3 4; 5 5 5 5 6 6 6 6]
        @test_throws ArgumentError mesh(crossing_nodes, crossing_faces)
        @test AS.gmsh.isInitialized() == 0
    end

    @testset "Gmsh surfaces against modal solutions" begin
        beta = pi / 3
        incident = [cos(beta), sin(beta), 0.0]
        for body in (Sphere(1.0),) # Spheroid(1.5, 1.0) requires SpheroidalWaves backend, not available locally
            imported = mesh(; provenance = "inline labeled canonical surface") do g
                g.model.add("canonical")
                volume = g.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
                body isa Spheroid &&
                    g.model.occ.dilate([(3, volume)], 0, 0, 0, body.b, body.b, body.a)
                g.model.occ.synchronize()
                surfaces = last.(g.model.getEntities(2))
                g.model.addPhysicalGroup(2, surfaces, 7)
                g.model.setPhysicalName(2, 7, "exterior interface")
                g.option.setNumber("Mesh.MeshSizeMin", 0.4)
                g.option.setNumber("Mesh.MeshSizeMax", 0.4)
                g.model.mesh.generate(2)
                g.model.mesh.setOrder(2)
                g.model.mesh.affineTransform([
                    0.0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
            end
            @test all(==([7 => "exterior interface"]), imported.body.labels)
            @test all(t -> length(t) == 6, imported.body.connectivity)
            for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02), GasFilled(0.0012, 0.23))
                options = boundary isa FluidFilled ? (;) :
                          (; gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
                solution = bem(imported, boundary, 1.0; incidence_angle = beta, options...)
                @test solution.body === imported.body
                @test solution.data.quad === imported.data
                @test diagnostics(solution).relative_residual < 1e-8
                for (angle, direction) in ((pi, -incident), (0.0, incident), (
                    pi / 2, [0.0, 0.0, 1.0]))
                    reference = body isa Sphere ? modal(body, boundary, 1.0; angle) :
                                modal(
                        body, boundary, 1.0; incidence_angle = beta, m_max = 6, n_max = 20,
                        scatter_angle = acos(direction[1]),
                        scatter_azimuth = atan(direction[3], direction[2]))
                    expected = scattering_amplitude(reference)
                    actual = scattering_amplitude(solution; direction)
                    @test abs(target_strength(actual) - target_strength(expected)) < 0.1
                    @test abs(actual - expected) / abs(expected) < 0.01
                end
            end
            reconstructed = mesh(imported.body.nodes, hcat(imported.body.connectivity...))
            @test AS.coordinates(reconstructed) ≈ AS.coordinates(imported)
            @test AS.normals(reconstructed) ≈ AS.normals(imported)
            if body isa Sphere
                nonconforming = hcat(imported.body.connectivity...)
                extra_nodes = hcat(imported.body.nodes, imported.body.nodes[:, nonconforming[4, 1]])
                nonconforming[4, 1] = size(extra_nodes, 2)
                @test_throws ArgumentError mesh(extra_nodes, nonconforming)
                folded = copy(imported.body.nodes)
                folded[:, imported.body.connectivity[1][4]] *= -2
                @test_throws ArgumentError mesh(folded, hcat(imported.body.connectivity...))
            end
        end
    end

    @testset "Nonconvex surface and translation" begin
        torus = mesh(; qorder = 4, provenance = "inline non-axisymmetric torus") do g
            g.model.add("torus")
            volume = g.model.occ.addTorus(0, 0, 0, 1.0, 0.35)
            g.model.occ.dilate([(3, volume)], 0, 0, 0, 1.2, 0.8, 1.0)
            g.model.occ.synchronize()
            g.option.setNumber("Mesh.MeshSizeMin", 0.5)
            g.option.setNumber("Mesh.MeshSizeMax", 0.5)
            g.model.mesh.generate(2)
            g.model.mesh.setOrder(2)
        end
        shift = [0.4, -0.2, 0.3]
        translated = mesh(torus.body.nodes .+ shift, hcat(torus.body.connectivity...); qorder = 4)
        beta, alpha, k = pi / 3, 0.4, 1.0
        incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
        solutions = [bem(m, Rigid(), k; incidence_angle = beta, incidence_azimuth = alpha,
                         compression = (method = :none,),
                         gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
                     for m in (torus, translated)]
        @test all(s -> diagnostics(s).converged, solutions)
        for direction in (-incident, incident, [0.0, 0.0, 1.0])
            original = scattering_amplitude(solutions[1]; direction)
            actual = scattering_amplitude(solutions[2]; direction)
            @test actual ≈ original * cis(k * AS.dot(incident - direction, shift)) rtol = 1e-6
        end
        matched = bem(torus, FluidFilled(1.0, 1.0), k)
        @test abs(scattering_amplitude(matched)) < 1e-11
    end

    @testset "Canonical mesh reuse" begin
        canonical = mesh(
            Sphere(1.0); resolution = 0.4, method = :full, mesh_order = 3, qorder = 4)
        solution = bem(canonical, Rigid(), 1.0; gmres_kwargs = (reltol = 1e-9,))
        reference = scattering_amplitude(modal(Sphere(1.0), Rigid(), 1.0))
        actual = scattering_amplitude(solution)
        @test solution.data.quad === canonical.data
        @test abs(target_strength(actual) - target_strength(reference)) < 0.1
        @test abs(actual - reference) / abs(reference) < 0.01
    end
end
