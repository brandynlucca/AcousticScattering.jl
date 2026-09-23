using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions
using CairoMakie

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Makie visualization: 3D model plots" @testset "Makie visualization: 3D model plots" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)

        @time "axisymmetric BEM: mesh and surface_field" @testset "axisymmetric BEM: mesh and surface_field" begin
            bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16, m_max = 2)
            @test plot(bem_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(bem_sol; kind = :surface_field) isa Makie.FigureAxisPlot
            @test plot(bem_sol; kind = :surface_field, field = :pressure_phase) isa
                  Makie.FigureAxisPlot
        end

        @time "spheroid, straight cylinder, and shell FEM geometries" @testset "spheroid, straight cylinder, and shell FEM geometries" begin
            spheroid = AS.Spheroid(0.05, 0.02)
            sph_sol = AS.bem(spheroid, AS.Rigid(), k; n = 16, m_max = 2)
            @test plot(sph_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(sph_sol; kind = :surface_field) isa Makie.FigureAxisPlot

            cyl = AS.Cylinder(0.01, 0.1)
            cyl_sol = AS.bem(cyl, AS.Rigid(), k; n = 16, m_max = 2)
            @test plot(cyl_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(cyl_sol; kind = :surface_field) isa Makie.FigureAxisPlot

            shell_body = AS.Shell(sphere, 0.001)
            shell_boundary = AS.Shelled(0.3, 7800.0, 2e11)
            # Rendering needs a small multi-mode solution, not a convergence-sized mesh.
            shell_sol = AS.fem(
                shell_body, shell_boundary, 1026.0, c_water, 1026.0, c_water, k;
                n_eta = 12, n_t = 2, m_max = 2)
            @test plot(shell_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(shell_sol; kind = :surface_field) isa Makie.FigureAxisPlot
        end

        @time "bent-cylinder MFS: both :mesh and :surface_field render as point clouds over the real solved points" @testset "bent-cylinder MFS: both :mesh and :surface_field render as point clouds over the real solved points" begin
            bent = AS.Cylinder(0.01, 0.1; radius_curvature = 0.5)
            k_bent = 2pi * 20000.0 / c_water
            mfs_sol = AS.mfs(bent, AS.Rigid(), k_bent; n_s = 8, n_φ = 8)
            fig_mesh = plot(mfs_sol; kind = :mesh)
            @test fig_mesh isa Makie.FigureAxisPlot
            @test length(fig_mesh.plot.points[]) == length(mfs_sol.data.points)
            @test plot(mfs_sol; kind = :surface_field) isa Makie.FigureAxisPlot
        end

        @time "ModalSolution/KirchhoffSolution: mesh works, surface_field errors" @testset "ModalSolution/KirchhoffSolution: mesh works, surface_field errors" begin
            modal_sol = AS.modal(sphere, AS.Rigid(), k)
            kirch_sol = AS.kirchhoff(sphere, AS.Rigid(), k)
            @test plot(modal_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(kirch_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test_throws ArgumentError plot(modal_sol; kind = :surface_field)
            @test_throws ArgumentError plot(kirch_sol; kind = :surface_field)
        end

        @time "Acoustic radial FEM: mesh works, surface_field errors" @testset "Acoustic radial FEM: mesh works, surface_field errors" begin
            fem_sol = AS.fem(sphere, AS.Rigid(), k)
            @test plot(fem_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test_throws ArgumentError plot(fem_sol; kind = :surface_field)
        end
    end
end

let
    @time "Makie visualization: full 3D BEM and mesh plots" @testset "Makie visualization: full 3D BEM and mesh plots" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)

        @time "full 3D BEM: mesh and surface_field" @testset "full 3D BEM: mesh and surface_field" begin
            full_sol = AS.bem(sphere, AS.Rigid(), k; method = :full,
                meshsize = AS.bem3d_elements_per_wavelength(k))
            @test plot(full_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(full_sol; kind = :surface_field) isa Makie.FigureAxisPlot
            @test plot(full_sol; kind = :mesh, show_edges = true) isa Makie.FigureAxisPlot
            @test plot(full_sol; kind = :mesh, show_nodes = true) isa Makie.FigureAxisPlot
            @test plot(full_sol; kind = :mesh, show_normals = true) isa Makie.FigureAxisPlot
        end

        @time "standalone Mesh: both representations" @testset "standalone Mesh: both representations" begin
            m1 = AS.mesh(sphere; k = k)
            m2 = AS.mesh(sphere; k = k, method = :full)
            @test plot(m1) isa Makie.FigureAxisPlot
            @test plot(m2) isa Makie.FigureAxisPlot
        end

        @time "Nested interfaces and cutaway" @testset "Nested interfaces and cutaway" begin
            surfaces = [AS.mesh(AS.Sphere(r); method = :full, resolution = 0.6r,
                            mesh_order = 3, qorder = 4) for r in (1.0, 0.3)]
            solution = bem(surfaces, [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)], 1.0)
            cutaway = (normal = (0.0, 1.0, 0.0), offset = 0.0)
            geometry = plot(solution; kind = :mesh, cutaway, show_edges = true)
            @test geometry isa Makie.FigureAxisPlot
            pieces = geometry.plot.pieces[]
            @test length(pieces) == 2
            @test all(p -> p[2] <= 1e-7, pieces[1].points)
            @test maximum(p[2] for p in pieces[2].points) > 0.29
            @test (pieces[1].index, pieces[1].exterior) == (1, 0)
            @test (pieces[2].index, pieces[2].exterior) == (2, 1)
            for field in (:pressure_magnitude, :pressure_phase, :pressure_real, :pressure_imag)
                rendered = plot(solution; kind = :surface_field, field, cutaway)
                @test all(piece -> all(isfinite, piece.values), rendered.plot.pieces[])
            end
            selected = plot(solution; kind = :surface_field, interfaces = [2])
            @test only(selected.plot.pieces[]).index == 2
            @test any(block -> block isa Colorbar, selected.figure.content)
            @test any(block -> block isa Legend, geometry.figure.content)
            compared = components(solution; labels = ["flesh", "bladder"])
            sweep = bistatic_sweep(compared, [0.0, pi / 2, pi])
            @test plot(sweep) isa Figure
            @test_throws ArgumentError plot(solution; kind = :mesh, interface_labels = ["one"])
            figure = Figure()
            axis = Axis3(figure[1, 1])
            @test plot!(axis, solution; kind = :mesh, cutaway) isa Makie.AbstractPlot
            @test plot!(axis, solution; kind = :surface_field, interfaces = [2]) isa
                  Makie.AbstractPlot
            @test_throws ArgumentError plot(solution; kind = :mesh, interfaces = [3])
            @test_throws ArgumentError plot(solution; kind = :mesh, interfaces = Int[])
            @test_throws ArgumentError plot(solution; kind = :mesh,
                cutaway = (normal = (0.0, 0.0, 0.0), offset = 0.0))
            mktempdir() do directory
                save(joinpath(directory, "cutaway.png"), geometry)
                @test filesize(joinpath(directory, "cutaway.png")) > 0
            end
        end
    end
end
