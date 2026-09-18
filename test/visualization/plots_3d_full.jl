using AcousticScattering
using CairoMakie
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "Makie visualization: full 3D BEM and mesh plots" begin
    a = 0.01
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = AS.Sphere(a)

    @testset "full 3D BEM: mesh and surface_field" begin
        full_sol = AS.bem(sphere, AS.Rigid(), k; method = :full,
            meshsize = AS.bem3d_elements_per_wavelength(k))
        @test plot(full_sol; kind = :mesh) isa Makie.FigureAxisPlot
        @test plot(full_sol; kind = :surface_field) isa Makie.FigureAxisPlot
        @test plot(full_sol; kind = :mesh, show_edges = true) isa Makie.FigureAxisPlot
        @test plot(full_sol; kind = :mesh, show_nodes = true) isa Makie.FigureAxisPlot
        @test plot(full_sol; kind = :mesh, show_normals = true) isa Makie.FigureAxisPlot
    end

    @testset "standalone Mesh: both representations" begin
        m1 = AS.mesh(sphere; k = k)
        m2 = AS.mesh(sphere; k = k, method = :full)
        @test plot(m1) isa Makie.FigureAxisPlot
        @test plot(m2) isa Makie.FigureAxisPlot
    end

    @testset "Nested interfaces and cutaway" begin
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
