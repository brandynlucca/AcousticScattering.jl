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
end
