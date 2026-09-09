using AcousticScattering
using CairoMakie
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "Makie visualization: 3D model plots" begin
    a = 0.01
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = AS.Sphere(a)

    @testset "axisymmetric BEM: mesh and surface_field" begin
        bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16, m_max = 2)
        @test plot(bem_sol; kind = :mesh) isa Makie.FigureAxisPlot
        @test plot(bem_sol; kind = :surface_field) isa Makie.FigureAxisPlot
        @test plot(bem_sol; kind = :surface_field, field = :pressure_phase) isa
              Makie.FigureAxisPlot
    end

    @testset "spheroid, straight cylinder, and shell FEM geometries" begin
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
        shell_sol = AS.fem(shell_body, shell_boundary, 1026.0, c_water, 1026.0, c_water, k;
            n_eta = 12, n_t = 2, m_max = 2)
        @test plot(shell_sol; kind = :mesh) isa Makie.FigureAxisPlot
        @test plot(shell_sol; kind = :surface_field) isa Makie.FigureAxisPlot
    end

    @testset "bent-cylinder MFS: mesh falls back, surface_field is a point cloud" begin
        bent = AS.Cylinder(0.01, 0.1; radius_curvature = 0.5)
        k_bent = 2pi * 20000.0 / c_water
        mfs_sol = AS.mfs(bent, AS.Rigid(), k_bent; n_s = 8, n_φ = 8)
        @test plot(mfs_sol; kind = :mesh) isa Makie.FigureAxisPlot
        @test plot(mfs_sol; kind = :surface_field) isa Makie.FigureAxisPlot
    end

    @testset "ModalSolution/KirchhoffSolution: mesh works, surface_field errors" begin
        modal_sol = AS.modal(sphere, AS.Rigid(), k)
        kirch_sol = AS.kirchhoff(sphere, AS.Rigid(), k)
        @test plot(modal_sol; kind = :mesh) isa Makie.FigureAxisPlot
        @test plot(kirch_sol; kind = :mesh) isa Makie.FigureAxisPlot
        @test_throws ArgumentError plot(modal_sol; kind = :surface_field)
        @test_throws ArgumentError plot(kirch_sol; kind = :surface_field)
    end

    @testset "FEMSolution{_ScalarFEMData}: mesh works, surface_field errors naming the gap" begin
        fem_sol = AS.fem(sphere, AS.Rigid(), k)
        @test plot(fem_sol; kind = :mesh) isa Makie.FigureAxisPlot
        @test_throws ArgumentError plot(fem_sol; kind = :surface_field)
    end
end
