using AcousticScattering
using CairoMakie
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "Makie visualization: 1D sweep plots" begin
    a = 0.01
    c_water = 1477.4
    sphere = AS.Sphere(a)
    spheroid = AS.Spheroid(0.05, 0.02)
    k38 = 2pi * 38000.0 / c_water

    @testset "plot(body, boundary, freqs; kind=:frequency) builds a figure" begin
        fap = plot(sphere, AS.Rigid(), 20e3:10e3:60e3;
            kind = :frequency, sound_speed = c_water)
        @test fap isa Makie.FigureAxisPlot
    end

    @testset "plot(body, boundary, angles; kind=:incidence_angle) builds a figure" begin
        fap = plot(spheroid, AS.Rigid(), 0:0.5:1.5; kind = :incidence_angle, k = k38)
        @test fap isa Makie.FigureAxisPlot
    end

    @testset "solver kwarg routes to a different dispatcher" begin
        fap = plot(sphere, AS.Rigid(), 20e3:10e3:60e3; kind = :frequency,
            sound_speed = c_water, solver = :bem, solver_kwargs = (n = 12,))
        @test fap isa Makie.FigureAxisPlot
    end

    @testset "plot! overlays onto an existing axis" begin
        fig = Figure()
        ax = Axis(fig[1, 1])
        plot!(ax, sphere, AS.Rigid(), 20e3:10e3:60e3;
            kind = :frequency, sound_speed = c_water)
        @test !isempty(ax.scene.plots)
    end

    @testset "unsupported kind errors clearly" begin
        @test_throws ArgumentError plot(
            sphere, AS.Rigid(), 20e3:10e3:60e3; kind = :bogus, sound_speed = c_water)
    end
end
