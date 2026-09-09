using AcousticScattering
using CairoMakie
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "Makie visualization: 2D bistatic plots" begin
    a = 0.01
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = AS.Sphere(a)
    bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)

    @testset "plot(sol; kind=:bistatic_polar) builds a figure with non-negative radius" begin
        fap = plot(bem_sol; kind = :bistatic_polar, angles = 0:0.2:(2pi))
        @test fap isa Makie.FigureAxisPlot
        @test fap.axis isa PolarAxis
    end

    @testset "plot(sol; kind=:bistatic_cartesian) builds a figure" begin
        fap = plot(bem_sol; kind = :bistatic_cartesian, angles = 0:0.2:(2pi))
        @test fap isa Makie.FigureAxisPlot
        @test fap.axis isa Axis
    end

    @testset "plot(sol; kind=:bistatic_map) builds a heatmap figure" begin
        fap = plot(bem_sol; kind = :bistatic_map, thetas = 0:0.3:pi, phis = 0:0.3:(2pi))
        @test fap isa Makie.FigureAxisPlot
    end

    @testset "show_incidence adds forward/backscatter reference lines" begin
        fig_with = plot(bem_sol; kind = :bistatic_cartesian,
            angles = 0:0.2:(2pi), show_incidence = true)
        fig_without = plot(bem_sol; kind = :bistatic_cartesian,
            angles = 0:0.2:(2pi), show_incidence = false)
        @test length(fig_with.plot.plots) > length(fig_without.plot.plots)
    end

    @testset "colorrange: default handles a deep null without erroring, explicit override works" begin
        fap = plot(bem_sol; kind = :bistatic_map, thetas = 0:0.3:pi,
            phis = 0:0.3:(2pi), colorrange = (-60.0, -40.0))
        @test fap isa Makie.FigureAxisPlot
    end

    @testset "unsupported kind errors clearly" begin
        @test_throws ArgumentError plot(bem_sol; kind = :bogus, angles = 0:0.2:(2pi))
    end
end
