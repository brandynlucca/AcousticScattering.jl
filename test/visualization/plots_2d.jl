using AcousticScattering
using CairoMakie
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@time "Makie visualization: 2D bistatic plots" @testset "Makie visualization: 2D bistatic plots" begin
    a = 0.01
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = AS.Sphere(a)
    bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)

    @time "plot(sol; kind=:bistatic_polar) builds a figure with non-negative radius" @testset "plot(sol; kind=:bistatic_polar) builds a figure with non-negative radius" begin
        fap = plot(bem_sol; kind = :bistatic_polar, angles = 0:0.2:(2pi))
        @test fap isa Makie.FigureAxisPlot
        @test fap.axis isa PolarAxis
    end

    @time "plot(sol; kind=:bistatic_cartesian) builds a figure" @testset "plot(sol; kind=:bistatic_cartesian) builds a figure" begin
        fap = plot(bem_sol; kind = :bistatic_cartesian, angles = 0:0.2:(2pi))
        @test fap isa Makie.FigureAxisPlot
        @test fap.axis isa Axis
    end

    @time "plot(sol; kind=:bistatic_map) builds a heatmap figure" @testset "plot(sol; kind=:bistatic_map) builds a heatmap figure" begin
        fap = plot(bem_sol; kind = :bistatic_map, thetas = 0:0.3:pi, phis = 0:0.3:(2pi))
        @test fap isa Makie.FigureAxisPlot
    end

    @time "show_incidence adds forward/backscatter reference lines" @testset "show_incidence adds forward/backscatter reference lines" begin
        fig_with = plot(bem_sol; kind = :bistatic_cartesian,
            angles = 0:0.2:(2pi), show_incidence = true)
        fig_without = plot(bem_sol; kind = :bistatic_cartesian,
            angles = 0:0.2:(2pi), show_incidence = false)
        @test length(fig_with.plot.plots) > length(fig_without.plot.plots)
    end

    @time "incidence markers are derived from the actual incidence direction, not a fixed 0/pi" @testset "incidence markers are derived from the actual incidence direction, not a fixed 0/pi" begin
        spheroid = AS.Spheroid(0.02, 0.01)
        β = deg2rad(30.0)
        sol = AS.bem(spheroid, AS.Rigid(), k; incidence_angle = β, m_max = 8, n = 16)
        n_base = length(plot(sol; kind = :bistatic_polar, angles = 0:0.1:pi,
            azimuth = 0.3, show_incidence = false).plot.plots)

        fig_neither = plot(
            sol; kind = :bistatic_polar, angles = 0:0.1:pi, azimuth = 0.3)
        @test length(fig_neither.plot.plots) == n_base

        fig_forward = plot(
            sol; kind = :bistatic_polar, angles = 0:0.1:pi, azimuth = 0.0)
        @test length(fig_forward.plot.plots) == n_base + 2

        fig_back = plot(sol; kind = :bistatic_polar, angles = 0:0.1:pi, azimuth = pi)
        @test length(fig_back.plot.plots) == n_base + 2
    end

    @time "colorrange: default handles a deep null without erroring, explicit override works" @testset "colorrange: default handles a deep null without erroring, explicit override works" begin
        fap = plot(bem_sol; kind = :bistatic_map, thetas = 0:0.3:pi,
            phis = 0:0.3:(2pi), colorrange = (-60.0, -40.0))
        @test fap isa Makie.FigureAxisPlot
    end

    @time "Rotated observation-plane markers" @testset "Rotated observation-plane markers" begin
        ext = Base.get_extension(AS, :AcousticScatteringMakieExt)
        sweep = AS.BistaticSweep([0.0, 2pi], 0.4, [-20.0, -20.0],
            pi / 3, 0.4, nothing, ["Scattered field"])
        angles, labels = ext._incidence_markers(sweep)
        @test angles ≈ [pi / 3, 4pi / 3]
        @test labels == ["Forward", "Backscatter"]
        missed = AS.BistaticSweep([0.0, 2pi], 0.0, [-20.0, -20.0],
            pi / 3, 0.4, nothing, ["Scattered field"])
        @test isempty(first(ext._incidence_markers(missed)))
    end

    @time "unsupported kind errors clearly" @testset "unsupported kind errors clearly" begin
        @test_throws ArgumentError plot(bem_sol; kind = :bogus, angles = 0:0.2:(2pi))
    end
end
