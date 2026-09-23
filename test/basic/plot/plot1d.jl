using AcousticScattering
import CairoMakie
using Test

@testset "1D plots" begin
    sphere = Sphere(0.01)
    frequencies = [20_000.0, 30_000.0]
    direct = CairoMakie.plot(sphere, Rigid(), frequencies;
        kind = :frequency, sound_speed = 1500.0)
    @test direct isa CairoMakie.Makie.FigureAxisPlot

    sweep = frequency_sweep(k -> modal(sphere, Rigid(), k), frequencies, 1500.0)
    @test CairoMakie.plot(sweep) isa CairoMakie.Figure
    @test CairoMakie.plot(sweep; quantity = :phase) isa CairoMakie.Figure
    @test_throws ArgumentError CairoMakie.plot(sweep; quantity = :unknown)

    angles = [pi / 4, pi / 2]
    angle_plot = CairoMakie.plot(Cylinder(0.01, 0.1), Rigid(), angles;
        kind = :incidence_angle, k = 2pi * 38_000.0 / 1500.0)
    @test angle_plot isa CairoMakie.Makie.FigureAxisPlot
end
