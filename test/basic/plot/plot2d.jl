using AcousticScattering
import CairoMakie
using Test

@testset "2D plots" begin
    solution = mfs(Sphere(1.0), Rigid(), 0.5; incidence_angle = 0.0,
        n = 12, offset = 0.2, condition_limit = 0)
    angles = [0.0, pi / 2, pi]
    polar = CairoMakie.plot(solution; kind = :bistatic_polar, angles)
    @test polar isa CairoMakie.Makie.FigureAxisPlot
    @test polar.axis isa CairoMakie.PolarAxis

    cartesian = CairoMakie.plot(solution; kind = :bistatic_cartesian, angles)
    @test cartesian isa CairoMakie.Makie.FigureAxisPlot
    @test cartesian.axis isa CairoMakie.Axis

    map_plot = CairoMakie.plot(solution; kind = :bistatic_map,
        thetas = [0.0, pi / 2], phis = [0.0, pi], interpolate = true)
    @test map_plot isa CairoMakie.Makie.FigureAxisPlot
    @test_throws ArgumentError CairoMakie.plot(solution; kind = :unknown, angles)
end
