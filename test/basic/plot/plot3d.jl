using AcousticScattering
import CairoMakie
using Test

@testset "3D plots" begin
    body = Sphere(0.01)
    k = 0.5
    surface = mesh(body; method = :axisymmetric, resolution = 8)
    @test CairoMakie.plot(surface) isa CairoMakie.Makie.FigureAxisPlot

    solution = mfs(Sphere(1.0), Rigid(), k; incidence_angle = 0.0,
        n = 12, offset = 0.2, condition_limit = 0)
    @test CairoMakie.plot(solution; kind = :mesh) isa CairoMakie.Makie.FigureAxisPlot
    @test CairoMakie.plot(solution; kind = :surface_field) isa
          CairoMakie.Makie.FigureAxisPlot

    modal_solution = modal(body, Rigid(), k; m_max = 2)
    @test CairoMakie.plot(modal_solution; kind = :mesh) isa
          CairoMakie.Makie.FigureAxisPlot
    @test_throws ArgumentError CairoMakie.plot(modal_solution; kind = :surface_field)

    fluid_solution = modal(body, FluidFilled(2.0, 0.65), k; m_max = 2)
    slices = ((axis = :z, at = 0.0, project_to = -0.03),
        (axis = :x, at = 0.003, project_to = 0.03))
    field_slices = CairoMakie.plot(fluid_solution; kind = :field_slices,
        slices, extent = 0.025, resolution = 7, colorbar = false)
    @test field_slices isa CairoMakie.Makie.FigureAxisPlot
    bad_slice = ((axis = :bad, at = 0.0, project_to = 0.03),)
    @test_throws ArgumentError CairoMakie.plot(fluid_solution; kind = :field_slices,
        slices = bad_slice, extent = 0.025, resolution = 7)
end
