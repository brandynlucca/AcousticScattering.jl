using AcousticScattering
import CairoMakie
using Test

const AS = AcousticScattering

@testset "Volume FEM and free-surface plots" begin
    k = 2.0
    material = AS.FluidFilled(1.05, 1.05)
    sphere = AS.Sphere(0.25)

    single = AS.fem(sphere, material, k; method = :volume, incidence_angle = pi / 2,
        incidence_azimuth = 0.0, points_per_wavelength = 6, closure = :dtn, domain_radius = 0.6)
    mesh_plot = CairoMakie.plot(single; kind = :mesh)
    @test mesh_plot isa CairoMakie.Makie.FigureAxisPlot
    @test_throws ArgumentError CairoMakie.plot(single; kind = :surface_field)

    angles = [0.0, pi / 2, pi]
    polar = CairoMakie.plot(single; kind = :bistatic_polar, angles)
    @test polar isa CairoMakie.Makie.FigureAxisPlot
    map_plot = CairoMakie.plot(single; kind = :bistatic_map, thetas = [0.0, pi / 2], phis = [0.0])
    @test map_plot isa CairoMakie.Makie.FigureAxisPlot

    slices = ((axis = :z, at = 0.0, project_to = -0.8),)
    slice_plot = CairoMakie.plot(single; kind = :field_slices, slices, extent = 0.35,
        resolution = 6, units = :m, colorbar = false, legend = false)
    @test slice_plot isa CairoMakie.Makie.FigureAxisPlot

    regions = AS.fem([sphere, AS.Sphere(0.1)], [material, material], k; method = :volume,
        parents = [0, 0], centers = [zeros(3), [0.0, 0.0, 1.5]], incidence_angle = 0.0,
        points_per_wavelength = 6)
    region_mesh_plot = CairoMakie.plot(regions; kind = :mesh)
    @test region_mesh_plot isa CairoMakie.Makie.FigureAxisPlot

    surface = AS.free_surface(sphere, material, k, 1.4; condition = :pressure_release,
        incidence_angle = 0.3, incidence_azimuth = 0.2, points_per_wavelength = 6)
    surface_mesh_plot = CairoMakie.plot(surface; kind = :mesh)
    @test surface_mesh_plot isa CairoMakie.Makie.FigureAxisPlot
    surface_polar = CairoMakie.plot(surface; kind = :bistatic_polar, angles)
    @test surface_polar isa CairoMakie.Makie.FigureAxisPlot
    surface_slices = CairoMakie.plot(surface; kind = :field_slices,
        slices = ((axis = :y, at = 0.0, project_to = 0.8),), extent = 0.5, resolution = 6,
        units = :m, colorbar = false, legend = false)
    @test surface_slices isa CairoMakie.Makie.FigureAxisPlot
end
