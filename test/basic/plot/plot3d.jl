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

    target = mesh(Sphere(0.05); method = :full, resolution = 0.04, qorder = 4)
    transmitter = TankTransducer(Transducer((-1, 0, 0), (1, 0, 0), 0.02),
        Wall((0, 0, -0.5), (0, 0, 1), -1))
    tank_solution = bem(target, Rigid(), 2.0; transducer = transmitter,
        compression = (method = :none,))
    tank_slices = ((axis = :z, at = 0.0, project_to = 0.1),)
    tank_plot = CairoMakie.plot(tank_solution; kind = :field_slices, slices = tank_slices,
        extent = 0.3, resolution = 5, units = :m, colorbar = false, mark_slices = false)
    @test tank_plot isa CairoMakie.Makie.FigureAxisPlot
    # Prescribed illumination has no single plane-wave direction to draw.
    @test !any(p->p isa CairoMakie.Makie.Arrows3D, tank_plot.axis.scene.plots)
    extension=Base.get_extension(AcousticScattering, :AcousticScatteringMakieExt)
    sampled=extension._slice_pressure(tank_solution, [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)], :total)
    @test isnan(sampled[1]) && isfinite(sampled[2])
    @test sampled[2] ≈ pressure(tank_solution, (0.1, 0.0, 0.0))

    tank=Tank((-1.5, 1.5), (-0.5, 0.5), (-0.5, 0.5))
    side=tank_field(tank_solution, tank; resolution = (15, 9), wall_images = false,
        transducers = (transmitter = transmitter,))
    side_plot=CairoMakie.plot(side; colorbar = false)
    @test side_plot.axis isa CairoMakie.Axis
    @test side_plot.axis.yreversed[]
    @test side_plot.axis.ylabel[] == "Depth below water surface (m)"
    @test side_plot.plot[3][] ≈ reverse(abs.(side.pressure); dims = 2) nans=true
    height_plot=CairoMakie.plot(side; depth = false, colorbar = false)
    @test !height_plot.axis.yreversed[]
    db_plot=CairoMakie.plot(side; field = :pressure_db, colorbar = false)
    finite=filter(isfinite, vec(abs.(side.pressure)))
    @test db_plot.plot[3][] ≈
          reverse(20 .* log10.(abs.(side.pressure) ./ maximum(finite)); dims = 2) nans=true
    @test_throws ArgumentError CairoMakie.plot(side; field = :pressure_db, pressure_reference = 0)
    @test_throws ArgumentError CairoMakie.plot(side; field = :pressure_db, dynamic_range = -1)
    explicit_db=CairoMakie.plot(side; field = :pressure_db, pressure_reference = 2.0, colorbar = false)
    @test explicit_db.plot[3][] ≈ reverse(20 .* log10.(abs.(side.pressure) ./ 2); dims = 2) nans=true
    cylinder=Tank(0.18, (-0.45, 0.45))
    cylinder_map=tank_field(Transducer((0.0, 0.0, 0.3), (0, 0, -1), 0.04), 10.0, cylinder;
        plane = :xy, at = 0.0, resolution = (9, 9))
    @test CairoMakie.plot(cylinder_map; colorbar = false) isa
          CairoMakie.Makie.FigureAxisPlot
    # Side-by-side apertures coincide in a side projection but remain distinct
    # in the complementary geometry view. Projection must preserve physical size.
    paired_tx=Transducer((-1, 0.04, 0.0), (cosd(10), 0, -sind(10)), 0.02)
    paired_rx=Transducer((-1, -0.04, 0.0), (cosd(10), 0, -sind(10)), 0.02)
    face=extension._tank_aperture_projection(paired_tx, 2, 1, identity)
    @test maximum(p[1] for p in face)-minimum(p[1] for p in face) ≈ 0.04
    @test maximum(p[2] for p in face)-minimum(p[2] for p in face) ≈ 0.04*sind(10)
    @test extension._tank_beam_halfangle(paired_tx, 1.0) === nothing
    beam_k=1.616339948310703/(paired_tx.radius*sind(4))
    halfangle=extension._tank_beam_halfangle(paired_tx, beam_k)
    @test rad2deg(2halfangle) ≈ 8
    u=beam_k*paired_tx.radius*sin(halfangle)
    @test abs2(2AcousticScattering.besselj(1, u)/u) ≈ 0.5
    paired=tank_field(tank_solution, tank; resolution = (15, 9), wall_images = false,
        transducers = (Tx = paired_tx, Rx = paired_rx))
    paired_plot=CairoMakie.plot(paired; colorbar = false)
    @test length(filter(x->x isa CairoMakie.Axis, paired_plot.figure.content))==2
    @test paired_plot.plot[3][] ≈ reverse(abs.(paired.pressure); dims = 2) nans=true
    before=length(paired_plot.axis.scene.plots)
    extension._tank_beam_guides!(paired_plot.axis, paired_tx, beam_k, 1, 3,
        z->0.5-z, :orangered, 1.0)
    @test length(paired_plot.axis.scene.plots)==before+2
    without_detail=CairoMakie.plot(paired; transducer_view = false, colorbar = false)
    @test length(filter(x->x isa CairoMakie.Axis, without_detail.figure.content))==1
end
