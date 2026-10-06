using AcousticScattering
using Test

const AS = AcousticScattering

@testset "Nested fluid MFS against Lavia confocal reference" begin
    outer = Spheroid(1.2, 1.0)
    boundary = Shelled(FluidLayer(1.2, 1.1), FluidInterior(0.7, 0.8), 0.55)
    k = 1.5
    reference_lines = split(raw"""
% Lavia three-fluid confocal prolate spheroid: outer a=1.2 b=1, inner b=0.55
% k=1.5; shell density/speed contrast 1.2/1.1; core 0.7/0.8; e^-iwt
% columns x y z real(p_scat) imag(p_scat); degree 14
1.5 0 0 -0.055997925374225807 0.041095621542758837
-1.5 0 0 -0.10140248464756568 0.082286451330468441
0 1.5 0 -0.077061616413034537 0.058151257694654207
1.5 0.40000000000000002 0 -0.05876161812074189 0.037458769525241124
""", '\n')
    rows = [parse.(Float64, split(line))
            for line in reference_lines
            if !isempty(line) && !startswith(line, "%")]
    points = [Tuple(row[1:3]) for row in rows]
    expected = complex.([row[4] for row in rows], [row[5] for row in rows])

    solution = mfs(outer, boundary, k; n = 40, oversampling = 2,
        offset_outer = 0.2, offset_inner = 0.12)
    actual = pressure(solution, points; field = :scattered)
    @test maximum(abs.(actual .- expected) ./ abs.(expected)) < 0.002
    @test isfinite(pressure(solution, (1.0, 0.0, 0.0); field = :shell))
    @test isfinite(pressure(solution, (0.0, 0.0, 0.0); field = :interior))
    @test_throws ArgumentError pressure(solution, (1.5, 0.0, 0.0); field = :shell)
    @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0); field = :scattered)

    report = only(diagnostics(solution).systems)
    @test report.source_count == 160
    @test report.collocation_count == 160
    @test report.check_count > report.collocation_count
    @test report.pressure_residual < 0.01
    @test report.velocity_residual < 0.01
    @test report.numerical_rank == report.source_count

    inner = AS._nested_fluid_inner(outer, boundary.radius_ratio)
    source_mesh = AS._axisymmetric_mesh(outer, 16)
    inner_source_mesh = AS._axisymmetric_mesh(inner, 16)
    sources = AS._nested_fluid_sources(source_mesh, inner_source_mesh, 0.2, 0.12)
    @test AS._check_nested_fluid_sources(outer, inner, sources) === nothing
    @test_throws ArgumentError AS._check_nested_fluid_sources(outer, inner,
        merge(sources, (; shell_outer = sources.shell_inner)))
end

@testset "Nested spherical fluid MFS against modal layers" begin
    body = Sphere(1.0)
    boundary = Shelled(FluidLayer(1.2, 1.1), FluidInterior(0.7, 0.8), 0.55)
    numerical = mfs(body, boundary, 1.5; n = 40, oversampling = 2,
        offset_outer = 0.2, offset_inner = 0.12, condition_limit = 0)
    reference = modal(body, boundary, 1.5; m_max = 16)
    for point in ((1.5, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 0.0, 0.0))
        @test isapprox(pressure(numerical, point; field = :total),
            pressure(reference, point; field = :total); rtol = 0.001)
    end
    @test isapprox(scattering_amplitude(numerical),
        scattering_amplitude(reference); rtol = 0.002)
    @test_throws ArgumentError mfs(Spheroid(1.0, 1.2), boundary, 1.5)
    @test_throws ArgumentError mfs(body, boundary, 1.5; offset_outer = -0.1)
    @test_throws ArgumentError mfs(body, boundary, 1.5; incidence_angle = pi / 2)
end
