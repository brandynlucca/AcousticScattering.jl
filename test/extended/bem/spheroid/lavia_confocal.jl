using AcousticScattering
using Test

@time "Lavia confocal two-fluid spheroid" @testset "Lavia confocal two-fluid spheroid" begin
    AS = AcousticScattering
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

    outer = Spheroid(1.2, 1.0)
    inner = Spheroid(sqrt(outer.q^2 + 0.55^2), 0.55)
    shell = Shelled(FluidLayer(1.2, 1.1), FluidInterior(0.7, 0.8), 0.55)
    errors = Float64[]
    for n in (128, 256)
        outer_mesh = AS.spheroid_mesh(outer.a, outer.b, n)
        inner_mesh = AS.spheroid_mesh(inner.a, inner.b, n)
        p, dp, panels = AS.solve_axial(shell, 1.5, outer_mesh, inner_mesh; rtol = 1e-7)
        actual = [AS._axisymmetric_pressure_traces(panels, [p], [dp], 1.5, point)
                  for point in points]
        push!(errors, maximum(abs.(actual .- expected) ./ abs.(expected)))
        if n == 256
            @test errors[end] < 1e-3
            @test maximum(abs.(20log10.(abs.(actual ./ expected)))) < 0.01
        end
    end
    @test errors[2] < errors[1]/3

    # Public multi-interface path on separately meshed confocal surfaces.
    surfaces = [mesh(body; method = :full, resolution = 0.25,
                    mesh_order = 2, qorder = 4) for body in (outer, inner)]
    full3d = bem(surfaces, [FluidFilled(1.2, 1.1), FluidFilled(0.7, 0.8)], 1.5;
        parents = [0, 1], incidence_angle = 0.0, condition_limit = 0,
        compression = (method = :hmatrix, tol = 1e-9))
    actual3d = pressure(full3d, points; field = :scattered)
    @test maximum(abs.(actual3d .- expected) ./ abs.(expected)) < 1e-3
    @test maximum(abs.(20log10.(abs.(actual3d ./ expected)))) < 0.01
end
