using AcousticScattering
using Test

@testset "Projected mixed spheroid T-matrix" begin
    body = Spheroid(1.2, 1.0)
    wall = ElasticLayer(2.7, 4.2, 2.1)
    core = FluidInterior(0.8, 0.9)
    boundary = Shelled(LayeredMaterial(FluidLayer(1.2, 1.1), wall, 0.8),
        core, 0.55)
    k = 1.2

    # Generated independently by the public multi-region full-3D volume FEM.
    fixture = split(
        raw"""
% incidence_angle scatter_angle scatter_azimuth real_f imag_f
0 0 0 -0.043066059014318654 0.05161493600338816
0 1.5707963267948966 0 -0.24436143272552882 0.05390666131509321
0 3.141592653589793 0 -0.1454786084267126 0.04986713181816339
1.5707963267948966 0 0 -0.2443607255739648 0.05390647041190504
1.5707963267948966 1.5707963267948966 0 -0.044288530897680846 0.06651668740841264
1.5707963267948966 1.5707963267948966 3.141592653589793 -0.24415882961995372 0.059315686973149706
1.5707963267948966 3.141592653589793 0 -0.24436021809111722 0.053905695024457974
""", '\n')
    rows = [parse.(Float64, split(line))
            for line in fixture
            if !isempty(line) && !startswith(line, "%")]
    @test length(rows) == 7
    for row in rows
        incidence_angle, scatter_angle, scatter_azimuth = row[1:3]
        expected = complex(row[4], row[5])
        actual = scattering_amplitude(tmatrix(body, boundary, k;
            incidence_angle, scatter_angle, scatter_azimuth,
            m_max = 4, n_max = 8, check = false))
        @test abs(actual - expected) / abs(expected) < 5e-3
    end

    # A matched outer fluid layer collapses to the established single
    # elastic-shell transition, including its complex phase.
    matched_body = Spheroid(sqrt(body.q^2 + 0.85^2), 0.85)
    for interior in (core, VacuumInterior())
        layered = Shelled(LayeredMaterial(FluidLayer(1.0, 1.0), wall, 0.85),
            interior, 0.55)
        direct = Shelled(wall, interior, 0.55 / 0.85)
        for incidence_angle in (0.0, pi / 2)
            actual = scattering_amplitude(tmatrix(body, layered, k;
                incidence_angle, m_max = 4, n_max = 8, check = false))
            expected = scattering_amplitude(tmatrix(matched_body, direct, k;
                incidence_angle, m_max = 4, n_max = 8, check = false))
            @test isapprox(actual, expected; rtol = 3e-3, atol = 1e-5)
        end
    end

    # Check the upper aspect limit against the independent single-shell route.
    edge = Spheroid(1.25, 1.0)
    edge_inner = Spheroid(sqrt(edge.q^2 + 0.85^2), 0.85)
    edge_layers = Shelled(LayeredMaterial(FluidLayer(1.0, 1.0), wall, 0.85),
        core, 0.55)
    edge_shell = Shelled(wall, core, 0.55 / 0.85)
    for incidence_angle in (0.0, pi / 2)
        actual = scattering_amplitude(tmatrix(edge, edge_layers, k;
            incidence_angle, check = false))
        expected = scattering_amplitude(tmatrix(edge_inner, edge_shell, k;
            incidence_angle, check = false))
        @test isapprox(actual, expected; rtol = 4e-3, atol = 1e-5)
    end
    @test isfinite(scattering_amplitude(tmatrix(edge, edge_layers, k;
        incidence_angle = pi / 2)))

    # The default reduced-degree check should accept the validated case.
    @test isfinite(scattering_amplitude(tmatrix(body, boundary, k)))
    @test_throws ArgumentError tmatrix(body, boundary, k; n_max = 10)
    @test_throws ArgumentError tmatrix(Spheroid(1.5, 1.0), boundary, k)
    @test_throws ArgumentError tmatrix(Spheroid(1.0, 1.2), boundary, k)
    @test_throws ArgumentError tmatrix(body, boundary, 2.0)
    @test_throws ArgumentError tmatrix(body,
        Shelled(LayeredMaterial(FluidLayer(1.2, 1.1), FluidLayer(0.9, 1.0), 0.8),
            core, 0.55), k)
    @test_throws ArgumentError modal(body, boundary, k)
end
