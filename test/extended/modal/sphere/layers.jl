using AcousticScattering
using Test

@testset "Composed spherical fluid layers" begin
    reference_lines = split(raw"""
% four-fluid sphere radii=[1 .8 .55] k=1.5 rho=[1 1.2 .9 .7] c=[1 1.1 .95 .8]
% columns r mu real(p_scat) imag(p_scat); truncation order 16
1.5 1 -0.065751917421210809 0.048226037944000814
1.5 -1 -0.11747515053867458 0.080474255487212154
1.5 0 -0.086204775893283792 0.066636799202467067
2 0.6 -0.069881288610435674 -0.0066265625723174537
% backscatter real imaginary
backscatter 0.20297954053756653 0.040137577979542767
""", '\n')
    lines = [strip(line)
             for line in reference_lines
             if !isempty(strip(line)) && !startswith(line, "%")]
    rows = [parse.(Float64, split(line)) for line in lines[1:4]]
    points = [(row[1]*row[2], row[1]*sqrt(1-row[2]^2), 0.0) for row in rows]
    expected_pressure = complex.([row[3] for row in rows], [row[4] for row in rows])
    farfield = split(lines[5])
    expected_amplitude = complex(parse(Float64, farfield[2]),
        parse(Float64, farfield[3]))

    outer = FluidLayer(1.2, 1.1)
    inner = FluidLayer(0.9, 0.95)
    core = FluidInterior(0.7, 0.8)
    boundary = Shelled(LayeredMaterial(outer, inner, 0.8), core, 0.55)
    solution = modal(Sphere(1.0), boundary, 1.5; m_max = 16)
    @test pressure(solution, points; field = :scattered) ≈ expected_pressure rtol = 1e-11
    @test scattering_amplitude(solution) ≈ expected_amplitude rtol = 1e-11
    @test AcousticScattering.form_function(boundary, 1.5, 1.0; m_max = 16) ≈
          expected_amplitude rtol = 1e-11

    # Continuity uses the same outward radial direction on each side, including
    # when the density and sound speed both change.
    for (radius, rho_out, rho_in) in ((0.8, 1.2, 0.9), (0.55, 0.9, 0.7))
        h = 1e-5
        p(r) = pressure(solution, (r, 0.0, 0.0); field = :total)
        @test p(radius + h) ≈ p(radius - h) rtol = 5e-5
        d_out = (p(radius + 2h) - p(radius + h)) / h / rho_out
        d_in = (p(radius - h) - p(radius - 2h)) / h / rho_in
        @test d_out ≈ d_in rtol = 2e-4
    end

    matched = FluidLayer(1.2, 1.1)
    collapsed = Shelled(LayeredMaterial(matched, matched, 0.8), core, 0.55)
    single = Shelled(matched, core, 0.55)
    @test scattering_amplitude(modal(Sphere(1.0), collapsed, 1.5; m_max = 16)) ≈
          scattering_amplitude(modal(Sphere(1.0), single, 1.5; m_max = 16)) rtol = 1e-11
    collapsed_vacuum = Shelled(LayeredMaterial(matched, matched, 0.8),
        VacuumInterior(), 0.55)
    single_vacuum = Shelled(matched, VacuumInterior(), 0.55)
    @test scattering_amplitude(modal(Sphere(1.0), collapsed_vacuum, 1.5; m_max = 16)) ≈
          scattering_amplitude(modal(Sphere(1.0), single_vacuum, 1.5; m_max = 16)) rtol = 1e-11

    nested = Shelled(
        LayeredMaterial(outer,
            LayeredMaterial(inner, FluidLayer(1.05, 1.02), 0.75), 0.8),
        core, 0.55)
    nested_solution = modal(Sphere(1.0), nested, 1.5; m_max = 16)
    @test isfinite(pressure(nested_solution, (0.7, 0.0, 0.0); field = :shell))
    @test isfinite(pressure(nested_solution, (0.59, 0.0, 0.0); field = :shell))
    @test_throws ArgumentError Shelled(nested.material, core, 0.61)
end

@testset "Composed spherical fluid and elastic layers" begin
    sphere = Sphere(1.0)
    wall = ElasticLayer(2.7, 4.2, 2.1)
    core = FluidInterior(0.8, 0.9)
    k = 1.2
    orders = 6

    # Removing a material interface must recover the independently implemented
    # single elastic-shell solution, including its complex phase.
    reductions = (
        (Sphere(1.0),
            Shelled(LayeredMaterial(FluidLayer(1.0, 1.0), wall, 0.85), core, 0.55),
            Sphere(0.85), Shelled(wall, core, 0.55 / 0.85)),
        (sphere,
            Shelled(LayeredMaterial(wall, FluidLayer(0.8, 0.9), 0.7), core, 0.55),
            sphere, Shelled(wall, core, 0.7)),
        (sphere, Shelled(LayeredMaterial(wall, wall, 0.8), core, 0.55),
            sphere, Shelled(wall, core, 0.55))
    )
    for (mixed_shape, mixed_boundary, reference_shape, reference_boundary) in reductions
        for angle in (0.0, pi / 2, pi)
            actual = scattering_amplitude(modal(mixed_shape, mixed_boundary, k;
                angle, m_max = orders))
            reference = scattering_amplitude(modal(reference_shape, reference_boundary,
                k; angle, m_max = orders))
            @test actual≈reference rtol=1e-11 atol=1e-12
        end
    end
    three_layers = Shelled(
        LayeredMaterial(FluidLayer(1.0, 1.0),
            LayeredMaterial(wall, wall, 0.9), 0.85),
        core, 0.55)
    @test isapprox(
        scattering_amplitude(modal(sphere, three_layers, k; m_max = orders)),
        scattering_amplitude(modal(Sphere(0.85),
            Shelled(wall, core, 0.55 / 0.85), k; m_max = orders));
        rtol = 1e-11, atol = 1e-12)
    legacy_wall = ElasticLayer(2.7, 4.2, 2.1;
        interior_coupling = :identical_fluid)
    @test_throws ArgumentError modal(sphere,
        Shelled(LayeredMaterial(FluidLayer(1.0, 1.0), legacy_wall, 0.85),
            core, 0.55), k; m_max = orders)

    mixed = Shelled(LayeredMaterial(FluidLayer(1.2, 1.1), wall, 0.8), core, 0.55)
    solution = modal(sphere, mixed, k; m_max = orders)
    @test isfinite(pressure(solution, (0.9, 0.0, 0.0); field = :shell))
    @test isfinite(pressure(solution, (0.4, 0.0, 0.0); field = :interior))
    @test_throws ArgumentError pressure(solution, (0.7, 0.0, 0.0))

    # A full-3D volume FEM assembly is independent of the modal interface matrix.
    volume = fem([Sphere(1.0), Sphere(0.8), Sphere(0.55)],
        [FluidFilled(1.2, 1.1), SolidElastic(2.7, 4.2, 2.1),
            FluidFilled(0.8, 0.9)], k;
        method = :volume, incidence_angle = 0.0, closure = :dtn,
        points_per_wavelength = 6)
    for angle in (0.0, pi / 2, pi)
        reference = scattering_amplitude(modal(sphere, mixed, k;
            angle, m_max = orders))
        @test scattering_amplitude(volume; angle)≈reference rtol=2e-3 atol=1e-5
    end

    vacuum = Shelled(LayeredMaterial(FluidLayer(1.0, 1.0), wall, 0.85),
        VacuumInterior(), 0.55)
    vacuum_amplitude = scattering_amplitude(modal(sphere, vacuum, k;
        m_max = orders))
    @test isfinite(vacuum_amplitude)
    dilute_core = Shelled(wall, FluidInterior(1e-6, 1.0), 0.55 / 0.85)
    @test vacuum_amplitude≈scattering_amplitude(modal(Sphere(0.85),
        dilute_core, k; m_max = orders)) rtol=1e-4 atol=1e-6
end
