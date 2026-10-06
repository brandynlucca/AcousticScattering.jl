using AcousticScattering
using Test

@testset "Confocal layered spheroid volume FEM" begin
    outer = Spheroid(1.2, 1.0)
    fluid_stack = Shelled(
        LayeredMaterial(FluidLayer(1.2, 1.1), FluidLayer(0.7, 0.8), 0.55),
        FluidInterior(0.7, 0.8), 0.2)
    fluid_solution = fem(outer, fluid_stack, 1.5;
        incidence_angle = 0.0, closure = :dtn, domain_radius = 1.8,
        points_per_wavelength = 6)

    # The independent Lavia confocal series uses the x axis; volume FEM uses z.
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
    points = [(row[2], row[3], row[1]) for row in rows]
    expected = complex.([row[4] for row in rows], [row[5] for row in rows])
    actual = pressure(fluid_solution, points; field = :scattered)
    @test maximum(abs.(actual .- expected) ./ abs.(expected)) < 5e-3

    wall = ElasticLayer(2.7, 4.2, 2.1)
    core = FluidInterior(0.8, 0.9)
    mixed = Shelled(LayeredMaterial(FluidLayer(1.2, 1.1), wall, 0.8),
        core, 0.55)
    near_sphere = Spheroid(1.001, 1.0)
    mixed_solution = fem(near_sphere, mixed, 1.2;
        incidence_angle = 0.0, closure = :dtn,
        points_per_wavelength = 6)
    for angle in (0.0, pi / 2, pi)
        reference = scattering_amplitude(modal(Sphere(1.0), mixed, 1.2;
            angle, m_max = 8))
        @test isapprox(scattering_amplitude(mixed_solution; angle), reference;
            rtol = 5e-3, atol = 1e-5)
    end
    @test mixed_solution.body.bodies[2].b ≈ 0.8
    @test mixed_solution.body.bodies[3].b ≈ 0.55
    @test mixed_solution.boundary.materials[2] isa SolidElastic

    @test_throws ArgumentError fem(near_sphere,
        Shelled(mixed.material, VacuumInterior(), 0.55), 1.2)
    legacy = ElasticLayer(2.7, 4.2, 2.1;
        interior_coupling = :identical_fluid)
    @test_throws ArgumentError fem(near_sphere,
        Shelled(LayeredMaterial(FluidLayer(1.2, 1.1), legacy, 0.8),
            core, 0.55), 1.2)
end
