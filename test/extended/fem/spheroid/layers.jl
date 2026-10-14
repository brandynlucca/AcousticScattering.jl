using AcousticScattering
using Test
using LinearAlgebra

BLAS.set_num_threads(1)
include("../../../helpers/layered_fem.jl")

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

@testset "Viscous/elastic confocal spheroids with a fluid core" begin
    boundary = Shelled(
        LayeredMaterial(
            ViscousLayer(1500.0, 1.1, 1.05, 1000.0, 300.0),
            ElasticLayer(1.3, 1.7, 0.6), 0.7),
        FluidInterior(0.2, 0.7),
        0.4)
    k = 1.0
    for body in (Spheroid(1.2, 1.0), Spheroid(0.95, 1.0))
        amplitudes = ComplexF64[]
        # The oblate fluid core has a smaller polar radius and needs a finer
        # starting mesh to meet the same refinement tolerance.
        coarse_h = body.kind === :oblate ? 0.15 : 0.20
        for h in (coarse_h, 0.10)
            solution = fem(body, boundary, k; incidence_angle = 0.4,
                closure = :dtn, h, h_body = h)
            push!(amplitudes, scattering_amplitude(solution; angle = 1.3, azimuth = 0.0))
            @test diagnostics(solution).residual < 1e-7
            @test solution.body.bodies[2].b ≈ 0.7
            @test solution.body.bodies[3].b ≈ 0.4
            @test solution.boundary.materials[1] isa ViscousLayer
            @test solution.boundary.materials[2] isa SolidElastic
            @test solution.boundary.materials[3] isa FluidFilled
            # Optical theorem: extinction minus scattered power is positive.
            mean_power = real(layered_angular_average(solution; transform = abs2))
            forward = scattering_amplitude(solution; angle = 0.4, azimuth = 0.0)
            @test imag(forward) - k * mean_power > 0
        end
        @test amplitudes[2] ≈ amplitudes[1] rtol = 0.01
        reciprocal = fem(body, boundary, k;
            incidence_angle = pi - 1.3, incidence_azimuth = pi,
            closure = :dtn, h = 0.10, h_body = 0.10)
        @test scattering_amplitude(reciprocal; angle = pi - 0.4, azimuth = pi) ≈
              amplitudes[2] rtol = 0.001
    end
    @test_throws ArgumentError fem(Spheroid(1.2, 1.0), boundary, k; method = :radial)
end
