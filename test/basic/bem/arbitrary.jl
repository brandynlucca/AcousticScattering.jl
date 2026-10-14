using AcousticScattering
using Test

@testset "Generated full-surface BEM" begin
    body = Torus(1.0, 0.3)
    full = bem(body, PressureRelease(), 0.5; method = :full, meshsize = 0.3,
        qorder = 4, incidence_angle = pi / 3, compression = (method = :none,))
    meridian = bem(body, PressureRelease(), 0.5; n = 64, m_max = 4,
        incidence_angle = pi / 3)
    @test scattering_amplitude(full) ≈ scattering_amplitude(meridian) rtol = 0.005
    @test isfinite(pressure(full, (0.0, 0.0, 0.0)))
    @test_throws ArgumentError pressure(full, (0.0, 1.0, 0.0))
    matched = bem(Ellipsoid(1.0, 0.7, 0.5), FluidFilled(1.0, 1.0), 0.5;
        meshsize = 0.5, qorder = 2, condition_limit = 0)
    @test abs(scattering_amplitude(matched)) < 1e-8
    @test pressure(matched, (0.0, 0.0, 0.0); field = :interior) ≈ 1 atol = 1e-4
end

@testset "Revolution-body BEM" begin
    # An independently known sphere checks orientation, normalization and all
    # three acoustic boundary conditions through the generic meridian path.
    sphere = RevolvedBody(t -> (cospi(t), sinpi(t)); n = 64)
    k = 0.5
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1))
        solution = bem(sphere, boundary, k; incidence_angle = pi / 3, n = 64, m_max = 4)
        reference = modal(Sphere(1.0), boundary, k)
        @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.005
        @test isfinite(pressure(solution, (2.0, 0.0, 0.0)))
    end

    body = Torus(1.0, 0.3)
    coarse = bem(
        body, FluidFilled(1.0, 1.0), k; n = 24, incidence_angle = pi / 3, m_max = 4)
    surface = mesh(body; resolution = 48)
    solution = bem(surface, FluidFilled(1.0, 1.0), k; incidence_angle = pi / 3, m_max = 4)
    # Constant-panel trace interpolation leaves a nonzero discretization error
    # even for a matched medium. It must decrease under meridian refinement.
    @test abs(scattering_amplitude(solution)) < abs(scattering_amplitude(coarse)) / 2
    @test abs(scattering_amplitude(solution)) < 5e-5
    # The hole belongs to the exterior and the tube to the fluid interior.
    @test pressure(solution, (0.0, 0.0, 0.0)) ≈ 1 atol = 1e-5
    @test pressure(solution, (0.0, 1.0, 0.0); field = :interior) ≈ cis(k * sin(pi / 3)) atol = 1e-4
    @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0); field = :interior)
    angles = [0.0, pi / 3]
    sweep = incidence_angle_sweep(body, PressureRelease(), k, angles; n = 16, m_max = 3)
    for (i, angle) in enumerate(angles)
        direct = bem(body, PressureRelease(), k; incidence_angle = angle, n = 16, m_max = 3)
        @test sweep.amplitudes[i] ≈ scattering_amplitude(direct) rtol = 1e-5
    end
    @test_throws ArgumentError bem(body, Rigid(), k; incidence_azimuth = 0.2)
    @test_throws ArgumentError bem(body, Rigid(), k; method = :unknown)
    @test_throws ArgumentError bem(Ellipsoid(1, 2, 3), Rigid(), k; method = :axisymmetric)
end

@testset "Arbitrary surfaces and fluid regions" begin
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    surface = mesh(nodes, triangles; qorder = 2)
    solution = bem(surface, PressureRelease(), 0.3;
        compression = (method = :none,))
    @test solution isa BEMSolution
    @test solution.method == :full
    @test isfinite(scattering_amplitude(solution))
    @test_throws ArgumentError bem(surface, PressureRelease(), -0.3;
        compression = (method = :none,))
    @test_throws ArgumentError bem(surface, PressureRelease(), 0.3;
        formulation = :unknown, compression = (method = :none,))
    @test_throws ArgumentError bem(surface, PressureRelease(), 0.3;
        correction = (method = :edge,), compression = (method = :none,))

    sphere_surface = mesh(Sphere(1.0); method = :full, resolution = 1.0,
        mesh_order = 2, qorder = 2)
    region = bem([sphere_surface], [FluidFilled(1.0, 1.0)], 0.3;
        condition_limit = 0)
    @test region isa BEMSolution
    @test diagnostics(region).interface_count == 1
    @test abs(scattering_amplitude(region)) < 1e-8
    @test pressure(region, (1.5, 0.0, 0.0)) ≈
          pressure(region, (1.5, 0.0, 0.0); field = :incident) atol = 1e-7
    @test pressure(region, (0.0, 0.0, 0.0); field = :interior, region = 1) ≈
          1.0 atol = 5e-5
end
