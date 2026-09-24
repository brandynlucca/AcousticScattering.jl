using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "special functions" @testset "special functions" begin
        for x in (0.5, 1.3, 4.2)
            @test AS.js(0, x) ≈ sin(x) / x
            @test AS.ys(0, x) ≈ -cos(x) / x
        end
        @test AS.js(0, 0.0) == 1.0
        @test AS.js(3, 0.0) == 0.0

        x, h = 2.7, 1e-6
        fd = (AS.js(2, x + h) - AS.js(2, x - h)) / 2h
        @test AS.jsd(2, x) ≈ fd atol = 1e-8

        fdc = (AS.besselj(2, x + h) - AS.besselj(2, x - h)) / 2h
        @test AS.jcd(2, x) ≈ fdc atol = 1e-8

        @test AS.legendre_p(0, 0.37) == 1.0
        @test AS.legendre_p(1, 0.37) == 0.37
        @test AS.legendre_p(2, 1.0) ≈ 1.0
        @test AS.legendre_p(2, -1.0) ≈ 1.0
        @test AS.legendre_p(3, -1.0) ≈ -1.0

        @test AS.neumann_factor(0) == 1
        @test AS.neumann_factor(1) == 2
    end

    @time "sphere modal series (golden values)" @testset "sphere modal series (golden values)" begin
        c_water = 1477.4
        a = 0.01

        # (freq_Hz, TS_rigid, TS_pressure_release)
        rigid_pr_cases = [
            (12000.0, -54.436999, -42.288189),
            (38000.0, -49.088291, -44.997865),
            (70000.0, -48.381378, -45.639879),
            (120000.0, -46.089148, -45.849080),
            (200000.0, -45.582900, -45.951078)
        ]
        sphere = AS.Sphere(a)
        for (freq, ts_rigid, ts_pr) in rigid_pr_cases
            k = 2pi * freq / c_water
            @test AS.target_strength(AS.modal(sphere, AS.Rigid(), k)) ≈ ts_rigid atol = 1e-4
            @test AS.target_strength(AS.modal(sphere, AS.PressureRelease(), k)) ≈ ts_pr atol = 1e-4
        end

        c_sphere = 1480.3
        rho_water = 1026.8
        rho_sphere = 1028.9
        g = rho_sphere / rho_water
        h = c_sphere / c_water
        fluid = AS.FluidFilled(g, h)

        fluid_cases = [
            (12000.0, -104.099721),
            (38000.0, -94.278687),
            (70000.0, -94.048765),
            (120000.0, -97.556969),
            (200000.0, -106.721041)
        ]
        for (freq, ts_fluid) in fluid_cases
            k = 2pi * freq / c_water
            @test AS.target_strength(AS.modal(sphere, fluid, k)) ≈ ts_fluid atol = 1e-3
        end

        @test AS.GasFilled === AS.FluidFilled
    end

    @time "sphere modal series (limits and consistency)" @testset "sphere modal series (limits and consistency)" begin
        a = 0.01
        c = 1477.4
        sphere = AS.Sphere(a)

        k_high = 2pi * 2.0e6 / c
        @test AS.target_strength(AS.modal(sphere, AS.Rigid(), k_high; m_max = 300)) ≈
              20 * log10(a / 2) atol = 0.05

        k = 2pi * 50000.0 / c
        ts1 = AS.target_strength(AS.modal(sphere, AS.Rigid(), k; m_max = 40))
        ts2 = AS.target_strength(AS.modal(sphere, AS.Rigid(), k; m_max = 80))
        @test ts1 ≈ ts2 atol = 1e-6

        f_back = AS.scattering_amplitude(AS.modal(sphere, AS.Rigid(), k; angle = pi))
        f_forward = AS.scattering_amplitude(AS.modal(sphere, AS.Rigid(), k; angle = 0.0))
        @test f_back != f_forward
    end
end

let
    @time "Matched fluid and pressure-release surface" @testset "Matched fluid and pressure-release surface" begin
        body, k = Sphere(1.0), 1.6
        matched = modal(body, FluidFilled(1.0, 1.0), k)
        points = [(0.0, 0.0, 0.0), (0.23, 0.34, 0.45), (1.0, 0.0, 0.0), (-2.0, 0.5, 0.0)]
        @test pressure(matched, points) ≈ pressure(matched, points; field = :incident) rtol = 1e-12
        @test iszero(scattering_amplitude(matched))
        @test iszero(pressure(matched, last(points); field = :scattered))
        @test pressure(matched, (0.0, 0.0, 0.0); field = :interior) ≈ 1.0
        @test_throws ArgumentError pressure(matched, (1.1, 0.0, 0.0); field = :interior)
        @test_throws ArgumentError pressure(matched, (0.1, 0.0, 0.0); field = :scattered)
        soft = modal(body, PressureRelease(), k)
        @test maximum(abs, pressure(soft, [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)])) < 1e-12
    end

    @time "Spherical pressure far-field limit and modal cutoff" @testset "Spherical pressure far-field limit and modal cutoff" begin
        body, k = Sphere(1.0), 1.6
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1)),
            theta in (0.0, pi / 3, pi)

            solution = modal(body, boundary, k; m_max = 12)
            r = 1e6
            point = (r * cos(theta), r * sin(theta), 0.0)
            expected = scattering_amplitude(modal(
                body, boundary, k; angle = theta, m_max = 12))
            @test r * cis(-k * r) * pressure(solution, point; field = :scattered) ≈ expected rtol = 1e-5
        end
        point = (1.01, 0.0, 0.0)
        coarse = pressure(modal(body, Rigid(), k; m_max = 0), point)
        fine = pressure(modal(body, Rigid(), k; m_max = 12), point)
        reference = pressure(modal(body, Rigid(), k; m_max = 18), point)
        @test abs(fine - reference) < abs(coarse - reference) / 1000
        @test_throws ArgumentError modal(body, Rigid(), k; m_max = -1)
    end
end
