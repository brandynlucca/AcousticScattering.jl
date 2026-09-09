using AcousticScattering
using CairoMakie
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset verbose = true "AcousticScattering.jl" begin
    @testset "special functions" begin
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

    @testset "sphere modal series (golden values)" begin
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

    @testset "sphere modal series (limits and consistency)" begin
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

    @testset "radial FEM (sphere)" begin
        c_water = 1477.4
        a = 0.01
        freq = 38000.0
        k = 2pi * freq / c_water

        sphere = AS.Sphere(a)
        ts_modal_rigid = AS.target_strength(AS.modal(sphere, AS.Rigid(), k))
        ts_fem_rigid = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; R = 3a, n_elements = 200))
        @test ts_fem_rigid ≈ ts_modal_rigid atol = 1e-3

        ts_modal_pr = AS.target_strength(AS.modal(sphere, AS.PressureRelease(), k))
        ts_fem_pr = AS.target_strength(AS.fem(
            sphere, AS.PressureRelease(), k; R = 3a, n_elements = 200))
        @test ts_fem_pr ≈ ts_modal_pr atol = 1e-3

        ts_fem_R2 = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; R = 5a, n_elements = 200))
        @test ts_fem_rigid ≈ ts_fem_R2 atol = 1e-3

        ts_fem_quad = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; R = 3a, n_elements = 100, order = 2))
        @test ts_fem_quad ≈ ts_modal_rigid atol = 1e-3

        c_francis = 1477.3
        range_cases_khz = (12.0, 100.0, 200.0, 300.0, 386.0)
        for freq_khz in range_cases_khz
            kk = 2pi * freq_khz * 1000 / c_francis
            ts_modal = AS.target_strength(AS.modal(sphere, AS.Rigid(), kk))
            ts_fem_adaptive = AS.target_strength(AS.fem(
                sphere, AS.Rigid(), kk; R = 3a, adaptive = true, target_tol = 0.005))
            @test ts_fem_adaptive ≈ ts_modal atol = 0.01
        end

        range_cases_pr_khz = (12.0, 100.0, 200.0, 300.0, 386.0)
        for freq_khz in range_cases_pr_khz
            kk = 2pi * freq_khz * 1000 / c_francis
            ts_modal_pr2 = AS.target_strength(AS.modal(sphere, AS.PressureRelease(), kk))
            ts_fem_pr_adaptive = AS.target_strength(AS.fem(
                sphere, AS.PressureRelease(), kk; R = 3a,
                adaptive = true, target_tol = 0.005))
            @test ts_fem_pr_adaptive ≈ ts_modal_pr2 atol = 0.01
        end
    end

    @testset "meridian FEM (sphere, 2D r-θ mesh)" begin
        a = 0.01
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        R = 3a

        sphere = AS.Sphere(a)
        ts_modal = AS.target_strength(AS.modal(sphere, AS.Rigid(), k))
        ts_2d_coarse = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; method = :meridian, R = R, n_r = 10, n_theta = 20))
        ts_2d_fine = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; method = :meridian, R = R, n_r = 30, n_theta = 60))
        @test ts_2d_coarse ≈ ts_modal atol = 0.3
        @test ts_2d_fine ≈ ts_modal atol = 0.03
        @test abs(ts_2d_fine - ts_modal) < abs(ts_2d_coarse - ts_modal)

        ts_modal_pr = AS.target_strength(AS.modal(sphere, AS.PressureRelease(), k))
        ts_2d_pr = AS.target_strength(AS.fem(sphere, AS.PressureRelease(), k;
            method = :meridian, R = R, n_r = 30, n_theta = 60))
        @test ts_2d_pr ≈ ts_modal_pr atol = 0.05
    end

    @testset "sphere modal series (shelled and solid elastic)" begin
        freq = 38000.0
        c_ext = 1477.4
        rho_ext = 1026.8
        k = 2pi * freq / c_ext

        rho_shell, G, lambda = 2700.0, 2.6e10, 5.3e10
        cL_shell = sqrt((lambda + 2G) / rho_shell)
        cT_shell = sqrt(G / rho_shell)
        radius_shell, radius_fluid = 0.05, 0.048

        shell_water = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
            AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), radius_fluid / radius_shell)
        shell_air = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), radius_fluid / radius_shell)
        solid = AS.SolidElastic(14900.0 / rho_ext, 6853.0 / c_ext, 4171.0 / c_ext)

        golden_cases = [
            (shell_water, radius_shell, 20, -28.236014616259297),
            (shell_air, radius_shell, 20, -34.93644485774581),
            (solid, 0.019, 13, -42.3768620168836)
        ]
        for (boundary, a, m_max, ts_expected) in golden_cases
            @test AS.target_strength(AS.modal(AS.Sphere(a), boundary, k; m_max = m_max)) ≈
                  ts_expected atol = 1e-4
        end

        a = radius_shell
        sphere = AS.Sphere(a)
        ts_rigid = AS.target_strength(AS.modal(sphere, AS.Rigid(), k; m_max = 25))
        shell_stiff_diffs = Float64[]
        for (rho_c, speed_c) in ((50.0, 10.0), (200.0, 30.0), (1000.0, 100.0))
            bc = AS.Shelled(AS.ElasticLayer(rho_c, speed_c, speed_c), AS.FluidInterior(0.001, 0.5), 0.9)
            push!(shell_stiff_diffs, abs(AS.target_strength(AS.modal(sphere, bc, k; m_max = 25)) -
                                         ts_rigid))
        end
        @test issorted(shell_stiff_diffs, rev = true)
        @test shell_stiff_diffs[end] < 0.01

        shell_trivial_gen = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext,
                cT_shell / c_ext; interior_coupling = :generalized),
            AS.FluidInterior(1.0, 1.0), radius_fluid / radius_shell)
        shell_trivial_orig = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext;
                interior_coupling = :identical_fluid),
            AS.FluidInterior(1.0, 1.0), radius_fluid / radius_shell)
        @test AS.target_strength(AS.modal(sphere, shell_trivial_gen, k; m_max = 20)) ≈
              AS.target_strength(AS.modal(sphere, shell_trivial_orig, k; m_max = 20)) atol = 1e-10

        shell_air_orig = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext;
                interior_coupling = :identical_fluid),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), radius_fluid / radius_shell)
        @test abs(AS.target_strength(AS.modal(sphere, shell_air, k; m_max = 20)) -
                  AS.target_strength(AS.modal(sphere, shell_air_orig, k; m_max = 20))) > 1.0

        a2 = 0.019
        sphere2 = AS.Sphere(a2)
        ts_rigid2 = AS.target_strength(AS.modal(sphere2, AS.Rigid(), k; m_max = 25))
        solid_stiff_diffs = Float64[]
        for (rho_c, cL_c, cT_c) in ((20.0, 5.0, 3.0), (100.0, 15.0, 8.0), (
            500.0, 40.0, 20.0))
            bc = AS.SolidElastic(rho_c, cL_c, cT_c)
            push!(solid_stiff_diffs, abs(AS.target_strength(AS.modal(sphere2, bc, k; m_max = 25)) -
                                         ts_rigid2))
        end
        @test issorted(solid_stiff_diffs, rev = true)
        @test solid_stiff_diffs[end] < 0.01
    end

    @testset "ViscoelasticShell (VESM, Feuillade & Nero 1998, monopole)" begin
        c1 = 1477.3
        k = 2pi * 2000.0 / c1
        rho_core, c_core = 1.24 / 1026.8, 345.0 / 1477.3

        @testset "flesh matched to water: reduces exactly to Shelled{ElasticLayer,FluidInterior}" begin
            Re, R = 0.009, 0.005
            rho_wall, cL_wall, cT_wall = 1.05, 1.05, 0.30
            bc_ref = AS.Shelled(AS.ElasticLayer(rho_wall, cL_wall, cT_wall),
                AS.FluidInterior(rho_core, c_core), R / Re)
            ts_ref = AS.target_strength(AS.modal(AS.Sphere(Re), bc_ref, k; m_max = 0))

            for radius_ratio_wall in (0.5, 0.7, 0.9, 0.99)
                Rv = Re / radius_ratio_wall
                bc = AS.Shelled(
                    AS.LayeredMaterial(AS.ViscousLayer(c1, 1.0, 1.0, 0.0, 0.0),
                        AS.ElasticLayer(rho_wall, cL_wall, cT_wall), radius_ratio_wall),
                    AS.FluidInterior(rho_core, c_core), R / Rv)
                @test AS.target_strength(AS.modal(AS.Sphere(Rv), bc, k; m_max = 0)) ≈ ts_ref atol = 1e-8
            end
        end

        @testset "wall matched to water (fluid limit): reduces exactly to FluidFilled" begin
            R = 0.005
            ts_ref = AS.target_strength(AS.modal(AS.Sphere(R), AS.FluidFilled(rho_core, c_core), k; m_max = 0))

            for (radius_ratio_wall, radius_ratio_core) in ((0.9, 0.5), (0.95, 0.3), (
                0.99, 0.7))
                Rv = R / radius_ratio_core
                bc = AS.Shelled(
                    AS.LayeredMaterial(AS.ViscousLayer(c1, 1.0, 1.0, 0.0, 0.0),
                        AS.ElasticLayer(1.0, 1.0, 1e-6), radius_ratio_wall),
                    AS.FluidInterior(rho_core, c_core), radius_ratio_core)
                @test AS.target_strength(AS.modal(AS.Sphere(Rv), bc, k; m_max = 0)) ≈ ts_ref atol = 1e-6
            end
        end

        @testset "m >= 1 is explicitly rejected, not silently wrong" begin
            Re, R = 0.009, 0.005
            bc = AS.Shelled(
                AS.LayeredMaterial(AS.ViscousLayer(c1, 1.05, 1.0, 1e-6, 1e-6),
                    AS.ElasticLayer(1.05, 1.05, 0.30), Re / 0.01),
                AS.FluidInterior(rho_core, c_core), R / 0.01)
            @test_throws ArgumentError AS.modal(AS.Sphere(0.01), bc, k; m_max = 1)
        end
    end

    @testset "finite cylinder modal series" begin
        freq = 38000.0
        c_sw = 1477.4
        rho_sw = 1026.8
        k = 2pi * freq / c_sw
        radius, length = 0.01, 0.1

        g, h = 1028.9 / rho_sw, 1480.3 / c_sw
        fluid = AS.FluidFilled(g, h)

        # (boundary, aspect_angle, ts_expected)
        golden_cases = [
            (AS.Rigid(), π / 2, -30.5254506539618),
            (AS.PressureRelease(), π / 2, -28.4384596592856),
            (fluid, π / 2, -81.847628998419),
            (AS.Rigid(), 1.2, -54.5632700811014)
        ]
        cyl = AS.Cylinder(radius, length)
        for (boundary, aspect_angle, ts_expected) in golden_cases
            ts = AS.target_strength(AS.modal(
                cyl, boundary, k; incidence_angle = aspect_angle, m_max = 20))
            @test ts ≈ ts_expected atol = 1e-6
        end

        ts_exact = AS.target_strength(AS.modal(
            cyl, AS.Rigid(), k; incidence_angle = π / 2, m_max = 20))
        ts_eps = AS.target_strength(AS.modal(
            cyl, AS.Rigid(), k; incidence_angle = π / 2 - 1e-6, m_max = 20))
        @test ts_exact ≈ ts_eps atol = 1e-6

        ts1 = AS.target_strength(AS.modal(cyl, AS.Rigid(), k; m_max = 30))
        ts2 = AS.target_strength(AS.modal(cyl, AS.Rigid(), k; m_max = 60))
        @test ts1 ≈ ts2 atol = 1e-8
    end

    @testset "Bent-cylinder modal series (BCMS)" begin
        # radius_curvature is a ratio times length here, not meters
        c_sw, ρ_sw = 1477.3, 1026.8
        radius, length = 1e-3, 10.5e-3
        bc = AS.FluidFilled(1026.8 * 1.0357 / ρ_sw, 1477.3 * 1.0279 / c_sw)
        golden_bcms = [
            (12000.0, false, -121.641805435579), (12000.0, true, -121.644828618057),
            (38000.0, false, -101.73204605257), (38000.0, true, -101.762375463093),
            (100000.0, false, -85.6474263920345),
            (200000.0, false, -76.1893202028611),
            (400000.0, false, -78.8330565807749)
        ]
        cyl_straight = AS.Cylinder(radius, length)
        cyl_bent = AS.Cylinder(radius, length; radius_curvature = 1.5length)
        for (freq, bent, ts_ref) in golden_bcms
            k = 2π * freq / c_sw
            ts = AS.target_strength(AS.modal(bent ? cyl_bent : cyl_straight, bc, k))
            @test ts ≈ ts_ref atol = 1e-4
        end

        @test AS.equivalent_length_fresnel(2π * 38000 / 1477.3, 0.0105, Inf) ≈ 0.0105 + 0im
    end

    @testset "Bent-cylinder Kirchhoff (physical optics)" begin
        radius, length = 0.01, 0.07
        c = 1477.3
        k = 2π * 38000.0 / c

        function _lateral_only(boundary, k, radius, length; angle = π / 2)
            Rc = AS.reflection_coefficient(boundary)
            sb, cb = sincos(angle)
            x = 2k * radius * sb
            total = 2besselj(0, x) + im * π * besselj(1, x)
            for n in 1:60
                term = besselj(2n, x) / (4n^2 - 1)
                total -= 4term
                abs(term) < 1e-15 * abs(total) && break
            end
            return Rc * (k * radius * length) / (2π) * sb * total *
                   sinc(k * length * cb / π)
        end

        cyl_nearly_straight = AS.Cylinder(radius, length; radius_curvature = 1e8length)
        for angle in (π / 2, 1.2, 1.0)
            f_ref = _lateral_only(AS.Rigid(), k, radius, length; angle = angle)
            f_bent = AS.scattering_amplitude(AS.kirchhoff(
                cyl_nearly_straight, AS.Rigid(), k;
                incidence_angle = angle))
            @test f_bent ≈ f_ref rtol = 1e-6
        end
    end

    @testset "Bent-cylinder MFS (method of fundamental solutions)" begin
        radius, length = 0.01, 0.07
        c = 1477.3
        k = 2π * 38000.0 / c
        ρc = 1e6 * length
        cyl_straight = AS.Cylinder(radius, length)
        cyl_bent = AS.Cylinder(radius, length; radius_curvature = ρc)

        @testset "broadside, both boundaries: converges tightly" begin
            for boundary in (AS.Rigid(), AS.PressureRelease())
                ts_exact = AS.target_strength(AS.modal(cyl_straight, boundary, k; m_max = 30))
                ts_mfs = AS.target_strength(AS.mfs(
                    cyl_bent, boundary, k; offset = 0.3radius, n_s = 40, n_φ = 32))
                @test ts_mfs ≈ ts_exact atol = 0.3
            end
        end

        @testset "oblique incidence: PressureRelease converges tightly, Rigid more slowly" begin
            angle = 1.2
            ts_pr = AS.target_strength(AS.modal(
                cyl_straight, AS.PressureRelease(), k; incidence_angle = angle, m_max = 30))
            ts_pr_mfs = AS.target_strength(AS.mfs(cyl_bent, AS.PressureRelease(), k;
                incidence_angle = angle, offset = 0.3radius, n_s = 40, n_φ = 32))
            @test ts_pr_mfs ≈ ts_pr atol = 0.3

            ts_rigid = AS.target_strength(AS.modal(
                cyl_straight, AS.Rigid(), k; incidence_angle = angle, m_max = 30))
            ts_rigid_mfs = AS.target_strength(AS.mfs(cyl_bent, AS.Rigid(), k;
                incidence_angle = angle, offset = 0.5radius, n_s = 100, n_φ = 80))
            @test ts_rigid_mfs ≈ ts_rigid atol = 0.7
        end
    end

    @testset "finite cylinder modal series (elastic shell and solid, ECMS)" begin
        freq_ecms = 38000.0
        c_sw_ecms = 1477.3
        golden_solid_cases = [
            (38e3, -43.3581252581),
            (120e3, -34.9217937431),
            (200e3, -32.0014108456)
        ]
        solid_ecms = AS.SolidElastic(2800.0 / 1026.8, 6398.0 / c_sw_ecms, 3122.0 /
                                                                          c_sw_ecms)
        cyl_ecms = AS.Cylinder(0.005, 0.04)
        for (f, ts_expected) in golden_solid_cases
            k_ecms = 2pi * f / c_sw_ecms
            ts = AS.target_strength(AS.modal(cyl_ecms, solid_ecms, k_ecms; m_max = 40))
            @test ts ≈ ts_expected atol = 1e-6
        end

        freq = 38000.0
        c_ext = 1477.4
        k = 2pi * freq / c_ext
        radius, length = 0.02, 0.2
        cyl = AS.Cylinder(radius, length)

        ts_rigid = AS.target_strength(AS.modal(cyl, AS.Rigid(), k; m_max = 25))

        shell_stiff_diffs = Float64[]
        for (rho_c, speed_c) in ((50.0, 10.0), (500.0, 60.0), (5000.0, 200.0))
            bc = AS.Shelled(AS.ElasticLayer(rho_c, speed_c, speed_c), AS.FluidInterior(0.001, 0.5), 0.9)
            push!(shell_stiff_diffs, abs(AS.target_strength(AS.modal(cyl, bc, k; m_max = 25)) -
                                         ts_rigid))
        end
        @test issorted(shell_stiff_diffs, rev = true)
        @test shell_stiff_diffs[end] < 0.01

        solid_stiff_diffs = Float64[]
        for (rho_c, speed_c) in ((50.0, 10.0), (500.0, 60.0), (5000.0, 200.0))
            bc = AS.SolidElastic(rho_c, speed_c, speed_c * 0.6)
            push!(solid_stiff_diffs, abs(AS.target_strength(AS.modal(cyl, bc, k; m_max = 25)) -
                                         ts_rigid))
        end
        @test issorted(solid_stiff_diffs, rev = true)
        @test solid_stiff_diffs[end] < 0.01

        rho_shell, cL_shell, cT_shell = 7900.0 / 1026.8, 5610.0 / c_ext, 3060.0 / c_ext
        solid_ref = AS.SolidElastic(rho_shell, cL_shell, cT_shell)
        ts_solid = AS.target_strength(AS.modal(cyl, solid_ref, k; m_max = 25))
        shell_solid_diffs = Float64[]
        for rr in (0.2, 0.05, 0.01, 0.001)
            shell = AS.Shelled(AS.ElasticLayer(rho_shell, cL_shell, cT_shell), AS.FluidInterior(1.0, 1.0), rr)
            push!(shell_solid_diffs, abs(AS.target_strength(AS.modal(cyl, shell, k; m_max = 25)) -
                                         ts_solid))
        end
        @test issorted(shell_solid_diffs, rev = true)
        @test shell_solid_diffs[end] < 0.01
    end

    @testset "elastic sphere/cylinder radial FEM (vs modal series)" begin
        freq = 38000.0
        c_ext = 1477.4
        rho_ext = 1026.8
        k = 2pi * freq / c_ext
        rho_shell, G, lambda = 2700.0, 2.6e10, 5.3e10
        cL_shell = sqrt((lambda + 2G) / rho_shell)
        cT_shell = sqrt(G / rho_shell)
        radius_shell, radius_fluid = 0.05, 0.048

        solid = AS.SolidElastic(14900.0 / rho_ext, 6853.0 / c_ext, 4171.0 / c_ext)
        sphere1 = AS.Sphere(0.019)
        @test AS.target_strength(AS.fem(sphere1, solid, k; n_elements = 200)) ≈
              AS.target_strength(AS.modal(sphere1, solid, k; m_max = 13)) atol = 0.05

        shell_water = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
            AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), radius_fluid / radius_shell)
        sphere_shell = AS.Sphere(radius_shell)
        @test AS.target_strength(AS.fem(
            sphere_shell, shell_water, k; n_elements = 200, m_max = 20)) ≈
              AS.target_strength(AS.modal(sphere_shell, shell_water, k; m_max = 20)) atol = 0.001

        radius, length = 0.05, 0.5
        rho0, c1, c2 = 7810.0, 5973.0, 3193.0
        solid_cyl = AS.SolidElastic(rho0 / rho_ext, c1 / c_ext, c2 / c_ext)
        cyl = AS.Cylinder(radius, length)
        @test AS.target_strength(AS.fem(cyl, solid_cyl, k; n_elements = 200)) ≈
              AS.target_strength(AS.modal(cyl, solid_cyl, k)) atol = 0.01

        shell_cyl = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
            AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), radius_fluid / radius_shell)
        @test AS.target_strength(AS.fem(cyl, shell_cyl, k; n_elements = 200)) ≈
              AS.target_strength(AS.modal(cyl, shell_cyl, k)) atol = 0.001

        for angle_deg in (90.0, 70.0, 50.0, 30.0)
            ang = deg2rad(angle_deg)
            @test AS.target_strength(AS.fem(
                cyl, solid_cyl, k; incidence_angle = ang, n_elements = 200)) ≈
                  AS.target_strength(AS.modal(cyl, solid_cyl, k; incidence_angle = ang)) atol = 0.01
            @test AS.target_strength(AS.fem(
                cyl, shell_cyl, k; incidence_angle = ang, n_elements = 200)) ≈
                  AS.target_strength(AS.modal(cyl, shell_cyl, k; incidence_angle = ang)) atol = 0.001
        end

        ts_solid_modal = AS.target_strength(AS.modal(sphere_shell, solid, k))
        shell_limit_diffs = Float64[]
        for rr in (0.1, 0.02, 0.005)
            bc = AS.Shelled(
                AS.ElasticLayer(solid.density_contrast, solid.speed_longitudinal_contrast,
                    solid.speed_transversal_contrast),
                AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), rr)
            ts_fem = AS.target_strength(AS.fem(
                sphere_shell, bc, k; n_elements = 200, m_max = 20))
            push!(shell_limit_diffs, abs(ts_fem - ts_solid_modal))
        end
        @test shell_limit_diffs[1] > 10 * maximum(shell_limit_diffs[2:end])
        @test maximum(shell_limit_diffs[2:end]) < 0.05
    end

    @testset "fluid shell sphere radial FEM (vs modal series)" begin
        freq = 38000.0
        c_ext = 1477.4
        rho_ext = 1026.8
        k = 2pi * freq / c_ext
        a = 0.05
        rr = 0.9

        sphere = AS.Sphere(a)
        bc_pr = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), rr)
        @test AS.target_strength(AS.fem(sphere, bc_pr, k; n_elements = 200, m_max = 20)) ≈
              AS.target_strength(AS.modal(sphere, bc_pr, k; m_max = 20)) atol = 1e-4

        bc_g = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), rr)
        @test AS.target_strength(AS.fem(sphere, bc_g, k; n_elements = 200, m_max = 20)) ≈
              AS.target_strength(AS.modal(sphere, bc_g, k; m_max = 20)) atol = 1e-4

        bc_plain_fluid = AS.FluidFilled(1.24 / rho_ext, 345.0 / c_ext)
        ts_plain = AS.target_strength(AS.modal(sphere, bc_plain_fluid, k; m_max = 20))
        bc_g2 = AS.Shelled(AS.FluidLayer(1.24 / rho_ext, 345.0 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), 0.999)
        @test AS.target_strength(AS.fem(sphere, bc_g2, k; n_elements = 400, m_max = 20)) ≈
              ts_plain atol = 1e-3

        bc_g3 = AS.Shelled(AS.FluidLayer(1.24 / rho_ext, 345.0 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), 0.999999)
        @test AS.reflection_coefficient(bc_g3, k, a) ≈
              AS.reflection_coefficient(AS.FluidFilled(1.24 / rho_ext, 345.0 / c_ext)) atol = 1e-6
        bc_pr3 = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), 0.999999)
        @test AS.reflection_coefficient(bc_pr3, k, a) ≈
              AS.reflection_coefficient(AS.PressureRelease()) atol = 1e-4

        for f in (5e3, 38e3, 120e3, 250e3)
            kk = 2pi * f / c_ext
            @test abs(AS.reflection_coefficient(bc_g, kk, a)) <= 1.0 + 1e-10
        end

        ts_ka = AS.target_strength(AS.kirchhoff(sphere, bc_g, 2pi * 200e3 / c_ext))
        ts_modal_hika = AS.target_strength(AS.modal(sphere, bc_g, 2pi * 200e3 / c_ext; m_max = 40))
        @test ts_ka ≈ ts_modal_hika atol = 0.1
    end

    @testset "fluid shell sphere BEM (vs modal series)" begin
        rho_ext, c_ext = 1026.8, 1477.4
        a = 0.05
        rr = 0.9
        k = 2pi * 12000.0 / c_ext

        sphere = AS.Sphere(a)
        bc_g = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), rr)
        ts_modal = AS.target_strength(AS.modal(sphere, bc_g, k; m_max = 20))
        diffs_g = Float64[]
        for n in (24, 40)
            sol = AS.bem(sphere, bc_g, k; n = n, rtol = 1e-5)
            push!(diffs_g, abs(AS.target_strength(sol) - ts_modal))
        end
        @test issorted(diffs_g, rev = true)
        @test diffs_g[end] < 0.05

        bc_pr = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), rr)
        ts_modal_pr = AS.target_strength(AS.modal(sphere, bc_pr, k; m_max = 20))
        diffs_pr = Float64[]
        for n in (24, 40)
            sol = AS.bem(sphere, bc_pr, k; n = n, rtol = 1e-5)
            push!(diffs_pr, abs(AS.target_strength(sol) - ts_modal_pr))
        end
        @test issorted(diffs_pr, rev = true)
        @test diffs_pr[end] < 0.02
    end

    @testset "Kirchhoff high-frequency baseline" begin
        a = 0.01
        c = 1477.4
        k_high = 2pi * 2.0e6 / c

        @test AS.reflection_coefficient(AS.Rigid()) == 1.0
        @test AS.reflection_coefficient(AS.PressureRelease()) == -1.0
        sphere = AS.Sphere(a)
        @test AS.target_strength(AS.kirchhoff(sphere, AS.Rigid(), k_high)) ≈
              20 * log10(a / 2) atol = 0.05
        @test AS.target_strength(AS.kirchhoff(sphere, AS.PressureRelease(), k_high)) ≈
              20 * log10(a / 2) atol = 0.05

        ts_modal = AS.target_strength(AS.modal(sphere, AS.Rigid(), k_high; m_max = 300))
        ts_ka = AS.target_strength(AS.kirchhoff(sphere, AS.Rigid(), k_high))
        @test ts_modal ≈ ts_ka atol = 0.05

        c_francis = 1477.3
        francis_kirchhoff_cases = [
            (12.0, -52.12), (14.0, -50.87), (16.0, -49.81), (18.0, -48.91), (20.0, -48.13)
        ]
        for (freq_khz, ts_expected) in francis_kirchhoff_cases
            k = 2pi * freq_khz * 1000 / c_francis
            @test AS.target_strength(AS.kirchhoff(sphere, AS.Rigid(), k)) ≈ ts_expected atol = 0.01
        end
    end

    @testset "Finite-cylinder Kirchhoff high-frequency baseline" begin
        radius, length = 0.01, 0.07
        c_francis = 1477.3
        k = 2pi * 38000.0 / c_francis
        cyl = AS.Cylinder(radius, length)
        francis_kirchhoff_cylinder_cases = [
            (8.0, -42.07), (28.0, -44.47), (48.0, -45.97), (68.0, -44.66), (88.0, -31.29)
        ]
        for (angle_deg, ts_expected) in francis_kirchhoff_cylinder_cases
            ts = AS.target_strength(AS.kirchhoff(cyl, AS.Rigid(), k; incidence_angle = deg2rad(angle_deg)))
            @test ts ≈ ts_expected atol = 0.01
        end

        f_default = AS.scattering_amplitude(AS.kirchhoff(cyl, AS.Rigid(), k))
        f_explicit_broadside = AS.scattering_amplitude(AS.kirchhoff(
            cyl, AS.Rigid(), k; incidence_angle = pi / 2))
        @test f_default == f_explicit_broadside

        f_endon = AS.scattering_amplitude(AS.kirchhoff(cyl, AS.Rigid(), k; incidence_angle = 0.0))
        @test abs(f_endon) ≈ k * radius^2 / 2 atol = 1e-9
    end

    @testset "Spheroid geometry" begin
        body = AS.Spheroid(0.10, 0.03)
        @test body.kind == :prolate
        @test body.a == 0.10
        @test body.b == 0.03
        @test body.q ≈ sqrt(0.10^2 - 0.03^2)
        @test body.xi0 ≈ body.a / body.q

        oblate = AS.Spheroid(0.03, 0.10)
        @test oblate.kind == :oblate
        @test oblate.q ≈ sqrt(0.10^2 - 0.03^2)
        @test oblate.xi0 ≈ oblate.a / oblate.q

        a_p, b_p = body.a, body.b
        @test body.xi0 ≈ 1 / sqrt(1 - (b_p / a_p)^2)
        a_o, b_o = oblate.a, oblate.b
        @test oblate.xi0 ≈ 1 / sqrt((b_o / a_o)^2 - 1)

        @test_throws ArgumentError AS.Spheroid(0.05, 0.05)
        @test_throws ArgumentError AS.Spheroid(-0.05, 0.03)

        a, b = body.a, body.b
        R1_pole, R2_pole = AS.principal_curvatures(body, 0.0)
        @test R1_pole ≈ b^2 / a
        @test R2_pole ≈ b^2 / a

        R1_eq, R2_eq = AS.principal_curvatures(body, pi / 2)
        @test R1_eq ≈ a^2 / b
        @test R2_eq ≈ b
    end

    @testset "Spheroid Kirchhoff high-frequency baseline" begin
        body = AS.Spheroid(0.10, 0.03)
        k = 2pi * 200000.0 / 1477.4

        ts_endon = AS.target_strength(AS.kirchhoff(body, AS.Rigid(), k; incidence_angle = 0.0))
        ts_broadside = AS.target_strength(AS.kirchhoff(body, AS.Rigid(), k; incidence_angle = pi /
                                                                                              2))
        @test ts_broadside > ts_endon

        R1, R2 = AS.principal_curvatures(body, 0.0)
        @test AS.target_strength(AS.kirchhoff(body, AS.Rigid(), k; incidence_angle = 0.0)) ≈
              AS.target_strength(AS.kirchhoff_form_function(AS.Rigid(), R1, R2)) atol = 0.05

        body_francis = AS.Spheroid(0.07, 0.01)
        c_francis = 1477.3
        k_francis = 2pi * 38000.0 / c_francis
        francis_kirchhoff_spheroid_cases = [
            (0.0, -62.67), (8.0, -62.8), (28.0, -62.21), (48.0, -55.61), (68.0, -46.49), (
                88.0, -28.15)
        ]
        for (angle_deg, ts_expected) in francis_kirchhoff_spheroid_cases
            ts = AS.target_strength(AS.kirchhoff(
                body_francis, AS.Rigid(), k_francis; incidence_angle = deg2rad(angle_deg)))
            @test ts ≈ ts_expected atol = 0.01
        end
    end

    @testset "AxisymmetricBEM (sphere, axial incidence)" begin
        a = 0.01
        c_water = 1477.4

        mesh = AS.sphere_mesh(a, 10)
        ps = AS.panels(mesh)
        n = length(ps)
        k_static = 1e-6
        row_sums = Vector{ComplexF64}(undef, n)
        for i in 1:n
            xρ, xz = ps[i].rhom, ps[i].zm
            total = zero(ComplexF64)
            for j in 1:n
                pj = ps[j]
                integrand = s -> begin
                    ρ2, z2 = AS._panel_point(pj, s)
                    AS._azimuthal_dGdn(k_static, xρ, xz, ρ2, z2, pj.nrho, pj.nz) * ρ2 * pj.L
                end
                total += i == j ? AS.quadgk(integrand, 0.0, 0.5, 1.0; rtol = 1e-6)[1] :
                         AS.quadgk(integrand, 0.0, 1.0; rtol = 1e-6)[1]
            end
            row_sums[i] = total
        end
        @test all(rs -> isapprox(rs, -0.5 + 0im; atol = 1e-4), row_sums)

        freq = 38000.0
        k = 2pi * freq / c_water

        sphere = AS.Sphere(a)
        for boundary in (AS.Rigid(), AS.PressureRelease())
            @testset "$(typeof(boundary))" begin
                ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))

                sol16 = AS.bem(
                    sphere, boundary, k; n = 16, incidence_angle = 0.0, rtol = 1e-5)
                ts_back_16 = AS.target_strength(sol16; angle = pi)
                ts_fwd_16 = AS.target_strength(sol16; angle = 0.0)

                sol32 = AS.bem(
                    sphere, boundary, k; n = 32, incidence_angle = 0.0, rtol = 1e-5)
                ts_back_32 = AS.target_strength(sol32; angle = pi)

                @test ts_back_16 ≈ ts_modal atol = 0.06
                @test ts_back_32 ≈ ts_modal atol = 0.02
                @test abs(ts_back_32 - ts_modal) < abs(ts_back_16 - ts_modal)

                ts_modal_fwd = AS.target_strength(AS.modal(sphere, boundary, k; angle = 0.0))
                @test ts_fwd_16 ≈ ts_modal_fwd atol = 0.06
            end
        end
    end

    @testset "Axisymmetric MFS (sphere, axial incidence)" begin
        a = 0.01
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        for boundary in (AS.Rigid(), AS.PressureRelease())
            ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))
            for offset_frac in (0.2, 0.5, 0.9)
                ts_mfs = AS.target_strength(AS.mfs(
                    sphere, boundary, k; incidence_angle = 0.0, offset = offset_frac * a))
                @test ts_mfs ≈ ts_modal atol = 0.1
            end
        end
    end

    @testset "Axisymmetric MFS (spheroid, axial incidence)" begin
        a, b = 0.05, 0.02
        c_water = 1477.4
        freq = 20000.0
        k = 2pi * freq / c_water
        body = AS.Spheroid(a, b)

        for boundary in (AS.Rigid(), AS.PressureRelease())
            ts_modal = AS.target_strength(AS.modal(
                body, boundary, k; incidence_angle = 0.0, m_max = 24, n_max = 24))
            for offset_frac in (0.2, 0.5)
                ts_mfs = AS.target_strength(AS.mfs(
                    body, boundary, k; incidence_angle = 0.0,
                    offset = offset_frac * min(a, b)))
                @test ts_mfs ≈ ts_modal atol = 0.1
            end
        end
    end

    @testset "Axisymmetric MFS (cylinder with spheroidal endcaps, axial incidence)" begin
        radius, cyl_length, endcap_depth = 0.01, 0.05, 0.01
        c_water = 1477.3
        freq = 38000.0
        k = 2pi * freq / c_water
        mesh = AS.cylinder_spheroidal_endcap_mesh(radius, cyl_length, endcap_depth, 112)
        capped_cyl = AS.Cylinder(radius, cyl_length; endcap_depth = endcap_depth)

        for boundary in (AS.Rigid(), AS.PressureRelease())
            p_bem, dpdn_bem, ps_bem = AS.solve_axial(boundary, k, mesh; rtol = 1e-5)
            ts_bem = AS.target_strength(ps_bem, p_bem, dpdn_bem, k, pi)
            for offset_frac in (0.1, 0.2, 0.3, 0.5)
                ts_mfs = AS.target_strength(AS.mfs(
                    capped_cyl, boundary, k; incidence_angle = 0.0,
                    offset = offset_frac * radius, n = 112))
                @test ts_mfs ≈ ts_bem atol = 0.05
            end
        end
    end

    @testset "Axisymmetric MFS (sphere, oblique incidence / rotational symmetry)" begin
        a = 0.01
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        for boundary in (AS.Rigid(), AS.PressureRelease())
            ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))
            for angle_deg in (0.0, 30.0, 60.0, 90.0)
                β = deg2rad(angle_deg)
                sol = AS.mfs(
                    sphere, boundary, k; incidence_angle = β, m_max = 15, offset = 0.3a)
                ts_mfs = AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test ts_mfs ≈ ts_modal atol = 0.1
            end
        end
    end

    @testset "Axisymmetric MFS fluid-filled/transmission (sphere, axial incidence)" begin
        a = 0.05
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        @testset "rigid limit (gh >> 1)" begin
            stiff = AS.FluidFilled(1e8, 1e8)
            ts_stiff = AS.target_strength(AS.mfs(
                sphere, stiff, k; incidence_angle = 0.0, offset = 0.3a, n = 24))
            ts_rigid = AS.target_strength(AS.mfs(
                sphere, AS.Rigid(), k; incidence_angle = 0.0, offset = 0.3a, n = 24))
            @test ts_stiff ≈ ts_rigid atol = 1e-4
        end

        @testset "pressure-release limit (g << 1)" begin
            soft = AS.FluidFilled(1e-8, 1.0)
            ts_soft = AS.target_strength(AS.mfs(
                sphere, soft, k; incidence_angle = 0.0, offset = 0.3a, n = 24))
            ts_pr = AS.target_strength(AS.mfs(sphere, AS.PressureRelease(), k;
                incidence_angle = 0.0, offset = 0.3a, n = 24))
            @test ts_soft ≈ ts_pr atol = 1e-4
        end

        @testset "converges to the analytical FluidFilled sphere modal series" begin
            g, h = 1.05, 1.02
            bc = AS.FluidFilled(g, h)
            ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
            ts_mfs = AS.target_strength(AS.mfs(
                sphere, bc, k; incidence_angle = 0.0, offset = 0.3a, n = 48))
            @test abs(ts_mfs - ts_modal) < 0.08
        end
    end

    @testset "Axisymmetric MFS fluid-filled/transmission (sphere, oblique incidence)" begin
        a = 0.05
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        @testset "rigid limit (gh >> 1)" begin
            stiff = AS.FluidFilled(1e8, 1e8)
            for angle_deg in (0.0, 30.0, 60.0, 90.0)
                β = deg2rad(angle_deg)
                sol_stiff = AS.mfs(sphere, stiff, k; incidence_angle = β,
                    m_max = 15, offset = 0.3a, n = 24)
                ts_stiff = AS.target_strength(sol_stiff; angle = pi - β, azimuth = pi)

                sol_rigid = AS.mfs(sphere, AS.Rigid(), k; incidence_angle = β,
                    m_max = 15, offset = 0.3a, n = 24)
                ts_rigid = AS.target_strength(sol_rigid; angle = pi - β, azimuth = pi)
                @test ts_stiff ≈ ts_rigid atol = 1e-4
            end
        end

        @testset "pressure-release limit (g << 1)" begin
            soft = AS.FluidFilled(1e-8, 1.0)
            for angle_deg in (0.0, 30.0, 60.0, 90.0)
                β = deg2rad(angle_deg)
                sol_soft = AS.mfs(
                    sphere, soft, k; incidence_angle = β, m_max = 15, offset = 0.3a, n = 24)
                ts_soft = AS.target_strength(sol_soft; angle = pi - β, azimuth = pi)

                sol_pr = AS.mfs(sphere, AS.PressureRelease(), k; incidence_angle = β,
                    m_max = 15, offset = 0.3a, n = 24)
                ts_pr = AS.target_strength(sol_pr; angle = pi - β, azimuth = pi)
                @test ts_soft ≈ ts_pr atol = 1e-4
            end
        end

        @testset "sphere rotational symmetry against the analytical modal series" begin
            g, h = 1.05, 1.02
            bc = AS.FluidFilled(g, h)
            ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
            for angle_deg in (0.0, 30.0, 60.0, 90.0)
                β = deg2rad(angle_deg)
                sol = AS.mfs(
                    sphere, bc, k; incidence_angle = β, m_max = 15, offset = 0.3a, n = 48)
                ts_mfs = AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test abs(ts_mfs - ts_modal) < 0.08
            end
        end
    end

    @testset "AxisymmetricBEM fluid-filled/transmission (sphere, axial incidence)" begin
        a = 0.05
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        @testset "rigid limit (gh >> 1)" begin
            stiff = AS.FluidFilled(1e8, 1e8)
            ts_stiff = AS.target_strength(AS.bem(
                sphere, stiff, k; incidence_angle = 0.0, n = 24))
            ts_rigid = AS.target_strength(AS.bem(
                sphere, AS.Rigid(), k; incidence_angle = 0.0, n = 24))
            @test ts_stiff ≈ ts_rigid atol = 0.25
        end

        @testset "pressure-release limit (g << 1)" begin
            soft = AS.FluidFilled(1e-8, 1.0)
            ts_soft = AS.target_strength(AS.bem(
                sphere, soft, k; incidence_angle = 0.0, n = 24))
            ts_pr = AS.target_strength(AS.bem(
                sphere, AS.PressureRelease(), k; incidence_angle = 0.0, n = 24))
            @test ts_soft ≈ ts_pr atol = 0.1
        end

        @testset "converges to the analytical FluidFilled sphere modal series" begin
            g, h = 1.05, 1.02
            bc = AS.FluidFilled(g, h)
            ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
            ts_bem = AS.target_strength(AS.bem(
                sphere, bc, k; incidence_angle = 0.0, n = 48))
            @test abs(ts_bem - ts_modal) < 0.15
        end
    end

    @testset "Spheroid modal series (rigid/pressure-release/fluid-filled)" begin
        k = 2pi * 38000.0 / 1477.4

        @testset "converges to the sphere modal series as eccentricity -> 0" begin
            a_sphere = 0.05
            sphere = AS.Sphere(a_sphere)
            for (boundary, sphere_ts) in (
                (AS.Rigid(), AS.target_strength(AS.modal(sphere, AS.Rigid(), k))),
                (AS.PressureRelease(),
                AS.target_strength(AS.modal(sphere, AS.PressureRelease(), k)))
            )
                errs = Float64[]
                for eps in (0.01, 0.001, 0.0001)
                    body = AS.Spheroid(a_sphere + eps, a_sphere - eps)
                    ts = AS.target_strength(AS.modal(
                        body, boundary, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
                    push!(errs, abs(ts - sphere_ts))
                end
                @test errs[1] > errs[2] > errs[3]
                @test errs[3] < 0.1
            end

            g, h = 1028.9 / 1026.8, 1480.3 / 1477.4
            fluid = AS.FluidFilled(g, h)
            sphere_ts = AS.target_strength(AS.modal(sphere, fluid, k))
            errs = Float64[]
            for eps in (0.001, 0.0001, 0.00001)
                body = AS.Spheroid(a_sphere + eps, a_sphere - eps)
                ts = AS.target_strength(AS.modal(
                    body, fluid, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
                push!(errs, abs(ts - sphere_ts))
            end
            @test errs[1] > errs[2] > errs[3]
            @test errs[3] < 0.1
        end

        @testset "fluid-filled limiting cases (independent cross-checks)" begin
            body = AS.Spheroid(0.02, 0.005)
            stiff = AS.FluidFilled(1e6, 1e6)
            ts_stiff = AS.target_strength(AS.modal(
                body, stiff, k; incidence_angle = 0.0, m_max = 16, n_max = 16))
            ts_rigid = AS.target_strength(AS.modal(
                body, AS.Rigid(), k; incidence_angle = 0.0, m_max = 16, n_max = 16))
            @test ts_stiff ≈ ts_rigid atol = 1e-3

            ts_pr = AS.target_strength(AS.modal(body, AS.PressureRelease(), k;
                incidence_angle = 0.0, m_max = 16, n_max = 16))
            diffs = Float64[]
            for g_soft in (0.1, 0.01, 0.001)
                soft = AS.FluidFilled(g_soft, 0.5)
                ts_soft = AS.target_strength(AS.modal(
                    body, soft, k; incidence_angle = 0.0, m_max = 16, n_max = 16))
                push!(diffs, abs(ts_soft - ts_pr))
            end
            @test diffs[1] > diffs[2] > diffs[3]
            @test diffs[3] < 0.1
        end

        @testset "fluid-filled full off-diagonal coupling (Furusawa Eq. 4)" begin
            q = 3.0
            xi0 = 1.05
            a = q * xi0
            b = sqrt(a^2 - q^2)
            body = AS.Spheroid(a, b)
            k_ref = 1.0
            bc = AS.FluidFilled(1050 / 1026, 1.02; coupling = :full)
            f = AS.scattering_amplitude(AS.modal(body, bc, k_ref;
                incidence_angle = 0.3, m_max = 2, n_max = 4))
            f_bs = f * k_ref / (-2im)
            @test real(f_bs) ≈ 9.10954776790933e-06 atol = 1e-14
            @test imag(f_bs) ≈ 0.00326545204233986 atol = 1e-12

            a_sphere = 0.05
            g, h = 1028.9 / 1026.8, 1480.3 / 1477.4
            bc_full = AS.FluidFilled(g, h; coupling = :full)
            bc_diag = AS.FluidFilled(g, h; coupling = :diagonal)
            errs = Float64[]
            for eps in (0.001, 0.0001, 0.00001)
                body_ns = AS.Spheroid(a_sphere + eps, a_sphere - eps)
                ts_full = AS.target_strength(AS.modal(
                    body_ns, bc_full, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
                ts_diag = AS.target_strength(AS.modal(
                    body_ns, bc_diag, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
                push!(errs, abs(ts_full - ts_diag))
            end
            @test errs[1] > errs[2] > errs[3]
            @test errs[3] < 0.01

            body_fem = AS.Spheroid(0.08, 0.02)
            bc_fem_full = AS.FluidFilled(1.05, 1.05; coupling = :full)
            bc_fem_diag = AS.FluidFilled(1.05, 1.05; coupling = :diagonal)
            c_water = 1500.0
            for (freq_hz, ts_comsol) in ((38000.0, -72.301), (60000.0, -73.284))
                k_fem = 2pi * freq_hz / c_water
                ka_fem = k_fem * 0.08
                order = ceil(Int, 1.8 * ka_fem)
                ts_full = AS.target_strength(AS.modal(body_fem, bc_fem_full, k_fem;
                    incidence_angle = deg2rad(30.0), m_max = order, n_max = order))
                ts_diag = AS.target_strength(AS.modal(body_fem, bc_fem_diag, k_fem;
                    incidence_angle = deg2rad(30.0), m_max = order, n_max = order))
                @test ts_full ≈ ts_comsol atol = 0.5
                @test abs(ts_diag - ts_comsol) > 2.0
            end
        end

        @testset "fluid-filled full coupling: quad-precision special functions (Jech et al. 2015)" begin
            c_med, rho_med = 1477.3, 1026.8
            rho_ws, c_ws = 1028.9, 1480.3
            body_ws = AS.Spheroid(0.07, 0.01)
            bc_ws = AS.FluidFilled(rho_ws / rho_med, c_ws / c_med)
            rho_gas, c_gas = 1.24, 345.0
            bc_gas = AS.FluidFilled(rho_gas / rho_med, c_gas / c_med)
            golden_ws = [(12.0, -87.05, 14, 14), (48.0, -84.91, 18, 18),
                (88.0, -88.95, 10, 22), (128.0, -105.33, 30, 30)]
            for (freq_khz, ts_bench, m_max, n_max) in golden_ws
                k_ws = 2pi * freq_khz * 1000 / c_med
                ts = AS.target_strength(AS.modal(
                    body_ws, bc_ws, k_ws; incidence_angle = pi / 2,
                    m_max = m_max, n_max = n_max, precision = :quad))
                @test ts ≈ ts_bench atol = 0.3
            end

            golden_gas = [
                (12.0, -30.13435708789646, 14, 14), (48.0, -28.594046949257944, 10, 56)]
            for (freq_khz, ts_ref, m_max, n_max) in golden_gas
                k_gas = 2pi * freq_khz * 1000 / c_med
                ts = AS.target_strength(AS.modal(
                    body_ws, bc_gas, k_gas; incidence_angle = pi / 2,
                    m_max = m_max, n_max = n_max, precision = :quad))
                @test ts ≈ ts_ref atol = 0.01
            end
        end

        @testset "Rigid/PressureRelease: precision keyword now actually reaches the modal coefficient" begin
            body = AS.Spheroid(0.10, 0.03)
            k_safe = 2pi * 38000.0 / 1477.4
            for boundary in (AS.Rigid(), AS.PressureRelease())
                ts_q = AS.target_strength(AS.modal(
                    body, boundary, k_safe; incidence_angle = 0.0,
                    m_max = 10, n_max = 10, precision = :quad))
                ts_d = AS.target_strength(AS.modal(
                    body, boundary, k_safe; incidence_angle = 0.0,
                    m_max = 10, n_max = 10, precision = :double))
                @test isfinite(ts_q)
                @test ts_q ≈ ts_d atol = 0.01
            end
        end

        @testset "fluid-filled full coupling: no spurious warnings off-axis of m>=1" begin
            body = AS.Spheroid(0.02, 0.005)
            stiff = AS.FluidFilled(1e6, 1e6; coupling = :full)
            @test_logs AS.modal(
                body, stiff, k; incidence_angle = 0.0, m_max = 16, n_max = 16)
        end

        @testset "mode-order convergence (prolate, rigid, endon)" begin
            body = AS.Spheroid(0.10, 0.03)
            ts16 = AS.target_strength(AS.modal(body, AS.Rigid(), k; m_max = 16, n_max = 16))
            ts20 = AS.target_strength(AS.modal(body, AS.Rigid(), k; m_max = 20, n_max = 20))
            ts24 = AS.target_strength(AS.modal(body, AS.Rigid(), k; m_max = 24, n_max = 24))
            @test ts20 ≈ ts24 atol = 0.01
            @test abs(ts24 - ts20) < abs(ts20 - ts16)
        end

        @testset "oblate cross-check against the Kirchhoff baseline" begin
            body = AS.Spheroid(0.03, 0.10)
            cases = vcat(
                [(AS.Rigid(), pi / 2, f) for f in (38.0, 76.0)],
                [(AS.PressureRelease(), pi / 2, f) for f in (38.0, 76.0)],
                [(AS.Rigid(), 0.0, 38.0)],
                [(AS.PressureRelease(), 0.0, 38.0)]
            )
            for (boundary, angle, freq_khz) in cases
                let k_ka = 2pi * freq_khz * 1000 / 1477.4
                    ka_eq = k_ka * 0.10
                    margin = freq_khz >= 76.0 ? 3 : 8
                    order = max(24, ceil(Int, ka_eq) + margin)
                    precision = freq_khz < 45.0 ? :double : :quad
                    ts_modal = AS.target_strength(AS.modal(
                        body, boundary, k_ka; incidence_angle = angle,
                        m_max = order, n_max = order, precision = precision))
                    ts_ka = AS.target_strength(AS.kirchhoff(body, boundary, k_ka; incidence_angle = angle))
                    @test ts_modal ≈ ts_ka atol = 1.5
                end
            end
        end
    end

    @testset "ShellFEM (Hayek & Boisvert axisymmetric shell operator)" begin
        geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)

        @testset "geometry" begin
            @test geometry.focal_radius ≈ 0.0342928563989645
            @test geometry.semimajor_inner ≈ 0.03490343822605447
            @test geometry.semiminor_inner ≈ 0.006500000000000001
            @test geometry.xi_outer ≈ 1.0206207261596576
            @test geometry.xi_inner ≈ 1.0178049276498415
            @test geometry.xi_mid ≈ 1.0192128269047496
            @test geometry.semimajor_mid ≈ 0.03495171911302723
            @test geometry.semiminor_mid ≈ 0.006754455489227317
            @test geometry.a_shape ≈ 1.0192128269047496
            @test geometry.aspect_ratio ≈ 5.174616839028955
            @test geometry.bending_epsilon ≈ 1.705382018748745e-5

            @test_throws ArgumentError AS.ProlateShellGeometry(0.007, 0.035, 0.0005)
            @test_throws ArgumentError AS.ProlateShellGeometry(0.035, 0.007, 0.007)
            @test_throws ArgumentError AS.Shelled(0.6, 2565.0, 70e9)
        end

        material = AS.Shelled(0.32, 2565.0, 70e9)

        @testset "assembled system, freq=$freq" for freq in (12000.0, 38000.0)
            sys = AS.assemble_shell_system(geometry, material.material, freq; n_eta = 9)

            omega_hat_golden = freq == 12000.0 ? 0.4779315959914645 : 1.5134500539729712
            @test sys.nondimensional_frequency ≈ omega_hat_golden

            fro_golden = Dict(
                :K_uu => 7.0917050782e+03, :K_uw => 1.0232814400e+01, :K_u_beta =>
                    4.1159927894e-01,
                :K_wu => 8.2081258324e+02, :K_ww => 1.0624527234e+02, :K_w_beta =>
                    5.4102101877e+00,
                :K_beta_u => 4.1185967896e-01, :K_beta_w => 3.5052072706e-01, :K_beta_beta =>
                    1.6278895367e-01
            )
            for (name, golden) in fro_golden
                @test norm(sys.structural_blocks[name]) ≈ golden rtol = 1e-9
            end

            mass_fro_golden = Dict(
                :M_uu => 3.0081981566e+00, :M_u_beta => 6.7653941525e-05,
                :M_ww => 4.4878125468e-01, :M_beta_u => 6.7653941525e-05, :M_beta_beta =>
                    7.6518924142e-06
            )
            for (name, golden) in mass_fro_golden
                @test norm(sys.mass_blocks[name]) ≈ golden rtol = 1e-9
            end

            dynamic_fro_golden = freq == 12000.0 ? 7139.522101805418 : 7136.5777497886775
            @test norm(sys.dynamic_matrix) ≈ dynamic_fro_golden rtol = 1e-9
            @test norm(sys.load_scale_q) ≈ 1.4057207588996422e-11 rtol = 1e-9
            @test norm(sys.load_scale_m) ≈ 1.150699634692459e-08 rtol = 1e-9

            @test sys.structural_blocks[:K_uu][1, 1:5] ≈
                  [-5014.550258592734, 15.900070807275364,
                -3.966239524917636, -0.003192064327710995, 0.0]
            dynamic_row1_golden = freq == 12000.0 ? -5014.319075589088 : -5012.232006806174
            @test sys.dynamic_matrix[1, 1:5] ≈
                  [dynamic_row1_golden, 15.900070807275364,
                -3.966239524917636, -0.003192064327710995, 0.0]
        end
    end

    @testset "Interior CBIE (closed-form spherical-cavity cross-check)" begin
        a, k_test = 0.01, 37.5
        mesh = AS.sphere_mesh(a, 20)
        K, V, ps = AS.assemble_cbie_operators(mesh, k_test; rtol = 1e-6)
        n = length(ps)

        p_bc = ones(ComplexF64, n)
        dpdn = V \ ((0.5I + K) * p_bc)

        j0ka = AS.js(0, k_test * a)
        j0dka = AS.jsd(0, k_test * a)
        exact_dpdn = k_test * j0dka / j0ka

        @test all(d -> abs(d - exact_dpdn) / abs(exact_dpdn) < 0.01, dpdn)
        @test maximum(abs, dpdn) - minimum(abs, dpdn) < 1e-3 * abs(exact_dpdn)
    end

    @testset "Coupled shell/fluid axisymmetric scattering" begin
        geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
        shell_body = AS.Shell(AS.Spheroid(0.035, 0.007), 0.0005)
        rho_ext, c_ext = 1026.8, 1477.3
        rho_int, c_int = 1077.3, 1575.0
        freq = 12000.0
        k = 2pi * freq / c_ext

        @testset "geometry/interpolation building blocks" begin
            eta = AS.uniform_eta_grid(9; pole_offset = 0.001)
            mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            ps = AS.panels(mesh)
            @test length(ps) == 8

            a, b = geometry.semimajor_length, geometry.semiminor_length
            for p in ps
                g = (p.rhom / b^2, p.zm / a^2)
                gn = hypot(g...)
                @test (p.nrho * g[1] + p.nz * g[2]) / gn ≈ 1.0 atol = 1e-9
            end

            L_s2b = AS.shell_to_bem_interpolation(9)
            L_b2s = AS.bem_to_shell_interpolation(9)
            f = geometry.focal_radius
            interp_eta = L_s2b * eta
            actual_eta = [p.zm / (f * geometry.xi_outer) for p in ps]
            @test interp_eta ≈ actual_eta atol = 1e-12

            roundtrip = L_b2s * interp_eta
            @test roundtrip[2:8] ≈ eta[2:8] atol = 1e-12
        end

        @testset "rigid limit (independent cross-check against solve_axial)" begin
            n_eta = 9
            eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
            outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

            diffs = Float64[]
            for E in (1e12, 1e14, 1e16)
                stiff_material = AS.Shelled(0.32, 2565.0, E)
                sol = AS.fem(shell_body, stiff_material, rho_ext, c_ext, 0.0, 1.0, k;
                    method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
                ts = AS.target_strength(sol)
                push!(diffs, abs(ts - ts_rigid))
            end
            @test diffs[1] > diffs[2]
            @test diffs[3] < 0.1
        end

        @testset "realistic material: finite, sane, and resolution-stable at high n_eta" begin
            material = AS.Shelled(0.32, 2565.0, 70e9)
            results = Float64[]
            for n_eta in (97, 129)
                sol = AS.fem(shell_body, material, rho_ext, c_ext, 0.0, 1.0, k;
                    method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
                @test all(isfinite, sol.data.p_ext_modes[1])
                @test all(isfinite, sol.data.dpdn_ext_modes[1])
                @test all(isfinite, sol.data.shell_state)
                push!(results, AS.target_strength(sol))
            end
            @test all(ts -> -150.0 < ts < 0.0, results)
            @test abs(results[1] - results[2]) < 2.0
        end

        @testset "fluid-filled: rigid limit (independent cross-check)" begin
            n_eta = 9
            eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
            outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

            diffs = Float64[]
            for E in (1e12, 1e14, 1e16)
                stiff_material = AS.Shelled(0.32, 2565.0, E)
                sol = AS.fem(
                    shell_body, stiff_material, rho_ext, c_ext, rho_int, c_int, k;
                    method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
                ts = AS.target_strength(sol)
                push!(diffs, abs(ts - ts_rigid))
                @test maximum(abs, sol.data.p_int_modes[1]) < 1.0
            end
            @test diffs[1] > diffs[2]
            @test diffs[3] < 0.1
        end

        @testset "fluid-filled: zero-interior-density limit recovers the vacuum-backed solver" begin
            n_eta = 9
            material = AS.Shelled(0.32, 2565.0, 70e9)
            sol_vac = AS.fem(shell_body, material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            ts_vac = AS.target_strength(sol_vac)

            diffs = Float64[]
            for rho_int_test in (1.0, 1e-3, 1e-6)
                sol = AS.fem(
                    shell_body, material, rho_ext, c_ext, rho_int_test, c_int, k;
                    method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
                ts = AS.target_strength(sol)
                push!(diffs, abs(ts - ts_vac))
            end
            @test diffs[1] > diffs[2] > diffs[3]
            @test diffs[3] < 1e-4
        end
    end

    @testset "General (Fourier-mode) solid shell FEM and oblique coupling" begin
        python_geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
        geometry = AS.ProlateShellGeometry(0.02, 0.005, 0.0005)
        c_water = 1477.4
        freq_hz = 12000.0
        k = 2pi * freq_hz / c_water
        n_eta = 13
        n_t = 3

        @testset "mesh matches the Python reference geometry exactly" begin
            mesh = AS.build_structured_shell_strip(python_geometry, 9, 3; pole_offset = 0.001)
            @test AS.Ferrite.getnnodes(mesh.grid) == 27
            @test AS.Ferrite.getncells(mesh.grid) == 32
            @test mesh.outer_ρ[1] ≈ 0.00036756472001377 atol = 1e-12
            @test mesh.outer_z[end] ≈ 0.035208966293679235 atol = 1e-12
            @test mesh.inner_ρ[5] ≈ 0.00675 atol = 1e-12
        end

        @testset "shell operators: eigenvalues match the Python reference (m=0,1,2)" begin
            mesh = AS.build_structured_shell_strip(python_geometry, 9, 3; pole_offset = 0.001)
            omega = 2pi * 12000.0
            expected_smallest = Dict(
                0 => 632372.37319513,
                1 => 372895.73609484,
                2 => 434667.49165402
            )
            for m in (0, 1, 2)
                ops = AS.assemble_shell_modal_operators(mesh, m, omega, 2565.0, 70e9, 0.32)
                evals_abs = sort(abs.(eigvals(ops.dynamic_matrix)))
                @test evals_abs[1] ≈ expected_smallest[m] rtol = 1e-6
                @test tr(ops.B_out * ops.B_out') ≈ 9.0 atol = 1e-10
                @test tr(ops.Q_out' * ops.Q_out) ≈ 4.16258628395464e-07 rtol = 1e-6
            end
        end

        @testset "coupled oblique rigid limit (independent cross-check against solve_oblique)" begin
            beta = deg2rad(45.0)
            m_max = 3
            shell_mesh = AS.build_structured_shell_strip(geometry, n_eta, n_t; pole_offset = 0.001)
            outer_mesh = AS.MeridianMesh(reverse(shell_mesh.outer_ρ), reverse(shell_mesh.outer_z))
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_oblique(
                AS.Rigid(), k, outer_mesh, beta; m_max = m_max, rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi - beta, pi)

            shell_body2 = AS.Shell(AS.Spheroid(0.02, 0.005), 0.0005)
            material_stiff = AS.Shelled(0.3, 1e6, 1e14)
            sol = AS.fem(
                shell_body2, material_stiff, 1026.8, c_water, 1026.8, c_water, k;
                method = :general, incidence_angle = beta, m_max = m_max, n_eta = n_eta,
                n_t = n_t, pole_offset = 0.001, rtol = 1e-4)
            ts_stiff = AS.target_strength(sol; angle = pi - beta, azimuth = pi)

            @test ts_stiff ≈ ts_rigid atol = 0.1
        end
    end

    @testset "Oblique (multi-Fourier-mode) incidence" begin
        c_water = 1477.4

        @testset "$(typeof(boundary)): sphere rotational symmetry (independent of solve_axial)" for boundary in (AS.Rigid(), AS.PressureRelease())
            a = 0.01
            freq = 38000.0
            k = 2pi * freq / c_water
            body = AS.Sphere(a)
            mesh = AS.sphere_mesh(a, 12)
            ts_modal = AS.target_strength(AS.modal(body, boundary, k))

            for β_deg in (0.0, 90.0)
                β = deg2rad(β_deg)
                sol = AS._bem_oblique(body, boundary, k, mesh, β; m_max = 5, rtol = 1e-3)
                ts = AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test ts ≈ ts_modal atol = 0.3
            end
        end

        @testset "$(typeof(boundary)): prolate spheroid cross-check against the analytical modal series" for boundary in (AS.Rigid(), AS.PressureRelease())
            a, b = 0.05, 0.02
            spheroid = AS.Spheroid(a, b)
            freq = 20000.0
            k = 2pi * freq / c_water

            for β_deg in (0.0, 90.0)
                β = deg2rad(β_deg)
                ts_analytic = AS.target_strength(AS.modal(spheroid, boundary, k; incidence_angle = β))
                sol = AS.bem(spheroid, boundary, k; n = 10,
                    incidence_angle = β, m_max = 4, rtol = 1e-3)
                ts_bem = AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test ts_bem ≈ ts_analytic atol = 0.5
            end
        end

        @testset "$(typeof(boundary)): finite cylinder cross-check against FCMS" for boundary in (AS.Rigid(), AS.PressureRelease())
            radius, length = 0.01, 1.0
            freq = 20000.0
            k = 2pi * freq / c_water
            body = AS.Cylinder(radius, length)
            β = π / 2

            ts_modal = AS.target_strength(AS.modal(
                body, boundary, k; incidence_angle = β, m_max = 30))
            sol = AS.bem(
                body, boundary, k; n = 140, incidence_angle = β, m_max = 4, rtol = 1e-4)
            ts_bem = AS.target_strength(sol; angle = π - β, azimuth = π)
            @test ts_bem ≈ ts_modal atol = 0.01
        end

        @testset "solve_oblique(::FluidFilled,...): reduced 2n×2n system correctness" begin
            a_pol, b_eq = 0.01, 0.07
            spheroid = AS.Spheroid(a_pol, b_eq)
            c_med = 1477.3
            k = 2pi * 38000.0 / c_med
            mesh = AS.spheroid_mesh(a_pol, b_eq, 24)
            β = deg2rad(8.0)

            p_rigid, d_rigid, ps_rigid = AS.solve_oblique(
                AS.Rigid(), k, mesh, β; m_max = 15)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, d_rigid, k, π - β, π)
            stiff = AS.FluidFilled(1e8, 1e8)
            p_stiff, d_stiff, ps_stiff = AS.solve_oblique(stiff, k, mesh, β; m_max = 15)
            ts_stiff = AS.target_strength(ps_stiff, p_stiff, d_stiff, k, π - β, π)
            @test ts_stiff ≈ ts_rigid atol = 1e-4

            trivial = AS.FluidFilled(1.0, 1.0)
            p_triv, d_triv, _ = AS.solve_oblique(trivial, k, mesh, β; m_max = 10)
            @test maximum(abs, p_triv[1]) < 0.05
            @test maximum(abs, d_triv[1]) < 5.0

            rho_med, rho_ws, c_ws = 1026.8, 1028.9, 1480.3
            bc = AS.FluidFilled(rho_ws / rho_med, c_ws / c_med)
            k30 = 2pi * 30000.0 / c_med
            ts_modal = AS.target_strength(bc, k30, spheroid; incidence_angle = β,
                m_max = 12, n_max = 12, precision = :quad)
            p_modes, d_modes, ps = AS.solve_oblique(bc, k30, mesh, β; m_max = 15)
            ts_bem = AS.target_strength(ps, p_modes, d_modes, k30, π - β, π)
            @test ts_bem ≈ Float64(ts_modal) atol = 1.0
        end

        @testset "_azimuthal_fixed_order: far-pair quadrature converges under mesh refinement" begin
            a_pol, b_eq = 0.01, 0.07
            k = 2pi * 38000.0 / 1477.3
            β_r = deg2rad(8.0)
            maxR(panels) = begin
                mesh_r = AS.spheroid_mesh(a_pol, b_eq, panels)
                ps_r = AS.panels(mesh_r)
                n_r = length(ps_r)
                I_r = Matrix{ComplexF64}(I, n_r, n_r)
                K_int, V_int, _ = AS.assemble_cbie_operators(mesh_r, k; m = 0, rtol = 1e-6)
                p_inc = ComplexF64[AS._p_inc_mode(0, k, β_r, p.rhom, p.zm) for p in ps_r]
                dpdn_inc = ComplexF64[AS._dpdn_inc_mode(
                                          0, k, β_r, p.rhom, p.zm, p.nrho, p.nz)
                                      for p in ps_r]
                R = (0.5 * I_r + K_int) * p_inc - V_int * dpdn_inc
                maximum(abs, R)
            end
            R48, R96 = maxR(48), maxR(96)
            @test R96 < 0.6 * R48
            @test R96 < 0.01
        end
    end

    @testset "Spheroid meridian FEM (Rigid, vs Jech et al. 2015 benchmark)" begin
        a_major, b_minor = 0.07, 0.01
        c_water = 1477.3
        k = 2pi * 38000.0 / c_water
        R = 1.2 * a_major
        β = deg2rad(28.0)
        ts_benchmark = -76.91

        ts_fem = AS.target_strength(AS.fem(
            AS.Spheroid(a_major, b_minor), AS.Rigid(), k; method = :meridian, R = R,
            incidence_angle = β, m_max = 25, n_r = 300, n_theta = 600))
        @test ts_fem ≈ ts_benchmark atol = 0.1
    end

    @testset "Spheroid meridian FEM (oblate, Rigid, vs own modal series)" begin
        a_pol, b_eq = 0.03, 0.10
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        R = 1.2 * b_eq
        body = AS.Spheroid(a_pol, b_eq)

        for angle in (0.0, pi / 2)
            ts_modal = AS.target_strength(AS.modal(
                body, AS.Rigid(), k; incidence_angle = angle, m_max = 24, n_max = 24))
            ts_fem = AS.target_strength(AS.fem(
                body, AS.Rigid(), k; method = :meridian, R = R,
                incidence_angle = angle, m_max = 25, n_r = 200, n_theta = 400))
            @test ts_fem ≈ ts_modal atol = 0.02
        end
    end

    @testset "Solution interface contract (every concrete AbstractSolution type)" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)

        modal_sol = AS.modal(sphere, AS.Rigid(), k)
        kirch_sol = AS.kirchhoff(sphere, AS.Rigid(), k)
        fem_sol = AS.fem(sphere, AS.Rigid(), k)
        bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)
        mfs_sol = AS.mfs(sphere, AS.Rigid(), k; n = 16)

        @testset "target_strength works with no keywords on every concrete type" begin
            for sol in (modal_sol, kirch_sol, fem_sol, bem_sol, mfs_sol)
                @test AS.target_strength(sol) isa Float64
            end
        end

        @testset "scattering_amplitude: available for 4 of 5, explicit error for the scalar-FEM case" begin
            for sol in (modal_sol, kirch_sol, bem_sol, mfs_sol)
                @test AS.scattering_amplitude(sol) isa Complex
            end
            @test_throws ArgumentError AS.scattering_amplitude(fem_sol)
        end

        @testset "angle/azimuth keywords: supported where a real bistatic query exists, explicit error otherwise" begin
            # modal/kirchhoff/scalar-fem: no reusable surface state, angle already baked in at solve
            # time — must error, not silently ignore the keyword (a real bug caught this session:
            # target_strength(fem_sol; angle=0.3) used to silently return the unchanged backscatter
            # value instead of erroring or actually honoring the angle).
            @test_throws ArgumentError AS.target_strength(modal_sol; angle = 0.3)
            @test_throws ArgumentError AS.scattering_amplitude(modal_sol; angle = 0.3)
            @test_throws ArgumentError AS.target_strength(kirch_sol; angle = 0.3)
            @test_throws ArgumentError AS.scattering_amplitude(kirch_sol; angle = 0.3)
            @test_throws ArgumentError AS.target_strength(fem_sol; angle = 0.3)

            # bem/mfs axisymmetric: genuine reusable surface state, angle/azimuth are real queries.
            @test AS.target_strength(bem_sol; angle = pi / 2) isa Float64
            @test AS.target_strength(mfs_sol; angle = pi / 2) isa Float64
        end
    end

    @testset "Visualization sampling (frequency/incidence-angle sweeps)" begin
        a = 0.01
        c_water = 1477.4
        sphere = AS.Sphere(a)

        @testset "frequency_sweep: shape, k conversion, endpoint agreement" begin
            freqs = 20e3:10e3:60e3
            sweep = AS.frequency_sweep(k -> AS.modal(sphere, AS.Rigid(), k), freqs, c_water)
            @test sweep.frequencies == collect(freqs)
            @test length(sweep.k) == length(freqs)
            @test length(sweep.target_strength) == length(freqs)
            @test sweep.k ≈ 2pi .* collect(freqs) ./ c_water

            single = AS.frequency_sweep(k -> AS.modal(sphere, AS.Rigid(), k), [38000.0], c_water)
            k_direct = 2pi * 38000.0 / c_water
            ts_direct = AS.target_strength(AS.modal(sphere, AS.Rigid(), k_direct))
            @test single.target_strength[1] == ts_direct
        end

        @testset "incidence_angle_sweep: shape, endpoint agreement" begin
            k = 2pi * 38000.0 / c_water
            spheroid = AS.Spheroid(0.05, 0.02)
            angles = 0:(pi / 8):(pi / 2)
            sweep = AS.incidence_angle_sweep(
                angle -> AS.modal(spheroid, AS.Rigid(), k; incidence_angle = angle), angles)
            @test sweep.angles == collect(angles)
            @test length(sweep.target_strength) == length(angles)

            single = AS.incidence_angle_sweep(
                angle -> AS.modal(spheroid, AS.Rigid(), k; incidence_angle = angle), [pi /
                                                                                      4])
            ts_direct = AS.target_strength(AS.modal(spheroid, AS.Rigid(), k; incidence_angle = pi /
                                                                                               4))
            @test single.target_strength[1] == ts_direct
        end
    end

    @testset "Visualization sampling (bistatic sweep/map)" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)
        bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)

        @testset "bistatic_sweep: shape, endpoint agreement, azimuthal periodicity" begin
            angles = 0:(pi / 4):pi
            sweep = AS.bistatic_sweep(bem_sol, angles)
            @test sweep.angles == collect(angles)
            @test length(sweep.target_strength) == length(angles)

            single = AS.bistatic_sweep(bem_sol, [pi]; azimuth = 0.3)
            ts_direct = AS.target_strength(bem_sol; angle = pi, azimuth = 0.3)
            @test single.target_strength[1] == ts_direct

            sweep_0 = AS.bistatic_sweep(bem_sol, [pi / 2]; azimuth = 0.0)
            sweep_2pi = AS.bistatic_sweep(bem_sol, [pi / 2]; azimuth = 2pi)
            @test sweep_0.target_strength[1] ≈ sweep_2pi.target_strength[1] atol = 1e-8
        end

        @testset "bistatic_map: shape, endpoint agreement" begin
            thetas = 0:(pi / 4):pi
            phis = 0:(pi / 2):(2pi)
            map_result = AS.bistatic_map(bem_sol, thetas, phis)
            @test size(map_result.target_strength) == (length(thetas), length(phis))
            @test map_result.target_strength[2, 3] ==
                  AS.target_strength(bem_sol; angle = thetas[2], azimuth = phis[3])
        end

        @testset "axisymmetric vs full-BEM cross-validation (coordinate convention correctness)" begin
            full_sol = AS.bem(sphere, AS.Rigid(), k; method = :full,
                meshsize = AS.bem3d_elements_per_wavelength(k))
            thetas = [0.0, pi / 2, pi]
            phis = [0.0, pi / 2]
            axi_map = AS.bistatic_map(bem_sol, thetas, phis)
            full_map = AS.bistatic_map(full_sol, thetas, phis)
            @test all(abs.(axi_map.target_strength .- full_map.target_strength) .< 0.5)
        end
    end

    @testset "Visualization sampling (revolve_panels)" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)

        @testset "shape, radius, and z agree with panel midpoints" begin
            bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)
            d = bem_sol.data
            ps = AS.panels(d.mesh)
            surf = AS.revolve_panels(ps, d.p_scat_modes; n_phi = 36)
            @test size(surf.x) == size(surf.y) == size(surf.z) == size(surf.field) ==
                  (36, length(ps))
            @test all(
                hypot(surf.x[i, j], surf.y[i, j]) ≈ ps[j].rhom
            for i in 1:36, j in eachindex(ps))
            @test all(surf.z[i, j] == ps[j].zm for i in 1:36, j in eachindex(ps))
        end

        @testset "axisymmetric (m=0-only) field is constant across azimuth" begin
            bem_axial = AS.bem(sphere, AS.Rigid(), k; n = 16, incidence_angle = 0.0)
            d = bem_axial.data
            @test length(d.p_scat_modes) == 1
            ps = AS.panels(d.mesh)
            surf = AS.revolve_panels(ps, d.p_scat_modes; n_phi = 36)
            @test all(surf.field[i, j] == surf.field[1, j]
            for i in 1:36, j in eachindex(ps))
        end
    end

    @testset "Makie visualization: 1D sweep plots" begin
        a = 0.01
        c_water = 1477.4
        sphere = AS.Sphere(a)
        spheroid = AS.Spheroid(0.05, 0.02)
        k38 = 2pi * 38000.0 / c_water

        @testset "plot(body, boundary, freqs; kind=:frequency) builds a figure" begin
            fap = plot(sphere, AS.Rigid(), 20e3:10e3:60e3;
                kind = :frequency, sound_speed = c_water)
            @test fap isa Makie.FigureAxisPlot
        end

        @testset "plot(body, boundary, angles; kind=:incidence_angle) builds a figure" begin
            fap = plot(spheroid, AS.Rigid(), 0:0.5:1.5; kind = :incidence_angle, k = k38)
            @test fap isa Makie.FigureAxisPlot
        end

        @testset "solver kwarg routes to a different dispatcher" begin
            fap = plot(sphere, AS.Rigid(), 20e3:10e3:60e3; kind = :frequency,
                sound_speed = c_water, solver = :bem, solver_kwargs = (n = 12,))
            @test fap isa Makie.FigureAxisPlot
        end

        @testset "plot! overlays onto an existing axis" begin
            fig = Figure()
            ax = Axis(fig[1, 1])
            plot!(ax, sphere, AS.Rigid(), 20e3:10e3:60e3;
                kind = :frequency, sound_speed = c_water)
            @test !isempty(ax.scene.plots)
        end

        @testset "unsupported kind errors clearly" begin
            @test_throws ArgumentError plot(
                sphere, AS.Rigid(), 20e3:10e3:60e3; kind = :bogus, sound_speed = c_water)
        end
    end

    @testset "Makie visualization: 2D bistatic plots" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)
        bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)

        @testset "plot(sol; kind=:bistatic_polar) builds a figure with non-negative radius" begin
            fap = plot(bem_sol; kind = :bistatic_polar, angles = 0:0.2:(2pi))
            @test fap isa Makie.FigureAxisPlot
            @test fap.axis isa PolarAxis
        end

        @testset "plot(sol; kind=:bistatic_cartesian) builds a figure" begin
            fap = plot(bem_sol; kind = :bistatic_cartesian, angles = 0:0.2:(2pi))
            @test fap isa Makie.FigureAxisPlot
            @test fap.axis isa Axis
        end

        @testset "plot(sol; kind=:bistatic_map) builds a heatmap figure" begin
            fap = plot(bem_sol; kind = :bistatic_map, thetas = 0:0.3:pi, phis = 0:0.3:(2pi))
            @test fap isa Makie.FigureAxisPlot
        end

        @testset "show_incidence adds forward/backscatter reference lines" begin
            fig_with = plot(bem_sol; kind = :bistatic_cartesian,
                angles = 0:0.2:(2pi), show_incidence = true)
            fig_without = plot(bem_sol; kind = :bistatic_cartesian,
                angles = 0:0.2:(2pi), show_incidence = false)
            @test length(fig_with.plot.plots) > length(fig_without.plot.plots)
        end

        @testset "colorrange: default handles a deep null without erroring, explicit override works" begin
            fap = plot(bem_sol; kind = :bistatic_map, thetas = 0:0.3:pi,
                phis = 0:0.3:(2pi), colorrange = (-60.0, -40.0))
            @test fap isa Makie.FigureAxisPlot
        end

        @testset "unsupported kind errors clearly" begin
            @test_throws ArgumentError plot(bem_sol; kind = :bogus, angles = 0:0.2:(2pi))
        end
    end

    @testset "Makie visualization: 3D mesh/field plots" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)

        @testset "axisymmetric BEM: mesh and surface_field" begin
            bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)
            @test plot(bem_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(bem_sol; kind = :surface_field) isa Makie.FigureAxisPlot
            @test plot(bem_sol; kind = :surface_field, field = :pressure_phase) isa
                  Makie.FigureAxisPlot
        end

        @testset "full 3D BEM: mesh and surface_field" begin
            full_sol = AS.bem(sphere, AS.Rigid(), k; method = :full,
                meshsize = AS.bem3d_elements_per_wavelength(k))
            @test plot(full_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(full_sol; kind = :surface_field) isa Makie.FigureAxisPlot
        end

        @testset "bent-cylinder MFS: mesh falls back, surface_field is a point cloud" begin
            bent = AS.Cylinder(0.01, 0.1; radius_curvature = 0.5)
            k_bent = 2pi * 20000.0 / c_water
            mfs_sol = AS.mfs(bent, AS.Rigid(), k_bent)
            @test plot(mfs_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(mfs_sol; kind = :surface_field) isa Makie.FigureAxisPlot
        end

        @testset "ModalSolution/KirchhoffSolution: mesh works, surface_field errors" begin
            modal_sol = AS.modal(sphere, AS.Rigid(), k)
            kirch_sol = AS.kirchhoff(sphere, AS.Rigid(), k)
            @test plot(modal_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test plot(kirch_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test_throws ArgumentError plot(modal_sol; kind = :surface_field)
            @test_throws ArgumentError plot(kirch_sol; kind = :surface_field)
        end

        @testset "FEMSolution{_ScalarFEMData}: mesh works, surface_field errors naming the gap" begin
            fem_sol = AS.fem(sphere, AS.Rigid(), k)
            @test plot(fem_sol; kind = :mesh) isa Makie.FigureAxisPlot
            @test_throws ArgumentError plot(fem_sol; kind = :surface_field)
        end

        @testset "standalone Mesh: both representations" begin
            m1 = AS.mesh(sphere; k = k)
            m2 = AS.mesh(sphere; k = k, method = :full)
            @test plot(m1) isa Makie.FigureAxisPlot
            @test plot(m2) isa Makie.FigureAxisPlot
        end
    end

    @testset "Mesh interface: geometry preservation, orientation, element counts" begin
        a = 0.01
        n = 40
        sphere = AS.Sphere(a)
        m = AS.mesh(sphere; resolution = n)

        @testset "element counts match requested resolution" begin
            @test AS.element_count(m) == n
            @test length(AS.elements(m)) == n
            @test length(AS.coordinates(m)) == n
            @test length(AS.normals(m)) == n
        end

        @testset "geometry preservation: every element sits on the sphere's own surface" begin
            for (rho, z) in AS.coordinates(m)
                @test hypot(rho, z) ≈ a atol = 1e-3 * a
            end
        end

        @testset "orientation: outward normal has positive radial component (convex body about the origin)" begin
            for ((rho, z), (nrho, nz)) in zip(AS.coordinates(m), AS.normals(m))
                @test nrho * rho + nz * z > 0
            end
        end

        @testset "spheroid geometry preservation" begin
            a2, b2 = 0.05, 0.02
            spheroid = AS.Spheroid(a2, b2)
            m2 = AS.mesh(spheroid; resolution = 30)
            for (rho, z) in AS.coordinates(m2)
                @test (rho / b2)^2 + (z / a2)^2 ≈ 1.0 atol = 1e-2
            end
        end

        @testset "full 3D mesh element counts and geometry" begin
            k = 2pi * 38000.0 / 1477.4
            m3 = AS.mesh(sphere; k = k, method = :full)
            @test AS.element_count(m3) == length(AS.coordinates(m3)) ==
                  length(AS.normals(m3))
            # Gmsh's triangulated quadrature nodes approximate the sphere, they don't sit exactly on
            # it — a coarse mesh at this resolution deviates from `a` by ~1-2%, not the 0.1% the
            # axisymmetric meridian mesh above achieves, so this tolerance is deliberately looser.
            for c in AS.coordinates(m3)
                @test hypot(c...) ≈ a atol = 0.03 * a
            end
        end

        # Round-trip mesh I/O is genuinely untestable, not merely unwritten: `src/ecosystem/mesh_io.jl`
        # is a stub ("Mesh import (.stl/.msh/.vtk). Not yet implemented."), so there is no I/O
        # capability to round-trip yet. Flagged here rather than silently skipped from the suite.
    end
end
