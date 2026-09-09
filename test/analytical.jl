using AcousticScattering
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

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
