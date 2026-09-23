using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "finite cylinder modal series" @testset "finite cylinder modal series" begin
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

    @time "Bent-cylinder modal series (BCMS)" @testset "Bent-cylinder modal series (BCMS)" begin
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

    @time "finite cylinder modal series (elastic shell and solid, ECMS)" @testset "finite cylinder modal series (elastic shell and solid, ECMS)" begin
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
end
