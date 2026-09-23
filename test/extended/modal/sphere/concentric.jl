using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "sphere modal series (shelled and solid elastic)" @testset "sphere modal series (shelled and solid elastic)" begin
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

        # The multi-mode stiff limit must agree in phase as well as target-strength magnitude.
        stiff_shell = Shelled(ElasticLayer(1000.0, 100.0, 100.0), FluidInterior(0.001, 0.5), 0.9)
        @test scattering_amplitude(modal(sphere, stiff_shell, k; m_max = 25)) ≈
              scattering_amplitude(modal(sphere, Rigid(), k; m_max = 25)) rtol = 0.01

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
end
