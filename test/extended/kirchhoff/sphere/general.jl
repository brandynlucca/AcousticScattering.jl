using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Kirchhoff high-frequency baseline" @testset "Kirchhoff high-frequency baseline" begin
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

        reference_soundspeed = 1477.3
        sphere_reference_cases = [
            (12.0, -52.12), (14.0, -50.87), (16.0, -49.81), (18.0, -48.91), (20.0, -48.13)
        ]
        for (freq_khz, ts_expected) in sphere_reference_cases
            k = 2pi * freq_khz * 1000 / reference_soundspeed
            @test AS.target_strength(AS.kirchhoff(sphere, AS.Rigid(), k)) ≈ ts_expected atol = 0.01
        end
    end
end
