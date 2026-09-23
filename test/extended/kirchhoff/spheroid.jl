using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Spheroid Kirchhoff high-frequency baseline" @testset "Spheroid Kirchhoff high-frequency baseline" begin
        body = AS.Spheroid(0.10, 0.03)
        k = 2pi * 200000.0 / 1477.4

        ts_endon = AS.target_strength(AS.kirchhoff(body, AS.Rigid(), k; incidence_angle = 0.0))
        ts_broadside = AS.target_strength(AS.kirchhoff(body, AS.Rigid(), k; incidence_angle = pi /
                                                                                              2))
        @test ts_broadside > ts_endon

        R1, R2 = AS.principal_curvatures(body, 0.0)
        @test AS.target_strength(AS.kirchhoff(body, AS.Rigid(), k; incidence_angle = 0.0)) ≈
              AS.target_strength(AS.kirchhoff_form_function(AS.Rigid(), R1, R2)) atol = 0.05

        reference_body = AS.Spheroid(0.07, 0.01)
        reference_soundspeed = 1477.3
        reference_wavenumber = 2pi * 38000.0 / reference_soundspeed
        spheroid_reference_cases = [
            (0.0, -62.67), (8.0, -62.8), (28.0, -62.21), (48.0, -55.61), (68.0, -46.49), (
                88.0, -28.15)
        ]
        for (angle_deg, ts_expected) in spheroid_reference_cases
            ts = AS.target_strength(AS.kirchhoff(
                reference_body, AS.Rigid(), reference_wavenumber; incidence_angle = deg2rad(angle_deg)))
            @test ts ≈ ts_expected atol = 0.01
        end
    end
end
