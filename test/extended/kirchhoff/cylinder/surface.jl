using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Finite-cylinder Kirchhoff high-frequency baseline" @testset "Finite-cylinder Kirchhoff high-frequency baseline" begin
        radius, length = 0.01, 0.07
        reference_soundspeed = 1477.3
        k = 2pi * 38000.0 / reference_soundspeed
        cyl = AS.Cylinder(radius, length)
        cylinder_reference_cases = [
            (8.0, -42.07), (28.0, -44.47), (48.0, -45.97), (68.0, -44.66), (88.0, -31.29)
        ]
        for (angle_deg, ts_expected) in cylinder_reference_cases
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
end
