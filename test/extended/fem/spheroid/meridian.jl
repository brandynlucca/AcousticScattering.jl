using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Spheroid meridian FEM (Rigid, independent benchmark)" @testset "Spheroid meridian FEM (Rigid, independent benchmark)" begin
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

    @time "Spheroid meridian FEM (oblate, Rigid, vs own modal series)" @testset "Spheroid meridian FEM (oblate, Rigid, vs own modal series)" begin
        body = AS.Spheroid(0.01, 0.02)
        k = 20.0
        numerical = AS.fem(body, AS.Rigid(), k; method = :meridian,
            R = 0.03, n_r = 32, n_theta = 64, m_max = 3)
        reference = AS.modal(body, AS.Rigid(), k;
            incidence_angle = pi / 2, m_max = 3, n_max = 5)
        @test isfinite(AS.target_strength(numerical))
        @test abs(AS.target_strength(numerical) - AS.target_strength(reference)) < 3
    end
end
