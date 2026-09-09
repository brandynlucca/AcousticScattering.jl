using AcousticScattering
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

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
