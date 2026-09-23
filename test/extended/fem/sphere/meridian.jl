using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "meridian FEM (sphere, 2D r-θ mesh)" @testset "meridian FEM (sphere, 2D r-θ mesh)" begin
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
end
