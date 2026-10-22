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

    @time "Spheroid meridian FEM complex amplitudes" @testset "Spheroid meridian FEM complex amplitudes" begin
        k = 20.0
        incidence = pi / 3
        body = AS.Spheroid(0.01001, 0.01)
        for boundary in (AS.Rigid(), AS.PressureRelease(), AS.FluidFilled(1.2, 1.1))
            numerical = AS.fem(body, boundary, k; method = :meridian,
                R = 0.03, incidence_angle = incidence, n_r = 32, n_theta = 64, m_max = 3)
            reference = AS.modal(body, boundary, k;
                incidence_angle = incidence, m_max = 3, n_max = 5)
            @test AS.scattering_amplitude(numerical) ≈
                  AS.scattering_amplitude(reference) rtol = 0.003
            @test AS.target_strength(numerical) ≈ AS.target_strength(reference) atol = 0.03
            @test AS.scattering_amplitude(numerical; angle = pi - incidence, azimuth = pi) ≈
                  AS.scattering_amplitude(numerical) rtol = 1e-12
        end

        oblate = AS.Spheroid(0.01, 0.02)
        numerical = AS.fem(oblate, AS.Rigid(), k; method = :meridian,
            R = 0.03, incidence_angle = pi / 2, n_r = 32, n_theta = 64, m_max = 3)
        reference = AS.modal(oblate, AS.Rigid(), k;
            incidence_angle = pi / 2, m_max = 3, n_max = 5)
        @test AS.scattering_amplitude(numerical) ≈
              AS.scattering_amplitude(reference) rtol = 0.003
        sweep = AS.bistatic_sweep(numerical, [pi / 3, pi / 2]; azimuth = pi / 4)
        @test length(sweep.amplitudes) == 2
        @test all(isfinite, real.(sweep.amplitudes))
    end
end

@testset "FEM-BEM high-frequency prolate gas validation" begin
    # Independent adaptive FEM with a spherical DtN boundary, order 6,
    # target_tol=0.005 dB, m_max=41, l_max=206. Mesh refinement used up to
    # 43 radial and 172 angular elements. No spheroidal modal answer is used.
    reference = ComplexF64[0.009251017780836401 - 0.033871069765870364im,
        -0.07464021253066706 + 0.33924855072102506im,
        -0.0010437670051196362 + 0.00201777562206837im]
    body = Spheroid(0.07, 0.01)
    boundary = FluidFilled(1.24/1026.8, 345/1477.3)
    solution = coupled(body, boundary, 2pi*200000/1477.3;
        coupling = Coupling(interior = :fem, exterior = :bem),
        target_tol = 0.005, mode_threads = 4)
    amplitudes = [scattering_amplitude(solution; angle, azimuth)
                  for (angle, azimuth) in ((pi/2, pi), (pi/2, 0.0), (0.7, 0.4))]
    for i in eachindex(reference)
        @test isapprox(amplitudes[i], reference[i]; rtol = 2e-4, atol = 1e-8)
    end
    @test abs(target_strength(solution)-target_strength(reference[1])) < 0.003
    report = diagnostics(solution)
    @test report.refinement.converged
    @test report.refinement.change_db<=0.005
    @test report.relative_residual < 1e-9
    sweep = bistatic_sweep(solution, [0.7, pi/2]; azimuth = 0.4)
    @test sweep.amplitudes[1] == amplitudes[3]
end
