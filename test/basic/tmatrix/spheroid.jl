using AcousticScattering
using Test

@testset "Elastic spheroid" begin
    boundary = SolidElastic(2.7, 6.4 / 1.5, 3.04 / 1.5)
    @test_throws ArgumentError tmatrix(Spheroid(3.0, 1.0), boundary, -1.0)
    shell = Shelled(ElasticLayer(2.7, 4.13, 2.08), FluidInterior(1.0, 1.0), 0.8)
    # The confocal inner surface of an oblate spheroid needs a radius ratio above the focal ratio.
    @test_throws ArgumentError tmatrix(Spheroid(1.0, 3.0),
        Shelled(ElasticLayer(2.7, 4.13, 2.08), FluidInterior(1.0, 1.0), 0.5), 1.0)

    radius = cbrt(1.01)
    k = 0.8 / radius
    solution = tmatrix(Spheroid(1.01, 1.0), boundary, k;
        incidence_angle = pi / 3, m_max = 5, n_max = 5)
    reference = modal(Sphere(radius), boundary, k)
    oblate = tmatrix(Spheroid(0.99, 1.0), boundary, k;
        incidence_angle = pi / 3, m_max = 5, n_max = 5)
    oblate_reference = modal(Sphere(cbrt(0.99)), boundary, k)
    @test abs(scattering_amplitude(oblate) - scattering_amplitude(oblate_reference)) /
          abs(scattering_amplitude(oblate_reference)) < 0.01
    @test solution isa TMatrixSolution
    @test isfinite(target_strength(solution))
    @test abs(scattering_amplitude(solution) - scattering_amplitude(reference)) /
          abs(scattering_amplitude(reference)) < 0.01

    @testset "Solution type and entry point" begin
        body = Spheroid(1.5, 1.0)
        solution = tmatrix(body, boundary, 1.0; incidence_angle = pi / 3)
        @test solution isa TMatrixSolution
        @test isfinite(target_strength(solution))
        @test scattering_amplitude(solution) isa ComplexF64
        @test_throws ArgumentError target_strength(solution; angle = 0.0)
        @test_throws ArgumentError scattering_amplitude(solution; angle = 0.0)
        @test_throws ArgumentError modal(body, boundary, 1.0)
        @test_throws ArgumentError modal(body, shell, 1.0)
        sweep = incidence_angle_sweep(body, boundary, 1.0, [pi / 4, pi / 2])
        @test sweep.amplitudes[2] ≈ scattering_amplitude(tmatrix(body, boundary, 1.0)) rtol = 1e-12
        @test_throws ArgumentError incidence_angle_sweep(body, boundary, 1.0, [pi / 4];
            method = :modal)
    end
end
