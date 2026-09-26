using AcousticScattering
using Test

@testset "Elastic spheroid" begin
    boundary = SolidElastic(2.7, 6.4 / 1.5, 3.04 / 1.5)
    @test_throws ArgumentError modal(Spheroid(3.0, 1.0), boundary, -1.0)
    shell = Shelled(ElasticLayer(2.7, 4.13, 2.08), FluidInterior(1.0, 1.0), 0.8)
    # The confocal inner surface of an oblate spheroid needs a radius ratio above the focal ratio.
    @test_throws ArgumentError modal(Spheroid(1.0, 3.0),
        Shelled(ElasticLayer(2.7, 4.13, 2.08), FluidInterior(1.0, 1.0), 0.5), 1.0)

    radius = cbrt(1.01)
    k = 0.8 / radius
    solution = modal(Spheroid(1.01, 1.0), boundary, k;
        incidence_angle = pi / 3, m_max = 5, n_max = 5)
    reference = modal(Sphere(radius), boundary, k)
    oblate = modal(Spheroid(0.99, 1.0), boundary, k;
        incidence_angle = pi / 3, m_max = 5, n_max = 5)
    oblate_reference = modal(Sphere(cbrt(0.99)), boundary, k)
    @test abs(scattering_amplitude(oblate) - scattering_amplitude(oblate_reference)) /
          abs(scattering_amplitude(oblate_reference)) < 0.01
    @test solution isa ModalSolution
    @test isfinite(target_strength(solution))
    @test abs(scattering_amplitude(solution) - scattering_amplitude(reference)) /
          abs(scattering_amplitude(reference)) < 0.01
end
