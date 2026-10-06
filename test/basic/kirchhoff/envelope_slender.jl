using AcousticScattering
using Test

@testset "Kirchhoff validity envelope, slender spheroid" begin
    b = 0.01
    # The higher-frequency kb=10 check remains in the Extended suite.
    k = 6.0 / b
    body = Spheroid(8.0 * b, b)
    solution = kirchhoff(body, Rigid(), k; incidence_angle = pi / 2)
    reference = modal(body, Rigid(), k; incidence_angle = pi / 2)
    @test abs(target_strength(solution) - target_strength(reference)) < 1.0
end
