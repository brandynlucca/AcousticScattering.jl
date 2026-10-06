using AcousticScattering
using Test

@testset "Kirchhoff high-frequency slender spheroid envelope" begin
    b = 0.01
    k = 10.0 / b
    body = Spheroid(8.0 * b, b)
    solution = kirchhoff(body, Rigid(), k; incidence_angle = pi / 2)
    reference = modal(body, Rigid(), k; incidence_angle = pi / 2)
    @test abs(target_strength(solution) - target_strength(reference)) < 1.0
end
