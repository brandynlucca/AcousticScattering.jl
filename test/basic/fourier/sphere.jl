using AcousticScattering
using Test

@testset "Sphere" begin
    body = Sphere(1.0)
    k = 0.5
    options = (; mapping_order = 2, continuation_steps = 1,
        m_max = 2, n_max = 2)
    for (name, boundary) in (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.05, 1.02)))
        @testset "$name" begin
            solution = fourier(body, boundary, k;
                incidence_angle = pi / 3, options...)
            reference = modal(body, boundary, k; m_max = 2)
            @test solution isa FMSolution
            @test solution.body === body
            @test diagnostics(solution).admissible
            @test scattering_amplitude(solution) isa ComplexF64
            @test scattering_amplitude(solution) ≈
                  scattering_amplitude(reference) rtol = 0.01
        end
    end
end
