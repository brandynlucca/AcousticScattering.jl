using AcousticScattering
using Test

@testset "Spheroid" begin
    radius = 0.02
    k = 2pi * 12000.0 / 1477.4
    sphere = Sphere(radius)
    bodies = (
        (:prolate, Spheroid(radius + 0.0002, radius - 0.0002)),
        (:oblate, Spheroid(radius - 0.0002, radius + 0.0002))
    )
    boundaries = (
        ("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled, full coupling", FluidFilled(1.05, 1.02; coupling = :full)),
        ("fluid-filled, diagonal coupling", FluidFilled(1.05, 1.02; coupling = :diagonal))
    )

    for (shape, body) in bodies
        @testset "$shape" begin
            @test body.kind == shape
            for (name, boundary) in boundaries
                @testset "$name" begin
                    solution = if boundary isa FluidFilled && boundary.coupling === :full
                        modal(body, boundary, k;
                            incidence_angle = pi / 3, m_max = 2, n_max = 6, n_quad = 12)
                    else
                        modal(body, boundary, k;
                            incidence_angle = pi / 3, m_max = 2, n_max = 6)
                    end
                    sphere_reference = modal(sphere, boundary, k; m_max = 6)
                    @test solution isa ModalSolution
                    @test scattering_amplitude(solution) isa ComplexF64
                    @test target_strength(solution) isa Float64
                    @test isfinite(target_strength(solution))
                    @test target_strength(solution) ≈
                          target_strength(sphere_reference) atol = 1.0
                end
            end
        end
    end

    @testset "Prolate full-coupling reference" begin
        q, xi0 = 3.0, 1.05
        body = Spheroid(q * xi0, sqrt((q * xi0)^2 - q^2))
        boundary = FluidFilled(1050 / 1026, 1.02; coupling = :full)
        solution = modal(body, boundary, 1.0;
            incidence_angle = 0.3, m_max = 2, n_max = 4)
        scaled = scattering_amplitude(solution) / (-2im)
        @test real(scaled) ≈ 9.10954776790933e-6 atol = 1e-14
        @test imag(scaled) ≈ 0.00326545204233986 atol = 1e-12
    end
end
