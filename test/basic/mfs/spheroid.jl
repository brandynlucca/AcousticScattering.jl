using AcousticScattering
using Test

@testset "Spheroid" begin
    for (shape, body) in ((:prolate, Spheroid(1.2, 1.0)),
        (:oblate, Spheroid(1.0, 1.2)))
        @testset "$shape" begin
            @test body.kind == shape
            for (name, boundary) in (("rigid", Rigid()),
                ("pressure-release", PressureRelease()),
                ("fluid-filled", FluidFilled(1.2, 1.1)))
                @testset "$name" begin
                    solution = mfs(body, boundary, 0.5;
                        incidence_angle = 0.0, n = 12, offset = 0.2,
                        condition_limit = 0)
                    @test solution isa MFSSolution
                    @test scattering_amplitude(solution) isa ComplexF64
                    @test isfinite(target_strength(solution))
                end
            end
        end
    end
end

@testset "Oblique incidence and threaded assembly" begin
    body = Spheroid(1.2, 1.0)
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1))
        solution = mfs(body, boundary, 0.5; incidence_angle = 0.5, n = 12,
            offset = 0.2, m_max = 2, condition_limit = 0)
        @test isfinite(target_strength(solution))
    end

    mesh_data = AcousticScattering.mesh(body; resolution = 12).data
    rho, z = AcousticScattering.mfs_source_points(mesh_data, 0.2)
    serial = AcousticScattering.assemble_mfs_operators(mesh_data, 0.5, rho, z;
        m = 1, threaded = false)
    threaded = AcousticScattering.assemble_mfs_operators(mesh_data, 0.5, rho, z;
        m = 1, threaded = true)
    @test serial[1] ≈ threaded[1]
    @test serial[2] ≈ threaded[2]
end
