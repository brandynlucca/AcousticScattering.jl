using AcousticScattering
using Test

@testset "Generated ellipsoid spherical limit" begin
    body = Ellipsoid(1.0, 1.0, 1.0)
    source_mesh = mesh(body; resolution = 0.5, mesh_order = 2, qorder = 1)
    solution = mfs(body, PressureRelease(), 0.5; meshsize = 0.4, qorder = 2,
        source_mesh, offset = 0.4, incidence_angle = 0.0, condition_limit = 0)
    reference = modal(Sphere(1.0), PressureRelease(), 0.5)
    @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.005
    @test pressure(solution, (2.0, 0.0, 0.0)) ≈ pressure(reference, (2.0, 0.0, 0.0)) rtol = 0.005
    @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0))
end

@testset "Sphere" begin
    body = Sphere(1.0)
    k = 0.5
    for (name, boundary) in (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.2, 1.1)))
        @testset "$name" begin
            solution = mfs(body, boundary, k; incidence_angle = 0.0,
                n = 12, offset = 0.2, condition_limit = 0)
            @test solution isa MFSSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
            @test diagnostics(solution).solver_options.incidence_angle == 0.0
        end
    end

    @testset "Impedance (locally reacting) boundary" begin
        rigid = mfs(body, Rigid(), k; incidence_angle = 0.3, n = 40, offset = 0.2)
        hard = mfs(body, Impedance(1e6), k; incidence_angle = 0.3, n = 40, offset = 0.2)
        @test scattering_amplitude(hard) ≈ scattering_amplitude(rigid) rtol = 1e-4

        zeta = 1.3 - 0.6im
        reference = modal(body, Impedance(zeta), k; m_max = 30)
        solution = mfs(body, Impedance(zeta), k; incidence_angle = 0.3, n = 40,
            m_max = 20, offset = 0.2)
        @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 1e-2

        axial = mfs(body, Impedance(zeta), k; incidence_angle = 0.0, n = 40, offset = 0.2)
        @test axial isa MFSSolution
        @test isfinite(target_strength(axial))
    end

    @testset "Full-surface source placement" begin
        collocation = mesh(Sphere(0.5); method = :full,
            resolution = 0.8, mesh_order = 2, qorder = 2)
        sources = mesh(Sphere(0.5); method = :full,
            resolution = 0.8, mesh_order = 2, qorder = 1)
        for boundary in (Rigid(), PressureRelease())
            solution = mfs(collocation, boundary, k;
                source_mesh = sources, offset = 0.2, condition_limit = 0)
            @test solution isa MFSSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
            @test diagnostics(solution).source_count <=
                  diagnostics(solution).collocation_count
            if boundary isa Rigid
                @test isfinite(pressure(solution, (1.0, 0.0, 0.0);
                    field = :scattered))
            end
        end
    end
end
