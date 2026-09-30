using AcousticScattering
using Test

@testset "Sphere" begin
    body = Sphere(1.0)
    k = 0.5

    @testset "Axisymmetric boundaries" begin
        for (name, boundary) in (("rigid", Rigid()),
            ("pressure-release", PressureRelease()),
            ("fluid-filled", FluidFilled(1.2, 1.1)))
            @testset "$name" begin
                incidence_angle = boundary isa Rigid ? pi / 3 : 0.0
                solution = bem(body, boundary, k; incidence_angle, n = 8, m_max = 1)
                @test solution isa BEMSolution
                @test solution.method == :axisymmetric
                @test scattering_amplitude(solution) isa ComplexF64
                @test isfinite(target_strength(solution))
                @test diagnostics(solution).relative_residual < 1e-5
                if boundary isa Rigid
                    @test isfinite(pressure(solution, (1.5, 0.0, 0.0);
                        field = :scattered))
                elseif boundary isa FluidFilled
                    @test isfinite(pressure(solution, (0.0, 0.0, 0.0);
                        field = :interior))
                end
            end
        end
    end

    @testset "Impedance (locally reacting) boundary" begin
        rigid = bem(body, Rigid(), k; incidence_angle = 0.0, n = 40)
        hard = bem(body, Impedance(1e6), k; incidence_angle = 0.0, n = 40)
        @test scattering_amplitude(hard) ≈ scattering_amplitude(rigid) rtol = 1e-4

        for zeta in (0.01, 1.3 - 0.6im)
            reference = modal(body, Impedance(zeta), k)
            solution = bem(body, Impedance(zeta), k; incidence_angle = 0.0, n = 40)
            @test scattering_amplitude(solution) ≈
                  scattering_amplitude(reference) rtol = 1e-2
        end

        @testset "Oblique incidence" begin
            zeta, incidence_angle = 1.3 - 0.6im, 0.3
            reference = modal(body, Impedance(zeta), k; m_max = 40)
            solution = bem(body, Impedance(zeta), k; incidence_angle, n = 40, m_max = 30)
            @test scattering_amplitude(solution) ≈
                  scattering_amplitude(reference) rtol = 1e-2
            angles = [0.2, incidence_angle, 1.4]
            sweep = incidence_angle_sweep(body, Impedance(zeta), k, angles;
                n = 40, m_max = 30, method = :axisymmetric)
            for (i, angle) in enumerate(angles)
                fresh = bem(
                    body, Impedance(zeta), k; incidence_angle = angle, n = 40, m_max = 30)
                @test sweep.amplitudes[i] ≈ scattering_amplitude(fresh) rtol = 1e-12
            end
        end
    end

    @testset "Fluid shells" begin
        layer = FluidLayer(1.1, 1.05)
        for interior in (VacuumInterior(), FluidInterior(1.2, 1.1))
            solution = bem(body, Shelled(layer, interior, 0.7), k; n = 8)
            @test solution isa BEMSolution
            @test solution.method == :axisymmetric
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
        end
    end

    @testset "Full 3D matched-medium limit" begin
        solution = bem(body, FluidFilled(1.0, 1.0), k;
            method = :full, meshsize = 1.0, mesh_order = 2, qorder = 2,
            condition_limit = 0)
        @test solution isa BEMSolution
        @test solution.method == :full
        @test diagnostics(solution).formulation == :muller
        @test abs(scattering_amplitude(solution)) < 1e-8
        @test isfinite(pressure(solution, (1.5, 0.0, 0.0);
            field = :scattered))
    end
end
