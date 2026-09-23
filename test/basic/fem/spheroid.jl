using AcousticScattering
using Test

@testset "Spheroid" begin
    k = 2pi * 12000.0 / 1477.4
    boundaries = (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.05, 1.02)))

    @testset "Meridian FEM: prolate and oblate" begin
        for (shape, body) in ((:prolate, Spheroid(0.02, 0.015)),
            (:oblate, Spheroid(0.015, 0.02)))
            @testset "$shape" begin
                @test body.kind == shape
                for (name, boundary) in boundaries
                    @testset "$name" begin
                        solution = fem(body, boundary, k; incidence_angle = pi / 4,
                            m_max = 1, n_r = 4, n_theta = 10, l_max = 4)
                        @test solution isa FEMSolution
                        @test solution.method == :meridian
                        @test target_strength(solution) isa Float64
                        @test isfinite(target_strength(solution))
                        @test -300.0 < target_strength(solution) < 50.0
                    end
                end
            end
        end
        @test_throws ArgumentError fem(Spheroid(0.02, 0.015), Rigid(), k;
            method = :radial)
    end

    @testset "Structural spheroidal shell FEM" begin
        shell = Shell(Spheroid(0.035, 0.007), 0.0005)
        boundary = Shelled(0.32, 2565.0, 70e9)
        k_shell = 2pi * 12000.0 / 1477.3

        vacuum = fem(shell, boundary, 1026.8, 1477.3, 0.0, 1.0, k_shell;
            method = :thin, incidence_angle = 0.0, n_eta = 9)
        filled = fem(shell, boundary, 1026.8, 1477.3, 1077.3, 1575.0, k_shell;
            method = :thin, incidence_angle = 0.0, n_eta = 9)
        general = fem(shell, boundary, 1026.8, 1477.3, 1077.3, 1575.0, k_shell;
            method = :general, incidence_angle = 0.0, m_max = 0,
            n_eta = 7, n_t = 2)

        for (method, solution) in ((:thin, vacuum), (:thin, filled),
            (:general, general))
            @test solution isa FEMSolution
            @test solution.method == method
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
        end
        @test_throws ArgumentError fem(shell, boundary,
            1026.8, 1477.3, 0.0, 1.0, k_shell;
            method = :thin, incidence_angle = pi / 4, n_eta = 9)
    end
end
