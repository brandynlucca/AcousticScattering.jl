using AcousticScattering
using Test

const AS = AcousticScattering

@testset "Adaptive axisymmetric BEM" begin
    k, a = 2pi * 12000.0 / 1477.4, 0.01
    mesh_at(n) = AS.mesh(Sphere(a); resolution = n).data
    start = AS.bem_panel_count(k, a)
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
        refined = AS.solve_axial_adaptive(boundary, k, mesh_at, a; target_tol = 1.0)
        @test length(refined[3]) == 2start
        unconverged = @test_logs (:warn, r"did not converge") AS.solve_axial_adaptive(
            boundary, k, mesh_at, a; target_tol = 1e-12, max_n = start)
        @test length(unconverged[3]) == start
        stalled = @test_logs (:warn, r"did not converge") AS.solve_axial_adaptive(
            boundary, k, mesh_at, a; target_tol = 1e-12, max_n = 3start)
        @test length(stalled[3]) == 3start
    end
end

@testset "CHIEF regularization of fluid-filled axisymmetric BEM" begin
    k, a = 2pi * 12000.0 / 1477.4, 0.01
    fluid = FluidFilled(1.05, 1.02)
    for options in ((;), (; incidence_angle = 0.5, m_max = 2))
        solution = bem(Sphere(a), fluid, k; n = 20, chief_points = 2, options...)
        @test isfinite(target_strength(solution))
    end
    K_row, V_row = AS.chief_row(AS.mesh(Sphere(a); resolution = 20).data, k, 0.002, 0.0)
    @test size(K_row) == size(V_row) == (1, 20)
    @test all(isfinite, K_row) && all(isfinite, V_row)
end
