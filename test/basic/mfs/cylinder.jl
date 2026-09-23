using AcousticScattering
using Test

@testset "Cylinder" begin
    body = Cylinder(0.5, 1.0; endcap_depth = 0.5)
    for (name, boundary) in (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.2, 1.1)))
        @testset "$name" begin
            solution = mfs(body, boundary, 0.3;
                incidence_angle = 0.0, n = 12, offset = 0.1,
                condition_limit = 0)
            @test solution isa MFSSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
        end
    end

    @testset "Bent cylinder" begin
        bent = Cylinder(0.01, 0.07; radius_curvature = 0.2)
        @test_throws ArgumentError mfs(bent, Rigid(), 100.0; n_s = 2)
        @test_throws ArgumentError mfs(bent, Rigid(), 100.0; n_phi = 2)
        @test_throws ArgumentError mfs(bent, Rigid(), 100.0;
            n_phi = 8, n_φ = 8)
        for boundary in (Rigid(), PressureRelease())
            solution = mfs(bent, boundary, 100.0;
                n_s = 6, n_phi = 8, offset = 0.003, condition_limit = 0)
            @test solution isa MFSSolution
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
            @test diagnostics(solution).solver_options.n_s == 6
        end
    end
end
