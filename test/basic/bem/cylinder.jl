using AcousticScattering
using Test

@testset "Cylinder" begin
    body = Cylinder(0.5, 1.0; endcap_depth = 0.5)
    k = 0.3
    for (name, boundary) in (("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled", FluidFilled(1.2, 1.1)))
        @testset "$name" begin
            solution = bem(body, boundary, k; incidence_angle = 0.0, n = 10)
            @test solution isa BEMSolution
            @test solution.method == :axisymmetric
            @test scattering_amplitude(solution) isa ComplexF64
            @test isfinite(target_strength(solution))
            if boundary isa Rigid
                @test isfinite(pressure(solution, (1.2, 0.0, 0.0);
                    field = :scattered))
            end
        end
    end

    bent = Cylinder(0.5, 1.0; radius_curvature = 2.0, endcap_depth = 0.5)
    @test_throws ArgumentError bem(bent, Rigid(), k; n = 10)
    full = bem(bent, Rigid(), k; method = :full, meshsize = 0.8,
        mesh_order = 2, qorder = 2, compression = (method = :none,))
    @test full isa BEMSolution
    @test full.method == :full
    @test isfinite(scattering_amplitude(full))
end
