using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Axisymmetric MFS (spheroid, axial incidence)" @testset "Axisymmetric MFS (spheroid, axial incidence)" begin
        for body in (Spheroid(1.2, 1.0), Spheroid(1.0, 1.2))
            solution = mfs(body, Rigid(), 0.5; incidence_angle = 0.0,
                n = 24, offset = 0.2, condition_limit = 0)
            @test solution isa MFSSolution
            @test isfinite(target_strength(solution))
            @test diagnostics(solution).relative_residual < 1e-4
        end
    end
end
