using AcousticScattering
using Test

const AS = AcousticScattering

@testset "Full BEM formulation variants" begin
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    tetra = mesh(nodes, triangles; qorder = 2)
    backscatter = AS.SVector(-1.0, 0.0, 0.0)
    for boundary in (Rigid(), PressureRelease())
        solution = bem(tetra, boundary, 0.3;
            formulation = :cbie, compression = (method = :none,))
        @test diagnostics(solution).formulation === :cbie
        @test isfinite(target_strength(solution))
        p, dp, quad = AS.solve_full_bem(boundary, 0.3, tetra.data;
            compression = (method = :none,), return_diagnostics = false)
        @test length(p) == length(dp) == length(quad)
        @test isfinite(AS.target_strength(quad, backscatter, 0.3, p, dp))
    end

    fluid = FluidFilled(1.2, 1.1)
    for (k, formulation) in ((0.3, :muller), (0.3, :cbie), (4.0, :muller))
        @test isfinite(target_strength(bem(tetra, fluid, k; formulation)))
    end
    @test diagnostics(bem(tetra, fluid, 4.0)).derivative_evaluation.exterior === :direct
    p, dp, quad = AS.solve_full_bem(fluid, 0.3, tetra.data; return_diagnostics = false)
    @test length(p) == length(dp) == length(quad)
end
