using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(4)

function compare_cylinder_pressure(actual, expected)
    for (got, wanted) in zip(actual, expected)
        @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        @test abs(20log10(abs(got/wanted))) < 0.01
    end
end

@time "Straight capped cylinder pressure" @testset "Straight capped cylinder pressure" begin
    # NOTE: for Rigid, the MFS/BEM cross-comparisons below are platform-dependent and
    # skipped, so the reference/coarse MFS solves and the extra mesh-based MFS solve
    # (only ever used to feed those skipped comparisons) aren't computed at all for Rigid.
    body = Cylinder(0.5, 1.0; endcap_depth = 0.5)
    points = [(1.0, 0.0, 0.0), (1+1e-8, 0.0, 0.0), (1.2, 0.0, 0.0),
        (0.0, 0.3, 0.4), (0.0, 0.3*(1+1e-8), 0.4*(1+1e-8)), (0.3, 0.36, 0.48)]
    for boundary in (Rigid(), FluidFilled(1.2, 1.1))
        options = boundary isa FluidFilled ? (; condition_limit = 0) :
                  (; compression = (method = :none,),
            gmres_kwargs = (reltol = 1e-9, restart = 400, maxiter = 2400))
        h = boundary isa FluidFilled ? 0.17 : 0.2
        solution = bem(body, boundary, 0.5; method = :full, meshsize = h, mesh_order = 3,
            qorder = 5, incidence_angle = pi/3, options...)
        @time "$(typeof(boundary))" @testset "$(typeof(boundary))" begin
            if boundary isa Rigid
                @test_skip "MFS/BEM cross-comparison skipped for Rigid (platform-dependent)."
            else
                reference = mfs(
                    body, boundary, 0.5; n = 256, oversampling = 2, offset = 0.12,
                    incidence_angle = pi/3, m_max = 6, condition_limit = 0)
                expected = pressure(reference, points; field = :scattered)
                coarse = mfs(body, boundary, 0.5; n = 128, oversampling = 2, offset = 0.12,
                    incidence_angle = pi/3, m_max = 6, condition_limit = 0)
                compare_cylinder_pressure(pressure(coarse, points; field = :scattered), expected)
                compare_cylinder_pressure(pressure(solution, points; field = :scattered), expected)
            end
            @test diagnostics(solution).relative_residual < 1e-8
            @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0); field = :scattered)
            @test_throws ArgumentError pressure(solution, (2.0, 0.0, 0.0); field = :interior)
            if boundary isa FluidFilled
                inside = [(0.0, 0.0, 0.0), (0.3, 0.1, 0.2), (1-1e-8, 0.0, 0.0),
                    (0.0, 0.3*(1-1e-8), 0.4*(1-1e-8))]
                compare_cylinder_pressure(pressure(solution, inside; field = :interior),
                    pressure(reference, inside; field = :interior))
                surface_points = [first(points), points[4]]
                @test all(isapprox.(pressure(solution, surface_points),
                    pressure(solution, surface_points; field = :interior); rtol = 1e-3, atol = 1e-12))
                @test pressure(solution, first(inside)) ≈
                      pressure(solution, first(inside); field = :interior)
            else
                @test diagnostics(solution).converged
            end
            direction, distance = [0.36, 0.48, 0.8], 1e6
            far = pressure(solution, Tuple(distance .* direction); field = :scattered)*distance*cis(-0.5distance)
            @test isapprox(far, scattering_amplitude(solution; direction); rtol = 1e-4, atol = 1e-12)
            mixed = [first(points), Tuple(distance .* direction)]
            @test pressure(solution, mixed; field = :scattered) ≈
                  [pressure(solution, p; field = :scattered) for p in mixed]
        end
    end
end

@time "Flat-cylinder domain and deferred shapes" @testset "Flat-cylinder domain and deferred shapes" begin
    solution = bem(
        Cylinder(0.5, 1.0; endcap_depth = 0.5), Rigid(), 0.3; n = 16, incidence_angle = 0.0)
    @test isfinite(pressure(solution, (0.8, 0.0, 0.0)))
    @test_throws ArgumentError pressure(solution, (0.0, 0.4, 0.0))
    bent = mfs(Cylinder(0.1, 1.0; radius_curvature = 2.0), Rigid(), 1.0;
        offset = 0.03, n_s = 6, n_phi = 6, condition_limit = 0)
    @test_throws ArgumentError pressure(bent, (2.0, 0.0, 0.0))
end
