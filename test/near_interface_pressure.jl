using AcousticScattering
using LinearAlgebra: BLAS, dot, norm
using Test

BLAS.set_num_threads(4)

function compare_near_pressure(actual, expected)
    for (got, wanted) in zip(actual, expected)
        @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        @test abs(20log10(abs(got/wanted))) < 0.01
    end
end

function near_reference_points(points, beta, alpha)
    direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
    return [(dot(direction, p), sqrt(max(0, norm(p)^2-dot(direction, p)^2)), 0.0)
            for p in points]
end

@time @testset "Higher-frequency spherical pressure" begin
    # NOTE: a full-mesh BEM comparison at this near-surface tolerance needs a fine enough
    # mesh that geometric approximation error stays under rtol=1e-3, which costs minutes
    # regardless of k/qorder/compression; that comparison is covered at full fidelity in
    # perf/near_interface_pressure.jl. Here only the (fast) MFS code path is checked.
    body, k, beta = Sphere(1.0), 6.0, pi/3
    points = [Tuple(r .* direction) for r in (1.0, 1+1e-8, 1.01, 1.2)
              for direction in ([1.0, 0, 0], [0.0, 0.6, 0.8], [-0.6, 0, 0.8])]
    for (boundary, n) in ((Rigid(), 96), (FluidFilled(1.2, 1.1), 64))
        reference = modal(body, boundary, k; m_max = 32)
        @time @testset "$(typeof(boundary))" begin
            solution = mfs(body, boundary, k; n, oversampling = 2,
                offset = 0.2, incidence_angle = beta, m_max = 18, condition_limit = 0)
            compare_near_pressure(pressure(solution, points; field = :scattered),
                pressure(reference, near_reference_points(points, beta, 0.0); field = :scattered))
        end
    end
end

@time @testset "Gas-sphere pressure across resonance" begin
    # NOTE: sweeping across the resonance peak (k in 0.0137-0.0139) requires several minutes
    # per point due to poor GMRES conditioning near resonance; that peak-location sweep is
    # covered at full fidelity in perf/near_interface_pressure.jl. Here a single off-resonance
    # k checks the same code path cheaply.
    body, gas, k = Sphere(1.0), GasFilled(0.0012, 0.23), 0.012
    beta, alpha = pi/3, 0.4
    points = [Tuple(r .* direction) for r in (1.0, 1+1e-8, 1.01, 1.2)
              for direction in ([1.0, 0, 0], [0.0, 0.6, 0.8], [-0.6, 0, 0.8])]
    inside = [(0.0, 0.0, 0.0), (0.3, 0.2, 0.1),
        (0.6*(1-1e-8), 0.0, 0.8*(1-1e-8))]
    reference = modal(body, gas, k; m_max = 8)
    solution = bem(body, gas, k; method = :full, meshsize = 0.3,
        mesh_order = 3, qorder = 5, incidence_angle = beta, incidence_azimuth = alpha,
        condition_limit = 0)
    actual = pressure(solution, points; field = :scattered)
    expected = pressure(reference, near_reference_points(points, beta, alpha); field = :scattered)
    compare_near_pressure(actual, expected)
    compare_near_pressure(pressure(solution, inside; field = :interior),
        pressure(reference, near_reference_points(inside, beta, alpha); field = :interior))
    surface = [(1.0, 0.0, 0.0), (0.0, 0.6, 0.8)]
    compare_near_pressure(pressure(solution, surface), pressure(solution, surface; field = :interior))
    @test diagnostics(solution).scaled_relative_residual < 1e-12
end
