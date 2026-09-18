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

@testset "Higher-frequency spherical pressure" begin
    body, k, beta = Sphere(1.0), 6.0, pi/3
    points = [Tuple(r .* direction) for r in (1.0, 1+1e-8, 1.01, 1.2)
              for direction in ([1.0, 0, 0], [0.0, 0.6, 0.8], [-0.6, 0, 0.8])]
    for boundary in (Rigid(), FluidFilled(1.2, 1.1))
        reference = modal(body, boundary, k; m_max = 32)
        @testset "$(typeof(boundary))" begin
            solution = mfs(body, boundary, k; n = 128, oversampling = 2,
                offset = 0.2, incidence_angle = beta, m_max = 18, condition_limit = 0)
            compare_near_pressure(pressure(solution, points; field = :scattered),
                pressure(reference, near_reference_points(points, beta, 0.0); field = :scattered))
            options = boundary isa FluidFilled ? (; condition_limit = 0) :
                      (; compression = (method = :none,),
                gmres_kwargs = (reltol = 1e-9, restart = 400, maxiter = 2400))
            solution = bem(body, boundary, k; method = :full,
                meshsize = boundary isa Rigid ? 0.25 : 0.3,
                mesh_order = 3, qorder = 7, incidence_angle = beta, incidence_azimuth = 0.4,
                options...)
            compare_near_pressure(pressure(solution, points; field = :scattered),
                pressure(reference, near_reference_points(points, beta, 0.4); field = :scattered))
            @test diagnostics(solution).relative_residual < 1e-8
            if boundary isa FluidFilled
                inside = [(0.0, 0.0, 0.0), (0.3, 0.2, 0.1),
                    (0.6*(1-1e-8), 0.0, 0.8*(1-1e-8))]
                compare_near_pressure(pressure(solution, inside),
                    pressure(reference, near_reference_points(inside, beta, 0.4)))
            else
                @test diagnostics(solution).converged
            end
        end
    end
end

@testset "Gas-sphere pressure across resonance" begin
    body, gas = Sphere(1.0), GasFilled(0.0012, 0.23)
    beta, alpha = pi/3, 0.4
    points = [Tuple(r .* direction) for r in (1.0, 1+1e-8, 1.01, 1.2)
              for direction in ([1.0, 0, 0], [0.0, 0.6, 0.8], [-0.6, 0, 0.8])]
    inside = [(0.0, 0.0, 0.0), (0.3, 0.2, 0.1),
        (0.6*(1-1e-8), 0.0, 0.8*(1-1e-8))]
    peaks, references = Float64[], Float64[]
    for k in (0.0137, 0.0138, 0.0139)
        reference = modal(body, gas, k; m_max = 8)
        solution = bem(body, gas, k; method = :full, meshsize = 0.22,
            mesh_order = 3, qorder = 5, incidence_angle = beta, incidence_azimuth = alpha,
            condition_limit = 0)
        @testset "k=$k" begin
            actual = pressure(solution, points; field = :scattered)
            expected = pressure(reference, near_reference_points(points, beta, alpha); field = :scattered)
            compare_near_pressure(actual, expected)
            compare_near_pressure(pressure(solution, inside; field = :interior),
                pressure(reference, near_reference_points(inside, beta, alpha); field = :interior))
            surface = [(1.0, 0.0, 0.0), (0.0, 0.6, 0.8)]
            compare_near_pressure(pressure(solution, surface), pressure(solution, surface; field = :interior))
            @test diagnostics(solution).scaled_relative_residual < 1e-12
            push!(peaks, abs(first(actual)))
            push!(references, abs(first(expected)))
        end
    end
    @test argmax(peaks) == argmax(references) == 2
end
