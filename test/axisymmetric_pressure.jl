using AcousticScattering
using LinearAlgebra: BLAS, dot, norm
using Test

BLAS.set_num_threads(2)

function rotated_pressure_points(points, beta)
    direction = [cos(beta), sin(beta), 0.0]
    return [(dot(direction, p), sqrt(max(0, norm(p)^2-dot(direction, p)^2)), 0.0)
            for p in points]
end

function compare_pressure_values(actual, expected)
    for (got, wanted) in zip(actual, expected)
        @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        @test abs(20log10(abs(got/wanted))) < 0.01
    end
end

@time "Axisymmetric spherical pressure" @testset "Axisymmetric spherical pressure" begin
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1)), beta in (0.0, pi/3)

        n = iszero(beta) ? 256 : 128
        solution = bem(
            Sphere(1.0), boundary, 0.3; n, incidence_angle = beta, m_max = 4, rtol = 1e-7)
        reference = modal(Sphere(1.0), boundary, 0.3)
        points = [(r, 0.0, 0.0) for r in (1.0, 1+1e-8, 1.01, 1.2)]
        append!(points, [Tuple(r .* [0.6, 0.48, 0.64]) for r in (1.0, 1+1e-8, 1.01, 1.2)])
        actual = pressure(solution, points; field = :scattered)
        expected = pressure(reference, rotated_pressure_points(points, beta); field = :scattered)
        @time "$(typeof(boundary)) beta=$beta" @testset "$(typeof(boundary)) beta=$beta" begin
            compare_pressure_values(actual, expected)
            @test pressure(solution, points) ≈
                  actual + pressure(solution, points; field = :incident)
            @test pressure(solution, first(points); field = :scattered) ≈ first(actual)
            @test pressure(solution, reduce(hcat, collect.(points)); field = :scattered) ≈
                  actual
            @test size(pressure(solution, reshape(points, 2, 4))) == (2, 4)
            @test isempty(pressure(solution, NTuple{3, Float64}[]))
            @test_throws ArgumentError pressure(solution, (NaN, 0.0, 0.0); field = :incident)
            direction, distance = [0.36, 0.48, 0.8], 1e6
            amplitude = scattering_amplitude(solution; angle = acos(direction[1]),
                azimuth = atan(direction[3], direction[2]))
            actual_far = pressure(solution, Tuple(distance .* direction); field = :scattered)*distance*cis(-0.3distance)
            @test isapprox(actual_far, amplitude; rtol = 1e-4, atol = 1e-12)
            if boundary isa FluidFilled
                inside = [(0.0, 0.0, 0.0), (0.4, 0.0, 0.0), (1-1e-8, 0.0, 0.0),
                    (0.6*(1-1e-8), 0.48*(1-1e-8), 0.64*(1-1e-8))]
                compare_pressure_values(pressure(solution, inside; field = :interior),
                    pressure(reference, rotated_pressure_points(inside, beta); field = :interior))
                surface = [(1.0, 0.0, 0.0), (0.6, 0.48, 0.64)]
                @test all(isapprox.(pressure(solution, surface),
                    pressure(solution, surface; field = :interior);
                    rtol = 1e-3, atol = 1e-12))
            else
                @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0))
            end
            if iszero(beta)
                coarse = bem(
                    Sphere(1.0), boundary, 0.3; n = 64, incidence_angle = 0.0, rtol = 1e-7)
                coarse_error = maximum(abs.((pressure(coarse, points; field = :scattered)-expected) ./
                                            expected))
                @test maximum(abs.((actual-expected) ./ expected)) < coarse_error
            end
        end
    end
end
