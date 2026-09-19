using AcousticScattering
using Test
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

@time @testset "Pressure sampling and incident coordinates" begin
    solution = modal(Sphere(1.0), Rigid(), 1.6)
    point = (1.3, 0.4, -0.2)
    @test pressure(solution, point; field = :incident) ≈ cis(1.6 * point[1])
    @test pressure(solution, point) ≈
          pressure(solution, point; field = :incident) +
          pressure(solution, point; field = :scattered)
    points = [point, (0.0, 1.4, 0.0)]
    @test pressure(solution, points) == [pressure(solution, p) for p in points]
    @test pressure(solution, hcat(collect.(points)...)) == pressure(solution, points)
    grid = reshape([point, point, point, point], 2, 2)
    @test size(pressure(solution, grid)) == (2, 2)
    @test pressure(solution, [1.3, 0.4, -0.2]) == pressure(solution, point)
    @test pressure(solution, (0.0, 1.4, 0.0)) ≈ pressure(solution, (0.0, 0.0, 1.4))
    @test isempty(pressure(solution, zeros(3, 0)))
    @test isempty(pressure(solution, NTuple{3, Float64}[]))
    @test pressure(modal(Sphere(1.0), Rigid(), 1.6; angle = 0.4), point) ==
          pressure(solution, point)
    for field in (:total, :scattered, :interior)
        @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0); field)
    end
    @test_throws ArgumentError pressure(solution, (Inf, 0.0, 0.0))
    @test_throws ArgumentError pressure(solution, (NaN, 0.0, 0.0); field = :incident)
    @test_throws ArgumentError pressure(solution, [1.0, 2.0])
    @test_throws ArgumentError pressure(solution, zeros(2, 4))
    @test_throws ArgumentError pressure(solution, point; field = :unknown)
    @test_throws ArgumentError pressure(kirchhoff(Sphere(1.0), Rigid(), 1.6), point)
    @test_throws ArgumentError pressure(modal(Sphere(1.0), SolidElastic(2.7, 4.0, 2.0), 1.6), (
        0.0, 0.0, 0.0))
end

@time @testset "Matched fluid and pressure-release surface" begin
    body, k = Sphere(1.0), 1.6
    matched = modal(body, FluidFilled(1.0, 1.0), k)
    points = [(0.0, 0.0, 0.0), (0.23, 0.34, 0.45), (1.0, 0.0, 0.0), (-2.0, 0.5, 0.0)]
    @test pressure(matched, points) ≈ pressure(matched, points; field = :incident) rtol = 1e-12
    @test iszero(scattering_amplitude(matched))
    @test iszero(pressure(matched, last(points); field = :scattered))
    @test pressure(matched, (0.0, 0.0, 0.0); field = :interior) ≈ 1.0
    @test_throws ArgumentError pressure(matched, (1.1, 0.0, 0.0); field = :interior)
    @test_throws ArgumentError pressure(matched, (0.1, 0.0, 0.0); field = :scattered)
    soft = modal(body, PressureRelease(), k)
    @test maximum(abs, pressure(soft, [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)])) < 1e-12
end

@time @testset "Exterior pressure against radial FEM" begin
    body, k = Sphere(1.0), 1.6
    points = [(r * cos(theta), r * sin(theta), 0.0)
              for r in (1.0, 1 + 1e-8, 1.01, 1.127, 1.2, 1.201, 2.0),
    theta in (0.0, pi / 3, pi)]
    for boundary in (Rigid(), PressureRelease()), order in (1, 2)

        reference = modal(body, boundary, k; m_max = 14)
        n_elements = order == 1 ? 640 : 80
        solution = fem(body, boundary, k; n_elements, order, m_max = 14)
        for point in points, field in (:scattered, :total)

            @test pressure(solution, point; field)≈pressure(reference, point; field) rtol=0.001 atol=1e-12
        end
    end
    boundary = Rigid()
    reference = pressure(modal(body, boundary, k; m_max = 12), (1.137, 0.0, 0.0))
    coarse = pressure(fem(body, boundary, k; R = 2.0, n_elements = 20, m_max = 12), (
        1.137, 0.0, 0.0))
    fine = pressure(fem(body, boundary, k; R = 2.0, n_elements = 80, m_max = 12), (
        1.137, 0.0, 0.0))
    @test abs(fine - reference) < abs(coarse - reference) / 5
    adaptive = fem(body, boundary, k; adaptive = true, target_tol = 1e-4, m_max = 12)
    @test pressure(adaptive, (1.137, 0.0, 0.0)) ≈ reference rtol = 0.001
end

@time @testset "Fluid interior and interface limits against radial FEM" begin
    body = Sphere(1.0)
    for (boundary, k) in ((FluidFilled(1.2, 1.1), 1.6), (GasFilled(0.0012, 0.23), 0.0138))
        reference = modal(body, boundary, k; m_max = 14)
        solution = fem(
            body, boundary, k; n_elements_int = 320, n_elements_ext = 160, m_max = 14)
        points = [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.321, 0.2, -0.1),
            (-0.7, 0.0, 0.0), (1 - 1e-8, 0.0, 0.0)]
        for point in points
            @test pressure(solution, point; field = :interior) ≈
                  pressure(reference, point; field = :interior) rtol = 0.001
        end
        @test pressure(solution, points) == pressure(solution, points; field = :interior)
        for r in (1.0, 1.137, 1.2, 2.0), theta in (0.0, pi / 3, pi)

            point = (r * cos(theta), r * sin(theta), 0.0)
            @test pressure(solution, point; field = :scattered) ≈
                  pressure(reference, point; field = :scattered) rtol = 0.001
        end
        for result in (reference, solution)
            surface = pressure(result, (1.0, 0.0, 0.0))
            @test pressure(result, (1.0, 0.0, 0.0); field = :interior) ≈ surface rtol = 1e-9
            for gap in (1e-4, 1e-6, 1e-8)
                @test pressure(result, (1 - gap, 0.0, 0.0)) ≈ surface rtol = 0.001
                @test pressure(result, (1 + gap, 0.0, 0.0)) ≈ surface rtol = 0.001
            end
        end
    end
end

@time @testset "Fluid pressure continuity at the center" begin
    body, boundary, k = Sphere(1.0), FluidFilled(1.2, 1.1), 1.6
    reference = modal(body, boundary, k; m_max = 14)
    solution = fem(body, boundary, k; n_elements_int = 80, n_elements_ext = 80, m_max = 14)
    center = pressure(solution, (0.0, 0.0, 0.0))
    @test center ≈ pressure(reference, (0.0, 0.0, 0.0)) rtol = 0.001
    for point in ((1e-12, 0.0, 0.0), (0.0, 1e-12, 0.0), (0.0, 0.0, -1e-12))
        @test pressure(solution, point) ≈ center atol = 1e-10
    end
end

@time @testset "Spherical pressure far-field limit and modal cutoff" begin
    body, k = Sphere(1.0), 1.6
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1)),
        theta in (0.0, pi / 3, pi)

        solution = modal(body, boundary, k; m_max = 12)
        r = 1e6
        point = (r * cos(theta), r * sin(theta), 0.0)
        expected = scattering_amplitude(modal(body, boundary, k; angle = theta, m_max = 12))
        @test r * cis(-k * r) * pressure(solution, point; field = :scattered) ≈ expected rtol = 1e-5
    end
    point = (1.01, 0.0, 0.0)
    coarse = pressure(modal(body, Rigid(), k; m_max = 0), point)
    fine = pressure(modal(body, Rigid(), k; m_max = 12), point)
    reference = pressure(modal(body, Rigid(), k; m_max = 18), point)
    @test abs(fine - reference) < abs(coarse - reference) / 1000
    @test_throws ArgumentError modal(body, Rigid(), k; m_max = -1)
end
