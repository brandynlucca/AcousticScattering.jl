using AcousticScattering
using LinearAlgebra: BLAS, dot, norm
using Test

BLAS.set_num_threads(2)

function reference_points(points, beta, alpha)
    direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
    return [(dot(direction, p), sqrt(max(0, norm(p)^2-dot(direction, p)^2)), 0.0)
            for p in points]
end

function check_boundary_pressure(solution, beta, alpha; full = false)
    reference = modal(solution.body, solution.boundary, solution.k)
    directions = ([1.0, 0, 0], [0.0, 0.6, 0.8], [-0.6, 0.0, 0.8])
    points = [Tuple(r .* v) for r in (1.0, 1+1e-8, 1.001, 1.1, 2.0) for v in directions]
    expected = pressure(reference, reference_points(points, beta, alpha); field = :scattered)
    actual = pressure(solution, points; field = :scattered)
    for (got, wanted) in zip(actual, expected)
        @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        @test abs(20log10(abs(got/wanted))) < 0.01
    end
    incident = pressure(solution, points; field = :incident)
    @test pressure(solution, points) ≈ incident + actual
    @test incident ≈
          pressure(reference, reference_points(points, beta, alpha); field = :incident)
    @test pressure(solution, first(points); field = :scattered) ≈ first(actual)
    @test pressure(solution, collect(first(points)); field = :scattered) ≈ first(actual)
    @test pressure(solution, reduce(hcat, collect.(points)); field = :scattered) ≈ actual
    @test vec(pressure(solution, reshape(points, 3, 5); field = :scattered)) ≈ actual
    @test isempty(pressure(solution, NTuple{3, Float64}[]))
    @test_throws ArgumentError pressure(solution, (NaN, 0.0, 0.0))
    @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0); field = :scattered)
    if solution.boundary isa FluidFilled
        inside = [(0.0, 0.0, 0.0);
                  [Tuple(r .* v) for r in (0.4, 1-1e-8, 1.0) for v in directions]]
        expected_inside = pressure(reference, reference_points(inside, beta, alpha); field = :interior)
        actual_inside = pressure(solution, inside; field = :interior)
        for (got, wanted) in zip(actual_inside, expected_inside)
            @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        end
        @test pressure(solution, first(inside)) ≈ first(actual_inside)
        surface = [Tuple(v) for v in directions]
        @test all(isapprox.(pressure(solution, surface),
            pressure(solution, surface; field = :interior); rtol = 1e-3, atol = 1e-12))
    else
        @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0))
    end
    direction, distance = [0.36, 0.48, 0.8], 1e6
    far_pressure = pressure(solution, Tuple(distance .* direction); field = :scattered) *
                   distance * cis(-solution.k*distance)
    amplitude = full ? scattering_amplitude(solution; direction) :
                scattering_amplitude(
        solution; angle = acos(direction[1]), azimuth = atan(direction[3], direction[2]))
    @test isapprox(far_pressure, amplitude; rtol = 1e-4, atol = 1e-12)
end

@testset "Spherical MFS pressure" begin
    for k in (0.3, 2.0), boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1)),
        beta in (0.0, pi/3)
        solution = mfs(Sphere(1.0), boundary, k; n = 96, oversampling = 2,
            offset = 0.2, incidence_angle = beta, m_max = 10, condition_limit = 0)
        @testset "k=$k $(typeof(boundary)) beta=$beta" begin
            check_boundary_pressure(solution, beta, 0.0)
        end
    end
end

@testset "Spherical full MFS pressure" begin
    surface = mesh(
        Sphere(1.0); method = :full, resolution = 0.3, mesh_order = 3, qorder = 4)
    sources = mesh(
        Sphere(1.0); method = :full, resolution = 0.35, mesh_order = 3, qorder = 1)
    for boundary in (Rigid(), PressureRelease())
        solution = mfs(surface, boundary, 1.0; source_mesh = sources, offset = 0.35,
            incidence_angle = pi/3, incidence_azimuth = 0.4, condition_limit = 0)
        check_boundary_pressure(solution, pi/3, 0.4; full = true)
    end
end

@testset "MFS pressure source correctness" begin
    points = [(1+1e-8, 0.0, 0.0), (0.0, 0.6*(1+1e-8), 0.8*(1+1e-8)), (1.2, 0.0, 0.0)]
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1))
        reference = pressure(modal(Sphere(1.0), boundary, 1.0),
            reference_points(points, pi/3, 0.0); field = :scattered)
        solution = mfs(Sphere(1.0), boundary, 1.0; n = 96, oversampling = 2,
            offset = 0.2, incidence_angle = pi/3, m_max = 8, condition_limit = 0)
        error = maximum(abs.((pressure(solution, points; field = :scattered)-reference) ./
                             reference))
        @test error < 1e-3
    end
end

@testset "Spherical full BEM pressure" begin
    for (k, boundary) in ((1.0, Rigid()), (1.0, PressureRelease()),
        (0.3, FluidFilled(1.2, 1.1)), (2.0, FluidFilled(1.2, 1.1)))
        options = boundary isa FluidFilled ? (; condition_limit = 0) :
                  (; gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
        meshsize = k < 1 ? 0.25 : 0.4
        solution = bem(Sphere(1.0), boundary, k; method = :full,
            meshsize, mesh_order = 3, qorder = 5, incidence_angle = pi/3, incidence_azimuth = 0.4,
            options...)
        @testset "k=$k $(typeof(boundary))" begin
            check_boundary_pressure(solution, pi/3, 0.4; full = true)
        end
    end
end

@testset "Boundary pressure geometry dispatch" begin
    solution = bem(Sphere(1.0), Rigid(), 1.0; n = 12, incidence_angle = 0.0)
    @test isfinite(pressure(solution, (2.0, 0.0, 0.0)))
    solution = mfs(Cylinder(0.1, 1.0; endcap_depth = 0.1), Rigid(), 1.0;
        n = 16, incidence_angle = 0.0, condition_limit = 0)
    @test isfinite(pressure(solution, (2.0, 0.0, 0.0)))
end

@testset "Full BEM pressure correctness" begin
    points = [(1+1e-8, 0.0, 0.0), (0.0, 0.6*(1+1e-8), 0.8*(1+1e-8)), (1.2, 0.0, 0.0)]
    reference = pressure(modal(Sphere(1.0), PressureRelease(), 1.0),
        reference_points(points, pi/3, 0.4); field = :scattered)
    solution = bem(Sphere(1.0), PressureRelease(), 1.0; method = :full,
        meshsize = 0.4, mesh_order = 3, qorder = 5, incidence_angle = pi/3, incidence_azimuth = 0.4,
        gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
    error = maximum(abs.((pressure(solution, points; field = :scattered)-reference) ./
                         reference))
    @test error < 1e-3
end
