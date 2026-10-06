using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
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
        @test pressure(solution, reduce(hcat, collect.(points)); field = :scattered) ≈
              actual
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

    @time "Spherical full MFS pressure" @testset "Spherical full MFS pressure" begin
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

    @time "MFS pressure source correctness" @testset "MFS pressure source correctness" begin
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
end

let
    function check_cylinder_amplitudes(actual, reference)
        for (a, b) in zip(actual, reference)
            @test abs(target_strength(a) - target_strength(b)) < 0.1
            @test abs(a - b) / abs(b) < 0.01
        end
    end

    function cylinder_amplitudes(solution, beta, alpha)
        d = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
        return [scattering_amplitude(solution; direction = q)
                for q in (-d, d, [0.0, 0.0, 1.0])]
    end

    @time "Closed-surface MFS against sphere modal solution" @testset "Closed-surface MFS against sphere modal solution" begin
        body = Sphere(0.5)
        collocation = mesh(body; method = :full, resolution = 0.2, mesh_order = 3)
        sources = mesh(body; method = :full, resolution = 0.25, qorder = 1, mesh_order = 2)
        checks = mesh(body; method = :full, resolution = 0.18, mesh_order = 3)
        beta, alpha = pi / 3, 0.4
        for boundary in (Rigid(), PressureRelease())
            solution = mfs(
                collocation, boundary, 1.0; source_mesh = sources, check_mesh = checks,
                offset = 0.3, incidence_angle = beta, incidence_azimuth = alpha, condition_limit = 0)
            angles = (pi, 0.0, acos(sin(beta) * sin(alpha)))
            reference = [scattering_amplitude(modal(body, boundary, 1.0; angle))
                         for angle in angles]
            check_cylinder_amplitudes(cylinder_amplitudes(solution, beta, alpha), reference)
            @test diagnostics(solution).boundary_residual.relative_residual < 0.01
            @test diagnostics(solution).check_count == length(checks.data)
            @test diagnostics(solution).condition_number === nothing
            @test target_strength(solution) ≈
                  first(AS.bistatic_map(solution, [pi - beta], [pi + alpha]).target_strength)
            @test_throws ArgumentError scattering_amplitude(solution; direction = zeros(3))
        end
        @test_throws ArgumentError mfs(collocation, Rigid(), 0.0; offset = 0.3)
        @test_throws ArgumentError mfs(collocation, Rigid(), 1.0; offset = -0.3)
        @test_throws ArgumentError mfs(
            sources, Rigid(), 1.0; offset = 0.3, source_mesh = collocation)
    end
end
