using AcousticScattering
using LinearAlgebra: BLAS, dot, norm
using Test

BLAS.set_num_threads(4)
const AS = AcousticScattering

function compare_region_pressure(actual, expected)
    for (got, wanted) in zip(actual, expected)
        @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        @test abs(20log10(abs(got/wanted))) < 0.01
    end
end

function modal_region_points(points, direction)
    [(dot(direction, p), sqrt(max(0, norm(p)^2-dot(direction, p)^2)), 0.0) for p in points]
end

@testset "Disconnected and branched fluid pressure" begin
    beta, alpha, k = pi/3, 0.4, 0.6
    direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
    center = [-0.35, 0.0, 0.0]
    active = mesh(; semiaxes = (0.25, 0.25, 0.25), center = Tuple(center),
        resolution = 0.36, mesh_order = 3, qorder = 5)
    passive = mesh(; semiaxes = (0.2, 0.2, 0.2), center = (0.4, 0.1, 0.0),
        resolution = 0.6, mesh_order = 3, qorder = 5)
    outer = mesh(Sphere(1.0); method = :full, resolution = 0.45, mesh_order = 3, qorder = 5)
    material = FluidFilled(0.4, 0.7)
    reference = modal(Sphere(0.25), material, k)
    phase = cis(k*dot(direction, center))
    points = [(1.1, 0.1, 0.0), (-0.35, 0.0, 0.0), (0.4, 0.1, 0.0),
        (-0.35, 0.4, 0.0), (0.0, -0.4, 0.2)]
    local_points = [Tuple(collect(p)-center) for p in points]
    expected = phase .* pressure(reference, modal_region_points(local_points, direction))
    for (surfaces, materials, parents, active_region, passive_region) in (
        ([passive, active], [FluidFilled(1, 1), material], [0, 0], 2, 1),
        ([outer, active, passive],
        [FluidFilled(1, 1), material, FluidFilled(1, 1)], [0, 1, 1], 2, 3))
        solution = bem(surfaces, materials, k; parents, incidence_angle = beta,
            incidence_azimuth = alpha, condition_limit = 0)
        @testset "parents=$parents" begin
            compare_region_pressure(pressure(solution, points), expected)
            @test pressure(solution, points[2]; region = active_region) ≈ expected[2] rtol = 1e-3
            @test pressure(solution, points[3]; region = passive_region) ≈ expected[3] rtol = 1e-3
            @test_throws ArgumentError pressure(solution, points[2]; region = passive_region)
            exterior = [points[1], (0.0, 0.0, 1.3)]
            local_exterior = [Tuple(collect(p)-center) for p in exterior]
            compare_region_pressure(pressure(solution, exterior; field = :scattered),
                phase .*
                pressure(reference, modal_region_points(local_exterior, direction); field = :scattered))
            if length(surfaces) == 3
                @test pressure(solution, points[4]; region = 1) ≈ expected[4] rtol = 1e-3
                @test_throws ArgumentError pressure(solution, points[4]; region = 0)
            end
            for (i, surface) in enumerate(surfaces)
                anchor = i == active_region ? AS.SVector(-0.35, 0.15, 0.2) :
                         i == passive_region ? AS.SVector(0.4, 0.22, 0.16) :
                         AS.SVector(0.6, 0.48, 0.64)
                q = surface.data[argmin(norm(node.coords-anchor) for node in surface.data)]
                compare_region_pressure([pressure(solution, q.coords; region = i)],
                    [pressure(solution, q.coords; region = parents[i])])
            end
            @test diagnostics(solution).relative_residual < 1e-9
        end
    end
end

@testset "Nested fluid pressure" begin
    beta, alpha = pi/3, 0.4
    direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
    outer = mesh(Sphere(1.0); method = :full, resolution = 0.3, mesh_order = 3, qorder = 5)
    inner = mesh(Sphere(0.5); method = :full, resolution = 0.2, mesh_order = 3, qorder = 5)
    materials = [FluidFilled(1.2, 1.1), FluidFilled(0.7, 0.8)]
    boundary = Shelled(FluidLayer(1.2, 1.1), FluidInterior(0.7, 0.8), 0.5)
    for k in (1.0,)
        solution = bem([outer, inner], materials, k; incidence_angle = beta,
            incidence_azimuth = alpha, condition_limit = 0)
        reference = modal(Sphere(1.0), boundary, k; m_max = 16)
        exterior = [(1.02, 0.0, 0.0), (0.0, 0.66, 0.88), (-1.2, 0.0, 1.6)]
        wall = [(0.75, 0.0, 0.0), (0.0, 0.48, 0.64)]
        cavity = [(0.0, 0.0, 0.0), (0.18, 0.0, 0.24)]
        @testset "k=$k" begin
            compare_region_pressure(pressure(solution, exterior; field = :scattered),
                pressure(reference, modal_region_points(exterior, direction); field = :scattered))
            for points in (exterior, wall, cavity)
                compare_region_pressure(pressure(solution, points),
                    pressure(reference, modal_region_points(points, direction)))
            end
            @test pressure(solution, wall; region = 1) ≈ pressure(solution, wall)
            @test pressure(solution, cavity; region = 2, field = :interior) ≈
                  pressure(solution, cavity)
            @test pressure(solution, exterior; region = 0) ≈ pressure(solution, exterior)
            for (i, surface) in enumerate((outer, inner))
                anchor = AS.SVector(0.6, 0.48, 0.64) * (i == 1 ? 1.0 : 0.5)
                q = surface.data[argmin(norm(node.coords-anchor) for node in surface.data)]
                point = Tuple(q.coords)
                pair = [point, Tuple(q.coords + 1e-8*q.normal)]
                parent = pressure(solution, pair; region = i-1)
                child = pressure(solution, [point, Tuple(q.coords - 1e-8*q.normal)]; region = i)
                compare_region_pressure(parent, child)
                compare_region_pressure(parent, pressure(reference, modal_region_points(pair, direction)))
                @test pressure(solution, point) ≈ first(parent)
                @test pressure(solution, point; field = :interior) ≈ first(child)
                @test_throws ArgumentError pressure(
                    solution, Tuple(q.coords -
                                    1e-8*q.normal); region = i-1)
                @test_throws ArgumentError pressure(
                    solution, Tuple(q.coords +
                                    1e-8*q.normal); region = i)
            end
            points = [exterior; wall; cavity]
            @test pressure(solution, reduce(hcat, collect.(points))) ≈
                  pressure(solution, points)
            @test vec(pressure(solution, reshape(points[1:6], 2, 3))) ≈
                  pressure(solution, points[1:6])
            @test pressure(solution, collect(first(exterior))) ≈
                  pressure(solution, first(exterior))
            @test isempty(pressure(solution, NTuple{3, Float64}[]))
            @test_throws ArgumentError pressure(solution, first(wall); field = :scattered)
            @test_throws ArgumentError pressure(solution, first(exterior); field = :interior)
            @test_throws ArgumentError pressure(solution, first(cavity); region = 1)
            @test_throws ArgumentError pressure(solution, first(wall); field = :shell)
            for region in (-1, 3, 1.5, :unknown)
                @test_throws ArgumentError pressure(solution, first(wall); region)
            end
            @test_throws ArgumentError pressure(solution, (NaN, 0.0, 0.0); field = :incident)
            @test pressure(solution, cavity; field = :incident, region = 2) ≈
                  pressure(reference, modal_region_points(cavity, direction); field = :incident)
            @test_throws ArgumentError pressure(solution, first(wall); field = :incident, region = 2)
            distance, observation = 1e6, [0.36, 0.48, 0.8]
            farpoint = Tuple(distance .* observation)
            far = pressure(solution, farpoint; field = :scattered)*distance*cis(-k*distance)
            @test isapprox(far, scattering_amplitude(solution; direction = observation);
                rtol = 1e-4, atol = 1e-12)
            mixed = [first(exterior), first(wall), first(cavity), farpoint]
            @test pressure(solution, mixed) ≈ [pressure(solution, p) for p in mixed]
            @test diagnostics(solution).relative_residual < 1e-9
        end
    end
end

@testset "Interacting fluid pressure" begin
    surfaces = [mesh(; semiaxes = (0.2, 0.2, 0.2), center,
                    resolution = 0.6, mesh_order = 3, qorder = 5)
                for center in ((-0.35, 0.0, 0.0), (0.35, 0.0, 0.0))]
    k = 0.6
    solution = bem(surfaces, [FluidFilled(0.7, 0.8), FluidFilled(1.4, 0.9)], k;
        parents = [0, 0], incidence_angle = pi/3, incidence_azimuth = 0.4,
        condition_limit = 0)
    for (i, surface) in enumerate(surfaces)
        anchor = AS.SVector(i == 1 ? -0.35 : 0.35, 0.12, 0.16)
        index = argmin(norm(node.coords-anchor) for node in surface.data)
        point = surface.data[index].coords
        expected = [solution.data.interfaces[i].pressure[index]]
        compare_region_pressure([pressure(solution, point; region = i)], expected)
        compare_region_pressure([pressure(solution, point; region = 0)], expected)
    end
    for direction in ([0.36, 0.48, 0.8], [-0.8, 0.6, 0.0])
        distance = 1e6
        value = pressure(solution, Tuple(distance .* direction); field = :scattered)
        amplitude = value * distance * cis(-k*distance)
        @test isapprox(amplitude, scattering_amplitude(solution; direction); rtol = 1e-4, atol = 1e-12)
    end
    @test_throws ArgumentError pressure(modal(Sphere(1.0), Rigid(), k), (2.0, 0.0, 0.0); region = 0)
end
