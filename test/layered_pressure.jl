using AcousticScattering
using Test
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

@time @testset "Elastic and layered spherical pressure against radial FEM" begin
    body = Sphere(1.0)
    cases = (
        (SolidElastic(2.7, 4.0, 2.0), (0.4, 1.6, 3.2), 640),
        (Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8),
            (0.4, 1.8, 1.92, 2.0, 2.1), 1280),
        (Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(0.0012, 0.23), 0.8),
            (0.4, 1.6, 3.2), 1280),
        (Shelled(FluidLayer(1.04, 1.04), VacuumInterior(), 0.8), (0.4, 1.6, 4.0), 640),
        (Shelled(FluidLayer(1.04, 1.04), FluidInterior(1.2, 1.1), 0.8),
            (0.4, 1.6, 4.0), 640),
        (Shelled(FluidLayer(1.04, 1.04), FluidInterior(0.0012, 0.23), 0.8),
            (0.016, 0.017, 0.018, 1.6), 640))
    for (boundary, ks, n_elements) in cases, k in ks

        reference = modal(body, boundary, k; m_max = 20)
        solution = fem(body, boundary, k; n_elements, m_max = 20)
        for r in (1.0, 1+1e-8, 1.13, 2.0), theta in (0.0, pi/3, pi),
            field in (:total, :scattered)
            point = (r*cos(theta), r*sin(theta), 0.0)
            @test pressure(solution, point; field)≈pressure(reference, point; field) rtol=0.001 atol=1e-12
        end
        if boundary isa Shelled && boundary.interior isa FluidInterior
            for r in (0.0, 0.17, 0.8-1e-8, 0.8), theta in (0.0, pi/3, pi)

                point = (r*cos(theta), 0.0, r*sin(theta))
                @test pressure(solution, point; field = :interior)≈pressure(reference, point; field = :interior) rtol=0.001 atol=1e-12
                @test pressure(solution, point) ==
                      pressure(solution, point; field = :interior)
            end
        end
        if boundary isa Shelled{FluidLayer}
            for r in (0.8, 0.8+1e-8, 0.9137, 1-1e-8, 1.0), theta in (0.0, pi/3, pi)

                point = (r*cos(theta), r*sin(theta), 0.0)
                @test pressure(solution, point; field = :shell)≈pressure(reference, point; field = :shell) rtol=0.001 atol=1e-12
            end
            for result in (reference, solution), theta in (0.0, pi/3, pi)

                outer = (cos(theta), sin(theta), 0.0)
                inner = 0.8 .* outer
                @test pressure(result, outer; field = :shell) ≈ pressure(result, outer) rtol = 1e-9
                if boundary.interior isa FluidInterior
                    @test pressure(result, inner; field = :shell) ≈
                          pressure(result, inner; field = :interior) rtol = 1e-9
                else
                    @test abs(pressure(result, inner; field = :shell)) < 1e-12
                end
            end
        end
    end
end

@time @testset "Layer regions, identical media and interface selection" begin
    body, k = Sphere(1.0), 1.6
    for solve in (modal, fem)
        wall = Shelled(FluidLayer(1.2, 1.1), FluidInterior(1.2, 1.1), 0.8)
        options = solve === fem ? (; n_elements = 640) : (;)
        layered = solve(body, wall, k; m_max = 16, options...)
        homogeneous = modal(body, FluidFilled(1.2, 1.1), k; m_max = 16)
        points = [(r, 0.0, 0.0)
                  for r in (0.0, 0.8-1e-8, 0.8, 0.8+1e-8, 0.93, 1.0, 1+1e-8, 2.0)]
        for point in points
            @test pressure(layered, point) ≈ pressure(homogeneous, point) rtol=0.001
        end
        @test_throws ArgumentError pressure(layered, (0.9, 0.0, 0.0); field = :interior)
        @test_throws ArgumentError pressure(layered, (0.7, 0.0, 0.0); field = :shell)
        @test_throws ArgumentError pressure(layered, (1.1, 0.0, 0.0); field = :shell)
        @test_throws ArgumentError pressure(layered, (0.9, 0.0, 0.0); field = :scattered)

        identical = Shelled(
            ElasticLayer(2.7, 4.0, 2.0; interior_coupling = :identical_fluid),
            FluidInterior(0.0012, 0.23), 0.8)
        water = Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8)
        first = solve(body, identical, k; m_max = 16, options...)
        second = solve(body, water, k; m_max = 16, options...)
        points = [(0.0, 0.0, 0.0), (0.3, 0.2, -0.1), (0.8, 0.0, 0.0),
            (1.0, 0.0, 0.0), (-2.0, 0.0, 0.0)]
        @test pressure(first, points) == pressure(second, points)
        @test_throws ArgumentError pressure(first, (0.9, 0.0, 0.0))
        @test_throws ArgumentError pressure(first, (0.8, 0.0, 0.0); field = :shell)
        @test pressure(first, (0.9, 0.0, 0.0); field = :incident) ≈ cis(k*0.9)
        vacuum = solve(body, Shelled(FluidLayer(1.04, 1.04), VacuumInterior(), 0.8), k; options...)
        @test_throws ArgumentError pressure(vacuum, (0.7, 0.0, 0.0))
        @test_throws ArgumentError pressure(vacuum, (0.8, 0.0, 0.0); field = :interior)
    end
end

@time @testset "Layered pressure refinement and far-field limit" begin
    body, k = Sphere(1.0), 1.8
    for layer in (FluidLayer(1.04, 1.04), ElasticLayer(2.7, 4.0, 2.0))
        boundary = Shelled(layer, FluidInterior(1.0, 1.0), 0.8)
        reference = modal(body, boundary, k; m_max = 16)
        coarse = fem(body, boundary, k; n_elements = 80, m_max = 16)
        fine = fem(body, boundary, k; n_elements = 640, m_max = 16)
        for point in ((0.31, 0.0, 0.0), (0.8-1e-8, 0.0, 0.0), (1+1e-8, 0.0, 0.0))
            expected = pressure(reference, point)
            @test abs(pressure(fine, point)-expected) <
                  abs(pressure(coarse, point)-expected)/8
        end
        for theta in (0.0, pi/3, pi)
            point = (1e6*cos(theta), 1e6*sin(theta), 0.0)
            expected = scattering_amplitude(modal(
                body, boundary, k; angle = theta, m_max = 16))
            @test 1e6*cis(-k*1e6)*pressure(reference, point; field = :scattered) ≈ expected rtol=1e-5
        end
    end
end
