using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Boundary pressure geometry dispatch" @testset "Boundary pressure geometry dispatch" begin
        solution = bem(Sphere(1.0), Rigid(), 1.0; n = 12, incidence_angle = 0.0)
        @test isfinite(pressure(solution, (2.0, 0.0, 0.0)))
        solution = mfs(Cylinder(0.1, 1.0; endcap_depth = 0.1), Rigid(), 1.0;
            n = 16, incidence_angle = 0.0, condition_limit = 0)
        @test isfinite(pressure(solution, (2.0, 0.0, 0.0)))
    end
end

let
    @time "Solution interface contract (every concrete AbstractSolution type)" @testset "Solution interface contract (every concrete AbstractSolution type)" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)

        modal_sol = AS.modal(sphere, AS.Rigid(), k)
        kirch_sol = AS.kirchhoff(sphere, AS.Rigid(), k)
        fem_sol = AS.fem(sphere, AS.Rigid(), k)
        bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)
        mfs_sol = AS.mfs(sphere, AS.Rigid(), k; n = 16)

        @time "target_strength works with no keywords on every concrete type" @testset "target_strength works with no keywords on every concrete type" begin
            for sol in (modal_sol, kirch_sol, fem_sol, bem_sol, mfs_sol)
                @test AS.target_strength(sol) isa Float64
            end
        end

        @time "Complex amplitude and scalar-only FEM fallback" @testset "Complex amplitude and scalar-only FEM fallback" begin
            for sol in (modal_sol, kirch_sol, fem_sol, bem_sol, mfs_sol)
                @test AS.scattering_amplitude(sol) isa Complex
            end
            scalar = AS.fem(sphere, AS.Rigid(), k;
                method = :meridian, n_r = 3, n_theta = 8, l_max = 3)
            @test_throws ArgumentError AS.scattering_amplitude(scalar)
        end

        @time "angle/azimuth keywords: supported where a real bistatic query exists, explicit error otherwise" @testset "angle/azimuth keywords: supported where a real bistatic query exists, explicit error otherwise" begin
            # These result paths reject unsupported observation queries.
            @test_throws ArgumentError AS.target_strength(modal_sol; angle = 0.3)
            @test_throws ArgumentError AS.scattering_amplitude(modal_sol; angle = 0.3)
            @test_throws ArgumentError AS.target_strength(kirch_sol; angle = 0.3)
            @test_throws ArgumentError AS.scattering_amplitude(kirch_sol; angle = 0.3)
            @test_throws ArgumentError AS.target_strength(fem_sol; angle = 0.3)

            # bem/mfs axisymmetric: genuine reusable surface state, angle/azimuth are real queries.
            @test AS.target_strength(bem_sol; angle = pi / 2) isa Float64
            @test AS.target_strength(mfs_sol; angle = pi / 2) isa Float64
        end
    end
end

let
    @time "Pressure sampling and incident coordinates" @testset "Pressure sampling and incident coordinates" begin
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
        @test_throws ArgumentError pressure(
            modal(Sphere(1.0), SolidElastic(2.7, 4.0, 2.0), 1.6), (
                0.0, 0.0, 0.0))
    end
end
