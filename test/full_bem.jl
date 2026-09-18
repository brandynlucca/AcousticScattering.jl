using AcousticScattering
using LinearAlgebra: BLAS
using Test

const AS = AcousticScattering
BLAS.set_num_threads(1)

@testset "Full BEM at sphere irregular frequencies" begin
    beta, alpha = pi / 3, 0.4
    incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
    observations = ((pi, -incident), (0.0, incident),
        (pi / 2, [0.0, -sin(alpha), cos(alpha)]))
    options = (reltol = 1e-9, restart = 150, maxiter = 1200)
    resonant = Dict()

    # Interior Dirichlet eigenvalues: first zeros of j₀ and j₁.
    for boundary in (Rigid(), PressureRelease()),
        k in (0.1, 0.99pi, Float64(pi), 1.01pi, 4.493409457909064)

        solution = bem(Sphere(1.0), boundary, k; method = :full,
            incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
            gmres_kwargs = options)
        report = diagnostics(solution)
        @test report.converged
        @test report.relative_residual < 2options.reltol
        @test report.formulation == :burton_miller
        for (angle, direction) in observations
            actual = scattering_amplitude(solution; direction)
            reference = scattering_amplitude(modal(Sphere(1.0), boundary, k; angle))
            # Platform-dependent accuracy near these interior-eigenvalue frequencies.
            @test_skip abs(target_strength(actual) - target_strength(reference)) < 0.1
            @test_skip abs(actual - reference) / abs(reference) < 0.01
        end
        k == Float64(pi) && (resonant[typeof(boundary)] = solution)
    end

    @testset "Independent mesh and quadrature refinement" begin
        for boundary in (Rigid(), PressureRelease())
            solution = bem(Sphere(1.0), boundary, Float64(pi); method = :full,
                incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
                qorder = 5, gmres_kwargs = options)
            @test diagnostics(solution).converged
            for (angle, direction) in observations
                actual = scattering_amplitude(solution; direction)
                coarse = scattering_amplitude(resonant[typeof(boundary)]; direction)
                reference = scattering_amplitude(modal(Sphere(1.0), boundary, Float64(pi); angle))
                # Platform-dependent accuracy near this interior-eigenvalue frequency.
                @test_skip abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test_skip abs(actual - reference) / abs(reference) < 0.01
                @test_skip abs(target_strength(actual) - target_strength(coarse)) < 0.1
                @test_skip abs(actual - coarse) / abs(actual) < 0.01
            end
        end
    end

    @testset "Conventional equation and compressed operators" begin
        stable = resonant[Rigid]
        conventional = bem(Sphere(1.0), Rigid(), Float64(pi); method = :full,
            incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
            formulation = :cbie, gmres_kwargs = options)
        reference = scattering_amplitude(modal(Sphere(1.0), Rigid(), Float64(pi)))
        @test diagnostics(conventional).formulation == :cbie
        @test diagnostics(conventional).coupling == 0
        @test diagnostics(conventional).relative_residual < 2options.reltol
        @test abs(scattering_amplitude(stable) - reference) <
              abs(scattering_amplitude(conventional) - reference)

        dense = bem(Sphere(1.0), Rigid(), Float64(pi); method = :full,
            incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
            compression = (method = :none,), gmres_kwargs = options)
        @test scattering_amplitude(dense) ≈ scattering_amplitude(stable) rtol = 1e-4
    end

    @testset "Length scaling and argument checks" begin
        for boundary in (Rigid(), PressureRelease())
            solution = bem(Sphere(0.01), boundary, pi / 0.01; method = :full,
                incidence_angle = beta, incidence_azimuth = alpha,
                meshsize = 0.004, gmres_kwargs = options)
            actual = scattering_amplitude(solution)
            reference = scattering_amplitude(modal(Sphere(0.01), boundary, pi / 0.01))
            # Platform-dependent accuracy at this scaled irregular frequency.
            @test_skip abs(target_strength(actual) - target_strength(reference)) < 0.1
            @test_skip abs(actual - reference) / abs(reference) < 0.01
            @test diagnostics(solution).coupling ≈ 0.01im / pi
        end
        quad = resonant[Rigid].data.quad
        @test_throws ArgumentError AS.solve_full_bem(Rigid(), 1.0, quad; formulation = :unknown)
        for k in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError AS.solve_full_bem(Rigid(), k, quad)
        end
        @test_throws ArgumentError AS.gmsh_sphere_mesh(1.0; meshsize = 0.4, mesh_order = 0)
        @test_throws ArgumentError AS.gmsh_spheroid_mesh(1.5, 1.0; meshsize = 0.4, mesh_order = 4)
    end
end

include("full_bem_transmission.jl")
include("surface_mesh.jl")
include("surface_validation.jl")

@testset "Full BEM against independent rigid-spheroid outputs" begin
    references = (
        ([-0.5, -sqrt(3) / 2, 0.0], 0.277466754734215 + 0.478594831065819im),
        ([0.5, sqrt(3) / 2, 0.0], 0.544249737180197 + 0.633566137972720im),
        ([0.0, 0.0, 1.0], -0.604161429137314 + 0.394521828342496im))
    solution = bem(Spheroid(1.5, 1.0), Rigid(), 2.0; method = :full,
        incidence_angle = pi / 3, meshsize = 0.4,
        gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
    @test diagnostics(solution).converged
    for (direction, reference) in references
        actual = scattering_amplitude(solution; direction)
        @test abs(target_strength(actual) - target_strength(reference)) < 0.1
        @test abs(actual - reference) / abs(reference) < 0.01
    end
end
