using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Full BEM against independent rigid-spheroid outputs" @testset "Full BEM against independent rigid-spheroid outputs" begin
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
end

let
    @time "Oblique gas spheroid against modal expansion" @testset "Oblique gas spheroid against modal expansion" begin
        k = 2pi * 500 / 1477.4
        gas = GasFilled(0.00129, 0.23)
        body = Spheroid(0.025, 0.0075)
        surface = mesh(; semiaxes = (0.025, 0.0075, 0.0075),
            resolution = 0.55, tip_ratio = 0.4, qorder = 5)
        beta, alpha = pi / 3, 0.4
        solutions = [bem([surface], [gas], k; formulation,
                         incidence_angle = beta, incidence_azimuth = alpha)
                     for formulation in (:muller, :cbie)]
        @test diagnostics(first(solutions)).formulation == :muller
        @test all(item -> item.method === :calderon,
            diagnostics(first(solutions)).derivative_evaluation)
        @test diagnostics(last(solutions)).interface_residuals[1].flux_interior === nothing
        reference = modal(body, gas, k; incidence_angle = beta, incidence_azimuth = alpha,
            m_max = 4, n_max = 6, precision = :quad)
        @test abs(target_strength(first(solutions)) - target_strength(reference)) < 2
    end
end
