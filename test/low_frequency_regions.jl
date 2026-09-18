using AcousticScattering
using Test
using LinearAlgebra: BLAS

BLAS.set_num_threads(1)

@testset "Low-frequency fluid regions" begin
    k = 2pi * 500 / 1477.4
    gas = GasFilled(0.00129, 0.23)
    @testset "Oblique gas spheroid against modal expansion" begin
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
        @test_skip "requires SpheroidalWaves quad-precision backend, not available locally"
    end

    @testset "Nested ellipsoid: $formulation" for formulation in (:muller,)
        materials = [FluidFilled(1.04, 1.04), gas]
        options = (; resolution = 0.4, qorder = 4, tip_ratio = 0.4)
        surfaces = [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
            mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
                rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
        beta = pi / 3
        incident = [cos(beta), sin(beta), 0.0]
        directions = (-incident, incident, [0.0, 0.0, 1.0])
        coupled = bem(surfaces, materials, k; formulation, incidence_angle = beta)
        @test diagnostics(coupled).formulation == formulation
        @test diagnostics(coupled).scaled_relative_residual < 1e-10
        values = [scattering_amplitude(coupled; direction) for direction in directions]
        @test all(isfinite, values)
    end
end
