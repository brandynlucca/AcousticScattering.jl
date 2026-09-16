using AcousticScattering
using Test
using LinearAlgebra: BLAS

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

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
        for (theta, phi) in ((pi - beta, alpha + pi), (beta, alpha), (pi / 2, pi / 2))
            direction = [cos(theta), sin(theta) * cos(phi), sin(theta) * sin(phi)]
            refs = [scattering_amplitude(modal(body, gas, k;
                        incidence_angle = beta, incidence_azimuth = alpha,
                        scatter_angle = theta, scatter_azimuth = phi,
                        n_max = order, m_max = order, precision = :quad))
                    for order in (4, 6)]
            @test abs(refs[1] - refs[2]) / abs(refs[2]) < 1e-5
            for sol in solutions
                actual, reference = scattering_amplitude(sol; direction), last(refs)
                db = abs(target_strength(actual) - target_strength(reference))
                relative = abs(actual - reference) / abs(reference)
                @test db < 0.1
                @test relative < 0.01
                @info "Low-frequency spheroid comparison" formulation=diagnostics(sol).formulation theta phi db relative
            end
        end
    end

    @testset "Nested ellipsoid refinement: $formulation" for formulation in (:muller, :cbie)
        materials = [FluidFilled(1.04, 1.04), gas]
        samples = Matrix{ComplexF64}[]
        refinements = formulation === :muller ?
                      ((0.55, 5), (0.5, 5), (0.55, 7)) : ((0.55, 4), (0.5, 4), (0.55, 5))
        for (resolution, qorder) in refinements
            options = (; resolution, qorder, tip_ratio = 0.4)
            surfaces = [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
                mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
                    rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
            values = zeros(ComplexF64, 6, 3)
            for (i, beta) in enumerate((pi / 3, pi / 2))
                coupled = bem(
                    surfaces, materials, k; formulation, incidence_angle = beta)
                comparison = components(coupled)
                @test all(sol -> diagnostics(sol).formulation == formulation, comparison.isolated)
                incident = [cos(beta), sin(beta), 0.0]
                for (j, sol) in enumerate((coupled, comparison.isolated...)),
                    (t, direction) in enumerate((-incident, incident, [0.0, 0.0, 1.0]))

                    values[3(i - 1) + t, j] = scattering_amplitude(sol; direction)
                end
                @test diagnostics(coupled).scaled_relative_residual < 1e-10
            end
            @test all(isfinite, values)
            push!(samples, values)
            @info "Low-frequency discretization" resolution qorder
        end
        for refined in samples[2:3]
            db = maximum(abs.(target_strength.(first(samples)) - target_strength.(refined)))
            relative = maximum(abs.(first(samples) - refined) ./ abs.(refined))
            @test db < 0.1
            @test relative < 0.01
            @info "Low-frequency component refinement" db relative
        end
    end
end
