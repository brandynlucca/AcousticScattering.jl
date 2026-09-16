using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

@testset "Oblique gas-spheroid resonance" begin
    body, gas = Spheroid(0.025, 0.0075), GasFilled(0.00129, 0.23)
    beta, alpha = pi / 3, 0.4
    frequencies = [300.0, 317.1, 317.4, 319.95, 320.0, 320.05, 322.65, 322.95, 340.0, 500.0]
    observations = ((pi - beta, alpha + pi), (beta, alpha), (pi / 2, pi / 2))
    surface = mesh(; semiaxes = (0.025, 0.0075, 0.0075),
        resolution = 0.45, qorder = 5, tip_ratio = 0.4)
    actual, reference = zeros(ComplexF64, length(frequencies), 3),
    zeros(ComplexF64, length(frequencies), 3)
    for (i, frequency) in enumerate(frequencies)
        k = 2pi * frequency / 1477.4
        sol = bem([surface], [gas], k; incidence_angle = beta, incidence_azimuth = alpha)
        for (j, (theta, phi)) in enumerate(observations)
            direction = [cos(theta), sin(theta) * cos(phi), sin(theta) * sin(phi)]
            actual[i, j] = scattering_amplitude(sol; direction)
            reference[i, j] = scattering_amplitude(modal(body, gas, k;
                incidence_angle = beta, incidence_azimuth = alpha,
                scatter_angle = theta, scatter_azimuth = phi,
                n_max = 6, m_max = 6, precision = :quad))
        end
    end
    db = maximum(abs.(target_strength.(actual) - target_strength.(reference)))
    relative = maximum(abs.(actual - reference) ./ abs.(reference))
    @test db < 0.1
    @test relative < 0.01
    @info "Resonance amplitudes" db relative

    peaks, widths = Float64[], Float64[]
    for curve in (actual[:, 1], reference[:, 1])
        power = abs2.(curve)
        index = argmax(power)
        push!(peaks, frequencies[index])
        half = power[index] / 2
        @test power[2] < half < power[3]
        @test power[7] > half > power[8]
        lower = frequencies[2] +
                (half - power[2]) / (power[3] - power[2]) *
                (frequencies[3] - frequencies[2])
        upper = frequencies[7] +
                (half - power[7]) / (power[8] - power[7]) *
                (frequencies[8] - frequencies[7])
        push!(widths, upper - lower)
    end
    @test isapprox(peaks[1], peaks[2]; atol = 0.05, rtol = 1e-12)
    @test abs(widths[1] - widths[2]) / widths[2] < 0.01
    @info "Sampled resonance metrics" peaks widths

    k = 2pi * 320 / 1477.4
    truncated = scattering_amplitude(modal(body, gas, k;
        incidence_angle = beta, incidence_azimuth = alpha,
        n_max = 4, m_max = 4, precision = :quad))
    @test abs(truncated - reference[5, 1]) / abs(reference[5, 1]) < 1e-5
    for (resolution, qorder) in ((0.4, 5), (0.45, 7))
        refined = mesh(; semiaxes = (0.025, 0.0075, 0.0075), resolution, qorder, tip_ratio = 0.4)
        sol = bem(refined, gas, k; incidence_angle = beta, incidence_azimuth = alpha)
        value = scattering_amplitude(sol)
        @test abs(target_strength(value) - target_strength(reference[5, 1])) < 0.1
        @test abs(value - reference[5, 1]) / abs(reference[5, 1]) < 0.01
        @test abs(target_strength(value) - target_strength(actual[5, 1])) < 0.1
        @test abs(value - actual[5, 1]) / abs(value) < 0.01
        @info "Resonance refinement" resolution qorder db=abs(target_strength(value)-target_strength(reference[5, 1])) relative=abs(value-reference[
            5, 1])/abs(reference[5, 1])
    end
end
