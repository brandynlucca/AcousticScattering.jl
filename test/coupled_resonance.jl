using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

@testset "Nested ellipsoids across gas resonance" begin
    materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
    frequencies = [320.0, 322.8, 325.0]
    beta = pi/3
    incident = [cos(beta), sin(beta), 0.0]
    samples = Array{ComplexF64, 3}[]
    for (resolution, qorder) in ((0.5, 5), (0.45, 5), (0.5, 7))
        options = (; resolution, qorder, tip_ratio = 0.4)
        surfaces = [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
            mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
                rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
        values = zeros(ComplexF64, 3, 3, 3)
        for (i, frequency) in enumerate(frequencies)
            coupled = bem(surfaces, materials, 2pi*frequency/1477.4; incidence_angle = beta)
            comparison = components(coupled)
            for (j, sol) in enumerate((coupled, comparison.isolated...)),
                (t, direction) in enumerate((-incident, incident, [0.0, 0.0, 1.0]))

                values[i, t, j] = scattering_amplitude(sol; direction)
            end
        end
        @test argmax(abs2.(values[:, 1, 1])) == 2
        push!(samples, values)
        @info "Coupled resonance discretization" resolution qorder
        GC.gc()
    end
    for refined in samples[2:3]
        db = maximum(abs.(target_strength.(first(samples))-target_strength.(refined)))
        errors = abs.(first(samples)-refined) ./ abs.(refined)
        relative = maximum(errors)
        worst = argmax(errors)
        @info "Coupled resonance refinement" db relative frequency=frequencies[worst[1]] model=(
            :coupled, :flesh, :bladder)[worst[3]]
        @test db < 0.1
        @test relative < 0.01
    end
end
