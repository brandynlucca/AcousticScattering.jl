using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

function fish_surfaces(; resolution = 0.4, qorder = 5)
    options = (; resolution, qorder, tip_ratio = 0.4)
    return [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
        mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
            rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
end

@testset "Synthetic fish and displaced bladder" begin
    materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
    surfaces = fish_surfaces()
    beta = deg2rad(90.0)
    incident = [cos(beta), sin(beta), 0.0]
    directions = (-incident, incident, [0.0, 0.0, 1.0])
    k = 2pi * 2250.0 / 1477.4

    coupled = bem(surfaces, materials, k; incidence_angle = beta)
    reference = bem(surfaces, materials, k; incidence_angle = beta, formulation = :cbie)
    actual = [scattering_amplitude(coupled; direction) for direction in directions]
    expected = [scattering_amplitude(reference; direction) for direction in directions]

    @test all(isfinite, actual)
    @test diagnostics(coupled).scaled_relative_residual < 1e-10
    db = maximum(abs.(target_strength.(actual) - target_strength.(expected)))
    relative = maximum(abs.(actual - expected) ./ abs.(expected))
    @test db < 0.1
    @test relative < 0.01
end
