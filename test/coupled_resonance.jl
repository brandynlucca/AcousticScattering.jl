using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(1)

@time "Nested ellipsoids across gas resonance" @testset "Nested ellipsoids across gas resonance" begin
    materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
    beta = pi/3
    direction = [-cos(beta), -sin(beta), 0.0]
    options = (; resolution = 0.6, qorder = 4, tip_ratio = 0.4)
    surfaces = [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
        mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
            rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
    coupled = bem(surfaces, materials, 2pi*322.8/1477.4; incidence_angle = beta)
    @test diagnostics(coupled).scaled_relative_residual < 1e-10
    @test isfinite(scattering_amplitude(coupled; direction))
end
