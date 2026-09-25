using AcousticScattering
using Test

const AS = AcousticScattering

@testset "VESM general-m building blocks" begin
    # The general-m coefficient is not exposed and not validated against a reference, so this only checks that its radial terms and boundary system evaluate to finite numbers.
    core = FluidInterior(71 / 1027, 325 / 1500)
    wall = ElasticLayer(1040 / 1027, 1520 / 1500, sqrt(0.2e6 / 1040) / 1500)
    flesh = ViscousLayer(1500.0, 1040 / 1027, 1510 / 1500, (4 / 3) / 1040, 1 / 1040)
    boundary = Shelled(LayeredMaterial(flesh, wall, 1.02 / 1.1), core, 1 / 1.1)
    a, k = 1.1e-3, 2pi * 23100.0 / 1500.0
    omega = k * flesh.soundspeed_exterior
    kL2, β2 = AS._vesm_flesh_wavenumbers(boundary, omega)
    kT2 = sqrt(im * omega / flesh.kinematic_viscosity_shear)
    kL3 = k / wall.speed_longitudinal_contrast
    kT3 = k / wall.speed_transversal_contrast
    β3 = (wall.speed_transversal_contrast / wall.speed_longitudinal_contrast)^2
    k4 = k / core.soundspeed_contrast

    for m in (1, 2)
        compressional = AS._vesm_compressional_terms(m, kL3, β3, a)
        shear = AS._vesm_shear_terms(m, kT3, a)
        @test length(compressional) == length(shear) == 8
        @test all(isfinite, compressional) && all(isfinite, shear)
        coefficient = AS._vesm_general_modal_coefficient(boundary, m, k, a, a,
            boundary.material.radius_ratio * a, boundary.radius_ratio * a,
            kL2, β2, kT2, kL3, kT3, β3, k4)
        @test isfinite(coefficient)
    end
end
