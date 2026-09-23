using AcousticScattering
using Test

@testset "Fluid-shell reflection across phase and impedance" begin
    body = Sphere(0.01)
    shell = FluidLayer(1.1, 1.05)
    ratio = 0.8
    interior = FluidInterior(1.2, 1.1)

    for (medium, inner_reflection) in (
            (VacuumInterior(), -1.0),
            (interior,
            (interior.density_contrast * interior.soundspeed_contrast -
             shell.density_contrast * shell.soundspeed_contrast) /
            (interior.density_contrast * interior.soundspeed_contrast +
             shell.density_contrast * shell.soundspeed_contrast))
        ),
        k in (0.5, 25.0, 100.0)

        boundary = Shelled(shell, medium, ratio)
        outer_reflection = (shell.density_contrast * shell.soundspeed_contrast - 1) /
                           (shell.density_contrast * shell.soundspeed_contrast + 1)
        phase = cis(2k * body.radius * (1 - ratio) / shell.soundspeed_contrast)
        expected = (outer_reflection + inner_reflection * phase) /
                   (1 + outer_reflection * inner_reflection * phase)
        rigid = scattering_amplitude(kirchhoff(body, Rigid(), k))
        actual = scattering_amplitude(kirchhoff(body, boundary, k))
        @test actual ≈ expected * rigid rtol = 1e-12
        @test target_strength(kirchhoff(body, boundary, k)) ≈
              target_strength(actual) rtol = 1e-12
    end
end
