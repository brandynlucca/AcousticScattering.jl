using AcousticScattering
using Test

const AS = AcousticScattering

@testset "High-frequency Kirchhoff building blocks" begin
    k = 2pi * 38e3 / 1477.4
    @test AS.reflection_coefficient(Rigid()) == 1
    @test AS.reflection_coefficient(PressureRelease()) == -1
    @test AS.kirchhoff_form_function(Rigid(), 0.02, 0.05) ≈ sqrt(0.02 * 0.05) / 2
    @test AS.kirchhoff_form_function(PressureRelease(), 0.02, 0.05) ≈
          -sqrt(0.02 * 0.05) / 2

    sphere = Sphere(0.01)
    @test AS.kirchhoff_target_strength(Rigid(), k, 0.01) ≈
          target_strength(kirchhoff(sphere, Rigid(), k))

    spheroid = Spheroid(0.03, 0.01)
    equator = AS.principal_curvatures(spheroid, pi / 2)
    pole = AS.principal_curvatures(spheroid, 0.0)
    @test all(>(0), equator) && all(>(0), pole)
    @test equator[1] ≈ 0.03^2 / 0.01 && equator[2] ≈ 0.01
    @test pole[1] ≈ 0.01^2 / 0.03 && pole[2] ≈ 0.01^2 / 0.03
    @test AS.kirchhoff_target_strength(Rigid(), k, spheroid) ≈
          target_strength(kirchhoff(spheroid, Rigid(), k))
    @test AS.kirchhoff_target_strength(PressureRelease(), k, spheroid; angle = 0.3) ≈
          target_strength(kirchhoff(spheroid, PressureRelease(), k; incidence_angle = 0.3))

    @test AS.kirchhoff_target_strength(Rigid(), k, 0.005, 0.03; angle = 0.4) ≈
          target_strength(kirchhoff(Cylinder(0.005, 0.03), Rigid(), k; incidence_angle = 0.4))
end

@testset "Layered sphere Kirchhoff reflection" begin
    shell = Shelled(FluidLayer(1.1, 1.02), VacuumInterior(), 0.9)
    @test isfinite(AcousticScattering.kirchhoff_target_strength(shell, 200.0, 0.01))
end
