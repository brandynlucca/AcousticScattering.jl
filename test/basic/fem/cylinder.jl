using AcousticScattering
using Test

@testset "Cylinder" begin
    body = Cylinder(0.01, 0.02)
    k = 100.0

    @testset "Meridian FEM: rigid, pressure-release, and fluid-filled" begin
        for (name, boundary) in (("rigid", Rigid()),
            ("pressure-release", PressureRelease()),
            ("fluid-filled", FluidFilled(1.05, 1.02)))
            @testset "$name" begin
                solution = fem(body, boundary, k; R = 0.018,
                    incidence_angle = pi / 4, m_max = 1,
                    n_r = 4, n_theta = 12, l_max = 4)
                @test solution isa FEMSolution
                @test solution.method == :meridian
                @test target_strength(solution) isa Float64
                @test isfinite(target_strength(solution))
                @test -300.0 < target_strength(solution) < 50.0
                @test scattering_amplitude(solution) isa ComplexF64
                @test target_strength(solution) ≈
                      20log10(abs(scattering_amplitude(solution)))
                # Off-backscatter observation direction exercises the retained per-mode traces,
                # not just the solve-time default.
                @test scattering_amplitude(solution; angle = pi / 3, azimuth = 0.5) isa
                      ComplexF64
            end
        end
    end

    @testset "Radial FEM: solid elastic and elastic shell" begin
        solid = SolidElastic(7810.0 / 1026.8,
            5973.0 / 1477.4, 3193.0 / 1477.4)
        shell = Shelled(ElasticLayer(2700.0 / 1026.8,
                5700.0 / 1477.4, 3100.0 / 1477.4),
            FluidInterior(1.0, 1.0), 0.8)
        for (name, boundary) in (("solid elastic", solid),
            ("elastic shell", shell))
            @testset "$name" begin
                solution = fem(body, boundary, k; n_elements = 20, m_max = 2)
                reference = modal(body, boundary, k; m_max = 2)
                @test solution isa FEMSolution
                @test solution.method == :radial
                @test target_strength(solution) isa Float64
                @test isfinite(target_strength(solution))
                @test abs(target_strength(solution) - target_strength(reference)) < 3.0
                @test scattering_amplitude(solution) isa ComplexF64
                @test target_strength(solution) ≈
                      20log10(abs(scattering_amplitude(solution)))
            end
        end
    end

    bent = Cylinder(0.01, 0.02; radius_curvature = 0.1)
    @test_throws ArgumentError fem(bent, Rigid(), k)
    @test_throws ArgumentError fem(bent, SolidElastic(2.0, 2.0, 1.0), k)
end

@testset "Meridian FEM complex amplitude matches modal" begin
    # modal(Cylinder, ...) is the DCM/FCMS reduction, not exact, and needs a long, slender
    # cylinder to be accurate (see the cylinder envelope tests); PressureRelease's DCM
    # approximation degrades faster than Rigid's at short aspect ratios (confirmed against
    # full 3D BEM directly: at aspect 2, FEM matches full BEM to ~1% for both boundaries while
    # matching modal to only ~4%/54%, so this uses a long enough cylinder for modal itself to
    # be trustworthy as the reference).
    radius = 0.01
    body = Cylinder(radius, 8 * 2radius)
    k = 100.0
    for boundary in (Rigid(), PressureRelease())
        solution = fem(body, boundary, k; R = 1.2 * (body.length / 2),
            incidence_angle = pi / 2, m_max = 8, n_r = 24, n_theta = 80, l_max = 30)
        reference = modal(body, boundary, k; incidence_angle = pi / 2, m_max = 20)
        @test scattering_amplitude(solution) ≈
              scattering_amplitude(reference) rtol = 0.08
    end
end

@testset "Axial meridian cylinder FEM" begin
    ts = AcousticScattering.cylinder_meridian_fem_target_strength(
        Rigid(), 0.5, 0.05, 0.1, 0.12; n_r = 6, n_theta = 12)
    @test isfinite(ts)
    ts_soft = AcousticScattering.cylinder_meridian_fem_target_strength(
        PressureRelease(), 0.5, 0.05, 0.1, 0.12; n_r = 6, n_theta = 12, l_max = 5)
    @test isfinite(ts_soft)
    @test ts != ts_soft
end
