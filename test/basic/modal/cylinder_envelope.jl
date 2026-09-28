using AcousticScattering
using Test

# Quantified finite-cylinder modal-series (FCMS/DCM) validity envelope against converged full 3D
# BEM. Jech et al. (2015) report the same reduction as a valid benchmark only within about 15 to
# 20 degrees of broadside incidence for rigid, pressure-release and fluid/gas-filled cylinders;
# this reproduces that pattern for this package's own implementation.
@testset "Finite-cylinder modal series validity envelope" begin
    radius, length = 0.01, 0.1
    k = 2pi * 38e3 / 1477.4
    body = Cylinder(radius, length)
    full_bem(boundary, angle) = bem(
        body, boundary, k; method = :full, incidence_angle = angle,
        meshsize = 0.006, mesh_order = 2, qorder = 2, compression = (method = :none,))

    @testset "Rigid: broadside agreement, end-on failure" begin
        broadside = deg2rad(90)
        m = modal(body, Rigid(), k; incidence_angle = broadside, m_max = 30)
        f = full_bem(Rigid(), broadside)
        @test abs(target_strength(m) - target_strength(f)) < 1.0

        near_endon = deg2rad(10)
        m2 = modal(body, Rigid(), k; incidence_angle = near_endon, m_max = 30)
        f2 = full_bem(Rigid(), near_endon)
        @test abs(target_strength(m2) - target_strength(f2)) > 20.0
    end

    @testset "Rigid: error grows monotonically away from broadside" begin
        errors = map((90, 60, 30, 10)) do deg
            angle = deg2rad(deg)
            m = modal(body, Rigid(), k; incidence_angle = angle, m_max = 30)
            f = full_bem(Rigid(), angle)
            abs(target_strength(m) - target_strength(f))
        end
        @test issorted(errors)
    end

    @testset "PressureRelease: broadside agreement" begin
        broadside = deg2rad(90)
        m = modal(body, PressureRelease(), k; incidence_angle = broadside, m_max = 30)
        f = full_bem(PressureRelease(), broadside)
        @test abs(target_strength(m) - target_strength(f)) < 1.0
    end

    @testset "Rigid: broadside robust to aspect ratio and ka" begin
        full_bem_body(other, angle) = bem(other, Rigid(), k; method = :full,
            incidence_angle = angle, meshsize = 0.006, mesh_order = 2, qorder = 2,
            compression = (method = :none,))
        for aspect in (3.0, 10.0, 20.0)
            other = Cylinder(radius, aspect * 2radius)
            m = modal(other, Rigid(), k; incidence_angle = pi / 2, m_max = 30)
            f = full_bem_body(other, pi / 2)
            @test abs(target_strength(m) - target_strength(f)) < 1.0
        end
        long_body = Cylinder(radius, 10 * 2radius)
        for ka in (0.3, 1.0, 5.0)
            kk = ka / radius
            m = modal(long_body, Rigid(), kk; incidence_angle = pi / 2, m_max = 30)
            f = bem(long_body, Rigid(), kk; method = :full, incidence_angle = pi / 2,
                meshsize = 0.006, mesh_order = 2, qorder = 2, compression = (method = :none,))
            @test abs(target_strength(m) - target_strength(f)) < 0.5
        end
    end
end
