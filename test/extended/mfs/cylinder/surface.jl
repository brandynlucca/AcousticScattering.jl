using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Bent-cylinder MFS (method of fundamental solutions)" @testset "Bent-cylinder MFS (method of fundamental solutions)" begin
        radius, length = 0.01, 0.07
        c = 1477.3
        k = 2π * 38000.0 / c
        ρc = 1e6 * length
        cyl_straight = AS.Cylinder(radius, length)
        cyl_bent = AS.Cylinder(radius, length; radius_curvature = ρc)

        @time "broadside, both boundaries: converges tightly" @testset "broadside, both boundaries: converges tightly" begin
            for boundary in (AS.Rigid(), AS.PressureRelease())
                ts_exact = AS.target_strength(AS.modal(cyl_straight, boundary, k; m_max = 30))
                ts_mfs = AS.target_strength(AS.mfs(
                    cyl_bent, boundary, k; offset = 0.3radius, n_s = 40, n_φ = 32))
                @test ts_mfs ≈ ts_exact atol = 0.3
            end
        end

        @time "oblique incidence: PressureRelease converges tightly, Rigid more slowly" @testset "oblique incidence: PressureRelease converges tightly, Rigid more slowly" begin
            angle = 1.2
            ts_pr = AS.target_strength(AS.modal(
                cyl_straight, AS.PressureRelease(), k; incidence_angle = angle, m_max = 30))
            ts_pr_mfs = AS.target_strength(AS.mfs(cyl_bent, AS.PressureRelease(), k;
                incidence_angle = angle, offset = 0.3radius, n_s = 40, n_φ = 32))
            @test ts_pr_mfs ≈ ts_pr atol = 0.3

            ts_rigid = AS.target_strength(AS.modal(
                cyl_straight, AS.Rigid(), k; incidence_angle = angle, m_max = 30))
            ts_rigid_mfs = AS.target_strength(AS.mfs(cyl_bent, AS.Rigid(), k;
                incidence_angle = angle, offset = 0.5radius, n_s = 100, n_φ = 80))
            @test ts_rigid_mfs ≈ ts_rigid atol = 0.7
        end
    end
end

let
    @time "Bent MFS independent checks" @testset "Bent MFS independent checks" begin
        body = Cylinder(0.01, 0.07; radius_curvature = 0.2)
        for boundary in (Rigid(), PressureRelease())
            square = mfs(body, boundary, 100.0; n_s = 4, n_phi = 4, offset = 0.003)
            sampled = mfs(body, boundary, 100.0;
                n_s = 4, n_phi = 4, offset = 0.003, oversampling = 2)
            d = only(diagnostics(sampled).systems)
            @test d.source_count == 16
            @test d.collocation_count == 64
            @test d.check_count == 256
            @test d.method == :least_squares
            @test isfinite(d.boundary_residual.relative_residual)
            checks, _, _ = AS.bent_cylinder_mfs_points(0.01, 0.07, 0.2, 16, 16)
            @test isempty(intersect(Set(checks), Set(sampled.data.points)))
            low = AS.solve_bent_cylinder_mfs(boundary, 100.0, 0.01, 0.07, 0.2;
                n_s = 4, n_φ = 4, offset = 0.003)
            @test length(low) == 5
            @test low[1] == square.data.p_scat
            @test low[2] == square.data.dpdn_scat
        end
    end
end

let
    # Included by the Interfaces group; calls below deliberately use ordinary user imports.

    @time "Bent MFS ASCII grid controls" @testset "Bent MFS ASCII grid controls" begin
        body = Cylinder(0.01, 0.07; radius_curvature = 0.20)
        ascii = mfs(body, PressureRelease(), 100.0; n_s = 6, n_phi = 8, offset = 0.003)
        legacy = mfs(body, PressureRelease(), 100.0; n_s = 6, n_φ = 8, offset = 0.003)
        @test scattering_amplitude(ascii) ≈ scattering_amplitude(legacy)
        @test length(ascii.data.points) == 6 * 8
        @test_throws ArgumentError mfs(body, Rigid(), 100.0; n_phi = 8, n_φ = 8)
        @test_throws ArgumentError mfs(body, Rigid(), 100.0; n_phi = 0)
        @test_throws ArgumentError mfs(body, Rigid(), 100.0; n_s = 0)
    end
end
