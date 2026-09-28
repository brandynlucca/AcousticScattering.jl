using AcousticScattering
using Test

const AS = AcousticScattering

let
    k = 2.0
    a = 0.25
    depth = 1.4
    material = AS.FluidFilled(1.05, 1.05)
    sphere = AS.Sphere(a)

    @time "Free surface (pressure-release interface condition)" @testset "Free surface (pressure-release interface condition)" begin
        solution = AS.free_surface(
            sphere, material, k, depth; condition = :pressure_release,
            incidence_angle = 0.3, incidence_azimuth = 0.2, points_per_wavelength = 6)
        plane_points = [[0.5, 0.1, 0.0], [-0.3, 0.4, 0.0], [0.2, -0.2, 0.0]]
        for p in AS.pressure(solution, plane_points; field = :total)
            @test abs(p) < 1e-2
        end
    end

    @time "Free surface (rigid interface symmetry)" @testset "Free surface (rigid interface symmetry)" begin
        solution = AS.free_surface(sphere, material, k, depth; condition = :rigid,
            incidence_angle = 0.3, incidence_azimuth = 0.2, points_per_wavelength = 6)
        above = [[0.5, 0.1, 0.6], [-0.3, 0.4, 0.9]]
        below = [[p[1], p[2], -p[3]] for p in above]
        values_above = AS.pressure(solution, above; field = :total)
        values_below = AS.pressure(solution, below; field = :total)
        for (pa, pb) in zip(values_above, values_below)
            @test pa≈pb rtol=1e-2
        end
    end

    @time "Free surface (matches two independent region solves)" @testset "Free surface (matches two independent region solves)" begin
        image = [0.0, 0.0, depth]
        centers = [[0.0, 0.0, -depth], image]
        orientations = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        combined(angle) = AS.fem(
            [sphere, sphere], [material, material], k; method = :volume,
            parents = [0, 0], centers, orientations, incidence_angle = angle,
            incidence_azimuth = 0.2, points_per_wavelength = 6)
        direct = combined(0.3)
        reflected = combined(π - 0.3)
        solution = AS.free_surface(
            sphere, material, k, depth; condition = :pressure_release,
            incidence_angle = 0.3, incidence_azimuth = 0.2, points_per_wavelength = 6)
        for (angle, azimuth) in ((0.0, 0.0), (π, 0.0), (π / 2, 0.5))
            reference = AS.scattering_amplitude(direct; angle, azimuth) -
                        AS.scattering_amplitude(reflected; angle, azimuth)
            @test AS.scattering_amplitude(solution; angle, azimuth)≈reference atol=2e-5 rtol=5e-3
        end
        @test isfinite(AS.target_strength(solution))
    end

    @testset "Free surface (argument validation)" begin
        @test_throws ArgumentError AS.free_surface(
            sphere, material, k, depth; condition = :bad)
        @test_throws ArgumentError AS.free_surface(sphere, material, k, -1.0)
        @test_throws ArgumentError AS.free_surface(sphere, material, k, 0.05)
    end
end
