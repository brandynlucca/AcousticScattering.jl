using AcousticScattering
using Test

const AS = AcousticScattering

let
    sphere = AS.Sphere(0.5)
    k = 2.0
    angles = (0.0, pi / 2, pi)
    relative_error(solution, boundary) = maximum(angles) do angle
        reference = AS.scattering_amplitude(AS.modal(sphere, boundary, k; angle))
        abs(AS.scattering_amplitude(solution; angle) - reference) / abs(reference)
    end
    volume(boundary; kwargs...) = AS.fem(sphere, boundary, k; method = :volume,
        incidence_angle = 0.0, points_per_wavelength = 6, kwargs...)

    @time "Volume FEM (spheres, PML closure)" @testset "Volume FEM (spheres, PML closure)" begin
        for boundary in (AS.PressureRelease(), AS.FluidFilled(1.05, 1.05))
            solution = volume(boundary; closure = :pml)
            @test solution isa AS.FEMSolution
            @test relative_error(solution, boundary) < 0.06
            @test AS.diagnostics(solution).residual < 1e-8
        end
    end

    @time "Volume FEM (spheres, DtN closure)" @testset "Volume FEM (spheres, DtN closure)" begin
        for boundary in (AS.Rigid(), AS.PressureRelease(), AS.FluidFilled(1.05, 1.05))
            solution = volume(boundary; closure = :dtn)
            @test relative_error(solution, boundary) < 0.02
        end
    end

    @time "Volume FEM (elastic sphere)" @testset "Volume FEM (elastic sphere)" begin
        boundary = AS.SolidElastic(2.7, 4.2, 2.1)
        solution = volume(boundary; closure = :dtn)
        @test relative_error(solution, boundary) < 0.05
        @test isfinite(AS.target_strength(solution))
    end

    @time "Volume FEM (rotated incidence)" @testset "Volume FEM (rotated incidence)" begin
        boundary = AS.FluidFilled(1.05, 1.05)
        axial = volume(boundary; closure = :dtn)
        oblique = AS.fem(sphere, boundary, k; method = :volume, closure = :dtn,
            incidence_angle = pi / 3, incidence_azimuth = 0.4, points_per_wavelength = 6)
        @test AS.scattering_amplitude(oblique; angle = pi - pi / 3, azimuth = 0.4 + pi) ≈
              AS.scattering_amplitude(axial; angle = pi) rtol = 0.02
    end

    @testset "Volume FEM (argument validation)" begin
        boundary = AS.Rigid()
        @test_throws ArgumentError AS.fem(
            sphere, boundary, k; method = :volume, closure = :bad)
        @test_throws ArgumentError AS.fem(
            sphere, boundary, k; method = :volume, domain_radius = 0.51)
        @test_throws ArgumentError AS.fem(sphere, boundary, -1.0; method = :volume)
        @test_throws ArgumentError AS.fem(
            AS.Spheroid(0.5, 0.2), AS.SolidElastic(2.7, 4.2, 2.1), k;
            method = :radial)
    end
end

let
    k = 2.0
    flesh = AS.FluidFilled(1.05, 1.05)
    gas = AS.FluidFilled(0.0012, 0.22)
    outer, inner = AS.Sphere(0.5), AS.Sphere(0.2)

    @time "Volume FEM (nested fluid spheres)" @testset "Volume FEM (nested fluid spheres)" begin
        boundary = AS.Shelled(AS.FluidLayer(1.05, 1.05), AS.FluidInterior(0.0012, 0.22), 0.4)
        solution = AS.fem([outer, inner], [flesh, gas], k; method = :volume,
            incidence_angle = 0.0, points_per_wavelength = 6, closure = :dtn)
        @test solution isa AS.FEMSolution
        for angle in (0.0, pi / 2, pi)
            reference = AS.scattering_amplitude(AS.modal(outer, boundary, k; angle))
            value = AS.scattering_amplitude(solution; angle)
            @test abs(value - reference) / abs(reference) < 0.02
        end
    end

    @testset "Volume FEM (region validation)" begin
        regions(; kwargs...) = AS.fem(
            [outer, inner], [flesh, gas], k; method = :volume, kwargs...)
        @test_throws ArgumentError regions(parents = [0])
        @test_throws ArgumentError regions(parents = [2, 1])
        @test_throws ArgumentError regions(parents = [0, 0])
        @test_throws ArgumentError regions(centers = [zeros(3), [0.4, 0.0, 0.0]])
        @test_throws ArgumentError AS.fem(
            [outer, outer], [flesh, gas], k; method = :volume,
            parents = [0, 0])
        @test_throws ArgumentError regions(closure = :pml_spheroidal)
    end
end
