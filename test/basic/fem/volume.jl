using AcousticScattering
using Test
using LinearAlgebra: dot

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

let
    k = 2.0
    solid = AS.SolidElastic(2.7, 2.0, 1.0)
    flesh = AS.FluidFilled(1.05, 1.05)

    @time "Volume FEM (elastic shell as regions)" @testset "Volume FEM (elastic shell as regions)" begin
        sphere = AS.Sphere(1.0)
        layer = AS.ElasticLayer(2.7, 2.0, 1.0)
        boundary = AS.Shelled(layer, AS.FluidInterior(1.0, 1.0), 0.8)
        solution = AS.fem([sphere, AS.Sphere(0.8)], [solid, AS.FluidFilled(1.0, 1.0)], 1.5;
            method = :volume, incidence_angle = 0.0, closure = :dtn)
        for angle in (0.0, pi / 2, pi)
            reference = AS.scattering_amplitude(AS.modal(sphere, boundary, 1.5; angle))
            value = AS.scattering_amplitude(solution; angle)
            @test abs(value - reference) / abs(reference) < 0.05
        end
    end

    @time "Volume FEM (rotation of an inclusion in flesh)" @testset "Volume FEM (rotation of an inclusion in flesh)" begin
        bodies = [AS.Sphere(0.5), AS.Spheroid(0.2, 0.08)]
        center, axis = [0.15, 0.0, 0.1], [sin(0.5), 0.0, cos(0.5)]
        direction = [sin(pi / 3) * cos(0.4), sin(pi / 3) * sin(0.4), cos(pi / 3)]
        th, ph = 0.7, 0.3
        Q = [cos(th) 0 sin(th); 0 1 0; -sin(th) 0 cos(th)] *
            [cos(ph) -sin(ph) 0; sin(ph) cos(ph) 0; 0 0 1]
        amplitude(c, a, d) = AS.scattering_amplitude(AS.fem(bodies, [flesh, solid], k;
            method = :volume, closure = :dtn, points_per_wavelength = 6,
            centers = [zeros(3), c], orientations = [[0.0, 0.0, 1.0], a],
            incidence_angle = acos(d[3]), incidence_azimuth = atan(d[2], d[1])))
        reference = amplitude(center, axis, direction)
        @test amplitude(Q * center, Q * axis, Q * direction) ≈ reference rtol = 0.01
    end

    @testset "Volume FEM (region material validation)" begin
        @test_throws ArgumentError AS.fem([AS.Sphere(0.5)], [AS.Rigid()], k; method = :volume)
        @test_throws ArgumentError AS.fem([AS.Sphere(0.5), AS.Cylinder(0.1, 0.3)],
            [flesh, flesh], k; method = :volume)
    end
end

let
    sphere = AS.Sphere(0.5)
    solid = AS.SolidElastic(2.7, 2.0, 1.0)

    @time "Volume FEM (iterative solver)" @testset "Volume FEM (iterative solver)" begin
        for boundary in (AS.Rigid(), solid)
            direct = AS.fem(sphere, boundary, 2.0; method = :volume, incidence_angle = 0.0,
                points_per_wavelength = 5, solver = :direct)
            iterative = AS.fem(
                sphere, boundary, 2.0; method = :volume, incidence_angle = 0.0,
                points_per_wavelength = 5, solver = :iterative)
            @test AS.diagnostics(direct).solver === :direct
            @test AS.diagnostics(iterative).solver === :iterative
            @test AS.diagnostics(iterative).iterations > 0
            for angle in (0.0, pi / 2, pi)
                @test AS.scattering_amplitude(iterative; angle) ≈
                      AS.scattering_amplitude(direct; angle) rtol = 1e-5
            end
        end
        @test_throws ArgumentError AS.fem(sphere, AS.Rigid(), 2.0; method = :volume,
            solver = :multigrid)
    end
end

let
    k = 2.0
    angles = [pi / 6, pi / 2]

    @time "Volume FEM (incidence sweep)" @testset "Volume FEM (incidence sweep)" begin
        body, boundary = AS.Spheroid(0.4, 0.2), AS.FluidFilled(1.05, 1.05)
        sweep = AS.incidence_angle_sweep(body, boundary, k, angles; method = :volume,
            points_per_wavelength = 5, solver = :direct)
        @test sweep isa AS.IncidenceAngleSweep
        for (i, angle) in enumerate(angles)
            single = AS.fem(body, boundary, k; method = :volume, incidence_angle = angle,
                points_per_wavelength = 5, solver = :direct)
            @test sweep.amplitudes[i] ≈ AS.scattering_amplitude(single) rtol = 1e-6
        end
        @test_throws ArgumentError AS.incidence_angle_sweep(body, boundary, k, angles;
            method = :bem)
        @test_throws ArgumentError AS.incidence_angle_sweep(body, boundary, k, Float64[])
    end

    @time "Volume FEM (incidence sweep of regions)" @testset "Volume FEM (incidence sweep of regions)" begin
        bodies = [AS.Sphere(0.5), AS.Sphere(0.2)]
        materials = [AS.FluidFilled(1.05, 1.05), AS.FluidFilled(0.0012, 0.22)]
        sweep = AS.incidence_angle_sweep(bodies, materials, k, angles; closure = :dtn,
            points_per_wavelength = 5, centers = [zeros(3), [0.1, 0.0, 0.0]])
        single = AS.fem(
            bodies, materials, k; method = :volume, incidence_angle = angles[1],
            closure = :dtn, points_per_wavelength = 5, centers = [
                zeros(3), [0.1, 0.0, 0.0]])
        @test sweep.amplitudes[1] ≈ AS.scattering_amplitude(single) rtol = 1e-6
    end
end

let
    sphere = AS.Sphere(0.5)

    @time "Volume FEM (automatic closure and matrix-free DtN)" @testset "Volume FEM (automatic closure and matrix-free DtN)" begin
        automatic = AS.fem(
            sphere, AS.Rigid(), 2.0; method = :volume, incidence_angle = 0.0,
            points_per_wavelength = 5)
        @test AS.diagnostics(automatic).closure === :dtn
        matrix_free = AS.fem(
            sphere, AS.Rigid(), 2.0; method = :volume, incidence_angle = 0.0,
            points_per_wavelength = 5, closure = :dtn, solver = :iterative)
        dense = AS.fem(sphere, AS.Rigid(), 2.0; method = :volume, incidence_angle = 0.0,
            points_per_wavelength = 5, closure = :dtn, solver = :direct)
        @test AS.diagnostics(matrix_free).solver === :iterative
        for angle in (0.0, pi / 2, pi)
            @test AS.scattering_amplitude(matrix_free; angle) ≈
                  AS.scattering_amplitude(dense; angle) rtol = 1e-5
        end
    end
end

let
    sphere = AS.Sphere(0.5)
    k = 2.0

    @time "Volume FEM (pressure at Cartesian points)" @testset "Volume FEM (pressure at Cartesian points)" begin
        fluid_boundary = AS.FluidFilled(1.05, 1.05)
        solution = AS.fem(sphere, fluid_boundary, k; method = :volume, closure = :dtn,
            incidence_angle = pi / 2, incidence_azimuth = 0.0, points_per_wavelength = 8)
        modal_solution = AS.modal(sphere, fluid_boundary, k; angle = 0.0)
        exterior_points = ((0.55, 0.0, 0.0), (0.0, 0.0, 0.55))
        interior_points = ((0.3, 0.0, 0.0), (0.0, 0.0, 0.0), (0.2, 0.2, 0.1))
        for (points, fields) in ((exterior_points, (:total, :scattered)),
            (interior_points, (:total, :interior)))
            for point in points, field in fields

                @test AS.pressure(solution, point; field) ≈
                      AS.pressure(modal_solution, point; field) rtol=1e-3
            end
        end
        @test AS.pressure(solution, (0.55, 0.0, 0.0); field = :incident) ≈ cis(k * 0.55) rtol=1e-12
        @test AS.pressure(solution, [(0.55, 0.0, 0.0), (0.3, 0.0, 0.0)]) isa
              Vector{ComplexF64}
        @test_throws ArgumentError AS.pressure(solution, (2.0, 0.0, 0.0))
        @test_throws ArgumentError AS.pressure(solution, (0.55, 0.0, 0.0); field = :shell)

        rigid = AS.fem(sphere, AS.Rigid(), k; method = :volume, closure = :dtn,
            incidence_angle = pi / 2, incidence_azimuth = 0.0, points_per_wavelength = 8)
        @test AS.pressure(rigid, (0.55, 0.0, 0.0)) ≈
              AS.pressure(AS.modal(sphere, AS.Rigid(), k; angle = 0.0), (0.55, 0.0, 0.0)) rtol=1e-2
        @test_throws ArgumentError AS.pressure(rigid, (0.2, 0.0, 0.0))

        solid = AS.fem(sphere, AS.SolidElastic(2.7, 4.2, 2.1), k; method = :volume,
            closure = :dtn, incidence_angle = pi / 2, incidence_azimuth = 0.0)
        @test_throws ArgumentError AS.pressure(solid, (0.2, 0.0, 0.0))
    end
end

let
    k = 2.0
    a = 0.3
    sep = 1.2
    material = AS.FluidFilled(1.05, 1.05)
    invisible = AS.FluidFilled(1.0, 1.0)

    @time "Volume FEM (disjoint multiple bodies)" @testset "Volume FEM (disjoint multiple bodies)" begin
        single = AS.fem(AS.Sphere(a), material, k; method = :volume, closure = :dtn,
            incidence_angle = 0.0, points_per_wavelength = 8)
        # A physically invisible sibling (zero contrast) must reduce exactly to the isolated body.
        disjoint = AS.fem([AS.Sphere(a), AS.Sphere(a)], [material, invisible], k;
            method = :volume, parents = [0, 0], centers = [zeros(3), [0.0, 0.0, sep]],
            incidence_angle = 0.0, closure = :dtn, points_per_wavelength = 8)
        for angle in (0.0, pi / 2, pi)
            @test AS.scattering_amplitude(disjoint; angle) ≈
                  AS.scattering_amplitude(single; angle) rtol=5e-3
        end

        # Two genuinely interacting disjoint bodies give a non-degenerate, direction-dependent result.
        interacting = AS.fem([AS.Sphere(a), AS.Sphere(a)], [material, material], k;
            method = :volume, parents = [0, 0],
            centers = [[0.0, 0.0, -sep / 2], [0.0, 0.0, sep / 2]], incidence_angle = 0.0,
            closure = :dtn, points_per_wavelength = 8)
        forward = AS.scattering_amplitude(interacting; angle = 0.0)
        backward = AS.scattering_amplitude(interacting; angle = pi)
        @test isfinite(forward) && isfinite(backward)
        @test abs(forward - backward) / abs(backward) > 0.1
    end
end

let
    k = 2.0
    a = 0.2
    sep = 6.0
    material = AS.FluidFilled(1.05, 1.05)
    centers = [[0.0, 0.0, -sep / 2], [0.0, 0.0, sep / 2]]
    incident = [0.0, 0.0, 1.0]

    @time "Volume FEM (multi-body translation theorem)" @testset "Volume FEM (multi-body translation theorem)" begin
        # At large separation, two-body scattering reduces to the standard far-field translation
        # theorem applied to each isolated body: f(direction) = sum_i f0(direction) *
        # exp(ik*dot(incident - direction, center_i)), an independent check of genuine multi-body
        # interaction physics, not just the trivial reduction with an invisible sibling.
        isolated = AS.fem(AS.Sphere(a), material, k; method = :volume, closure = :dtn,
            incidence_angle = 0.0, incidence_azimuth = 0.0, points_per_wavelength = 8)
        direction_of(angle, azimuth) = [
            sin(angle) * cos(azimuth), sin(angle) * sin(azimuth),
            cos(angle)]
        function superpose(angle, azimuth)
            direction = direction_of(angle, azimuth)
            f0 = AS.scattering_amplitude(isolated; angle, azimuth)
            return sum(c -> f0 * cis(k * dot(incident .- direction, c)), centers)
        end
        interacting = AS.fem([AS.Sphere(a), AS.Sphere(a)], [material, material], k;
            method = :volume, parents = [0, 0], centers, incidence_angle = 0.0,
            incidence_azimuth = 0.0, closure = :dtn, points_per_wavelength = 8)
        for (angle, azimuth) in ((0.0, 0.0), (pi, 0.0), (pi / 2, 0.0))
            @test AS.scattering_amplitude(interacting; angle, azimuth) ≈
                  superpose(angle, azimuth) rtol=1e-2
        end
    end
end
