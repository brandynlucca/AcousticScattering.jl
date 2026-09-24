using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
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
end

let
    function fish_surfaces(; resolution = 0.6, qorder = 4)
        options = (; resolution, qorder, tip_ratio = 0.4)
        return [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
            mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
                rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
    end

    @time "Synthetic fish and displaced bladder" @testset "Synthetic fish and displaced bladder" begin
        materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
        surfaces = fish_surfaces()
        beta = deg2rad(90.0)
        incident = [cos(beta), sin(beta), 0.0]
        directions = (-incident, incident, [0.0, 0.0, 1.0])
        k = 2pi * 2250.0 / 1477.4

        coupled = bem(surfaces, materials, k; incidence_angle = beta)
        actual = [scattering_amplitude(coupled; direction) for direction in directions]

        @test all(isfinite, actual)
        @test diagnostics(coupled).scaled_relative_residual < 1e-10
    end
end

let
    @time "Low-frequency nested fluid regions" @testset "Low-frequency nested fluid regions" begin
        k = 2pi * 500 / 1477.4
        gas = GasFilled(0.00129, 0.23)
        @time @testset "Nested ellipsoid: $formulation" for formulation in (:muller,)
            materials = [FluidFilled(1.04, 1.04), gas]
            options = (; resolution = 0.4, qorder = 4, tip_ratio = 0.4)
            surfaces = [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
                mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
                    rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
            beta = pi / 3
            incident = [cos(beta), sin(beta), 0.0]
            directions = (-incident, incident, [0.0, 0.0, 1.0])
            coupled = bem(surfaces, materials, k; formulation, incidence_angle = beta)
            @test diagnostics(coupled).formulation == formulation
            @test diagnostics(coupled).scaled_relative_residual < 1e-10
            values = [scattering_amplitude(coupled; direction) for direction in directions]
            @test all(isfinite, values)
        end
    end
end

let
    function ellipsoid_surface(axes, center; resolution, qorder = 4)
        return mesh(; qorder) do g
            g.model.add("ellipsoid")
            volume = g.model.occ.addSphere(center..., 1.0)
            g.model.occ.dilate([(3, volume)], center..., axes...)
            g.model.occ.synchronize()
            g.option.setNumber("Mesh.MeshSizeMin", resolution)
            g.option.setNumber("Mesh.MeshSizeMax", resolution)
            g.model.mesh.generate(2)
            g.model.mesh.setOrder(3)
        end
    end

    @time "Coupled fluid regions" @testset "Coupled fluid regions" begin
        beta, alpha = pi / 3, 0.4
        incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
        observations = ((pi, -incident), (0.0, incident),
            (pi / 2, [0.0, -sin(alpha), cos(alpha)]))
        outer = mesh(
            Sphere(1.0); method = :full, resolution = 0.5, mesh_order = 3, qorder = 5)
        inner = mesh(
            Sphere(0.5); method = :full, resolution = 0.25, mesh_order = 3, qorder = 5)

        @time "Single interface and material limits" @testset "Single interface and material limits" begin
            material = FluidFilled(1.2, 1.1)
            single = bem(
                outer, material, 1.0; incidence_angle = beta, incidence_azimuth = alpha)
            coupled = bem(
                [outer], [material], 1.0; incidence_angle = beta, incidence_azimuth = alpha)
            @test scattering_amplitude(coupled) ≈ scattering_amplitude(single) rtol = 1e-10
            map = AcousticScattering.bistatic_map(coupled, [0.0, pi], [0.0])
            @test map.target_strength[1, 1] ≈
                  target_strength(coupled; direction = [1.0, 0.0, 0.0])
            @test_throws ArgumentError scattering_amplitude(coupled; direction = [
                0.0, 0.0, 2.0])
            same = bem([outer, inner], [material, material], 1.0;
                incidence_angle = beta, incidence_azimuth = alpha)
            for (_, direction) in observations
                @test scattering_amplitude(same; direction) ≈
                      scattering_amplitude(single; direction) rtol = 0.01
                @test abs(target_strength(same; direction) -
                          target_strength(single; direction)) < 0.1
            end
            report = diagnostics(same)
            @test report.interface_count == 2
            @test report.unknown_count == 2report.quadrature_nodes
            @test report.relative_residual < 1e-9
            @test report.scaled_relative_residual < 1e-11
            @test report.conditioning == :not_computed
            @test maximum(maximum(values(r)) for r in report.interface_residuals) < 0.01
            for (i, interface) in enumerate(same.data.interfaces)
                @test interface.interior == i
                @test interface.exterior == i - 1
                @test all(isfinite, interface.pressure)
                @test interface.normal_derivative_interior / material.density_contrast ≈
                      interface.normal_derivative_exterior /
                      (i == 1 ? 1.0 : material.density_contrast)
            end
        end

        @time "Spherical layers and strong contrasts" @testset "Spherical layers and strong contrasts" begin
            g, h, k = 0.0012, 0.23, 1.0
            solution = bem([outer, inner], [FluidFilled(1.2, 1.1), FluidFilled(g, h)], k;
                incidence_angle = beta, incidence_azimuth = alpha)
            boundary = Shelled(FluidLayer(1.2, 1.1), FluidInterior(g, h), 0.5)
            for (angle, direction) in observations
                actual = scattering_amplitude(solution; direction)
                reference = scattering_amplitude(modal(Sphere(1.0), boundary, k; angle))
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
        end

        @time "Three-interface reference amplitudes" @testset "Three-interface reference amplitudes" begin
            reference_outer = mesh(
                Sphere(1.0); method = :full, resolution = 0.6, mesh_order = 3, qorder = 5)
            middle = mesh(
                Sphere(0.65); method = :full, resolution = 0.39, mesh_order = 3, qorder = 5)
            core = mesh(
                Sphere(0.3); method = :full, resolution = 0.18, mesh_order = 3, qorder = 5)
            materials = [
                FluidFilled(1.1, 1.05), FluidFilled(0.8, 0.9), FluidFilled(0.02, 0.4)]
            k, amplitudes = 0.7,
            (-0.3953599802493457 + 0.1290315899313256im,
                -0.4228617984591631 + 0.1291200358732379im,
                -0.4097734508458774 + 0.1290757719448574im)
            solution = bem([reference_outer, middle, core], materials, k; incidence_angle = 0.0)
            for (direction, reference) in zip(
                ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), amplitudes)
                actual = scattering_amplitude(solution; direction)
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
        end

        @time "Displaced inner body and invisible interfaces" @testset "Displaced inner body and invisible interfaces" begin
            center = [0.25, -0.1, 0.2]
            shifted = ellipsoid_surface((0.3, 0.3, 0.3), center; resolution = 0.15, qorder = 5)
            material = FluidFilled(0.0012, 0.23)
            solution = bem([outer, shifted], [FluidFilled(1, 1), material], 1.0;
                incidence_angle = beta, incidence_azimuth = alpha)
            for (angle, direction) in observations
                reference = scattering_amplitude(modal(Sphere(0.3), material, 1.0; angle)) *
                            cis(sum((incident - direction) .* center))
                actual = scattering_amplitude(solution; direction)
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
            invisible = bem([outer, shifted], fill(FluidFilled(1, 1), 2), 1.0)
            @test abs(scattering_amplitude(invisible)) < 1e-5
        end

        @time "Close interfaces" @testset "Close interfaces" begin
            close = mesh(
                Sphere(0.9); method = :full, resolution = 0.45, mesh_order = 3, qorder = 5)
            material = FluidFilled(0.0012, 0.23)
            solution = bem([outer, close], [FluidFilled(1.2, 1.1), material], 1.0;
                incidence_angle = beta, incidence_azimuth = alpha)
            boundary = Shelled(FluidLayer(1.2, 1.1), FluidInterior(0.0012, 0.23), 0.9)
            for (angle, direction) in observations
                reference = scattering_amplitude(modal(Sphere(1.0), boundary, 1.0; angle))
                actual = scattering_amplitude(solution; direction)
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
        end

        @time "Equilibration and conditioning" @testset "Equilibration and conditioning" begin
            coarse = mesh(Sphere(1.0); method = :full, resolution = 1.0, qorder = 2)
            material = GasFilled(0.0012, 0.23)
            balanced = bem([coarse], [material], 1.0)
            unscaled = bem(
                [coarse], [material], 1.0; equilibrate = false, condition_limit = 0)
            report = diagnostics(balanced)
            @test report.conditioning == :svd
            @test isfinite(report.condition_number)
            @test report.scaled_condition_number < report.condition_number
            @test diagnostics(unscaled).condition_number === nothing
            @test scattering_amplitude(balanced) ≈ scattering_amplitude(unscaled) rtol = 1e-9
        end

        @time "Invalid region inputs" @testset "Invalid region inputs" begin
            materials = [FluidFilled(1.2, 1.1), FluidFilled(0.7, 0.8)]
            @test_throws ArgumentError bem(Mesh[], FluidFilled[], 1.0)
            @test_throws ArgumentError bem([outer, inner], materials[1:1], 1.0)
            for parents in ([1, 0], [0, 2], [0, -1], [0])
                @test_throws ArgumentError bem([outer, inner], materials, 1.0; parents)
            end
            @test_throws ArgumentError bem([outer, inner], materials, 1.0; parents = [0, 0])
            @test_throws ArgumentError bem([inner, outer], materials, 1.0)
            @test_throws ArgumentError bem([outer, outer], materials, 1.0; validation = (maxdepth = 0,))
            @test_throws ArgumentError bem([outer], [FluidFilled(Inf, 1)], 1.0)
            @test_throws ArgumentError bem([outer], materials[1:1], 0.0)
            @test_throws ArgumentError bem([outer], materials[1:1], 1.0; incidence_angle = NaN)
            @test_throws ArgumentError bem([outer], materials[1:1], 1.0; condition_limit = -1)
            @test_throws ArgumentError bem([outer], materials[1:1], 1.0; formulation = :unknown)
            @test_throws ArgumentError bem([outer], materials[1:1], 1.0; validation = (maxwork = 0,))
            @test_throws ArgumentError bem([mesh(Sphere(1.0); resolution = 16)], materials[1:1], 1.0)
        end
    end
end

let
    function ellipsoid_surface(axes, center; resolution, qorder)
        return mesh(; qorder) do g
            g.model.add("ellipsoid")
            g.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
            g.model.occ.synchronize()
            g.option.setNumber("Mesh.MeshSizeMin", resolution / minimum(axes))
            g.option.setNumber("Mesh.MeshSizeMax", resolution / minimum(axes))
            g.model.mesh.generate(2)
            g.model.mesh.setOrder(3)
            g.model.mesh.affineTransform([0, 0, axes[1], center[1],
                axes[2], 0, 0, center[2], 0, axes[3], 0, center[3], 0, 0, 0, 1])
        end
    end

    @time "Nonspherical fluid regions" @testset "Nonspherical fluid regions" begin
        beta, alpha = pi / 3, 0.4
        incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
        observations = (-incident, incident, [0.0, 0.0, 1.0])
        materials = [FluidFilled(1.05, 1.02), GasFilled(0.0012, 0.23)]

        @time "Independent geometry and quadrature" @testset "Independent geometry and quadrature" begin
            for (name, body, axes, center) in (
                (:displaced, Spheroid(1.4, 1.0), (0.35, 0.35, 0.35), (0.3, 0.2, 0.1)),
                (:bent, Cylinder(0.7, 1.6; radius_curvature = 3.0, endcap_depth = 0.7),
                (0.3, 0.3, 0.3), (0.0, 0.08, 0.0)))
                resolution, qorder = 0.6, 4
                outer = mesh(body; method = :full, resolution, mesh_order = 3, qorder)
                inner = ellipsoid_surface(axes, center; resolution = 0.35resolution, qorder)
                solution = bem([outer, inner], materials, 1.0;
                    incidence_angle = beta, incidence_azimuth = alpha)
                amplitudes = [scattering_amplitude(solution; direction)
                              for direction in observations]
                @test diagnostics(solution).scaled_relative_residual < 1e-10
                @test all(isfinite, amplitudes)
            end
        end

        @time "Sibling interfaces and exterior interactions" @testset "Sibling interfaces and exterior interactions" begin
            outer = mesh(
                Sphere(1.0); method = :full, resolution = 0.6, mesh_order = 3, qorder = 4)
            left = ellipsoid_surface((0.15, 0.15, 0.15), (0.0, -0.4, 0.0); resolution = 0.09, qorder = 4)
            right = ellipsoid_surface((0.15, 0.15, 0.15), (0.0, 0.4, 0.0); resolution = 0.09, qorder = 4)
            material = FluidFilled(1.2, 1.1)
            solution = bem(
                [outer, left, right], fill(material, 3), 1.0; parents = [0, 1, 1],
                incidence_angle = beta, incidence_azimuth = alpha)
            for (angle, direction) in zip((pi, 0.0), observations[1:2])
                actual = scattering_amplitude(solution; direction)
                reference = scattering_amplitude(modal(Sphere(1.0), material, 1.0; angle))
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
            disjoint = bem(
                [left, right], [material, FluidFilled(1, 1)], 1.0; parents = [0, 0],
                incidence_angle = beta, incidence_azimuth = alpha)
            single = bem(
                left, material, 1.0; incidence_angle = beta, incidence_azimuth = alpha)
            for direction in observations
                actual = scattering_amplitude(disjoint; direction)
                reference = scattering_amplitude(single; direction)
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end
            @test_throws ArgumentError bem([outer, left, right], fill(material, 3), 1.0)
            crossing = ellipsoid_surface((0.7, 0.7, 0.7), (0.0, 0.7, 0.0); resolution = 0.42, qorder = 4)
            @test_throws ArgumentError bem([outer, crossing], [material, material], 1.0;
                validation = (maxdepth = 4, maxwork = 5000))
        end
    end
end

let
    @time "Rotated geometry preserves complex amplitudes" @testset "Rotated geometry preserves complex amplitudes" begin
        original = mesh(; semiaxes = (0.06, 0.018, 0.025), resolution = 0.8, qorder = 5, tip_ratio = 0.4)
        angle = 0.37
        rotation = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1]
        nodes = rotation*original.body.nodes
        rotated = mesh(nodes, hcat(original.body.connectivity...); qorder = 5)
        @test rotated.body.nodes ≈ nodes
        @test maximum(norm.(AS.normals(rotated) .-
                            [rotation*n for n in AS.normals(original)])) < 1e-10
        beta, alpha = pi/3, 0.4
        incident = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
        rotated_incident = rotation*incident
        for material in (FluidFilled(1.04, 1.04),)
            a = bem(
                original, material, 1.0; incidence_angle = beta, incidence_azimuth = alpha)
            b = bem(rotated, material, 1.0; incidence_angle = acos(rotated_incident[1]),
                incidence_azimuth = atan(rotated_incident[3], rotated_incident[2]))
            for direction in (-incident, incident, [0.0, 0.0, 1.0])
                reference = scattering_amplitude(a; direction)
                actual = scattering_amplitude(b; direction = rotation*direction)
                @test actual ≈ reference rtol=1e-5
            end
        end
    end
end

let
    function compare_region_pressure(actual, expected)
        for (got, wanted) in zip(actual, expected)
            @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
            @test abs(20log10(abs(got/wanted))) < 0.01
        end
    end

    function modal_region_points(points, direction)
        [(dot(direction, p), sqrt(max(0, norm(p)^2-dot(direction, p)^2)), 0.0)
         for p in points]
    end

    @time "Disconnected and branched fluid pressure" @testset "Disconnected and branched fluid pressure" begin
        beta, alpha, k = pi/3, 0.4, 0.6
        direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
        center = [-0.35, 0.0, 0.0]
        active = mesh(; semiaxes = (0.25, 0.25, 0.25), center = Tuple(center),
            resolution = 0.36, mesh_order = 3, qorder = 5)
        passive = mesh(; semiaxes = (0.2, 0.2, 0.2), center = (0.4, 0.1, 0.0),
            resolution = 0.6, mesh_order = 3, qorder = 5)
        outer = mesh(
            Sphere(1.0); method = :full, resolution = 0.45, mesh_order = 3, qorder = 5)
        material = FluidFilled(0.4, 0.7)
        reference = modal(Sphere(0.25), material, k)
        phase = cis(k*dot(direction, center))
        points = [(1.1, 0.1, 0.0), (-0.35, 0.0, 0.0), (0.4, 0.1, 0.0),
            (-0.35, 0.4, 0.0), (0.0, -0.4, 0.2)]
        local_points = [Tuple(collect(p)-center) for p in points]
        expected = phase .*
                   pressure(reference, modal_region_points(local_points, direction))
        for (surfaces, materials, parents, active_region, passive_region) in (
            ([passive, active], [FluidFilled(1, 1), material], [0, 0], 2, 1),
            ([outer, active, passive],
            [FluidFilled(1, 1), material, FluidFilled(1, 1)], [0, 1, 1], 2, 3))
            solution = bem(surfaces, materials, k; parents, incidence_angle = beta,
                incidence_azimuth = alpha, condition_limit = 0)
            @time "parents=$parents" @testset "parents=$parents" begin
                compare_region_pressure(pressure(solution, points), expected)
                @test pressure(solution, points[2]; region = active_region) ≈ expected[2] rtol = 1e-3
                @test pressure(solution, points[3]; region = passive_region) ≈ expected[3] rtol = 1e-3
                @test_throws ArgumentError pressure(solution, points[2]; region = passive_region)
                exterior = [points[1], (0.0, 0.0, 1.3)]
                local_exterior = [Tuple(collect(p)-center) for p in exterior]
                compare_region_pressure(pressure(solution, exterior; field = :scattered),
                    phase .*
                    pressure(reference, modal_region_points(local_exterior, direction); field = :scattered))
                if length(surfaces) == 3
                    @test pressure(solution, points[4]; region = 1) ≈ expected[4] rtol = 1e-3
                    @test_throws ArgumentError pressure(solution, points[4]; region = 0)
                end
                for (i, surface) in enumerate(surfaces)
                    anchor = i == active_region ? AS.SVector(-0.35, 0.15, 0.2) :
                             i == passive_region ? AS.SVector(0.4, 0.22, 0.16) :
                             AS.SVector(0.6, 0.48, 0.64)
                    q = surface.data[argmin(norm(node.coords-anchor)
                    for node in surface.data)]
                    compare_region_pressure([pressure(solution, q.coords; region = i)],
                        [pressure(solution, q.coords; region = parents[i])])
                end
                @test diagnostics(solution).relative_residual < 1e-9
            end
        end
    end

    @time "Nested fluid pressure" @testset "Nested fluid pressure" begin
        outer = mesh(
            Sphere(1.0); method = :full, resolution = 0.6, mesh_order = 3, qorder = 4)
        inner = mesh(
            Sphere(0.5); method = :full, resolution = 0.3, mesh_order = 3, qorder = 4)
        solution = bem([outer, inner],
            [FluidFilled(1.2, 1.1), FluidFilled(0.7, 0.8)], 1.0; condition_limit = 0)
        exterior = (1.2, 0.0, 0.0)
        wall = (0.75, 0.0, 0.0)
        cavity = (0.0, 0.0, 0.0)
        @test all(isfinite, pressure(solution, [exterior, wall, cavity]))
        @test pressure(solution, exterior; region = 0) ≈ pressure(solution, exterior)
        @test pressure(solution, wall; region = 1) ≈ pressure(solution, wall)
        @test pressure(solution, cavity; region = 2) ≈ pressure(solution, cavity)
    end

    @time "Interacting fluid pressure" @testset "Interacting fluid pressure" begin
        surfaces = [mesh(; semiaxes = (0.2, 0.2, 0.2), center,
                        resolution = 0.6, mesh_order = 3, qorder = 5)
                    for center in ((-0.35, 0.0, 0.0), (0.35, 0.0, 0.0))]
        k = 0.6
        solution = bem(surfaces, [FluidFilled(0.7, 0.8), FluidFilled(1.4, 0.9)], k;
            parents = [0, 0], incidence_angle = pi/3, incidence_azimuth = 0.4,
            condition_limit = 0)
        for (i, surface) in enumerate(surfaces)
            anchor = AS.SVector(i == 1 ? -0.35 : 0.35, 0.12, 0.16)
            index = argmin(norm(node.coords-anchor) for node in surface.data)
            point = surface.data[index].coords
            expected = [solution.data.interfaces[i].pressure[index]]
            compare_region_pressure([pressure(solution, point; region = i)], expected)
            compare_region_pressure([pressure(solution, point; region = 0)], expected)
        end
        for direction in ([0.36, 0.48, 0.8], [-0.8, 0.6, 0.0])
            distance = 1e6
            value = pressure(solution, Tuple(distance .* direction); field = :scattered)
            amplitude = value * distance * cis(-k*distance)
            @test isapprox(amplitude, scattering_amplitude(solution; direction); rtol = 1e-4, atol = 1e-12)
        end
        @test_throws ArgumentError pressure(modal(Sphere(1.0), Rigid(), k), (2.0, 0.0, 0.0); region = 0)
    end
end
