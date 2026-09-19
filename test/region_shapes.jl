using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(1)

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

@time @testset "Nonspherical fluid regions" begin
    beta, alpha = pi / 3, 0.4
    incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
    observations = (-incident, incident, [0.0, 0.0, 1.0])
    materials = [FluidFilled(1.05, 1.02), GasFilled(0.0012, 0.23)]

    @time @testset "Independent geometry and quadrature" begin
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

    @time @testset "Sibling interfaces and exterior interactions" begin
        outer = mesh(
            Sphere(1.0); method = :full, resolution = 0.6, mesh_order = 3, qorder = 4)
        left = ellipsoid_surface((0.15, 0.15, 0.15), (0.0, -0.4, 0.0); resolution = 0.09, qorder = 4)
        right = ellipsoid_surface((0.15, 0.15, 0.15), (0.0, 0.4, 0.0); resolution = 0.09, qorder = 4)
        material = FluidFilled(1.2, 1.1)
        solution = bem([outer, left, right], fill(material, 3), 1.0; parents = [0, 1, 1],
            incidence_angle = beta, incidence_azimuth = alpha)
        for (angle, direction) in zip((pi, 0.0), observations[1:2])
            actual = scattering_amplitude(solution; direction)
            reference = scattering_amplitude(modal(Sphere(1.0), material, 1.0; angle))
            @test abs(target_strength(actual) - target_strength(reference)) < 0.1
            @test abs(actual - reference) / abs(reference) < 0.01
        end
        disjoint = bem([left, right], [material, FluidFilled(1, 1)], 1.0; parents = [0, 0],
            incidence_angle = beta, incidence_azimuth = alpha)
        single = bem(left, material, 1.0; incidence_angle = beta, incidence_azimuth = alpha)
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
