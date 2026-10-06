using AcousticScattering
using Test

@testset "Kirchhoff source-facing surface and translation" begin
    # An axis-aligned box has just one illuminated face for axial incidence.
    # Its physical-optics integral is analytic, including the face's position.
    k, area = 2.3, 0.6 * 0.4
    for x0 in (0.0, 0.37)
        surface = mesh(; qorder = 4) do g
            g.model.add("kirchhoff-source-facing-box")
            g.model.occ.addBox(x0, -0.3, -0.2, 1.0, 0.6, 0.4)
            g.model.occ.synchronize()
            g.option.setNumber("Mesh.MeshSizeMin", 0.3)
            g.option.setNumber("Mesh.MeshSizeMax", 0.3)
            g.model.mesh.generate(2)
            g.model.mesh.setOrder(2)
        end
        for (beta, phase) in ((0.0, 2k*x0), (pi, -2k*(x0+1)))
            expected = -im*k*area/(2pi) * cis(phase)
            for (boundary, reflection) in ((Rigid(), 1), (PressureRelease(), -1))
                actual = scattering_amplitude(kirchhoff(surface, boundary, k;
                    incidence_angle = beta))
                @test actual ≈ reflection*expected rtol = 1e-11
            end
        end
    end
end

# A supplied-mesh kirchhoff() of a canonical shape should reproduce the closed-form kirchhoff()
# of that same shape to mesh-discretization accuracy: the two are independent code paths for the
# same physical-optics surface integral.
@testset "Kirchhoff on a supplied mesh" begin
    k = 2pi * 38e3 / 1477.4

    @testset "Sphere" begin
        a = 0.01
        surface = mesh(
            Sphere(a); method = :full, resolution = 0.001, qorder = 4, mesh_order = 2)
        for boundary in (Rigid(), PressureRelease())
            closed = kirchhoff(Sphere(a), boundary, k)
            meshed = kirchhoff(surface, boundary, k; incidence_angle = pi / 2)
            @test meshed isa KirchhoffSolution
            @test scattering_amplitude(meshed) ≈ scattering_amplitude(closed) rtol = 1e-4
        end
    end

    @testset "Spheroid, several incidence angles" begin
        body = Spheroid(0.03, 0.01)
        surface = mesh(
            body; method = :full, resolution = 0.0015, qorder = 4, mesh_order = 2)
        for boundary in (Rigid(), PressureRelease()), angle in (pi / 2, pi / 3, 0.15)

            closed = kirchhoff(body, boundary, k; incidence_angle = angle)
            meshed = kirchhoff(surface, boundary, k; incidence_angle = angle)
            @test scattering_amplitude(meshed) ≈ scattering_amplitude(closed) rtol = 1e-3
        end
    end

    @testset "Cylinder (with endcaps)" begin
        body = Cylinder(0.01, 0.05)
        surface = mesh(body; method = :full, resolution = 0.003, qorder = 4, mesh_order = 2)
        for boundary in (Rigid(), PressureRelease()), angle in (pi / 2, pi / 3)

            closed = kirchhoff(body, boundary, k; incidence_angle = angle)
            meshed = kirchhoff(surface, boundary, k; incidence_angle = angle)
            @test scattering_amplitude(meshed) ≈ scattering_amplitude(closed) rtol = 1e-3
        end
    end

    @testset "Solved body records the supplied surface" begin
        surface = mesh(Sphere(0.01); method = :full, resolution = 0.003)
        solution = kirchhoff(surface, Rigid(), k; incidence_angle = pi / 2)
        @test solution.body === surface.body
    end
end
