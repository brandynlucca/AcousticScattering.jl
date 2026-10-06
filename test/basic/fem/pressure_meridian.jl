using AcousticScattering
using Test

@testset "Meridian and shell FEM pressure" begin
    k = 2.0
    for boundary in (Rigid(), PressureRelease())
        solution = fem(Sphere(1.0), boundary, k; method = :meridian,
            R = 1.4, n_r = 18, n_theta = 32, l_max = 12)
        reference = modal(Sphere(1.0), boundary, k)
        @test isapprox(target_strength(solution), target_strength(reference); atol = 0.8)
        for point in ((1.08, 0.0, 0.0), (0.4, 1.2, 0.0), (2.0, 0.2, 0.0))
            @test abs(pressure(solution, point)-pressure(reference, point)) < 0.12
        end
    end

    for body in (Spheroid(1.0, 0.7), Cylinder(0.7, 1.4))
        # Unit contrasts make the coupled problem transparent, including its
        # interior source term and interface forcing.
        solution = fem(body, FluidFilled(1.0, 1.0), k; R = 1.5,
            incidence_angle = 0.6, n_r = 10, n_theta = 20,
            m_max = 3, l_max = 10)
        for point in ((0.1, 0.1, 0.1), (1.2, 0.2, 0.15), (1.6, 0.2, 0.15))
            @test abs(pressure(solution, point) -
                      pressure(solution, point; field = :incident)) < 1e-9
        end
        @test_throws ArgumentError pressure(solution, (0.1, 0.1, 0.1);
            field = :scattered)
    end

    body = Spheroid(1.0, 0.7)
    boundary = FluidFilled(1.1, 0.9)
    fem_solution = fem(body, boundary, k; R = 1.5,
        incidence_angle = 0.6, n_r = 24, n_theta = 48,
        m_max = 5, l_max = 14)
    bem_solution = bem(body, boundary, k; n = 96,
        incidence_angle = 0.6, m_max = 5, rtol = 1e-7)
    for point in ((1.15, 0.15, 0.1), (0.2, 0.1, 0.05))
        reference = pressure(bem_solution, point)
        @test abs(pressure(fem_solution, point)-reference)/abs(reference) < 3e-3
    end

    shell = Shell(Spheroid(0.035, 0.007), 0.0005)
    material = Shelled(0.32, 2565.0, 70e9)
    k_shell = 2π*12000/1477.3
    for method in (:thin, :general)
        solution = fem(shell, material, 1026.8, 1477.3, 1077.3, 1575.0,
            k_shell; method, incidence_angle = 0.0, n_eta = 9,
            m_max = 0, n_t = 2)
        for point in ((0.05, 0.0, 0.0), (0.0, 0.007, 0.0))
            @test isfinite(pressure(solution, point))
        end
        for point in ((0.0, 0.0, 0.0), (0.0, 0.0065, 0.0))
            @test isfinite(pressure(solution, point; field = :interior))
        end
        @test_throws ArgumentError pressure(solution, (0.0, 0.0068, 0.0))
    end

    spherical_shell = fem(Shell(Sphere(0.02), 0.001), material,
        1026.8, 1477.3, 1077.3, 1575.0, k_shell;
        method = :general, incidence_angle = 0.0,
        n_eta = 7, n_t = 2, m_max = 0)
    @test isfinite(pressure(spherical_shell, (0.03, 0.0, 0.0)))
    @test isfinite(pressure(spherical_shell, (0.0, 0.0, 0.0);
        field = :interior))
end
