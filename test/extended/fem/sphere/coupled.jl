using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Interior CBIE (closed-form spherical-cavity cross-check)" @testset "Interior CBIE (closed-form spherical-cavity cross-check)" begin
        a, k_test = 0.01, 37.5
        mesh = AS.sphere_mesh(a, 20)
        K, V, ps = AS.assemble_cbie_operators(mesh, k_test; rtol = 1e-6)
        n = length(ps)

        p_bc = ones(ComplexF64, n)
        dpdn = V \ ((0.5I + K) * p_bc)

        j0ka = AS.js(0, k_test * a)
        j0dka = AS.jsd(0, k_test * a)
        exact_dpdn = k_test * j0dka / j0ka

        @test all(d -> abs(d - exact_dpdn) / abs(exact_dpdn) < 0.01, dpdn)
        @test maximum(abs, dpdn) - minimum(abs, dpdn) < 1e-3 * abs(exact_dpdn)
    end

    @time "Finite-stiffness shell scattering against spherical modal solutions" @testset "Finite-stiffness shell scattering against spherical modal solutions" begin
        rho, youngs_modulus, poisson = 2700.0, 70e9, 0.33
        c_longitudinal = sqrt(youngs_modulus * (1 - poisson) /
                              (rho * (1 + poisson) * (1 - 2poisson)))
        c_transverse = sqrt(youngs_modulus / (2rho * (1 + poisson)))
        wall = ElasticLayer(rho / 1000, c_longitudinal / 1500, c_transverse / 1500)
        for (rho_inside, c_inside, beta) in ((1000.0, 1500.0, pi / 3),)
            reference_boundary = Shelled(
                wall, FluidInterior(rho_inside / 1000, c_inside /
                                                       1500), 0.8)
            solution = fem(
                Shell(Sphere(0.01), 0.002), Shelled(poisson, rho, youngs_modulus),
                1000.0, 1500.0, rho_inside, c_inside, 100.0;
                method = :general, incidence_angle = beta, n_eta = 49, n_t = 5,
                m_max = 5, pole_offset = 1e-4, rtol = 1e-5)
            for (angle, azimuth, scattering_angle) in ((pi - beta, pi, pi), (
                beta, 0.0, 0.0), (
                pi / 2, pi / 2, pi / 2))
                reference = modal(Sphere(0.01), reference_boundary, 100.0;
                    m_max = 12, angle = scattering_angle)
                @test scattering_amplitude(solution; angle, azimuth) ≈
                      scattering_amplitude(reference) rtol = 0.01
            end
        end

        @time @testset "Water-filled spherical-shell resonance: kR=$ka" for ka in (1.92,)
            beta = pi / 3
            reference_boundary = Shelled(wall, FluidInterior(1.0, 1.0), 0.8)
            solution = fem(
                Shell(Sphere(0.01), 0.002), Shelled(poisson, rho, youngs_modulus),
                1000.0, 1500.0, 1000.0, 1500.0, ka / 0.01;
                method = :general, incidence_angle = beta, n_eta = 193, n_t = 25,
                m_max = 6, pole_offset = 1e-4, rtol = 1e-5)
            for (angle, azimuth, scattering_angle) in ((pi - beta, pi, pi),
                (beta, 0.0, 0.0), (pi / 2, pi / 2, pi / 2))
                reference = modal(Sphere(0.01), reference_boundary, ka / 0.01;
                    m_max = 16, angle = scattering_angle)
                @test abs(target_strength(solution; angle, azimuth) -
                          target_strength(reference)) < 0.1
                @test scattering_amplitude(solution; angle, azimuth) ≈
                      scattering_amplitude(reference) rtol = 0.01
            end
        end

        # Thin-shell approximation: nearly spherical, 1% thickness, air-filled, kR=0.5.
        wall = ElasticLayer(2.7, sqrt(70e9 * 0.7 / (2700 * 1.3 * 0.4)) / 1500,
            sqrt(70e9 / (2 * 2700 * 1.3)) / 1500)
        reference = modal(
            Sphere(0.01), Shelled(wall, FluidInterior(0.0012, 343 / 1500), 0.99),
            50.0; m_max = 8)
        solution = fem(
            Shell(Spheroid(0.01000001, 0.01), 0.0001), Shelled(0.3, 2700.0, 70e9),
            1000.0, 1500.0, 1.2, 343.0, 50.0;
            method = :thin, incidence_angle = 0.0, n_eta = 65, rtol = 1e-5)
        @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.01
    end
end
