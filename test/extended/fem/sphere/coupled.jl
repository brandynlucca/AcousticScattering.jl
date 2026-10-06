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

        @time @testset "Additional water-filled shell resonances" begin
            body = Sphere(0.01)
            shell = Shell(body, 0.002)
            material = Shelled(poisson, rho, youngs_modulus)
            reference_boundary = Shelled(wall, FluidInterior(1.0, 1.0), 0.8)
            beta = pi / 3

            # The three samples around each peak check the local spectral shape.
            # The higher-frequency peak is shallow, so retain complex amplitudes
            # rather than inferring accuracy from its target strength alone.
            for ka_samples in ((2.78, 2.84, 2.90), (4.62, 4.71, 4.80))
                modal_ts = Float64[]
                radial_ts = Float64[]
                for ka in ka_samples
                    k = ka / body.radius
                    reference = modal(body, reference_boundary, k; m_max = 24)
                    radial = fem(body, reference_boundary, k;
                        m_max = 24, n_elements = 1280)
                    @test scattering_amplitude(radial) ≈ scattering_amplitude(reference) rtol = 0.001
                    @test abs(target_strength(radial) - target_strength(reference)) < 0.01
                    push!(modal_ts, target_strength(reference))
                    push!(radial_ts, target_strength(radial))
                end
                @test modal_ts[2] > max(modal_ts[1], modal_ts[3])
                @test radial_ts[2] > max(radial_ts[1], radial_ts[3])
            end

            for (ka, coarse_grid, fine_grid) in (
                (2.84, (193, 25), (257, 33)),
                (4.71, (257, 33), (321, 41)))
                k = ka / body.radius
                solve(grid) = fem(shell, material,
                    1000.0, 1500.0, 1000.0, 1500.0, k;
                    method = :general, incidence_angle = beta,
                    n_eta = grid[1], n_t = grid[2], m_max = 10,
                    pole_offset = 1e-4, rtol = 1e-5)
                coarse = solve(coarse_grid)
                fine = solve(fine_grid)
                reference_back = scattering_amplitude(modal(body, reference_boundary, k;
                    m_max = 24, angle = pi))
                @test abs(scattering_amplitude(fine) - reference_back) <
                      abs(scattering_amplitude(coarse) - reference_back)

                for (angle, azimuth, scattering_angle) in (
                    (pi - beta, pi, pi), (beta, 0.0, 0.0), (pi / 2, pi / 2, pi / 2))
                    reference = modal(body, reference_boundary, k;
                        m_max = 24, angle = scattering_angle)
                    @test abs(target_strength(fine; angle, azimuth) -
                              target_strength(reference)) < 0.1
                    @test scattering_amplitude(fine; angle, azimuth) ≈
                          scattering_amplitude(reference) rtol = 0.01
                end
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
