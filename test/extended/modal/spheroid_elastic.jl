using AcousticScattering
using Test
using LinearAlgebra

const AS = AcousticScattering

let
    boundary = SolidElastic(2.7, 6.4 / 1.5, 3.04 / 1.5)

    @time "Modal spheroid: elastic solid approaches the elastic sphere" @testset "Modal spheroid: elastic solid approaches the elastic sphere" begin
        for (ka, aspect, tolerance) in ((0.8, 1.005, 2e-3), (2.0, 1.005, 4e-3))
            radius = cbrt(aspect)
            k = ka / radius
            solution = modal(Spheroid(aspect, 1.0), boundary, k; incidence_angle = pi / 3)
            reference = modal(Sphere(radius), boundary, k)
            difference = abs(scattering_amplitude(solution) -
                             scattering_amplitude(reference))
            @test difference / abs(scattering_amplitude(reference)) < tolerance
        end
    end

    @time "Modal spheroid: stiff dense solid approaches the rigid spheroid" @testset "Modal spheroid: stiff dense solid approaches the rigid spheroid" begin
        stiff = SolidElastic(1e4, 300.0, 150.0)
        for aspect in (3.0, 5.0)
            body = Spheroid(aspect, 1.0)
            k = 2.0 / aspect
            for incidence_angle in (pi / 3, pi / 2)
                reference = modal(body, Rigid(), k; incidence_angle)
                solution = modal(body, stiff, k; incidence_angle)
                @test abs(scattering_amplitude(solution) -
                          scattering_amplitude(reference)) /
                      abs(scattering_amplitude(reference)) < 1e-3
            end
        end
    end

    @time "Modal spheroid: elastic transition matrix is unitary and reciprocal" @testset "Modal spheroid: elastic transition matrix is unitary and reciprocal" begin
        body = Spheroid(3.0, 1.0)
        k = 1.0
        for m in 0:2
            T = AS._elastic_layered_transition(m, 14, k, body.q, body.xi0, nothing,
                (2.7, 6.4 / 1.5, 3.04 / 1.5), nothing, :prolate)
            @test norm(T' * T + (T + T') / 2) / norm(T) < 1e-8
        end
        # Swapping source and receiver directions leaves the bistatic amplitude unchanged.
        options = (; incidence_azimuth = 0.0, scatter_azimuth = 0.7, m_max = 14, n_max = 14)
        forward = AS.form_function(boundary, k, body; incidence_angle = 1.0,
            scatter_angle = 2.2, options...)
        reversed = AS.form_function(boundary, k, body; incidence_angle = pi - 2.2,
            scatter_angle = pi - 1.0, options...)
        @test abs(forward - reversed) / abs(forward) < 1e-8
    end

    @time "Modal spheroid: elastic solid against independent ASIGA amplitudes" @testset "Modal spheroid: elastic solid against independent ASIGA amplitudes" begin
        # ASIGA infinite-element IGA, mesh level 4, degree 4, 8 infinite-element functions. Level 3 differs by at most 1.1e-4 relative.
        # Aluminium E=70 GPa, nu=0.33, rho=2700 in water (1000 kg/m^3, 1500 m/s). Incident direction (sqrt(3)/2, 0, 1/2), exp(-i omega t).
        aluminium = SolidElastic(2.7, 6197.824298019838 / 1500, 3121.9527052723133 / 1500)
        directions = ((2pi / 3, pi), (pi / 3, 0.0), (pi / 2, pi / 2))
        cases = (
            (1.5, 0.5, 1.0,
                (-0.147449368472606 + 0.00592487785177542im,
                    -0.00379603924306052 + 0.00998574496572924im,
                    -0.100996250821922 + 0.0085435104260526im)),
            (1.5, 0.5, 2.0,
                (-0.118512217954551 + 0.0523856629853087im,
                    0.0719088583398809 + 0.116063337703658im,
                    -0.293582362826262 + 0.0938820306544933im)),
            (1.5, 0.3, 1.0,
                (-0.0597969017874388 + 0.00088434096305577im,
                    -0.00297089889010796 + 0.00152857065308666im,
                    -0.0387677245490206 + 0.00130733586017213im)))
        for (a, b, k, values) in cases,
            ((angle, azimuth), expected) in zip(directions, values)

            actual = AS.form_function(aluminium, k, Spheroid(a, b);
                incidence_angle = pi / 3, incidence_azimuth = 0.0,
                scatter_angle = angle, scatter_azimuth = azimuth)
            @test abs(actual - expected) / abs(expected) < 1e-3
            @test abs(target_strength(actual) - target_strength(expected)) < 0.01
        end
    end

    @time "Modal spheroid: elastic shell against independent ASIGA amplitudes" @testset "Modal spheroid: elastic shell against independent ASIGA amplitudes" begin
        # Same ASIGA setup as above with a confocal shell whose equatorial semi-axis is 0.8 of the outer one. Level 3 to 4 changes are at most 3e-4.
        layer = ElasticLayer(2.7, 6197.824298019838 / 1500, 3121.9527052723133 / 1500)
        directions = ((2pi / 3, pi), (pi / 3, 0.0), (pi / 2, pi / 2))
        # (semi-axes a, b, interior, truncation, tolerance, amplitudes)
        cases = (
            (1.5, 1.0, FluidInterior(1.0, 1.0), 16, 1e-3,
                (-0.106487925705214 + 0.0747097439019188im,
                    0.0814480252840919 + 0.0804910782181885im,
                    -0.467543104429619 + 0.0525831637378128im)),
            (1.5, 1.0, VacuumInterior(), 18, 1e-4,
                (-0.0203600241972294 + 0.0661992781692976im,
                    -0.0990561548063969 + 0.0683892400198043im,
                    -0.387924155701646 + 0.0577926663978733im)),
            (1.5, 0.5, FluidInterior(1.0, 1.0), 26, 1e-2,
                (-0.0959171719579387 + 0.00536761112067266im,
                    -0.0216530144915487 + 0.0064494458393739im,
                    -0.0986395731466778 + 0.0062811929301369im)),
            (1.5, 0.5, VacuumInterior(), 22, 1e-2,
                (-0.0497503336091472 + 0.00516407289162214im,
                    -0.0719781836877906 + 0.00537249772622912im,
                    -0.0896064378721947 + 0.00568337555768539im)))
        for (a, b, interior, order, tolerance, values) in cases
            shell = Shelled(layer, interior, 0.8)
            for ((angle, azimuth), expected) in zip(directions, values)
                actual = AS.form_function(
                    shell, 1.0, Spheroid(a, b); incidence_angle = pi / 3,
                    incidence_azimuth = 0.0, scatter_angle = angle, scatter_azimuth = azimuth,
                    m_max = order, n_max = order, check = false)
                @test abs(actual - expected) / abs(expected) < tolerance
            end
        end
    end

    @time "Modal spheroid: elastic shell limits and consistency" @testset "Modal spheroid: elastic shell limits and consistency" begin
        layer = ElasticLayer(2.7, 6197.824298019838 / 1500, 3121.9527052723133 / 1500)
        body = Spheroid(1.5, 1.0)
        k = 1.0
        material = (2.7, 6197.824298019838 / 1500, 3121.9527052723133 / 1500)
        xi_inner = AS._confocal_inner_xi(body, 0.8)
        for interior in ((1.0, 1.0), :vacuum), m in 0:2

            T = AS._elastic_layered_transition(
                m, 16, k, body.q, body.xi0, xi_inner, material, interior, :prolate)
            @test norm(T' * T + (T + T') / 2) / norm(T) < 1e-6
        end
        # Deformation of a spherical shell shrinks linearly to the exact spherical shell.
        shell = Shelled(layer, FluidInterior(1.0, 1.0), 0.8)
        differences = map((1.02, 1.01)) do aspect
            radius = cbrt(aspect)
            ka = 1.5
            solution = AS.form_function(shell, ka / radius, Spheroid(aspect, 1.0);
                incidence_angle = pi / 3, check = false)
            reference = scattering_amplitude(modal(Sphere(radius), shell, ka / radius))
            abs(solution - reference) / abs(reference)
        end
        @test differences[2] < 0.6 * differences[1]
        # The truncation guard flags an elongated shell at the default truncation.
        elongated = Shelled(layer, VacuumInterior(), 0.8)
        @test_logs (:warn, r"Elastic shell amplitude changes") AS.form_function(
            elongated, 2.0, Spheroid(3.0, 1.0); incidence_angle = pi / 3)
    end

    @time "Modal spheroid: oblate elastic solid and shell" @testset "Modal spheroid: oblate elastic solid and shell" begin
        material = (2.7, 6197.824298019838 / 1500, 3121.9527052723133 / 1500)
        stiff = SolidElastic(1e4, 300.0, 150.0)
        for aspect in (2.0, 3.0)
            body = Spheroid(1.0, aspect)
            for incidence_angle in (pi / 3, pi / 2)
                reference = modal(body, Rigid(), 1.0; incidence_angle)
                solution = modal(body, stiff, 1.0; incidence_angle)
                @test abs(scattering_amplitude(solution) -
                          scattering_amplitude(reference)) /
                      abs(scattering_amplitude(reference)) < 1e-3
            end
        end
        body = Spheroid(1.0, 2.0)
        for m in 0:2
            T = AS._elastic_layered_transition(
                m, 14, 1.0, body.q, body.xi0, nothing, material, nothing, :oblate)
            @test norm(T' * T + (T + T') / 2) / norm(T) < 1e-8
        end
        shell_body = Spheroid(0.98, 1.01)
        xi_inner = AS._confocal_inner_xi(shell_body, 0.8)
        for interior in ((1.0, 1.0), :vacuum)
            T = AS._elastic_layered_transition(
                0, 14, 1.5, shell_body.q, shell_body.xi0,
                xi_inner, material, interior, :oblate)
            @test norm(T' * T + (T + T') / 2) / norm(T) < 1e-6
        end
        # A small deformation of a sphere is odd in its sign, so prolate and oblate amplitudes average to the sphere at second order.
        solid = SolidElastic(material...)
        reference = scattering_amplitude(modal(Sphere(1.0), solid, 1.5))
        means = map((0.02, 0.01)) do eps
            prolate = AS.form_function(solid, 1.5, Spheroid(1 + eps, 1 - eps / 2);
                incidence_angle = pi / 3)
            oblate = AS.form_function(solid, 1.5, Spheroid(1 - eps, 1 + eps / 2);
                incidence_angle = pi / 3)
            abs((prolate + oblate) / 2 - reference) / abs(reference)
        end
        @test means[1] < 2e-3
        @test means[1] / means[2] > 3.5
    end

    @time "Modal spheroid: incidence angle sweep reuses the transition matrices" @testset "Modal spheroid: incidence angle sweep reuses the transition matrices" begin
        angles = [0.4, 0.9, 1.3]
        solid = SolidElastic(2.7, 6197.824298019838 / 1500, 3121.9527052723133 / 1500)
        for body in (Spheroid(1.5, 0.5), Spheroid(0.5, 1.5))
            sweep = AS.incidence_angle_sweep(body, solid, 1.0, angles)
            for (i, angle) in enumerate(angles)
                @test sweep.amplitudes[i] ≈
                      AS.form_function(solid, 1.0, body; incidence_angle = angle) rtol = 1e-12
            end
        end
        shell = Shelled(ElasticLayer(2.7, 4.131882865346559, 2.0813018035148754),
            FluidInterior(1.0, 1.0), 0.8)
        sweep = AS.incidence_angle_sweep(Spheroid(1.5, 1.0), shell, 1.0, angles[1:2];
            m_max = 10, n_max = 10, check = false)
        for (i, angle) in enumerate(angles[1:2])
            @test sweep.amplitudes[i] ≈ AS.form_function(shell, 1.0, Spheroid(1.5, 1.0);
                incidence_angle = angle, m_max = 10, n_max = 10, check = false) rtol = 1e-12
        end
    end

    @time "Modal spheroid: elastic truncation convergence at 5:1" @testset "Modal spheroid: elastic truncation convergence at 5:1" begin
        body = Spheroid(5.0, 1.0)
        k = 3.0 / 5.0
        default = modal(body, boundary, k; incidence_angle = pi / 3)
        refined = modal(body, boundary, k; incidence_angle = pi / 3, m_max = 18, n_max = 18)
        @test abs(target_strength(default) - target_strength(refined)) < 0.01
    end
end
