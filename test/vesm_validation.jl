# An independent scalar pressure/admittance calculation for the shear-free shell limit.
# It uses elementary l=0 radial functions, not the VESM displacement/stress determinant.
function fluid_shell_monopole_reference(k, outer_radius, inner_radius,
        density_shell, k_shell, density_core, k_core)
    # Densities are ratios to the exterior. Match (1/rho) * dp/dr and p at the inner surface.
    inner_log_derivative = (k_core / tan(k_core * inner_radius) - 1 / inner_radius) /
                           density_core
    impedance = density_shell * inner_log_derivative + 1 / inner_radius
    x_inner = k_shell * inner_radius
    ratio = (k_shell * cos(x_inner) - impedance * sin(x_inner)) /
            (k_shell * sin(x_inner) + impedance * cos(x_inner))
    x_outer = k_shell * outer_radius
    outer_log_derivative = (k_shell * (cos(x_outer) - ratio * sin(x_outer)) /
                            (sin(x_outer) + ratio * cos(x_outer)) - 1 / outer_radius) /
                           density_shell
    x = k * outer_radius
    j = sin(x) / x
    dj = k * (x * cos(x) - sin(x)) / x^2
    h = -im * cis(x) / x
    dh = k * cis(x) * (x + im) / x^2
    coefficient = (outer_log_derivative * j - dj) / (dh - outer_log_derivative * h)
    return -im * coefficient / k
end

@testset "Shell complex phase and lossy VESM limits" begin
    sound_speed = 1500.0
    core = FluidInterior(0.00126, 330 / sound_speed)
    gas = FluidFilled(core.density_contrast, core.soundspeed_contrast)
    radius = 0.01

    @testset "Elastic and VESM transparent layers preserve phase" begin
        wall = ElasticLayer(1.0, 1.0, 1e-6)
        for k in (20.0, 100.0, 250.0)
            reference = scattering_amplitude(modal(Sphere(0.005), gas, k; m_max = 0))
            elastic = Shelled(wall, core, 0.005 / 0.009)
            viscous = Shelled(
                LayeredMaterial(
                    ViscousLayer(sound_speed, 1.0, 1.0, 0.0, 0.0), wall, 0.9),
                core, 0.5)
            @test scattering_amplitude(modal(Sphere(0.009), elastic, k; m_max = 0)) ≈
                  reference rtol = 1e-8
            @test scattering_amplitude(modal(Sphere(radius), viscous, k)) ≈ reference rtol = 1e-8
        end
    end

    @testset "Nonzero bulk loss agrees with independent fluid-shell pressure matching" begin
        # Match wall and core fluids, so their interface disappears in the zero-shear limit.
        wall = ElasticLayer(core.density_contrast, core.soundspeed_contrast, 1e-6)
        inner_radius = 0.009
        for viscosity in (0.0, 0.5, 5.0, 50.0), frequency in (200.0, 500.0, 1000.0, 4000.0)

            omega = 2pi * frequency
            k = omega / sound_speed
            flesh = ViscousLayer(sound_speed, 1.05, 1.01, viscosity, 0.0)
            boundary = Shelled(LayeredMaterial(flesh, wall, inner_radius / radius), core, 0.5)
            k_shell = omega / sqrt(complex((1.01sound_speed)^2, -omega * viscosity))
            reference = fluid_shell_monopole_reference(k, radius, inner_radius,
                1.05, k_shell, core.density_contrast, k / core.soundspeed_contrast)
            amplitude = scattering_amplitude(modal(Sphere(radius), boundary, k))
            @test amplitude ≈ reference rtol = 1e-7
        end
    end

    @testset "Nonzero shear loss has outgoing attenuation and passive partial waves" begin
        # SI materials inspired by Feuillade & Nero §III, with both viscosities explicit.
        # This is a physical consistency check, not a digitization of their Figure 6.
        outer_radius, gas_radius = 0.12, 0.02
        wall = ElasticLayer(1.05, sqrt((1149.3e6 + 2 * 1.06e6) / 1050) / sound_speed,
            sqrt(1.06e6 / 1050) / sound_speed)
        boundary(loss) = Shelled(
            LayeredMaterial(
                ViscousLayer(sound_speed, 1.05, 1.0, loss * 50 / 1050, loss * 25 / 1050),
                wall, 1.1gas_radius / outer_radius),
            core,
            gas_radius / outer_radius)
        for kr in range(0.01, 0.06; length = 41)
            k = kr / gas_radius
            omega = k * sound_speed
            for loss in (0.0, 1.0)
                bc = boundary(loss)
                amplitude = scattering_amplitude(modal(Sphere(outer_radius), bc, k))
                coefficient = im * k * amplitude
                absorbed = -real(coefficient) - abs2(coefficient)
                @test isfinite(amplitude)
                @test absorbed >= -1e-12
                if iszero(loss)
                    @test abs(absorbed) < 1e-12
                else
                    k_longitudinal, beta = AS._vesm_flesh_wavenumbers(bc, omega)
                    @test imag(k_longitudinal) > 0
                    @test imag(beta) < 0
                end
            end
            lossless = scattering_amplitude(modal(Sphere(outer_radius), boundary(0.0), k))
            vanishing = scattering_amplitude(modal(Sphere(outer_radius), boundary(1e-9), k))
            @test vanishing ≈ lossless rtol = 1e-6
        end
    end
end
