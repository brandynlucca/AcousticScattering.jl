@time "Lossy spherical-shell target-strength reference" @testset "Lossy spherical-shell target-strength reference" begin
    cases = (
        (1.1, (-40.689, -40.305, -40.460)),
        (1.2, (-41.011, -40.674, -40.834)),
        (1.5, (-41.510, -41.246, -41.419)),
        (2.0, (-41.790, -41.576, -41.767)),
        (2.5, (-41.882, -41.692, -41.895)),
        (3.0, (-41.919, -41.743, -41.956)),
        (4.0, (-41.943, -41.784, -42.010)),
        (5.0, (-41.950, -41.800, -42.033)))
    core = FluidInterior(71 / 1027, 325 / 1500)
    wall = ElasticLayer(1040 / 1027, 1520 / 1500, sqrt(0.2e6 / 1040) / 1500)
    for bulk_viscosity in (0.0, 10.0), (radius_ratio, expected) in cases

        flesh = ViscousLayer(1500.0, 1040 / 1027, 1510 / 1500,
            (bulk_viscosity + 4 / 3) / 1040, 1 / 1040)
        boundary = Shelled(LayeredMaterial(flesh, wall, 1.02 / radius_ratio),
            core, 1 / radius_ratio)
        for (frequency, reference_ts) in zip((23100.0, 23400.0, 23700.0), expected)
            solution = modal(Sphere(radius_ratio * 0.001), boundary, 2pi * frequency / 1500)
            @test target_strength(solution)≈reference_ts atol=0.02 rtol=0
        end
    end
end

@time "Lossy VESM reference outputs" @testset "Lossy VESM reference outputs" begin
    # q = k_water * gas_radius; amplitudes are in meters.
    cases = (
        (shear_modulus = 0.3e6,
            bulk = 50 / 3,
            shear = 25.0,
            points = (
                (0.01626841315035625, 0.058624661855765856 + 0.0659892164775021im),
                (0.017485150695167997, -0.007567729953482754 + 0.12460170464872054im),
                (0.01902372390771196, -0.06618975733694968 + 0.058398148024561045im))),
        (shear_modulus = 1.06e6,
            bulk = 50 / 3,
            shear = 25.0,
            points = (
                (0.023672017496512644, 0.07732791526936858 + 0.08306631417552834im),
                (0.025052784506221705, -0.005801332441953473 + 0.16039196551913582im),
                (0.02670792530218686, -0.08312381728088634 + 0.07726609884854825im))),
        (shear_modulus = 2.22e6,
            bulk = 50 / 3,
            shear = 25.0,
            points = (
                (0.031822365977184096, 0.09061807284543588 + 0.09440094154051638im),
                (0.033434073359267784, -0.0037380960115807097 + 0.18501992438134737im),
                (0.03531952660808097, -0.09434643583284394 + 0.09067481974395489im))),
        (shear_modulus = 1.06e6,
            bulk = 25 / 3,
            shear = 12.5,
            points = (
                (0.024156677746378686, 0.13196904385321812 + 0.13524442813929605im),
                (0.02500958462787464, -0.0032247845786163073 + 0.26721408743016417im),
                (0.025959930510248024, -0.13518835790467312 + 0.13202648130377162im))),
        (shear_modulus = 1.06e6,
            bulk = 100 / 3,
            shear = 50.0,
            points = (
                (0.02285479311625998, 0.040933091259550906 + 0.04821650485536376im),
                (0.025227774722815885, -0.00756157921193882 + 0.08912643334539794im),
                (0.02853907076954408, -0.04848936593764797 + 0.040609490165444774im))),
        (shear_modulus = 1.06e6,
            bulk = 40.0,
            shear = 7.5,
            points = (
                (0.02436676514477617, 0.18163077584523055 + 0.182618689186266im),
                (0.025000389544852154, -0.0008904556953923901 + 0.3642497163169916im),
                (0.025686270656262562, -0.18251574525938685 + 0.18173422106664275im))))

    body = Sphere(0.12)
    core = FluidInterior(1.26 / 1000, 330 / 1500)
    for c in cases
        wall = ElasticLayer(1.05, sqrt((1149.3e6 + 2c.shear_modulus) / 1050) / 1500,
            sqrt(c.shear_modulus / 1050) / 1500)
        flesh = ViscousLayer(
            1500.0, 1.05, 1.0, (c.bulk + 4c.shear / 3) / 1050, c.shear / 1050)
        boundary = Shelled(LayeredMaterial(flesh, wall, 0.022 / 0.12), core, 0.02 / 0.12)
        for (q, expected) in c.points
            solution = modal(body, boundary, q / 0.02)
            @test scattering_amplitude(solution)≈expected rtol=1e-7 atol=1e-11
            @test target_strength(solution) ≈ 20log10(abs(expected)) atol = 1e-5
        end
    end
end

@time "Shell complex phase and lossy VESM limits" @testset "Shell complex phase and lossy VESM limits" begin
    sound_speed = 1500.0
    core = FluidInterior(0.00126, 330 / sound_speed)
    gas = FluidFilled(core.density_contrast, core.soundspeed_contrast)
    radius = 0.01

    @time "Elastic and VESM transparent layers preserve phase" @testset "Elastic and VESM transparent layers preserve phase" begin
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

    @time "Bulk-loss reference amplitudes" @testset "Bulk-loss reference amplitudes" begin
        # Match wall and core fluids, so their interface disappears in the zero-shear limit.
        wall = ElasticLayer(core.density_contrast, core.soundspeed_contrast, 1e-6)
        inner_radius = 0.009
        # Fixed pressure-matching results, in viscosity-major, frequency-minor order.
        expected = (
            (0.004066385431193231 + 1.3852900874787224e-5im,
                -0.018335442225923763 + 0.0007051528437476446im,
                -0.010255164984947402 + 0.00044134431359675786im,
                -0.00888823181850429 + 0.001354404714811826im),
            (0.0040663854309588255 + 1.3852933949762766e-5im,
                -0.018335442209007683 + 0.0007051530653290586im,
                -0.010255164983579168 + 0.00044134432972138397im,
                -0.008888231694409504 + 0.0013544051203368645im),
            (0.004066385428034266 + 1.3853231616536258e-5im,
                -0.018335442070417044 + 0.0007051550605102261im,
                -0.010255164973253341 + 0.00044134447498435245im,
                -0.008888230780139337 + 0.0013544088201230666im),
            (0.004066385318198575 + 1.3856205285487478e-5im,
                -0.01833544203635399 + 0.0007051750137036485im,
                -0.010255165065092763 + 0.0004413459149023501im,
                -0.008888239154852732 + 0.0013544411823030198im))
        for (i, viscosity) in enumerate((0.0, 0.5, 5.0, 50.0)),
            (j, frequency) in enumerate((200.0, 500.0, 1000.0, 4000.0))

            omega = 2pi * frequency
            k = omega / sound_speed
            flesh = ViscousLayer(sound_speed, 1.05, 1.01, viscosity, 0.0)
            boundary = Shelled(LayeredMaterial(flesh, wall, inner_radius / radius), core, 0.5)
            amplitude = scattering_amplitude(modal(Sphere(radius), boundary, k))
            @test amplitude ≈ expected[i][j] rtol = 1e-7
        end
    end

    @time "Nonzero shear loss has outgoing attenuation and passive partial waves" @testset "Nonzero shear loss has outgoing attenuation and passive partial waves" begin
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
