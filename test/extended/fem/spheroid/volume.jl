using AcousticScattering
using Test

const AS = AcousticScattering

let
    E, nu, rho = 70e9, 0.33, 2700.0
    cL = sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2nu))) / 1500
    cT = sqrt(E / (2rho * (1 + nu))) / 1500
    solid = AS.SolidElastic(2.7, cL, cT)
    layer = AS.ElasticLayer(2.7, cL, cT)
    incidence = pi / 3
    k = 1.0
    directions = (back = (2pi / 3, pi), forward = (pi / 3, 0.0), side = (pi / 2, pi / 2))

    function errors(solution, boundary, body)
        return map(directions) do (angle, azimuth)
            reference = AS.form_function(boundary, k, body; incidence_angle = incidence,
                incidence_azimuth = 0.0, scatter_angle = angle, scatter_azimuth = azimuth,
                check = false)
            value = AS.scattering_amplitude(solution; angle, azimuth)
            abs(value - reference) / abs(reference)
        end
    end

    @time "Spheroid volume FEM (solid, prolate, vs transition matrix)" @testset "Spheroid volume FEM (solid, prolate, vs transition matrix)" begin
        body = AS.Spheroid(1.5, 0.5)
        solution = AS.fem(body, solid, k; method = :volume, incidence_angle = incidence)
        e = errors(solution, solid, body)
        @test e.back < 5e-3
        @test e.side < 5e-3
        @test e.forward < 0.05
    end

    @time "Spheroid volume FEM (solid, oblate, vs transition matrix)" @testset "Spheroid volume FEM (solid, oblate, vs transition matrix)" begin
        body = AS.Spheroid(0.6, 1.2)
        solution = AS.fem(body, solid, k; method = :volume, incidence_angle = incidence)
        e = errors(solution, solid, body)
        @test e.back < 5e-3
        @test e.side < 5e-3
        @test e.forward < 0.05
    end

    @time "Spheroid volume FEM (empty elastic shell, vs transition matrix)" @testset "Spheroid volume FEM (empty elastic shell, vs transition matrix)" begin
        body = AS.Spheroid(1.5, 1.0)
        boundary = AS.Shelled(layer, AS.VacuumInterior(), 0.8)
        solution = AS.fem(body, boundary, k; method = :volume, incidence_angle = incidence,
            h_body = 0.16)
        e = errors(solution, boundary, body)
        @test maximum(e) < 0.08
    end
end

let
    E, nu, rho = 70e9, 0.33, 2700.0
    cL = sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2nu))) / 1500
    cT = sqrt(E / (2rho * (1 + nu))) / 1500
    solid = AS.SolidElastic(2.7, cL, cT)
    incidence = pi / 3
    directions = ((2pi / 3, pi), (pi / 3, 0.0), (pi / 2, pi / 2))

    for (label, body, k) in (("prolate", AS.Spheroid(1.5, 0.5), 2.0),
        ("oblate", AS.Spheroid(0.6, 1.2), 1.5))
        @time "Spheroid volume FEM (solid, $label, confocal PML, vs transition matrix)" @testset "Spheroid volume FEM (solid, $label, confocal PML, vs transition matrix)" begin
            solution = AS.fem(body, solid, k; method = :volume, closure = :pml_spheroidal,
                incidence_angle = incidence, points_per_wavelength = 6)
            @test AS.diagnostics(solution).shape ===
                  (label == "prolate" ? :prolate : :oblate)
            for (angle, azimuth) in directions
                reference = AS.form_function(solid, k, body; incidence_angle = incidence,
                    incidence_azimuth = 0.0, scatter_angle = angle, scatter_azimuth = azimuth)
                value = AS.scattering_amplitude(solution; angle, azimuth)
                @test abs(value - reference) / abs(reference) < 0.08
            end
        end
    end
end

let
    c0 = 1477.4
    k = 2pi * 2250.0 / c0
    materials = [AS.FluidFilled(1.04, 1.04), AS.GasFilled(0.00129, 0.23)]
    tilt = deg2rad(10)

    @time "Spheroid volume FEM (fish with tilted bladder, vs coupled BEM)" @testset "Spheroid volume FEM (fish with tilted bladder, vs coupled BEM)" begin
        solution = AS.fem(
            [AS.Spheroid(0.1, 0.02), AS.Spheroid(0.025, 0.007)], materials, k;
            method = :volume, incidence_angle = pi / 2,
            centers = [[0.0, 0.0, 0.0], [0.003, 0.0, 0.010]],
            orientations = [[0.0, 0.0, 1.0], [sin(tilt), 0.0, cos(tilt)]])
        options = (; resolution = 0.4, qorder = 5, tip_ratio = 0.4)
        surfaces = [AS.mesh(; semiaxes = (0.1, 0.02, 0.02), options...),
            AS.mesh(; semiaxes = (0.025, 0.007, 0.007), center = (0.010, 0.003, 0),
                rotation = (axis = (0, 0, 1), angle = tilt), options...)]
        reference = AS.bem(
            surfaces, materials, k; parents = [0, 1], incidence_angle = pi / 2)
        a, b = AS.scattering_amplitude(solution), AS.scattering_amplitude(reference)
        @test abs(a - b) / abs(b) < 0.03
        @test abs(AS.target_strength(solution) - AS.target_strength(reference)) < 0.3
    end
end

let
    E, nu, rho = 70e9, 0.33, 2700.0
    cL = sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2nu))) / 1500
    cT = sqrt(E / (2rho * (1 + nu))) / 1500
    k, incidence = 1.5, pi / 3

    @time "Spheroid volume FEM (elastic shell as regions, vs transition matrix)" @testset "Spheroid volume FEM (elastic shell as regions, vs transition matrix)" begin
        body = AS.Spheroid(1.5, 1.0)
        boundary = AS.Shelled(AS.ElasticLayer(2.7, cL, cT), AS.FluidInterior(1.0, 1.0), 0.7)
        axes = AS._volume_geometry(body, boundary, k).inner_axes
        solution = AS.fem([body, AS.Spheroid(axes[2], axes[1])],
            [AS.SolidElastic(2.7, cL, cT), AS.FluidFilled(1.0, 1.0)], k; method = :volume,
            incidence_angle = incidence)
        for (angle, azimuth) in ((2pi / 3, pi), (pi / 3, 0.0), (pi / 2, pi / 2))
            reference = AS.form_function(boundary, k, body; incidence_angle = incidence,
                incidence_azimuth = 0.0, scatter_angle = angle, scatter_azimuth = azimuth,
                check = false)
            value = AS.scattering_amplitude(solution; angle, azimuth)
            @test abs(value - reference) / abs(reference) < 0.08
        end
    end
end
