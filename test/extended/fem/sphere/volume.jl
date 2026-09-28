using AcousticScattering
using Test

const AS = AcousticScattering

let
    E, nu, rho = 70e9, 0.33, 2700.0
    cL = sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2nu))) / 1500
    cT = sqrt(E / (2rho * (1 + nu))) / 1500
    layer = AS.ElasticLayer(2.7, cL, cT)
    sphere = AS.Sphere(1.0)
    k = 1.5

    @time "Sphere volume FEM (fluid-filled elastic shell, vs modal)" @testset "Sphere volume FEM (fluid-filled elastic shell, vs modal)" begin
        boundary = AS.Shelled(layer, AS.FluidInterior(1.0, 1.0), 0.8)
        solution = AS.fem(sphere, boundary, k; method = :volume, incidence_angle = 0.0,
            h_body = 0.16)
        for angle in (0.0, pi / 2, pi)
            reference = AS.scattering_amplitude(AS.modal(sphere, boundary, k; angle))
            value = AS.scattering_amplitude(solution; angle)
            @test abs(value - reference) / abs(reference) < 0.03
        end
    end
end
