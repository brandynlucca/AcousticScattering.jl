using AcousticScattering
using Test

let AS = AcousticScattering
    let
        k = 2.0
        solid = AS.SolidElastic(2.7, 2.0, 1.0)
        flesh = AS.FluidFilled(1.05, 1.05)

        @time "Volume FEM (elastic shell as regions)" @testset "Volume FEM (elastic shell as regions)" begin
            sphere = AS.Sphere(1.0)
            layer = AS.ElasticLayer(2.7, 2.0, 1.0)
            boundary = AS.Shelled(layer, AS.FluidInterior(1.0, 1.0), 0.8)
            solution = AS.fem(
                [sphere, AS.Sphere(0.8)], [solid, AS.FluidFilled(1.0, 1.0)], 1.5;
                method = :volume, incidence_angle = 0.0, closure = :dtn)
            for angle in (0.0, pi / 2, pi)
                reference = AS.scattering_amplitude(AS.modal(sphere, boundary, 1.5; angle))
                value = AS.scattering_amplitude(solution; angle)
                @test abs(value - reference) / abs(reference) < 0.05
            end
        end

        @testset "Volume FEM (region material validation)" begin
            @test_throws ArgumentError AS.fem([AS.Sphere(0.5)], [AS.Rigid()], k; method = :volume)
            @test_throws ArgumentError AS.fem([AS.Sphere(0.5), AS.Cylinder(0.1, 0.3)],
                [flesh, flesh], k; method = :volume)
        end
    end
end
