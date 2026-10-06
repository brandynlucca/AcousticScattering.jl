using AcousticScattering
using Test

let AS = AcousticScattering
    let
        k = 2.0
        solid = AS.SolidElastic(2.7, 2.0, 1.0)
        flesh = AS.FluidFilled(1.05, 1.05)

        @time "Volume FEM (rotation of an inclusion in flesh)" @testset "Volume FEM (rotation of an inclusion in flesh)" begin
            bodies = [AS.Sphere(0.5), AS.Spheroid(0.2, 0.08)]
            center, axis = [0.15, 0.0, 0.1], [sin(0.5), 0.0, cos(0.5)]
            direction = [sin(pi / 3) * cos(0.4), sin(pi / 3) * sin(0.4), cos(pi / 3)]
            th, ph = 0.7, 0.3
            Q = [cos(th) 0 sin(th); 0 1 0; -sin(th) 0 cos(th)] *
                [cos(ph) -sin(ph) 0; sin(ph) cos(ph) 0; 0 0 1]
            amplitude(c, a, d) = AS.scattering_amplitude(AS.fem(bodies, [flesh, solid], k;
                method = :volume, closure = :dtn, points_per_wavelength = 6,
                centers = [zeros(3), c], orientations = [[0.0, 0.0, 1.0], a],
                incidence_angle = acos(d[3]), incidence_azimuth = atan(d[2], d[1])))
            reference = amplitude(center, axis, direction)
            @test amplitude(Q * center, Q * axis, Q * direction) ≈ reference rtol = 0.01
        end
    end
end
