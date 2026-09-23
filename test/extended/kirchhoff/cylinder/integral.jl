using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Bent-cylinder Kirchhoff (physical optics)" @testset "Bent-cylinder Kirchhoff (physical optics)" begin
        radius, length = 0.01, 0.07
        c = 1477.3
        k = 2π * 38000.0 / c

        function _lateral_only(boundary, k, radius, length; angle = π / 2)
            Rc = AS.reflection_coefficient(boundary)
            sb, cb = sincos(angle)
            x = 2k * radius * sb
            total = 2besselj(0, x) + im * π * besselj(1, x)
            for n in 1:60
                term = besselj(2n, x) / (4n^2 - 1)
                total -= 4term
                abs(term) < 1e-15 * abs(total) && break
            end
            return Rc * (k * radius * length) / (2π) * sb * total *
                   sinc(k * length * cb / π)
        end

        cyl_nearly_straight = AS.Cylinder(radius, length; radius_curvature = 1e8length)
        for angle in (π / 2, 1.2, 1.0)
            f_ref = _lateral_only(AS.Rigid(), k, radius, length; angle = angle)
            f_bent = AS.scattering_amplitude(AS.kirchhoff(
                cyl_nearly_straight, AS.Rigid(), k;
                incidence_angle = angle))
            @test f_bent ≈ f_ref rtol = 1e-6
        end
    end
end
