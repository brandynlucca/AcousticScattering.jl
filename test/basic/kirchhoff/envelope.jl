using AcousticScattering
using Test

# Quantified Kirchhoff validity envelope against the exact modal series (Jech et al., 2015,
# report a similar pattern: KA matches within about 1 dB near broadside/high ka, and degrades
# off-broadside or at low ka).
@testset "Kirchhoff validity envelope" begin
    @testset "Sphere: low-ka failure and high-ka convergence" begin
        a = 0.01
        for boundary in (Rigid(), PressureRelease())
            low, low_ref = kirchhoff(Sphere(a), boundary, 0.1 / a),
            modal(Sphere(a), boundary, 0.1 / a)
            @test abs(target_strength(low) - target_strength(low_ref)) > 5.0

            high, high_ref = kirchhoff(Sphere(a), boundary, 20.0 / a),
            modal(Sphere(a), boundary, 20.0 / a)
            @test abs(target_strength(high) - target_strength(high_ref)) < 0.5
        end
    end

    @testset "Spheroid: broadside convergence, aspect-ratio insensitivity" begin
        b = 0.01
        k = 10.0 / b
        for aspect in (1.5, 3.0, 8.0)
            body = Spheroid(aspect * b, b)
            solution = kirchhoff(body, Rigid(), k; incidence_angle = pi / 2)
            reference = modal(body, Rigid(), k; incidence_angle = pi / 2)
            @test abs(target_strength(solution) - target_strength(reference)) < 1.0
        end
    end

    @testset "Spheroid: error grows from broadside toward end-on" begin
        b = 0.01
        k = 10.0 / b
        body = Spheroid(3b, b)
        error_at(angle) = abs(target_strength(kirchhoff(body, Rigid(), k; incidence_angle = angle)) -
                              target_strength(modal(body, Rigid(), k; incidence_angle = angle)))
        broadside_error, endon_error = error_at(pi / 2), error_at(0.0)
        @test broadside_error < 0.5
        @test endon_error > broadside_error
    end
end
