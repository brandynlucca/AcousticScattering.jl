using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Fourier matching: pressure-release, spheroid vs. independent axisymmetric BEM" @testset "Fourier matching: pressure-release, spheroid vs. independent axisymmetric BEM" begin
        a, b = 1.5, 1.0
        k = 1.0
        theta0 = pi / 3
        mapping_order = 24
        Rtheta(theta) = 1 / sqrt((cos(theta) / a)^2 + (sin(theta) / b)^2)
        profile = AS.Irregular(Rtheta, mapping_order; npoints = 800)
        mapping = AS.solve_mapping(profile, mapping_order; continuation_steps = 12)
        @test AS.is_admissible(mapping)

        n_max = 6
        bcoef = AS.solve_pressure_release(mapping, k, theta0; m_max = n_max, n_max = n_max)

        spheroid = Spheroid(a, b)
        boundary = PressureRelease()
        bem_sol = AS.bem(spheroid, boundary, k;
            method = :axisymmetric, incidence_angle = theta0, n = 160, m_max = n_max)
        for (angle, azimuth) in (
            (pi - theta0, pi), # backscatter
            (theta0, 0.0), # forward
            (pi / 2, pi / 2), # side, off incidence plane
            (2pi / 3, pi / 4) # side, another azimuth
        )
            actual = AS.fourier_matching_amplitude(bcoef, k, angle, azimuth)
            expected = AS.scattering_amplitude(bem_sol; angle, azimuth)
            @test abs(AS.target_strength(actual) - AS.target_strength(expected)) < 0.1
            @test abs(actual - expected) / abs(expected) < 0.01
        end
    end

    @time "Fourier matching: rigid, spheroid vs. independent axisymmetric BEM" @testset "Fourier matching: rigid, spheroid vs. independent axisymmetric BEM" begin
        a, b = 1.5, 1.0
        k = 1.0
        theta0 = pi / 3
        mapping_order = 24
        Rtheta(theta) = 1 / sqrt((cos(theta) / a)^2 + (sin(theta) / b)^2)
        profile = AS.Irregular(Rtheta, mapping_order; npoints = 800)
        mapping = AS.solve_mapping(profile, mapping_order; continuation_steps = 12)
        @test AS.is_admissible(mapping)

        n_max = 6
        bcoef = AS.solve_rigid(mapping, k, theta0; m_max = n_max, n_max = n_max)

        spheroid = Spheroid(a, b)
        boundary = Rigid()
        bem_sol = AS.bem(spheroid, boundary, k;
            method = :axisymmetric, incidence_angle = theta0, n = 160, m_max = n_max)
        for (angle, azimuth) in (
            (pi - theta0, pi), # backscatter
            (theta0, 0.0), # forward
            (pi / 2, pi / 2), # side, off incidence plane
            (2pi / 3, pi / 4) # side, another azimuth
        )
            actual = AS.fourier_matching_amplitude(bcoef, k, angle, azimuth)
            expected = AS.scattering_amplitude(bem_sol; angle, azimuth)
            @test abs(AS.target_strength(actual) - AS.target_strength(expected)) < 0.1
            @test abs(actual - expected) / abs(expected) < 0.01
        end
    end

    @time "Fourier matching: fluid, spheroid vs. independent axisymmetric BEM" @testset "Fourier matching: fluid, spheroid vs. independent axisymmetric BEM" begin
        a, b = 1.5, 1.0
        k = 1.0
        theta0 = pi / 3
        mapping_order = 24
        Rtheta(theta) = 1 / sqrt((cos(theta) / a)^2 + (sin(theta) / b)^2)
        profile = AS.Irregular(Rtheta, mapping_order; npoints = 800)
        mapping = AS.solve_mapping(profile, mapping_order; continuation_steps = 12)
        @test AS.is_admissible(mapping)

        n_max = 6
        dens, ss = 1.05, 1.02
        bcoef = AS.solve_fluid(mapping, k, theta0, dens, ss; m_max = n_max, n_max = n_max)

        spheroid = Spheroid(a, b)
        boundary = FluidFilled(dens, ss)
        bem_sol = AS.bem(spheroid, boundary, k;
            method = :axisymmetric, incidence_angle = theta0, n = 160, m_max = n_max)
        for (angle, azimuth) in (
            (pi - theta0, pi), # backscatter
            (theta0, 0.0), # forward
            (pi / 2, pi / 2), # side, off incidence plane
            (2pi / 3, pi / 4) # side, another azimuth
        )
            actual = AS.fourier_matching_amplitude(bcoef, k, angle, azimuth)
            expected = AS.scattering_amplitude(bem_sol; angle, azimuth)
            @test abs(AS.target_strength(actual) - AS.target_strength(expected)) < 0.1
            @test abs(actual - expected) / abs(expected) < 0.01
        end
    end
end

let
    @time "Fourier matching: truncation guard against axisymmetric BEM across aspect ratio" @testset "Fourier matching: truncation guard against axisymmetric BEM across aspect ratio" begin
        theta0 = pi / 3
        directions = ((pi - theta0, pi), (theta0, 0.0), (pi / 2, pi / 2))
        # (aspect ratio, ka, boundary, n_max) spanning converged, slowly converging and diverging truncations.
        cases = ((2.0, 3.0, PressureRelease(), 12), (2.0, 3.0, Rigid(), 16),
            (3.0, 3.0, PressureRelease(), 16), (3.0, 3.0, Rigid(), 10),
            (2.0, 3.0, FluidFilled(1.05, 1.02), 10), (5.0, 3.0, PressureRelease(), 8))
        for (aspect, ka, boundary, n_max) in cases
            k = ka / aspect
            body = Spheroid(aspect, 1.0)
            bem_sol = AS.bem(body, boundary, k; method = :axisymmetric,
                incidence_angle = theta0, n = 240, m_max = 12)
            logs, solution = Test.collect_test_logs() do
                fourier(body, boundary, k; incidence_angle = theta0, m_max = n_max, n_max)
            end
            report = diagnostics(solution)
            rel, db = 0.0, 0.0
            for (angle, azimuth) in directions
                expected = AS.scattering_amplitude(bem_sol; angle, azimuth)
                actual = AS.scattering_amplitude(solution; angle, azimuth)
                rel = max(rel, abs(actual - expected) / abs(expected))
                db = max(db, abs(AS.target_strength(actual) - AS.target_strength(expected)))
            end
            # The guard must never pass a solution outside the 1% amplitude and 0.1 dB gates.
            report.converged && @test rel < 0.01 && db < 0.1
            rel >= 0.01 && @test !report.converged
            @test any(log -> log.level >= Base.CoreLogging.Warn, logs) == !report.converged
        end
    end
end

let
    @time "Fourier matching: prolate spheroid, ellipse-equation check" @testset "Fourier matching: prolate spheroid, ellipse-equation check" begin
        a, b = 3.0, 1.0
        order = 32
        Rtheta(theta) = 1 / sqrt((cos(theta) / a)^2 + (sin(theta) / b)^2)
        profile = AS.Irregular(Rtheta, order; npoints = 800)
        mapping = AS.solve_mapping(profile, order; continuation_steps = 16)

        @test AS.is_admissible(mapping)
        for w in range(0, π; length = 73)
            g, f = AS.mapping_surface(mapping, w)
            # Independent check: the reconstructed (g,f) must satisfy the exact ellipse equation, without reference to theta(w) or the profile fit at all.
            @test (g / a)^2 + (f / b)^2 ≈ 1 atol = 1e-4
        end
    end
end

let
    @time "Fourier matching: default settings for elongated spheroids against axisymmetric BEM" @testset "Fourier matching: default settings for elongated spheroids against axisymmetric BEM" begin
        theta0 = pi / 3
        directions = ((pi - theta0, pi), (theta0, 0.0), (pi / 2, pi / 2))
        # (aspect ratio, ka, boundary) at the default mapping order and mode count.
        cases = ((3.0, 3.0, Rigid()), (3.0, 1.0, FluidFilled(1.05, 1.02)),
            (4.0, 3.0, PressureRelease()), (5.0, 3.0, PressureRelease()),
            (5.0, 1.0, FluidFilled(1.05, 1.02)))
        for (aspect, ka, boundary) in cases, body in (Spheroid(aspect, 1.0),)

            k = ka / aspect
            bem_sol = AS.bem(body, boundary, k; method = :axisymmetric,
                incidence_angle = theta0, n = 240, m_max = 12)
            solution = fourier(body, boundary, k; incidence_angle = theta0)
            @test diagnostics(solution).converged
            for (angle, azimuth) in directions
                expected = AS.scattering_amplitude(bem_sol; angle, azimuth)
                actual = AS.scattering_amplitude(solution; angle, azimuth)
                @test abs(AS.target_strength(actual) - AS.target_strength(expected)) < 0.1
                @test abs(actual - expected) / abs(expected) < 0.01
            end
        end
    end
end
