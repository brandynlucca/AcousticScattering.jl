using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Fourier matching: public API, fourier(Irregular, ...) against modal(Sphere, ...)" @testset "Fourier matching: public API, fourier(Irregular, ...) against modal(Sphere, ...)" begin
        a = 2.0
        k = 1.3
        theta0 = pi / 3
        body = Irregular(a, Float64[], Float64[])
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
            sol = fourier(body, boundary, k; incidence_angle = theta0, mapping_order = 4,
                continuation_steps = 1)
            @test diagnostics(sol).admissible
            # Sphere modal's `angle` is the incidence-observation angle, so backscatter is always pi, matching sol's default.
            ref = modal(Sphere(a), boundary, k; angle = pi)
            @test abs(target_strength(sol) - target_strength(ref)) < 0.01
            @test abs(scattering_amplitude(sol) - scattering_amplitude(ref)) /
                  abs(scattering_amplitude(ref)) < 0.01
        end
        @test_throws ArgumentError fourier(body, Rigid(), -1.0)
    end
end

let
    @time "Fourier matching: incidence_angle_sweep reuses the transition operator" @testset "Fourier matching: incidence_angle_sweep reuses the transition operator" begin
        a = 2.0
        k = 1.3
        body = Irregular(a, Float64[], Float64[])
        angles = [pi / 6, pi / 3, pi / 2, 2pi / 3]
        # This checks transition reuse, not truncation convergence; the public API test above
        # exercises the default order against the exact sphere solution.
        m_max = n_max = 6
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
            sweep = incidence_angle_sweep(body, boundary, k, angles;
                mapping_order = 4, continuation_steps = 1, m_max, n_max)
            for (i, angle) in enumerate(angles)
                individual = fourier(body, boundary, k; incidence_angle = angle,
                    mapping_order = 4, continuation_steps = 1, m_max, n_max)
                @test sweep.target_strength[i] ≈ target_strength(individual) atol = 1e-10
                @test sweep.amplitudes[i] ≈ scattering_amplitude(individual) atol = 1e-10
            end
        end
        @test_throws ArgumentError incidence_angle_sweep(body, Rigid(), k, Float64[])
    end
end

let
    @time "Fourier matching: mapping order convergence" @testset "Fourier matching: mapping order convergence" begin
        a, b = 3.0, 1.0
        Rtheta(theta) = 1 / sqrt((cos(theta) / a)^2 + (sin(theta) / b)^2)

        function ellipse_residual(order)
            profile = AS.Irregular(Rtheta, order; npoints = 800)
            mapping = AS.solve_mapping(profile, order; continuation_steps = 16)
            return maximum(range(0, π; length = 37)) do w
                g, f = AS.mapping_surface(mapping, w)
                abs((g / a)^2 + (f / b)^2 - 1)
            end
        end

        residuals = [ellipse_residual(order) for order in (8, 16, 24, 32)]
        # Monotonic, and each doubling of order should cut the residual by well over an order of magnitude, per both source papers; a plateauing or non-monotonic sequence would indicate a bug.
        @test issorted(residuals; rev = true)
        @test residuals[end] < residuals[1] / 100
    end

    @time "Fourier matching: noncanonical bumpy profile" @testset "Fourier matching: noncanonical bumpy profile" begin
        a = 1.0
        Rtheta(theta) = a * (1 + 0.12 * cos(3theta) - 0.05 * sin(2theta))
        order = 24
        profile = AS.Irregular(Rtheta, order; npoints = 400)
        mapping = AS.solve_mapping(profile, order; continuation_steps = 12)

        @test AS.is_admissible(mapping)
        # Self-consistency: the mapped radial distance from the origin at w must match the supplied profile evaluated at theta(w) (Eq. (28)'s relationship, checked pointwise).
        maxerr = maximum(range(0, π; length = 73)) do w
            g, f = AS.mapping_surface(mapping, w)
            theta = AS.mapping_theta(mapping, w)
            abs(hypot(g, f) - Rtheta(theta))
        end
        @test maxerr < 1e-3

        # Convergence: more modes should reduce the same self-consistency residual.
        function self_consistency_residual(ord)
            p = AS.Irregular(Rtheta, ord; npoints = 400)
            m = AS.solve_mapping(p, ord; continuation_steps = 12)
            return maximum(range(0, π; length = 37)) do w
                g, f = AS.mapping_surface(m, w)
                abs(hypot(g, f) - Rtheta(AS.mapping_theta(m, w)))
            end
        end
        coarse = self_consistency_residual(8)
        fine = self_consistency_residual(24)
        @test fine < coarse / 20
    end

    @time "Fourier matching: input validation and admissibility rejection" @testset "Fourier matching: input validation and admissibility rejection" begin
        @test_throws ArgumentError AS.Irregular(1.0, [1.0], Float64[])
        @test_throws ArgumentError AS.Irregular(-1.0, Float64[], Float64[])
        @test_throws ArgumentError AS.Irregular(theta -> 1.0, -1)
        @test_throws ArgumentError AS.solve_mapping(AS.Irregular(1.0, Float64[], Float64[]), 0)
        @test_throws ArgumentError AS.solve_mapping(
            AS.Irregular(1.0, Float64[], Float64[]), 4; continuation_steps = 0)

        # Directly construct a mapping whose Jacobian is known analytically to vanish (c_(-1)=1, c_1=1, else 0, so dG/drho|_{u=0}=2im*sin(w)=0 at w=0,pi), to test is_admissible in isolation from solve_mapping.
        dummy_profile = AS.Irregular(1.0, Float64[], Float64[])
        inadmissible = AS.ConformalMapping(
            dummy_profile, Float64[], Float64[], ComplexF64[1.0, 0.0, 1.0])
        @test AS.mapping_jacobian_squared(inadmissible, 0.0) == 0
        @test !AS.is_admissible(inadmissible)
    end
end
