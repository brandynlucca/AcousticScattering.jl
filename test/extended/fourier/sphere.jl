using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Fourier matching: pressure-release, sphere diagonal exact match" @testset "Fourier matching: pressure-release, sphere diagonal exact match" begin
        a = 2.0
        k = 1.3
        profile = AS.Irregular(a, Float64[], Float64[])
        mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

        n_max = 6
        for m in 0:2
            R, Q = AS._boundary_matrices(mapping, k, m; n_max)
            @test maximum(abs, R - AS.Diagonal(AS.diag(R))) < 1e-10
            @test maximum(abs, Q - AS.Diagonal(AS.diag(Q))) < 1e-10
            for (i, n) in enumerate(m:n_max)
                ratio = -R[i, i] / Q[i, i]
                expected = AS._modal_coefficient(PressureRelease(), n, k, a)
                @test ratio ≈ expected atol = 1e-8
            end
        end
    end

    @time "Fourier matching: pressure-release, sphere full pipeline (oblique, bistatic)" @testset "Fourier matching: pressure-release, sphere full pipeline (oblique, bistatic)" begin
        a = 2.0
        k = 1.3
        theta0 = pi / 3
        profile = AS.Irregular(a, Float64[], Float64[])
        mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

        n_max = 10
        bcoef = AS.solve_pressure_release(mapping, k, theta0; m_max = n_max, n_max = n_max)

        sphere = Sphere(a)
        boundary = PressureRelease()
        d_inc = [cos(theta0), sin(theta0), 0.0]
        for (angle, azimuth) in (
            (pi - theta0, pi), # backscatter
            (theta0, 0.0), # forward
            (pi / 2, pi / 2) # broadside, off incidence plane
        )
            actual = AS.fourier_matching_amplitude(bcoef, k, angle, azimuth)
            d_obs = [cos(angle), sin(angle) * cos(azimuth), sin(angle) * sin(azimuth)]
            cospsi = clamp(sum(d_inc .* d_obs), -1.0, 1.0)
            # A sphere's response depends only on the angle between incidence and observation.
            expected = scattering_amplitude(modal(sphere, boundary, k; angle = acos(cospsi)))
            @test abs(target_strength(actual) - target_strength(expected)) < 0.01
            @test abs(actual - expected) / abs(expected) < 1e-3
        end
    end

    @time "Fourier matching: rigid, sphere diagonal exact match" @testset "Fourier matching: rigid, sphere diagonal exact match" begin
        a = 2.0
        k = 1.3
        profile = AS.Irregular(a, Float64[], Float64[])
        mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

        n_max = 6
        for m in 0:2
            R, Q = AS._rigid_boundary_matrices(mapping, k, m; n_max)
            @test maximum(abs, R - AS.Diagonal(AS.diag(R))) < 1e-8
            @test maximum(abs, Q - AS.Diagonal(AS.diag(Q))) < 1e-8
            for (i, n) in enumerate(m:n_max)
                ratio = -R[i, i] / Q[i, i]
                expected = AS._modal_coefficient(Rigid(), n, k, a)
                @test ratio ≈ expected atol = 1e-6
            end
        end
    end

    @time "Fourier matching: rigid, sphere full pipeline (oblique, bistatic)" @testset "Fourier matching: rigid, sphere full pipeline (oblique, bistatic)" begin
        a = 2.0
        k = 1.3
        theta0 = pi / 3
        profile = AS.Irregular(a, Float64[], Float64[])
        mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

        n_max = 10
        bcoef = AS.solve_rigid(mapping, k, theta0; m_max = n_max, n_max = n_max)

        sphere = Sphere(a)
        boundary = Rigid()
        d_inc = [cos(theta0), sin(theta0), 0.0]
        for (angle, azimuth) in (
            (pi - theta0, pi), # backscatter
            (theta0, 0.0), # forward
            (pi / 2, pi / 2) # broadside, off incidence plane
        )
            actual = AS.fourier_matching_amplitude(bcoef, k, angle, azimuth)
            d_obs = [cos(angle), sin(angle) * cos(azimuth), sin(angle) * sin(azimuth)]
            cospsi = clamp(sum(d_inc .* d_obs), -1.0, 1.0)
            expected = scattering_amplitude(modal(sphere, boundary, k; angle = acos(cospsi)))
            @test abs(target_strength(actual) - target_strength(expected)) < 0.01
            @test abs(actual - expected) / abs(expected) < 1e-3
        end
    end

    @time "Fourier matching: fluid, sphere diagonal exact match" @testset "Fourier matching: fluid, sphere diagonal exact match" begin
        a = 2.0
        k = 1.3
        profile = AS.Irregular(a, Float64[], Float64[])
        mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

        n_max = 6
        for (dens, ss) in ((1.05, 1.02), (0.00126, 0.22)), m in 0:2

            bc = FluidFilled(dens, ss)
            R, Q = AS._boundary_matrices(mapping, k, m; n_max)
            Rp, Qp = AS._rigid_boundary_matrices(mapping, k, m; n_max)
            S, Sp = AS._interior_boundary_matrices(mapping, k / ss, m; n_max)
            SinvQ = S \ Q
            SinvR = S \ R
            M1 = dens .* Qp .- Sp * SinvQ
            M2 = Sp * SinvR .- dens .* Rp
            ratio = M1 \ M2
            for (i, n) in enumerate(m:n_max)
                expected = AS._modal_coefficient(bc, n, k, a)
                @test ratio[i, i] ≈ expected atol = 1e-6
            end
        end
    end

    @time "Fourier matching: fluid, sphere full pipeline (weak and gas contrast)" @testset "Fourier matching: fluid, sphere full pipeline (weak and gas contrast)" begin
        a = 2.0
        k = 1.3
        theta0 = pi / 3
        profile = AS.Irregular(a, Float64[], Float64[])
        mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

        n_max = 8
        sphere = Sphere(a)
        d_inc = [cos(theta0), sin(theta0), 0.0]
        for (dens, ss) in ((1.05, 1.02), (0.00126, 0.22))
            bc = FluidFilled(dens, ss)
            bcoef = AS.solve_fluid(
                mapping, k, theta0, dens, ss; m_max = n_max, n_max = n_max)
            for (angle, azimuth) in (
                (pi - theta0, pi), # backscatter
                (theta0, 0.0), # forward
                (pi / 2, pi / 2) # broadside, off incidence plane
            )
                actual = AS.fourier_matching_amplitude(bcoef, k, angle, azimuth)
                d_obs = [cos(angle), sin(angle) * cos(azimuth), sin(angle) * sin(azimuth)]
                cospsi = clamp(sum(d_inc .* d_obs), -1.0, 1.0)
                expected = scattering_amplitude(modal(sphere, bc, k; angle = acos(cospsi)))
                @test abs(target_strength(actual) - target_strength(expected)) < 0.01
                @test abs(actual - expected) / abs(expected) < 1e-3
            end
        end
    end
end

let
    @time "Fourier matching: sphere limit" @testset "Fourier matching: sphere limit" begin
        a = 2.0
        profile = AS.Irregular(a, Float64[], Float64[])
        mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

        @test all(iszero, mapping.delta_c)
        @test all(iszero, mapping.delta_s)
        @test mapping.c[1] ≈ a atol = 1e-10
        @test all(c -> abs(c) < 1e-10, mapping.c[2:end])
        @test AS.is_admissible(mapping)

        for w in range(0, π; length = 41)
            @test AS.mapping_theta(mapping, w) ≈ w atol = 1e-12
            g, f = AS.mapping_surface(mapping, w)
            @test g ≈ a * cos(w) atol = 1e-10
            @test f ≈ a * sin(w) atol = 1e-10
        end
    end
end
