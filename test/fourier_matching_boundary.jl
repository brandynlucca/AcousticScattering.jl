using AcousticScattering
using Test
using LinearAlgebra: BLAS

const AS = AcousticScattering

BLAS.set_num_threads(1)

@time @testset "Fourier matching: pressure-release, sphere diagonal exact match" begin
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

@time @testset "Fourier matching: pressure-release, sphere full pipeline (oblique, bistatic)" begin
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

@time @testset "Fourier matching: pressure-release, spheroid vs. independent axisymmetric BEM" begin
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

@time @testset "Fourier matching: rigid, sphere diagonal exact match" begin
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

@time @testset "Fourier matching: rigid, sphere full pipeline (oblique, bistatic)" begin
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

@time @testset "Fourier matching: rigid, spheroid vs. independent axisymmetric BEM" begin
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

@time @testset "Fourier matching: fluid, sphere diagonal exact match" begin
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

@time @testset "Fourier matching: fluid, sphere full pipeline (weak and gas contrast)" begin
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
        bcoef = AS.solve_fluid(mapping, k, theta0, dens, ss; m_max = n_max, n_max = n_max)
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

@time @testset "Fourier matching: fluid, spheroid vs. independent axisymmetric BEM" begin
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

@time @testset "Fourier matching: public API, fourier(Irregular, ...) against modal(Sphere, ...)" begin
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

@time @testset "Fourier matching: incidence_angle_sweep reuses the transition operator" begin
    a = 2.0
    k = 1.3
    body = Irregular(a, Float64[], Float64[])
    angles = [pi / 6, pi / 3, pi / 2, 2pi / 3]
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
        sweep = incidence_angle_sweep(body, boundary, k, angles;
            mapping_order = 4, continuation_steps = 1)
        for (i, angle) in enumerate(angles)
            individual = fourier(body, boundary, k; incidence_angle = angle,
                mapping_order = 4, continuation_steps = 1)
            @test sweep.target_strength[i] ≈ target_strength(individual) atol = 1e-10
            @test sweep.amplitudes[i] ≈ scattering_amplitude(individual) atol = 1e-10
        end
    end
    @test_throws ArgumentError incidence_angle_sweep(body, Rigid(), k, Float64[])
end
