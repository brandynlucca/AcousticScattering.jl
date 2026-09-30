using AcousticScattering
using Test

@testset "Bounded axisymmetric BEM mode storage" begin
    AS = AcousticScattering
    angles = [0.0, pi / 3, pi / 2, pi, -0.3, pi / 3]
    for body in (Sphere(1.0), Spheroid(1.4, 0.7)),
        boundary in (Rigid(), PressureRelease(), Impedance(1.3 - 0.6im)),
        m_max in (0, 8, 16)
        k, n, rtol = 0.7, 12, 1e-5
        mesh = AS._axisymmetric_mesh(body, n)
        ps, factors = AS._oblique_mode_factors(boundary, k, mesh; m_max, rtol)
        result = incidence_angle_sweep(body, boundary, k, angles;
            method = :axisymmetric, n, m_max, rtol)
        @test result.angles == angles
        @test result.labels == ["Scattered field"]
        @test result.amplitudes[2] == result.amplitudes[end]
        for (i, beta) in enumerate(angles)
            reports = AS._SolveReports()
            p, dp = AS._oblique_solve_factored(boundary, k, ps, factors, beta;
                rtol, solve_reports = reports)
            wanted = m_max == 0 ? AS.far_field(ps, only(p), only(dp), k, pi-beta) :
                     AS.far_field(ps, p, dp, k, pi-beta, pi)
            @test result.amplitudes[i]≈wanted rtol=1e-8 atol=1e-12
            @test result.target_strength[i] ≈ target_strength(wanted) atol = 1e-7
            if boundary isa Impedance
                streamed_reports = AS._SolveReports()
                actual = AS.solve_oblique(boundary, k, mesh, beta;
                    m_max, rtol, solve_reports = streamed_reports)
                @test actual[1] == p
                @test actual[2] == dp
                @test streamed_reports.systems == reports.systems
            end
        end
    end
    body = Cylinder(0.5, 1.0)
    for boundary in (Rigid(), PressureRelease(), Impedance(1.3-0.6im))
        sweep = incidence_angle_sweep(body, boundary, 0.7, angles; n = 12, m_max = 8)
        reversed = incidence_angle_sweep(
            body, boundary, 0.7, reverse(angles); n = 12, m_max = 8)
        @test reversed.amplitudes == reverse(sweep.amplitudes)
        @test all(isfinite, sweep.target_strength)
    end
    @test_throws ArgumentError incidence_angle_sweep(body, Rigid(), 0.7, angles; m_max = -1)
    @test_throws ArgumentError incidence_angle_sweep(body, Rigid(), 0.7, Float64[])
    @test_throws ArgumentError incidence_angle_sweep(body, Rigid(), 0.7, [NaN])
    @test_throws ArgumentError incidence_angle_sweep(
        Cylinder(0.1, 0.5; radius_curvature = 2.0), Rigid(), 0.7, [0.3])

    # Force the cancellation safeguard and check only the flagged angle is rebuilt.
    mesh = AS._axisymmetric_mesh(Sphere(1.0), 12)
    samples = [0.3, 0.7]
    amplitudes = ComplexF64[99 + im, 77 - im]
    AS._axisymmetric_sweep_check!(amplitudes, [Inf, 0.0], Rigid(), 0.7, mesh, samples;
        m_max = 8, rtol = 1e-5)
    ps, factors = AS._oblique_mode_factors(Rigid(), 0.7, mesh; m_max = 8)
    p, dp = AS._oblique_solve_factored(Rigid(), 0.7, ps, factors, first(samples))
    @test amplitudes[1] == AS.far_field(ps, p, dp, 0.7, pi-first(samples), pi)
    @test amplitudes[2] == 77-im
    # The factor batch preserves absolute Fourier indices and retains only its range.
    for boundary in (Rigid(), PressureRelease(), Impedance(1.3-0.6im))
        _, full = AS._oblique_mode_factors(boundary, 0.7, mesh; m_max = 16)
        _, batch = AS._oblique_mode_factors(boundary, 0.7, mesh; m_max = 15, modes = 8:15)
        @test length(batch) == AS._MODE_CHUNK
        @test Base.summarysize(batch) < 0.5 * Base.summarysize(full)
        full_reports, batch_reports = AS._SolveReports(), AS._SolveReports()
        p, dp = AS._oblique_solve_factored(
            boundary, 0.7, ps, full, 0.3; solve_reports = full_reports)
        pb, dpb = AS._oblique_solve_factored(boundary, 0.7, ps, batch, 0.3;
            first_mode = 8, solve_reports = batch_reports)
        @test pb == p[9:16]
        @test dpb == dp[9:16]
        @test batch_reports.systems == full_reports.systems[9:16]
    end
end

@testset "Far-field Fourier batches" begin
    AS = AcousticScattering
    ps = AS.panels(AS._axisymmetric_mesh(Spheroid(1.4, 0.7), 12))
    p = [ComplexF64[cis(0.2j+0.3m)/(m+1) for j in eachindex(ps)] for m in 0:16]
    dp = [ComplexF64[cis(0.7j-0.1m)/(m+1) for j in eachindex(ps)] for m in 0:16]
    for theta in (0.0, 1e-8, 0.7, pi, -0.4), phi in (0.0, 0.4, pi), k in (0.01, 3.0)
        whole = AS.far_field(ps, p, dp, k, theta, phi)
        pieces = sum(AS._far_field_modes(ps, p[(first + 1):(last + 1)],
                         dp[(first + 1):(last + 1)], k, theta, phi, first)
        for (first, last) in ((0, 7), (8, 15), (16, 16)))
        @test pieces≈whole rtol=1e-8 atol=1e-12
        value, error = AS._far_field_modes(ps, p, dp, k, theta, phi, 0; return_error = true)
        @test value == whole
        @test isfinite(error) && error >= 0
    end
end
