using AcousticScattering
using Test

@testset "Analytic azimuthal far-field reduction" begin
    AS = AcousticScattering
    # Independent representation integral, retained as a reference for phases,
    # cosine-mode normalization and both components of the surface normal.
    function integrated(ps, p, dp, k, theta, phi; rtol = 1e-10)
        total = 0.0im
        for (j, panel) in enumerate(ps)
            value, _ = AS.quadgk(0.0, 1.0; rtol, atol = 1e-11) do s
                rho, z = AS._panel_point(panel, s)
                azimuthal, _ = AS.quadgk(0.0, 2pi; rtol, atol = 1e-12) do alpha
                    pressure = sum(p[m][j]*cos((m-1)*alpha) for m in eachindex(p))
                    derivative = sum(dp[m][j]*cos((m-1)*alpha) for m in eachindex(dp))
                    radial = sin(theta)*cos(alpha-phi)
                    phase = cis(-k*(rho*radial + z*cos(theta)))
                    normal = panel.nrho*radial + panel.nz*cos(theta)
                    (derivative + im*k*normal*pressure)*phase*rho
                end
                azimuthal*panel.L
            end
            total += value
        end
        return -total/(4pi)
    end

    for body in (Sphere(1.0), Spheroid(1.5, 0.6), Cylinder(0.7, 2.0))
        ps = AS.panels(mesh(body; resolution = 6).data)
        p = [ComplexF64[cis(0.3j + m)/(m+1) for j in eachindex(ps)] for m in 0:6]
        dp = [ComplexF64[cis(0.7j - m)/(m+1) for j in eachindex(ps)] for m in 0:6]
        saved_p, saved_dp = deepcopy(p), deepcopy(dp)
        for (k, theta, phi) in ((1e-6, 0.7, 0.4), (0.8, 0.0, 1.3),
            (0.8, pi, 0.6), (2.0, pi/2, 0.0), (2.0, 0.9, 1.1),
            (12.0, 1.2, 2.3), (2.0, -0.9, -1.1))
            expected = integrated(ps, p, dp, k, theta, phi)
            actual = AS.far_field(ps, p, dp, k, theta, phi; rtol = 1e-10)
            @test actual≈expected rtol=2e-8 atol=2e-10
        end
        @test p == saved_p
        @test dp == saved_dp
        for theta in (0.0, 0.7, pi)
            @test AS.far_field(ps, [p[1]], [dp[1]], 2.0, theta, 1.3; rtol = 1e-10)≈
            AS.far_field(ps, p[1], dp[1], 2.0, theta) rtol=1e-9 atol=1e-12
        end
        # Pure modes isolate the phase and azimuthal parity of each order.
        for m in (1, 2, 3, 8, 24)
            pure_p = [zeros(ComplexF64, length(ps)) for _ in 0:m]
            pure_dp = deepcopy(pure_p)
            pure_p[end] .= p[1]
            pure_dp[end] .= dp[1]
            k = m == 24 ? 0.05 : 4.0
            reference = integrated(ps, pure_p, pure_dp, k, 0.8, 0.3)
            value = AS.far_field(ps, pure_p, pure_dp, k, 0.8, 0.3; rtol = 1e-10)
            @test value≈reference rtol=2e-8 atol=2e-10
            @test AS.far_field(ps, pure_p, pure_dp, k, 0.0, 0.3) == 0
            @test AS.far_field(ps, pure_p, pure_dp, k, 0.8, 0.3+pi)≈
            (-1)^m*value rtol=1e-9 atol=1e-14
            m == 24 && @test abs(value) < 1e-50
        end
        @test AS.far_field(ps, p, dp, 2.0, 0.8, -0.3) ≈
              AS.far_field(ps, p, dp, 2.0, 0.8, 2pi + 0.3) rtol = 1e-12
    end

    # Public oblique amplitudes and maps retain body-frame angle conventions.
    beta, k = pi/3, 0.8
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1))
        sol = bem(Sphere(1.0), boundary, k; n = 32, m_max = 6, incidence_angle = beta)
        thetas, phis = [0.0, 0.4, pi/2, pi], [0.0, 0.9, pi]
        values = [scattering_amplitude(sol; angle = t, azimuth = p)
                  for t in thetas, p in phis]
        exact = [scattering_amplitude(modal(Sphere(1.0), boundary, k;
                     angle = acos(clamp(cos(beta)*cos(t) + sin(beta)*sin(t)*cos(p), -1, 1))))
                 for t in thetas, p in phis]
        @test maximum(abs.(values-exact))/maximum(abs, exact) < 0.01
        @test bistatic_map(sol, thetas, phis).target_strength ≈ target_strength.(values)
        ps = AS.panels(sol.data.mesh)
        for (t, p) in ((0.4, 0.9), (pi-beta, pi))
            @test scattering_amplitude(sol; angle = t, azimuth = p)≈
            integrated(ps, sol.data.p_scat_modes, sol.data.dpdn_scat_modes,
                k, t, p) rtol=2e-8 atol=1e-11
        end
    end
end
