using AcousticScattering
using Test

@testset "Spheroidal coupling at angular nodes" begin
    body = Spheroid(0.02, 0.005)
    k = 2pi * 38000 / 1500
    for bc in (FluidFilled(1.05, 1.05), FluidFilled(1.24 / 1026, 345 / 1500))
        solve(beta; precision = :double, m = 12, n = 12,
            quadrature = 96) = scattering_amplitude(
            modal(body, bc, k; incidence_angle = beta, m_max = m, n_max = n,
            n_quad = quadrature, precision = precision))
        f = solve(pi / 2)
        # The old incident-weighted matrix lost modes at exact broadside.
        @test solve(pi / 2 - 1e-5) ≈ f rtol = 1e-8
        @test solve(pi / 2 + 1e-5) ≈ f rtol = 1e-8
        @test solve(pi / 2; precision = :quad) ≈ f rtol = 1e-8
        @test solve(pi / 2; precision = :quad, quadrature = 95) ≈ f rtol = 1e-8
        @test solve(pi / 2; n = 14) ≈ f rtol = 1e-6
        # An azimuthal ceiling must not add degrees beyond n_max.
        @test solve(0.7; m = 8, n = 4) ≈ solve(0.7; m = 4, n = 4) rtol = 1e-13
    end
    @test abs(scattering_amplitude(modal(body, FluidFilled(1, 1), k;
        incidence_angle = 0.7, m_max = 8, n_max = 12))) < 1e-14
end
