using AcousticScattering
using Test

function spheroid_near_points(body, deltas)
    surface = [(body.a*cos(theta), body.b*sin(theta)*cos(phi),
                   body.b*sin(theta)*sin(phi))
               for theta in (pi/4, pi/2) for phi in (0.0, pi/3)]
    return [Tuple((1+delta) .* point) for point in surface for delta in deltas]
end

function check_near_pressure(actual, reference)
    @test all(isapprox.(actual, reference; rtol = 1e-3, atol = 1e-12))
    @test maximum(abs.(20log10.(abs.(actual ./ reference)))) < 0.01
end

@time "Gas spheroid near fields across resonance" @testset "Gas spheroid near fields across resonance" begin
    body = Spheroid(0.025, 0.0075)
    boundary = GasFilled(0.00129, 0.23)
    beta = pi/3
    exterior = spheroid_near_points(body, (1e-3, 1e-2))
    interior = spheroid_near_points(body, (-1e-3, -1e-2))

    for frequency in (315, 320, 325)
        k = 2pi*frequency/1477.4
        n_bem = frequency == 320 ? (256, 384) : (128, 192)
        n_mfs = frequency == 320 ? (256, 384) : (192, 256)
        bem_solutions = [bem(body, boundary, k; method = :axisymmetric,
                             n, m_max = 4, incidence_angle = beta) for n in n_bem]
        mfs_solutions = [mfs(body, boundary, k; n, m_max = 4,
                             incidence_angle = beta, oversampling = 2,
                             offset = 0.002, condition_limit = 0) for n in n_mfs]
        for (points, field) in ((exterior, :scattered), (interior, :interior))
            bem_values = [pressure(solution, points; field) for solution in bem_solutions]
            mfs_values = [pressure(solution, points; field) for solution in mfs_solutions]
            check_near_pressure(bem_values[1], bem_values[2])
            check_near_pressure(mfs_values[1], mfs_values[2])
            check_near_pressure(bem_values[2], mfs_values[2])
        end
        reference = modal(body, boundary, k; incidence_angle = beta,
            m_max = 4, n_max = 6, precision = :quad)
        @test isapprox(scattering_amplitude(last(bem_solutions)),
            scattering_amplitude(reference); rtol = 1e-3)
        @test isapprox(scattering_amplitude(last(mfs_solutions)),
            scattering_amplitude(reference); rtol = 1e-3)
        @test pressure(last(bem_solutions), first(exterior)) ≈
              pressure(last(bem_solutions), first(exterior); field = :incident) +
              pressure(last(bem_solutions), first(exterior); field = :scattered)
    end
end

@time "High-frequency rigid spheroid near fields" @testset "High-frequency rigid spheroid near fields" begin
    body, k, beta = Spheroid(1.5, 1.0), 4.0, pi/3
    points = spheroid_near_points(body, (1e-3, 1e-2))
    bem_coarse = bem(body, Rigid(), k; method = :axisymmetric,
        n = 384, m_max = 16, incidence_angle = beta)
    bem_fine = bem(body, Rigid(), k; method = :axisymmetric,
        n = 768, m_max = 16, incidence_angle = beta)
    mfs_coarse = mfs(body, Rigid(), k; n = 192, m_max = 16,
        incidence_angle = beta, oversampling = 2, offset = 0.2,
        condition_limit = 0)
    mfs_fine = mfs(body, Rigid(), k; n = 384, m_max = 16,
        incidence_angle = beta, oversampling = 2, offset = 0.2,
        condition_limit = 0)
    bem_values = pressure(bem_coarse, points; field = :scattered)
    bem_refined = pressure(bem_fine, points; field = :scattered)
    mfs_values = pressure(mfs_coarse, points; field = :scattered)
    mfs_refined = pressure(mfs_fine, points; field = :scattered)
    check_near_pressure(bem_values, bem_refined)
    check_near_pressure(mfs_values, mfs_refined)
    check_near_pressure(bem_refined, mfs_refined)
    @test_throws ArgumentError pressure(bem_fine, (0.0, 0.0, 0.0))
    @test_throws ArgumentError pressure(mfs_fine, (0.0, 0.0, 0.0))
end

@time "Oblate spheroid pressure region" @testset "Oblate spheroid pressure region" begin
    body = Spheroid(1.0, 1.2)
    solution = mfs(body, Rigid(), 0.5; incidence_angle = 0.0,
        n = 32, offset = 0.2, condition_limit = 0)
    @test isfinite(pressure(solution, (0.0, 1.21, 0.0); field = :scattered))
    @test_throws ArgumentError pressure(solution, (0.0, 1.19, 0.0))
end
