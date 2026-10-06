using AcousticScattering
using Test
using LinearAlgebra

const AS = AcousticScattering
BLAS.set_num_threads(1)

@testset "Full-surface fluid MFS against spherical modal transmission" begin
    body = Sphere(0.5)
    boundary = FluidFilled(1.2, 1.1)
    collocation = mesh(body; method = :full, resolution = 0.2,
        mesh_order = 3, qorder = 2)
    sources = mesh(body; method = :full, resolution = 0.25,
        mesh_order = 3, qorder = 1)
    checks = mesh(body; method = :full, resolution = 0.18,
        mesh_order = 3, qorder = 2)
    beta, alpha = pi / 3, 0.4
    direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
    solved = mfs(collocation, boundary, 1.0; source_mesh = sources,
        check_mesh = checks, offset_ext = 0.15, offset_int = 0.15,
        incidence_angle = beta, incidence_azimuth = alpha, condition_limit = 0)
    exact = modal(body, boundary, 1.0)
    @test isapprox(scattering_amplitude(solved; direction = -direction),
        scattering_amplitude(exact); rtol = 0.01)
    outside = Tuple(1.5 .* direction)
    @test isapprox(pressure(solved, outside; field = :scattered),
        pressure(exact, (1.5, 0.0, 0.0); field = :scattered); rtol = 0.03)
    @test isapprox(pressure(solved, (0.0, 0.0, 0.0); field = :interior),
        pressure(exact, (0.0, 0.0, 0.0); field = :interior); rtol = 0.001)
    @test pressure(solved, (0.0, 0.0, 0.0); field = :total) ≈
          pressure(solved, (0.0, 0.0, 0.0); field = :interior)
    @test pressure(solved, outside; field = :total) ≈
          pressure(solved, outside; field = :scattered) +
          pressure(solved, outside; field = :incident)
    @test_throws ArgumentError pressure(solved, outside; field = :interior)
    @test_throws ArgumentError pressure(solved, (0.0, 0.0, 0.0); field = :scattered)
    @test_throws ArgumentError mfs(collocation, boundary, 1.0; source_mesh = sources,
        offset_ext = 1.2, offset_int = 0.15, condition_limit = 0)
    @test diagnostics(solved).boundary_residual.relative_residual < 0.03
end

@testset "Penetrable capped bent cylinder: full MFS against BEM" begin
    body = Cylinder(0.5, 1.0; radius_curvature = 2.0, endcap_depth = 0.5)
    boundary = FluidFilled(1.2, 1.1)
    collocation = mesh(body; method = :full, resolution = 0.28,
        mesh_order = 3, qorder = 4)
    sources = mesh(body; method = :full, resolution = 0.2,
        mesh_order = 3, qorder = 1)
    checks = mesh(body; method = :full, resolution = 0.252,
        mesh_order = 3, qorder = 4)
    beta, alpha = pi / 3, 0.4
    solved = mfs(collocation, boundary, 0.5; source_mesh = sources,
        check_mesh = checks, offset_ext = 0.2, offset_int = 0.2,
        incidence_angle = beta, incidence_azimuth = alpha, condition_limit = 0)
    reference = bem(collocation, boundary, 0.5; incidence_angle = beta,
        incidence_azimuth = alpha, condition_limit = 0)
    directions = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])
    actual = [scattering_amplitude(solved; direction = collect(d)) for d in directions]
    expected = [scattering_amplitude(reference; direction = collect(d)) for d in directions]
    @test maximum(abs.(actual .- expected) ./ abs.(expected)) < 0.005
    exterior = (1.5, 0.2, 0.3)
    interior = (0.0, 0.0, 0.0)
    @test isapprox(pressure(solved, exterior; field = :scattered),
        pressure(reference, exterior; field = :scattered); rtol = 0.005)
    @test isapprox(pressure(solved, interior; field = :interior),
        pressure(reference, interior; field = :interior); rtol = 0.001)
    report = diagnostics(solved)
    @test report.source_count == 2length(sources.data)
    @test report.equation_count == 2length(collocation.data)
    @test report.check_count == length(checks.data)
    @test report.boundary_residual.relative_residual < 0.005
    @test report.pressure_residual.relative_residual < 0.002
    @test report.velocity_residual.relative_residual < 0.01
    @test diagnostics(reference).relative_residual < 1e-8
end
