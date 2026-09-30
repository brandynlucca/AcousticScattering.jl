using AcousticScattering
using Test
using LinearAlgebra: norm, dot, cross

const AS = AcousticScattering

@testset "Small numerical helpers" begin
    @test AS.backscattering_cross_section(2.0 + 1.0im) ≈ 5.0
    @test AS.bem3d_elements_per_wavelength(2pi) ≈ 0.1
    @test AS.bem3d_elements_per_wavelength(2pi; elements_per_wavelength = 5) ≈ 0.2

    A = [3.0 0.0; 0.0 1.0; 0.0 0.0]
    report = AS._mfs_matrix_diagnostics(A)
    @test report.conditioning === :svd
    @test report.condition_number ≈ 3.0
    @test report.numerical_rank == 2
    skipped = AS._mfs_matrix_diagnostics(A; condition_limit = 1)
    @test skipped.conditioning === :not_computed && skipped.condition_number === nothing
    @test_throws ArgumentError AS._mfs_matrix_diagnostics(A; condition_limit = -1)
    @test AS._mfs_matrix_diagnostics([1.0 1.0; 1.0 1.0]).condition_number == Inf ||
          AS._mfs_matrix_diagnostics([1.0 1.0; 1.0 1.0]).numerical_rank == 1

    @test AS._record_refinement!(nothing, true, 0.1, 0.5, 10) === nothing
    reports = AS._SolveReports()
    AS._record_refinement!(reports, true, 0.1, 0.5, 10)
    @test reports.refinement.n_elements == 10 && reports.refinement.converged
end

@testset "Baffled circular piston Rayleigh integral" begin
    a, k = 0.05, 2pi / 0.0125
    for z in (0.01, 0.05, 0.2, 1.0, 5.0)
        closed = AS._piston_pressure_axial(k, a, z)
        general = AS._piston_pressure(k, a, (0.0, 0.0, z))
        @test closed ≈ general rtol = 1e-10
    end
    x0 = (0.02, 0.015, 0.15)
    h = 1e-6
    g_ad = AS._piston_gradient(k, a, x0)
    for (i, dx) in enumerate(((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h)))
        plus = AS._piston_pressure(k, a, x0 .+ dx)
        minus = AS._piston_pressure(k, a, x0 .- dx)
        @test g_ad[i] ≈ (plus - minus) / 2h rtol = 1e-5
    end
    # Zemanek (1971) puts the last on-axis pressure maximum near a^2/lambda.
    lambda = 2pi / k
    zs = range(0.15, 0.25; length = 4000)
    mags = [abs(AS._piston_pressure_axial(k, a, z)) for z in zs]
    peaks = [i
             for i in 2:(length(mags) - 1)
             if mags[i] > mags[i - 1] && mags[i] > mags[i + 1]]
    @test !isempty(peaks)
    z_last = zs[last(peaks)]
    @test isapprox(z_last, a^2 / lambda; rtol = 0.02)
end

@testset "Tank wall images" begin
    normalize_manual(v) = v ./ norm(v)

    w = Wall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
    t = Transducer((0.3, 0.1, -1.0), (0.0, 0.0, 1.0), 0.05)
    img = AS._image(w, t)
    @test img.position == (0.3, 0.1, 1.0)
    @test img.axis == (0.0, 0.0, -1.0)

    k = 1.0
    tank = TankTransducer(t, w)
    pinc_tank, _ = AS._transducer_field(tank, k)
    pinc_direct, _ = AS._transducer_field(t, k)
    pinc_image, _ = AS._transducer_field(img, k)
    x = (0.5, 0.2, 0.4)
    @test pinc_tank(x) ≈ pinc_direct(x) + pinc_image(x) rtol = 1e-12

    # Exact, no scatterer needed. The combined field satisfies the wall's own boundary condition.
    wall_point = (0.0, 0.0, 0.5)
    normal = (0.3, 0.4, 0.5)
    src = Transducer((2.0, -1.0, 3.0), (-1.0, 0.5, -1.5), 0.08)
    n = collect(normal)
    p0 = collect(wall_point)
    u = normalize_manual([-n[2], n[1], 0.0] .- dot([-n[2], n[1], 0.0], n) .* n)
    v = normalize_manual(cross(n, u))
    offsets = ((0.0, 0.0), (0.3, -0.2), (-0.4, 0.5), (0.1, 0.1))

    wall_rigid = Wall(wall_point, normal, 1.0)
    _, gradinc_r = AS._transducer_field(TankTransducer(src, wall_rigid), 1.3)
    for (a, b) in offsets
        xp = Tuple(p0 .+ a .* u .+ b .* v)
        @test abs(dot(gradinc_r(xp), n)) < 1e-12
    end

    wall_soft = Wall(wall_point, normal, -1.0)
    pinc_s, _ = AS._transducer_field(TankTransducer(src, wall_soft), 1.3)
    for (a, b) in offsets
        xp = Tuple(p0 .+ a .* u .+ b .* v)
        @test abs(pinc_s(xp)) < 1e-12
    end

    @test_throws ArgumentError Wall((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    @test_throws ArgumentError Wall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.5)

    # Omitting the wall's effect (reflection=0) must reproduce the direct field exactly.
    tank_zero = TankTransducer(t, Wall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0))
    pinc_zero, _ = AS._transducer_field(tank_zero, k)
    @test pinc_zero(x) == pinc_direct(x)
end
