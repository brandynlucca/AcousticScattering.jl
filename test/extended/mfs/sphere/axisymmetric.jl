using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Axisymmetric MFS (sphere, axial incidence)" @testset "Axisymmetric MFS (sphere, axial incidence)" begin
        a = 0.01
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        for boundary in (AS.Rigid(), AS.PressureRelease())
            ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))
            for offset_frac in (0.2, 0.9)
                ts_mfs = AS.target_strength(AS.mfs(
                    sphere, boundary, k; incidence_angle = 0.0, offset = offset_frac * a))
                @test ts_mfs ≈ ts_modal atol = 0.1
            end
        end
    end

    @time "Axisymmetric MFS (sphere, oblique incidence / rotational symmetry)" @testset "Axisymmetric MFS (sphere, oblique incidence / rotational symmetry)" begin
        a = 0.01
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        for boundary in (AS.Rigid(), AS.PressureRelease())
            ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))
            for angle_deg in (0.0, 90.0)
                β = deg2rad(angle_deg)
                sol = AS.mfs(
                    sphere, boundary, k; incidence_angle = β, m_max = 15, offset = 0.3a)
                ts_mfs = AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test ts_mfs ≈ ts_modal atol = 0.1
            end
        end
    end

    @time "Axisymmetric MFS fluid-filled/transmission (sphere, axial incidence)" @testset "Axisymmetric MFS fluid-filled/transmission (sphere, axial incidence)" begin
        a = 0.05
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        @time "rigid limit (gh >> 1)" @testset "rigid limit (gh >> 1)" begin
            stiff = AS.FluidFilled(1e8, 1e8)
            ts_stiff = AS.target_strength(AS.mfs(
                sphere, stiff, k; incidence_angle = 0.0, offset = 0.3a, n = 24))
            ts_rigid = AS.target_strength(AS.mfs(
                sphere, AS.Rigid(), k; incidence_angle = 0.0, offset = 0.3a, n = 24))
            @test ts_stiff ≈ ts_rigid atol = 1e-4
        end

        @time "pressure-release limit (g << 1)" @testset "pressure-release limit (g << 1)" begin
            soft = AS.FluidFilled(1e-8, 1.0)
            ts_soft = AS.target_strength(AS.mfs(
                sphere, soft, k; incidence_angle = 0.0, offset = 0.3a, n = 24))
            ts_pr = AS.target_strength(AS.mfs(sphere, AS.PressureRelease(), k;
                incidence_angle = 0.0, offset = 0.3a, n = 24))
            @test ts_soft ≈ ts_pr atol = 1e-4
        end

        @time "converges to the analytical FluidFilled sphere modal series" @testset "converges to the analytical FluidFilled sphere modal series" begin
            g, h = 1.05, 1.02
            bc = AS.FluidFilled(g, h)
            ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
            ts_mfs = AS.target_strength(AS.mfs(
                sphere, bc, k; incidence_angle = 0.0, offset = 0.3a, n = 48))
            @test abs(ts_mfs - ts_modal) < 0.08
        end
    end

    @time "Axisymmetric MFS fluid-filled/transmission (sphere, oblique incidence)" @testset "Axisymmetric MFS fluid-filled/transmission (sphere, oblique incidence)" begin
        a = 0.05
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        @time "rigid limit (gh >> 1)" @testset "rigid limit (gh >> 1)" begin
            stiff = AS.FluidFilled(1e8, 1e8)
            for angle_deg in (0.0, 90.0)
                β = deg2rad(angle_deg)
                sol_stiff = AS.mfs(sphere, stiff, k; incidence_angle = β,
                    m_max = 15, offset = 0.3a, n = 24)
                ts_stiff = AS.target_strength(sol_stiff; angle = pi - β, azimuth = pi)

                sol_rigid = AS.mfs(sphere, AS.Rigid(), k; incidence_angle = β,
                    m_max = 15, offset = 0.3a, n = 24)
                ts_rigid = AS.target_strength(sol_rigid; angle = pi - β, azimuth = pi)
                @test ts_stiff ≈ ts_rigid atol = 1e-4
            end
        end

        @time "pressure-release limit (g << 1)" @testset "pressure-release limit (g << 1)" begin
            soft = AS.FluidFilled(1e-8, 1.0)
            for angle_deg in (0.0, 90.0)
                β = deg2rad(angle_deg)
                sol_soft = AS.mfs(
                    sphere, soft, k; incidence_angle = β, m_max = 15, offset = 0.3a, n = 24)
                ts_soft = AS.target_strength(sol_soft; angle = pi - β, azimuth = pi)

                sol_pr = AS.mfs(sphere, AS.PressureRelease(), k; incidence_angle = β,
                    m_max = 15, offset = 0.3a, n = 24)
                ts_pr = AS.target_strength(sol_pr; angle = pi - β, azimuth = pi)
                @test ts_soft ≈ ts_pr atol = 1e-4
            end
        end

        @time "sphere rotational symmetry against the analytical modal series" @testset "sphere rotational symmetry against the analytical modal series" begin
            g, h = 1.05, 1.02
            bc = AS.FluidFilled(g, h)
            ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
            for angle_deg in (0.0, 90.0)
                β = deg2rad(angle_deg)
                sol = AS.mfs(
                    sphere, bc, k; incidence_angle = β, m_max = 15, offset = 0.3a, n = 48)
                ts_mfs = AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test abs(ts_mfs - ts_modal) < 0.08
            end
        end
    end
end

let
    @time "MFS independent checks and oversampling" @testset "MFS independent checks and oversampling" begin
        square = mfs(Sphere(1.0), PressureRelease(), 1.0; n = 12, incidence_angle = 0.0)
        sampled = mfs(Sphere(1.0), PressureRelease(), 1.0;
            n = 12, incidence_angle = 0.0, oversampling = 2)
        d1, d2 = only(diagnostics(square).systems), only(diagnostics(sampled).systems)
        @test d1.relative_residual < 1e-12
        @test d1.boundary_residual.relative_residual > 1e-3
        @test d2.boundary_residual.relative_residual <
              d1.boundary_residual.relative_residual / 2
        @test d1.unknown_count == d2.unknown_count == d2.source_count == 12
        @test d2.equation_count == d2.collocation_count == 24
        @test d2.check_count == 48
        @test d2.method == :least_squares
        @test d2.numerical_rank == 12
        @test d2.condition_number > 1
        @test d2.rank_tolerance > 0
        @test d2.offset_ext == 0.3
        @test d2.offset_int === nothing
        @test d2.conditioning == :svd
        checks = AS.panels(AS._mfs_check_mesh(sampled.data.mesh))
        collocation = AS.panels(sampled.data.mesh)
        @test isempty(intersect(Set((p.rhom, p.zm) for p in checks),
            Set((p.rhom, p.zm) for p in collocation)))
        low = AS.solve_axial_mfs(PressureRelease(), 1.0, square.data.mesh; offset = 0.3)
        @test length(low) == 3
        @test low[1] == only(square.data.p_scat_modes)
        @test low[2] == only(square.data.dpdn_scat_modes)

        for beta in (0.0, pi / 3)
            transmission = mfs(Sphere(1.0), FluidFilled(1.2, 1.1), 1.0;
                n = 12, incidence_angle = beta, m_max = 2, oversampling = 2)
            for report in diagnostics(transmission).systems
                @test report.unknown_count == report.source_count == 24
                @test report.equation_count == 48
                @test isfinite(report.pressure_residual.relative_residual)
                @test isfinite(report.velocity_residual.relative_residual)
                @test report.offset_int == 0.3
            end
        end
        skipped = mfs(
            Sphere(1.0), Rigid(), 1.0; n = 8, incidence_angle = 0.0, condition_limit = 0)
        report = only(diagnostics(skipped).systems)
        @test report.conditioning == :not_computed
        @test report.condition_number === report.numerical_rank ===
              report.rank_tolerance ===
              nothing
        @test isfinite(report.boundary_residual.relative_residual)
        @test_throws ArgumentError mfs(Sphere(1.0), Rigid(), 1.0; oversampling = 0)
        @test_throws ArgumentError mfs(Sphere(1.0), Rigid(), 1.0; condition_limit = -1)
        @test_throws ArgumentError AS._mfs_matrix_diagnostics([1.0;;]; condition_limit = -1)
        deficient = AS._mfs_matrix_diagnostics([1.0 0.0; 0.0 0.0])
        @test deficient.numerical_rank == 1
        @test deficient.condition_number == Inf
    end

    @time "Oversampled MFS near gas-sphere resonance" @testset "Oversampled MFS near gas-sphere resonance" begin
        body, boundary, k = Sphere(1.0), FluidFilled(0.0012, 0.23), 0.014
        reference = modal(body, boundary, k)
        solution = mfs(body, boundary, k; incidence_angle = 0.0, n = 96, oversampling = 2)
        @test abs(target_strength(solution) - target_strength(reference)) < 0.1
        @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.01
    end
end

let
    function reference_points(points, beta, alpha)
        direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
        return [(dot(direction, p), sqrt(max(0, norm(p)^2-dot(direction, p)^2)), 0.0)
                for p in points]
    end

    function check_boundary_pressure(solution, beta, alpha; full = false)
        reference = modal(solution.body, solution.boundary, solution.k)
        directions = ([1.0, 0, 0], [0.0, 0.6, 0.8], [-0.6, 0.0, 0.8])
        points = [Tuple(r .* v) for r in (1.0, 1+1e-8, 1.001, 1.1, 2.0) for v in directions]
        expected = pressure(reference, reference_points(points, beta, alpha); field = :scattered)
        actual = pressure(solution, points; field = :scattered)
        for (got, wanted) in zip(actual, expected)
            @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
            @test abs(20log10(abs(got/wanted))) < 0.01
        end
        incident = pressure(solution, points; field = :incident)
        @test pressure(solution, points) ≈ incident + actual
        @test incident ≈
              pressure(reference, reference_points(points, beta, alpha); field = :incident)
        @test pressure(solution, first(points); field = :scattered) ≈ first(actual)
        @test pressure(solution, collect(first(points)); field = :scattered) ≈ first(actual)
        @test pressure(solution, reduce(hcat, collect.(points)); field = :scattered) ≈
              actual
        @test vec(pressure(solution, reshape(points, 3, 5); field = :scattered)) ≈ actual
        @test isempty(pressure(solution, NTuple{3, Float64}[]))
        @test_throws ArgumentError pressure(solution, (NaN, 0.0, 0.0))
        @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0); field = :scattered)
        if solution.boundary isa FluidFilled
            inside = [(0.0, 0.0, 0.0);
                      [Tuple(r .* v) for r in (0.4, 1-1e-8, 1.0) for v in directions]]
            expected_inside = pressure(reference, reference_points(inside, beta, alpha); field = :interior)
            actual_inside = pressure(solution, inside; field = :interior)
            for (got, wanted) in zip(actual_inside, expected_inside)
                @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
            end
            @test pressure(solution, first(inside)) ≈ first(actual_inside)
            surface = [Tuple(v) for v in directions]
            @test all(isapprox.(pressure(solution, surface),
                pressure(solution, surface; field = :interior); rtol = 1e-3, atol = 1e-12))
        else
            @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0))
        end
        direction, distance = [0.36, 0.48, 0.8], 1e6
        far_pressure = pressure(solution, Tuple(distance .* direction); field = :scattered) *
                       distance * cis(-solution.k*distance)
        amplitude = full ? scattering_amplitude(solution; direction) :
                    scattering_amplitude(
            solution; angle = acos(direction[1]), azimuth = atan(direction[3], direction[2]))
        @test isapprox(far_pressure, amplitude; rtol = 1e-4, atol = 1e-12)
    end

    @time "Spherical MFS pressure" @testset "Spherical MFS pressure" begin
        for k in (0.3, 2.0), boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1))

            beta = pi/3
            solution = mfs(Sphere(1.0), boundary, k; n = 96, oversampling = 2,
                offset = 0.2, incidence_angle = beta, m_max = 10, condition_limit = 0)
            @time "k=$k $(typeof(boundary)) beta=$beta" @testset "k=$k $(typeof(boundary)) beta=$beta" begin
                check_boundary_pressure(solution, beta, 0.0)
            end
        end
    end
end

let
    function compare_near_pressure(actual, expected)
        for (got, wanted) in zip(actual, expected)
            @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
            @test abs(20log10(abs(got/wanted))) < 0.01
        end
    end

    function near_reference_points(points, beta, alpha)
        direction = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
        return [(dot(direction, p), sqrt(max(0, norm(p)^2-dot(direction, p)^2)), 0.0)
                for p in points]
    end

    @time "Higher-frequency spherical pressure" @testset "Higher-frequency spherical pressure" begin
        # NOTE: a full-mesh BEM comparison at this near-surface tolerance needs a fine enough
        # mesh that geometric approximation error stays under rtol=1e-3, which costs minutes
        # regardless of k/qorder/compression; that comparison is covered at full fidelity in
        # perf/near_interface_pressure.jl. Here only the (fast) MFS code path is checked.
        body, k, beta = Sphere(1.0), 6.0, pi/3
        points = [Tuple(r .* direction) for r in (1.0, 1+1e-8, 1.01, 1.2)
                  for direction in ([1.0, 0, 0], [0.0, 0.6, 0.8], [-0.6, 0, 0.8])]
        for (boundary, n) in ((Rigid(), 96), (FluidFilled(1.2, 1.1), 64))
            reference = modal(body, boundary, k; m_max = 32)
            @time "$(typeof(boundary))" @testset "$(typeof(boundary))" begin
                solution = mfs(body, boundary, k; n, oversampling = 2,
                    offset = 0.2, incidence_angle = beta, m_max = 18, condition_limit = 0)
                compare_near_pressure(pressure(solution, points; field = :scattered),
                    pressure(reference, near_reference_points(points, beta, 0.0); field = :scattered))
            end
        end
    end
end
