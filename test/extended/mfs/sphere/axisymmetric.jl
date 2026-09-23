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
