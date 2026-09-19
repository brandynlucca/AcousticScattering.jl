@time @testset "Axisymmetric BEM diagnostic contracts" begin
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1)),
        beta in (0.0, pi / 3)

        solution = bem(
            Sphere(1.0), boundary, 1.0; n = 12, incidence_angle = beta, m_max = 2)
        report = diagnostics(solution)
        @test report.converged === report.iterations === nothing
        @test report.relative_residual < 1e-10
        @test Set(r.mode for r in report.systems) == (iszero(beta) ? Set([0]) : Set(0:2))
        @test report.solver_options.n == 12
        low = iszero(beta) ? AS.solve_axial(boundary, 1.0, solution.data.mesh) :
              AS.solve_oblique(boundary, 1.0, solution.data.mesh, beta; m_max = 2)
        @test solution.data.p_scat_modes == (iszero(beta) ? [low[1]] : low[1])
        @test solution.data.dpdn_scat_modes == (iszero(beta) ? [low[2]] : low[2])
    end
end

@time @testset "MFS independent checks and oversampling" begin
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
    @test report.condition_number === report.numerical_rank === report.rank_tolerance ===
          nothing
    @test isfinite(report.boundary_residual.relative_residual)
    @test_throws ArgumentError mfs(Sphere(1.0), Rigid(), 1.0; oversampling = 0)
    @test_throws ArgumentError mfs(Sphere(1.0), Rigid(), 1.0; condition_limit = -1)
    @test_throws ArgumentError AS._mfs_matrix_diagnostics([1.0;;]; condition_limit = -1)
    deficient = AS._mfs_matrix_diagnostics([1.0 0.0; 0.0 0.0])
    @test deficient.numerical_rank == 1
    @test deficient.condition_number == Inf
end

@time @testset "Bent MFS independent checks" begin
    body = Cylinder(0.01, 0.07; radius_curvature = 0.2)
    for boundary in (Rigid(), PressureRelease())
        square = mfs(body, boundary, 100.0; n_s = 4, n_phi = 4, offset = 0.003)
        sampled = mfs(body, boundary, 100.0;
            n_s = 4, n_phi = 4, offset = 0.003, oversampling = 2)
        d = only(diagnostics(sampled).systems)
        @test d.source_count == 16
        @test d.collocation_count == 64
        @test d.check_count == 256
        @test d.method == :least_squares
        @test isfinite(d.boundary_residual.relative_residual)
        checks, _, _ = AS.bent_cylinder_mfs_points(0.01, 0.07, 0.2, 16, 16)
        @test isempty(intersect(Set(checks), Set(sampled.data.points)))
        low = AS.solve_bent_cylinder_mfs(boundary, 100.0, 0.01, 0.07, 0.2;
            n_s = 4, n_φ = 4, offset = 0.003)
        @test length(low) == 5
        @test low[1] == square.data.p_scat
        @test low[2] == square.data.dpdn_scat
    end
end

@time @testset "FEM reports across discretizations" begin
    sphere, cylinder, spheroid = Sphere(1.0), Cylinder(1.0, 3.0), Spheroid(1.2, 1.0)
    solid = SolidElastic(2.7, 4.0, 2.0)
    elastic_shell = Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8)
    fluid_shell = Shelled(FluidLayer(1.2, 1.1), FluidInterior(1.0, 1.0), 0.8)
    for body in (sphere, cylinder), boundary in (solid, elastic_shell)

        solution = fem(body, boundary, 1.0; n_elements = 8, m_max = 2)
        d = diagnostics(solution)
        @test d.converged === nothing
        @test d.relative_residual < 1e-9
        @test Set(r.component for r in d.systems) == Set((:radial_basis, :interface))
        @test Set(r.mode for r in d.systems) == Set(0:2)
    end
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1), fluid_shell,
        Shelled(FluidLayer(1.2, 1.1), VacuumInterior(), 0.8))
        solution = boundary isa FluidFilled ?
                   fem(
            sphere, boundary, 1.0; n_elements_int = 8, n_elements_ext = 8, m_max = 2) :
                   fem(sphere, boundary, 1.0; n_elements = 8, m_max = 2)
        @test diagnostics(solution).relative_residual < 1e-10
        @test Set(r.mode for r in diagnostics(solution).systems) == Set(0:2)
    end
    for body in (sphere, cylinder, spheroid), boundary in (Rigid(), PressureRelease())

        solution = body isa Sphere ?
                   fem(
            body, boundary, 1.0; method = :meridian, n_r = 3, n_theta = 8, l_max = 3) :
                   fem(body, boundary, 1.0; n_r = 3, n_theta = 8, l_max = 3, m_max = 2)
        d = diagnostics(solution)
        @test d.relative_residual < 1e-10
        @test Set(r.mode for r in d.systems) == (body isa Sphere ? Set([0]) : Set(0:2))
        @test all(r.n_r == 3 && r.n_theta == 8 for r in d.systems)
    end
    for method in (:thin, :general)
        solution = fem(Shell(Spheroid(1.2, 1.0), 0.02), Shelled(0.3, 2700.0, 70e9),
            1000.0, 1500.0, method == :thin ? 0.0 : 1000.0, 1500.0, 1.0;
            method, incidence_angle = 0.0, n_eta = 9, n_t = 3, m_max = 0)
        d = diagnostics(solution)
        @test d.discretization == method
        @test d.relative_residual < 1e-8
        @test only(d.systems).mode == 0
        @test d.solver_options.n_eta == 9
    end
end

@time @testset "Adaptive FEM refinement status" begin
    for boundary in (Rigid(), FluidFilled(1.2, 1.1))
        solution = fem(Sphere(1.0), boundary, 1.0; adaptive = true, m_max = 3,
            n_elements_start = 16, max_n_elements = 128, target_tol = 0.01)
        report = diagnostics(solution)
        @test report.converged === nothing
        @test report.refinement.converged
        @test report.refinement.change_db < report.refinement.target_tol == 0.01
        @test 16 < report.refinement.n_elements <= 128
        @test abs(target_strength(solution) -
                  target_strength(modal(Sphere(1.0), boundary, 1.0; m_max = 3))) < 0.1
        failed = @test_logs (:warn, r"did not converge") fem(Sphere(1.0), boundary, 1.0;
            adaptive = true, m_max = 2, n_elements_start = 8, max_n_elements = 8)
        @test !diagnostics(failed).refinement.converged
        @test diagnostics(failed).refinement.change_db === nothing
    end
end

@time @testset "Complex sphere references near irregular frequencies and at strong contrast" begin
    for (boundary, k) in ((PressureRelease(), pi - 0.02), (PressureRelease(), pi + 0.02),
        (FluidFilled(10.0, 0.5), 1.0))
        reference = modal(Sphere(1.0), boundary, k)
        for solver in (bem, mfs)
            solution = solver(Sphere(1.0), boundary, k; incidence_angle = 0.0, n = 48)
            @test abs(target_strength(solution) - target_strength(reference)) < 0.1
            @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.01
        end
    end
end

@time @testset "Oblique spheroid complex references at strong contrast" begin
    @test_skip "requires SpheroidalWaves backend, not available locally"
end

@time @testset "Oversampled MFS near gas-sphere resonance" begin
    body, boundary, k = Sphere(1.0), FluidFilled(0.0012, 0.23), 0.014
    reference = modal(body, boundary, k)
    solution = mfs(body, boundary, k; incidence_angle = 0.0, n = 96, oversampling = 2)
    @test abs(target_strength(solution) - target_strength(reference)) < 0.1
    @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.01
end

@time @testset "BEM gas-sphere resonance and mesh refinement" begin
    # NOTE: the fine (n=192) and resonance-peak (n=256) refinement checks are covered at
    # full fidelity in perf/diagnostics.jl; here a single coarse off-resonance k is cheap.
    body, boundary, k = Sphere(1.0), FluidFilled(0.0012, 0.23), 0.014
    reference = modal(body, boundary, k)
    solution = bem(body, boundary, k; incidence_angle = 0.0, n = 96, rtol = 1e-8)
    @test abs(target_strength(solution) - target_strength(reference)) < 0.1
    @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.01
    scaled = bem(Sphere(0.01), boundary, 1.4;
        incidence_angle = pi / 3, n = 96, m_max = 2, rtol = 1e-8)
    scaled_reference = modal(Sphere(0.01), boundary, 1.4)
    @test abs(target_strength(scaled) - target_strength(scaled_reference)) < 0.1
    @test scattering_amplitude(scaled) ≈ scattering_amplitude(scaled_reference) rtol = 0.01
end
