using AcousticScattering
using LinearAlgebra
using Test

@testset "Compressed coupled-fluid equations and diagnostics" begin
    AS = AcousticScattering
    function sphere_at(radius, center; resolution = 1.0)
        mesh(; qorder = 4) do g
            g.model.add("compressed-regions")
            g.model.occ.addSphere(center..., radius)
            g.model.occ.synchronize()
            g.option.setNumber("Mesh.MeshSizeMin", radius*resolution)
            g.option.setNumber("Mesh.MeshSizeMax", radius*resolution)
            g.model.mesh.generate(2)
            g.model.mesh.setOrder(2)
        end
    end
    outer = sphere_at(1.0, (0, 0, 0))
    inner = sphere_at(0.5, (0, 0, 0))
    offset = sphere_at(0.5, (0.3, 0.1, 0))
    close = sphere_at(0.9, (0, 0, 0))
    core = sphere_at(0.2, (0, 0, 0))
    sibling = sphere_at(0.3, (1.6, 0, 0))
    branch = [outer, sphere_at(0.2, (-0.3, 0, 0)), sphere_at(0.2, (0.3, 0, 0))]
    ellipsoids = [
        mesh(; semiaxes = (1.0, 0.7, 0.8), center = (0.2, -0.1, 0.15),
            resolution = 0.9, qorder = 4),
        mesh(; semiaxes = (0.25, 0.16, 0.18), center = (0.3, -0.06, 0.1),
            resolution = 0.9, qorder = 4)]
    weak = FluidFilled(1.05, 1.02)
    gas = GasFilled(0.0012, 0.23)
    compression = (method = :hmatrix, tol = 1e-9)
    direction = AS._bem3d_incidence_direction(pi/3, 0.4)
    cases = (([outer], [weak], [0], 0.3),
        ([outer, inner], [weak, FluidFilled(1.1, 1.03)], [0, 1], 0.3),
        ([outer, inner], [weak, FluidFilled(1.1, 1.03)], [0, 1], 1.0),
        ([outer, inner], [weak, gas], [0, 1], 1.0),
        ([outer, offset], [FluidFilled(1, 1), gas], [0, 1], 0.0276),
        ([outer, close], [weak, gas], [0, 1], 1.0),
        ([outer, inner], fill(FluidFilled(1, 1), 2), [0, 1], 0.3),
        ([outer, inner], fill(FluidFilled(1.0001, 1.0001), 2), [0, 1], 0.3),
        ([outer, inner, core], [weak, FluidFilled(0.8, 0.9), gas], [0, 1, 2], 0.3),
        ([outer, sibling], [weak, weak], [0, 0], 0.3),
        (branch, [weak, weak, gas], [0, 1, 1], 0.3),
        (ellipsoids, [weak, gas], [0, 1], 0.8),
        ([outer, inner], [FluidFilled(1.2, 1.0), FluidFilled(0.8, 1.0)], [0, 1], 0.3),
        ([outer, inner], [FluidFilled(1.2, 1.0), FluidFilled(0.8, 1.0)], [0, 1], 2.0),
        ([outer, inner], [gas, weak], [0, 1], 0.0276),
        ([outer, close], [weak, gas], [0, 1], 0.03))
    selected_case = get(ENV, "TEST_REGION_BEM_CASE", "")
    case_indices = if isempty(selected_case)
        eachindex(cases)
    else
        index = parse(Int, selected_case)
        index in eachindex(cases) || throw(ArgumentError("Unknown region BEM case $index"))
        (index,)
    end
    @testset "case $case_index" for case_index in case_indices
        surfaces, materials, parents, k = cases[case_index]
        # The reversed-density case is ill-conditioned
        gmres_kwargs = case_index == 15 ? (; reltol = 1e-12) : (;)
        options = (;
            parents, incidence_angle = pi/3, incidence_azimuth = 0.4, condition_limit = 0,
            gmres_kwargs)
        dense = AS._assemble_region_bem(surfaces, materials, k; parents)
        dense_factor = AS._factor_full_fluid(
            dense; equilibrate = true, condition_limit = 0,
            norm_floor = eps(Float64))
        reference = AS._solve_region_bem(dense, dense_factor; incidence_angle = pi/3, incidence_azimuth = 0.4)
        solution = bem(surfaces, materials, k; options..., compression)
        report = diagnostics(solution)
        @test report.converged
        @test report.method === :gmres
        reconstructed = all(d -> d.target != d.source || d.method === :calderon,
            report.derivative_evaluation)
        @test report.local_preconditioner === (reconstructed ? :calderon : :block_jacobi)
        @test report.solver_options.restart == 600
        @test report.conditioning === :not_computed
        @test report.derivative_evaluation == diagnostics(reference).derivative_evaluation
        @test report.scaled_relative_residual < 1e-8
        @test report.unknown_count == 2sum(length(s.data) for s in surfaces)
        p = reduce(vcat, [s.pressure for s in solution.data.interfaces])
        v = reduce(vcat,
            [s.normal_derivative_interior ./ materials[i].density_contrast
             for (i, s) in enumerate(solution.data.interfaces)])
        x = [p; v]
        b = AS._region_incident_rhs(dense, pi/3, 0.4)
        r = dense.A*x-b
        @test AS._fluid_residual_report(r, b, dense_factor.row_norms).scaled_relative_residual <
              2e-7
        ri = dense.inner*x
        for (i, interface) in enumerate(solution.data.interfaces)
            expected = reference.data.interfaces[i]
            @test interface.pressure≈expected.pressure rtol=2e-5 atol=1e-8
            # The small interior derivative in case 15 amplifies platform-dependent
            # differences between the dense and compressed solves
            interior_rtol = case_index == 15 ? 1e-4 : 2e-5
            @test interface.normal_derivative_interior≈expected.normal_derivative_interior rtol=interior_rtol atol=1e-8
            @test interface.normal_derivative_exterior≈expected.normal_derivative_exterior rtol=2e-5 atol=1e-8
            rows = dense.ranges[i]
            pscale = max(norm(interface.pressure), eps(Float64))
            vscale = max(norm(v[rows]), k*pscale/materials[i].density_contrast, eps(Float64))
            expected_residuals = (norm(ri[rows])/pscale, norm((r - ri)[rows])/pscale,
                norm(ri[rows .+ dense.n])/vscale, norm((r - ri)[rows .+ dense.n])/vscale)
            @test collect(values(report.interface_residuals[i]))≈collect(expected_residuals) rtol=2e-4 atol=2e-7
        end
        for observation in (-direction, direction, AS.SVector(0.0, 0.0, 1.0))
            @test scattering_amplitude(solution; direction = observation)≈
            scattering_amplitude(reference; direction = observation) rtol=2e-5 atol=2e-9
        end
        if case_index in (1, 5, 12)
            compressed = AS._assemble_region_bem(
                surfaces, materials, k; parents, compression)
            prepared = AS._factor_full_fluid(compressed;
                equilibrate = true, condition_limit = 0, norm_floor = eps(Float64))
            recycled = AS._fluid_sweep_factor(prepared, 3, 6)
            # Include resonance, nonspherical interfaces, angular jumps, repetition
            # and more samples than retained vectors. Compare original dense equations.
            for angle in (pi/3, pi/3 + 0.01, pi, 0.0, pi/3, pi/2)
                current = AS._solve_region_bem(compressed, recycled;
                    incidence_angle = angle, incidence_azimuth = 0.4)
                expected = AS._solve_region_bem(dense, dense_factor;
                    incidence_angle = angle, incidence_azimuth = 0.4)
                @test diagnostics(current).converged
                @test diagnostics(current).scaled_relative_residual < 1.1e-10
                @test scattering_amplitude(current)≈scattering_amplitude(expected) rtol=2e-5 atol=2e-9
                traces = [reduce(vcat, [s.pressure for s in current.data.interfaces]);
                          reduce(vcat,
                              [s.normal_derivative_interior ./
                               materials[j].density_contrast
                               for (j, s) in enumerate(current.data.interfaces)])]
                rhs = AS._region_incident_rhs(dense, angle, 0.4)
                @test AS._fluid_residual_report(dense.A*traces-rhs, rhs,
                    dense_factor.row_norms).scaled_relative_residual < 2e-7
            end
        end
    end
    surfaces, materials = [outer, offset], [weak, gas]
    # Unscaled solves, near fields, and reuse are independent of the matrix representation.
    options = (; compression, equilibrate = false, condition_limit = 0)
    solution = bem(surfaces, materials, 0.3; options...)
    reference = bem(surfaces, materials, 0.3; condition_limit = 0)
    @test diagnostics(solution).converged
    @test diagnostics(solution).relative_residual ≈
          diagnostics(solution).scaled_relative_residual
    @test pressure(solution, (1.5, 0.0, 0.0)) ≈ pressure(reference, (1.5, 0.0, 0.0)) rtol = 2e-5
    @test pressure(solution, (0.0, 0.7, 0.0); region = 1) ≈
          pressure(reference, (0.0, 0.7, 0.0); region = 1) rtol = 2e-5
    @test pressure(solution, (0.3, 0.1, 0.0); field = :interior, region = 2) ≈
          pressure(reference, (0.3, 0.1, 0.0); field = :interior, region = 2) rtol = 2e-5
    angles = [0.0, pi/3, 0.0]
    sweep = incidence_angle_sweep(surfaces, materials, 0.3, angles; compression,
        condition_limit = 0, components = true)
    fresh = [AS._comparison_amplitudes(components(bem(surfaces, materials, 0.3;
                 compression, condition_limit = 0, incidence_angle)))
             for incidence_angle in angles]
    @test sweep.amplitudes≈permutedims(reduce(hcat, fresh)) rtol=2e-7 atol=1e-10
    @test sweep.amplitudes[1, :] ≈ sweep.amplitudes[end, :] rtol = 1e-10
    unrecycled = incidence_angle_sweep(surfaces, materials, 0.3, reverse(angles);
        compression, condition_limit = 0, components = true, recycle_dimension = 0)
    @test unrecycled.amplitudes≈reverse(sweep.amplitudes; dims = 1) rtol=2e-7 atol=1e-10
    @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, 0.3, angles;
        recycle_dimension = -1)
    @test_throws ArgumentError incidence_angle_sweep(outer, weak, 0.3, angles;
        recycle_dimension = -1)
    separated = components(solution)
    @test all(s -> diagnostics(s).method === :gmres, separated.isolated)
    dense_isolated = components(solution; solver_kwargs = (;
        compression = (method = :none,)))
    @test all(s -> diagnostics(s).method === :direct, dense_isolated.isolated)
    for keywords in ((; compression, formulation = :cbie),
        (; compression, correction = (method = :edge,)),
        (; compression = (method = :fmm,)),
        (; compression = (method = :hmatrix, tol = NaN)),
        (; compression, gmres_kwargs = (; log = false)))
        @test_throws ArgumentError bem([outer], [weak], 0.3; keywords...)
    end
    @test_logs (:warn, r"GMRES did not converge") begin
        failed = bem(surfaces, materials, 0.3; compression,
            gmres_kwargs = (; maxiter = 1, reltol = 1e-14))
        @test !diagnostics(failed).converged
    end
end
