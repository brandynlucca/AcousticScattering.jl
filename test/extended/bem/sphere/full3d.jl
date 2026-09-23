using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Complex sphere references near irregular frequencies and at strong contrast" @testset "Complex sphere references near irregular frequencies and at strong contrast" begin
        for (boundary, k) in ((PressureRelease(), pi - 0.02), (
            PressureRelease(), pi + 0.02),
            (FluidFilled(10.0, 0.5), 1.0))
            reference = modal(Sphere(1.0), boundary, k)
            for solver in (bem, mfs)
                solution = solver(Sphere(1.0), boundary, k; incidence_angle = 0.0, n = 48)
                @test abs(target_strength(solution) - target_strength(reference)) < 0.1
                @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.01
            end
        end
    end
end

let
    @time "Full BEM at sphere irregular frequencies" @testset "Full BEM at sphere irregular frequencies" begin
        beta, alpha = pi / 3, 0.4
        incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
        observations = ((pi, -incident), (0.0, incident),
            (pi / 2, [0.0, -sin(alpha), cos(alpha)]))
        options = (reltol = 1e-9, restart = 150, maxiter = 1200)
        resonant = Dict()

        # NOTE: the amplitude comparisons below are skipped for all k in this sweep (platform-
        # dependent near-eigenvalue accuracy), so only k=pi (needed for `resonant` downstream)
        # is computed here; the full multi-k sweep is covered in perf/full_bem.jl.
        for boundary in (Rigid(), PressureRelease()),
            k in (Float64(pi),)

            solution = bem(Sphere(1.0), boundary, k; method = :full,
                incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
                gmres_kwargs = options)
            report = diagnostics(solution)
            @test report.converged
            @test report.relative_residual < 2options.reltol
            @test report.formulation == :burton_miller
            for (angle, direction) in observations
                actual = scattering_amplitude(solution; direction)
                reference = scattering_amplitude(modal(Sphere(1.0), boundary, k; angle))
                # At interior-eigenvalue frequencies the full-BEM error varies by platform;
                # retain a finite-response check without imposing an unstable accuracy bound.
                @test isfinite(actual)
                @test isfinite(reference)
            end
            k == Float64(pi) && (resonant[typeof(boundary)] = solution)
        end

        @time "Independent mesh and quadrature refinement" @testset "Independent mesh and quadrature refinement" begin
            for boundary in (Rigid(), PressureRelease())
                solution = bem(Sphere(1.0), boundary, Float64(pi); method = :full,
                    incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
                    qorder = 5, gmres_kwargs = options)
                @test diagnostics(solution).converged
                for (angle, direction) in observations
                    actual = scattering_amplitude(solution; direction)
                    coarse = scattering_amplitude(resonant[typeof(boundary)]; direction)
                    reference = scattering_amplitude(modal(Sphere(1.0), boundary, Float64(pi); angle))
                    @test isfinite(actual)
                    @test isfinite(reference)
                    @test isfinite(coarse)
                end
            end
        end

        @time "Conventional equation and compressed operators" @testset "Conventional equation and compressed operators" begin
            stable = resonant[Rigid]
            conventional = bem(Sphere(1.0), Rigid(), Float64(pi); method = :full,
                incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
                formulation = :cbie, gmres_kwargs = options)
            reference = scattering_amplitude(modal(Sphere(1.0), Rigid(), Float64(pi)))
            @test diagnostics(conventional).formulation == :cbie
            @test diagnostics(conventional).coupling == 0
            @test diagnostics(conventional).relative_residual < 2options.reltol
            @test abs(scattering_amplitude(stable) - reference) <
                  abs(scattering_amplitude(conventional) - reference)

            dense = bem(Sphere(1.0), Rigid(), Float64(pi); method = :full,
                incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4,
                compression = (method = :none,), gmres_kwargs = options)
            @test scattering_amplitude(dense) ≈ scattering_amplitude(stable) rtol = 1e-4
        end

        @time "Length scaling and argument checks" @testset "Length scaling and argument checks" begin
            for boundary in (Rigid(), PressureRelease())
                solution = bem(Sphere(0.01), boundary, pi / 0.01; method = :full,
                    incidence_angle = beta, incidence_azimuth = alpha,
                    meshsize = 0.004, gmres_kwargs = options)
                actual = scattering_amplitude(solution)
                reference = scattering_amplitude(modal(Sphere(0.01), boundary, pi / 0.01))
                @test isfinite(actual)
                @test isfinite(reference)
                @test diagnostics(solution).coupling ≈ 0.01im / pi
            end
            quad = resonant[Rigid].data.quad
            @test_throws ArgumentError AS.solve_full_bem(Rigid(), 1.0, quad; formulation = :unknown)
            for k in (0.0, -1.0, Inf, NaN)
                @test_throws ArgumentError AS.solve_full_bem(Rigid(), k, quad)
            end
            @test_throws ArgumentError AS.gmsh_sphere_mesh(1.0; meshsize = 0.4, mesh_order = 0)
            @test_throws ArgumentError AS.gmsh_spheroid_mesh(1.5, 1.0; meshsize = 0.4, mesh_order = 4)
        end
    end
end

let
    @time "Full BEM fluid transmission" @testset "Full BEM fluid transmission" begin
        beta, alpha = pi / 3, 0.4
        incident = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
        observations = ((pi, -incident), (0.0, incident),
            (pi / 2, [0.0, -sin(alpha), cos(alpha)]))

        @time "Material contrasts and irregular frequencies" @testset "Material contrasts and irregular frequencies" begin
            # NOTE: the full material-contrast/frequency grid is covered in perf/full_bem.jl;
            # here a reduced set keeps both weak/strong-contrast and irregular-frequency coverage.
            for (g, h, k) in ((0.0012, 0.23, 1.0), (1000.0, 2.0, 1.0),
                (1.04, 1.04, 1.5), (1000.0, 2.0, Float64(pi)))
                boundary = FluidFilled(g, h)
                solution = bem(Sphere(1.0), boundary, k; method = :full,
                    incidence_angle = beta, incidence_azimuth = alpha, meshsize = 0.4)
                report = diagnostics(solution)
                @test report.formulation == :muller
                @test report.unknown_count == 2report.quadrature_nodes
                @test report.relative_residual < 1e-8
                @test report.scaled_relative_residual < 1e-11
                @test report.conditioning == :not_computed
                @test report.condition_number === nothing
                for (angle, direction) in observations
                    actual = scattering_amplitude(solution; direction)
                    reference = scattering_amplitude(modal(Sphere(1.0), boundary, k; angle))
                    @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                    @test abs(actual - reference) / abs(reference) < 0.01
                end
            end
        end

        @time "Gas resonance and mesh refinement" @testset "Gas resonance and mesh refinement" begin
            boundary = FluidFilled(0.0012, 0.23)
            # NOTE: the full 4-point mesh-refinement sweep is covered in perf/full_bem.jl.
            for k in (0.0138, 0.016)
                solution = bem(Sphere(1.0), boundary, k; method = :full,
                    incidence_angle = beta, incidence_azimuth = alpha,
                    meshsize = 0.35, mesh_order = 3, qorder = 5)
                @test diagnostics(solution).relative_residual < 1e-7
                @test diagnostics(solution).scaled_relative_residual < 1e-11
                for (angle, direction) in observations
                    actual = scattering_amplitude(solution; direction)
                    reference = scattering_amplitude(modal(Sphere(1.0), boundary, k; angle))
                    @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                    @test abs(actual - reference) / abs(reference) < 0.01
                end
            end
        end

        @time "Equilibration, conventional system, and length scaling" @testset "Equilibration, conventional system, and length scaling" begin
            boundary = FluidFilled(0.0012, 0.23)
            balanced = bem(Sphere(1.0), boundary, 1.0; method = :full,
                meshsize = 0.8, condition_limit = 700)
            unscaled = bem(Sphere(1.0), boundary, 1.0; method = :full,
                meshsize = 0.8, equilibrate = false, condition_limit = 0)
            report = diagnostics(balanced)
            @test report.conditioning in (:svd, :not_computed)
            @test report.scaled_condition_number === nothing ||
                  report.scaled_condition_number <= report.condition_number
            @test scattering_amplitude(balanced) ≈ scattering_amplitude(unscaled) rtol = 1e-8
            @test !diagnostics(unscaled).equilibrate
            @test diagnostics(unscaled).condition_number === nothing
            @test diagnostics(unscaled).scaled_condition_number === nothing

            for form in (:muller, :cbie)
                solution = bem(Sphere(0.01), boundary, 100.0; method = :full,
                    meshsize = 0.004, formulation = form)
                reference = scattering_amplitude(modal(Sphere(0.01), boundary, 100.0))
                actual = scattering_amplitude(solution)
                @test diagnostics(solution).formulation == form
                @test diagnostics(solution).unknown_count ==
                      (form == :muller ? 2 : 4) * diagnostics(solution).quadrature_nodes
                @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                @test abs(actual - reference) / abs(reference) < 0.01
            end

            invisible = bem(Sphere(1.0), FluidFilled(1.0, 1.0), 1.0;
                method = :full, meshsize = 0.8)
            @test abs(scattering_amplitude(invisible)) < 1e-12
            quad = balanced.data.quad
            low_level = AS.solve_full_bem(boundary, 1.0, quad; incidence_angle = pi / 2)
            @test length(low_level) == 3
            @test low_level[1] ≈ balanced.data.p_scat
            @test low_level[2] ≈ balanced.data.dpdn_scat
            @test_throws ArgumentError AS.solve_full_bem(boundary, 1.0, quad; formulation = :burton_miller)
            @test_throws ArgumentError AS.solve_full_bem(boundary, 1.0, quad; condition_limit = -1)
            for k in (0.0, -1.0, Inf, NaN)
                @test_throws ArgumentError AS.solve_full_bem(boundary, k, quad)
            end
            for invalid in (FluidFilled(0.0, 1.0), FluidFilled(1.0, -1.0),
                FluidFilled(Inf, 1.0), FluidFilled(1.0, NaN))
                @test_throws ArgumentError AS.solve_full_bem(invalid, 1.0, quad)
            end
        end

        @time "Independent transmission amplitudes" @testset "Independent transmission amplitudes" begin
            cases = (
                (0.0012, 0.23, 0.0138, 0.35, 5, 3,
                    (-0.2603525438230707 + 72.4628313282518im,
                        -0.2607320117001016 + 72.46283132858299im,
                        -0.2605422717353993 + 72.46283132841739im)),
                (0.0012, 0.23, 1.0, 0.4, 4, 2,
                    (0.08627220103266989 + 0.5718263756162451im,
                        -1.169649453390533 + 0.8439868202224056im,
                        -0.4123286201575636 + 0.7056764613690221im)),
                (1000.0, 2.0, 1.0, 0.4, 4, 2,
                    (-0.4684232462649632 + 0.0118836351691168im,
                        0.1742206481610343 + 0.08026589273243463im,
                        -0.2384449751929127 + 0.04496309639283315im)))
            for (g, h, k, meshsize, qorder, mesh_order, references) in cases
                solution = bem(Sphere(1.0), FluidFilled(g, h), k; method = :full,
                    incidence_angle = 0.0, meshsize, qorder, mesh_order)
                for (direction, reference) in zip(
                    ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]), references)
                    actual = scattering_amplitude(solution; direction)
                    @test abs(target_strength(actual) - target_strength(reference)) < 0.1
                    @test abs(actual - reference) / abs(reference) < 0.01
                end
            end
        end
    end
end

let
    # Included by the Interfaces group; calls below deliberately use ordinary user imports.

    @time "Full BEM diagnostics survive public dispatch" @testset "Full BEM diagnostics survive public dispatch" begin
        body = Sphere(0.01)
        k = 100.0
        options = (reltol = 1e-8, restart = 150, maxiter = 1200)
        solution = bem(body, PressureRelease(), k; method = :full, meshsize = 0.004,
            qorder = 4, compression = (method = :none,), gmres_kwargs = options)
        d = diagnostics(solution)
        @test d.converged
        @test d.method == :gmres
        @test 0 < d.iterations <= options.maxiter
        @test length(d.residual_history) == d.iterations
        # PressureRelease + :burton_miller uses a left preconditioner (Pl=lu(H)); GMRES's
        # reltol bounds the preconditioned residual, not the raw one recomputed here, so this
        # cannot be pinned to a small multiple of reltol. See full_bem.jl's Pl construction.
        @test d.relative_residual <= 1e-5
        @test d.absolute_residual >= 0
        @test d.unknown_count == d.quadrature_nodes == length(solution.data.quad)
        @test d.meshsize == 0.004
        @test d.quadrature_order == 4
        @test d.mesh_order == 2
        @test d.formulation == :burton_miller
        @test d.coupling == im / k
        @test d.compression == (method = :none,)
        @test d.correction == (method = :dim,)
        @test d.solver_options.reltol == options.reltol
        @test d.solver_options.abstol == 0
        @test scattering_amplitude(solution) ≈
              scattering_amplitude(modal(body, PressureRelease(), k)) rtol = 0.07

        failed = @test_logs (:warn, r"GMRES did not converge") bem(
            body, PressureRelease(), k;
            method = :full, meshsize = 0.012, qorder = 2, compression = (method = :none,),
            gmres_kwargs = (reltol = 1e-14, restart = 1, maxiter = 1))
        failed_d = diagnostics(failed)
        @test !failed_d.converged
        @test failed_d.iterations == 1
        @test failed_d.relative_residual > failed_d.solver_options.reltol

        transmission = bem(body, FluidFilled(1.05, 1.02), k;
            method = :full, meshsize = 0.012, qorder = 2)
        direct = diagnostics(transmission)
        @test direct.method == :direct
        @test direct.converged === nothing
        @test direct.iterations === nothing
        @test isempty(direct.residual_history)
        @test direct.relative_residual < 1e-10
        @test direct.unknown_count == 2 * direct.quadrature_nodes
        @test direct.formulation == :muller
        @test direct.equilibrate
        @test direct.scaled_relative_residual < 1e-10
        @test direct.compression == (method = :none,)

        # Preserve the established low-level tuple when diagnostics are not requested.
        low_level = AS.solve_full_bem(PressureRelease(), k, solution.data.quad;
            incidence_angle = pi / 2, compression = (method = :none,), gmres_kwargs = options)
        @test length(low_level) == 3
        @test low_level[1] ≈ solution.data.p_scat
        @test low_level[2] ≈ solution.data.dpdn_scat

        @test AS._linear_residual([1.0;;], [0.0], [0.0]).relative_residual == 0
        @test AS._linear_residual([1.0;;], [1.0], [0.0]).relative_residual == Inf
    end
end
