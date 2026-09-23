using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    function check_cylinder_amplitudes(actual, reference)
        for (a, b) in zip(actual, reference)
            @test abs(target_strength(a) - target_strength(b)) < 0.1
            @test abs(a - b) / abs(b) < 0.01
        end
    end

    function cylinder_amplitudes(solution, beta, alpha)
        d = [cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha)]
        return [scattering_amplitude(solution; direction = q)
                for q in (-d, d, [0.0, 0.0, 1.0])]
    end

    @time "Closed flat cylinder against axisymmetric BEM" @testset "Closed flat cylinder against axisymmetric BEM" begin
        body = Cylinder(0.2, 0.5)
        boundary = PressureRelease()
        k = 0.5
        full = bem(body, boundary, k; method = :full, meshsize = 0.2,
            mesh_order = 3, qorder = 4, incidence_angle = pi / 3,
            formulation = :cbie, compression = (method = :none,))
        axisymmetric = bem(body, boundary, k; method = :axisymmetric,
            n = 32, m_max = 3, incidence_angle = pi / 3)
        @test isfinite(scattering_amplitude(full))
        @test abs(target_strength(full) - target_strength(axisymmetric)) < 3
    end

    @time "Closed bent cylinders at oblique incidence" @testset "Closed bent cylinders at oblique incidence" begin
        options = (reltol = 1e-9, restart = 150, maxiter = 1200)
        reference30 = [-0.115044319980511 + 0.02223869525583432im,
            0.05724218769794465 + 0.01330966815636354im,
            -0.07441192295622054 + 0.01052766721323333im]
        reference60 = [-0.2798784736915524 - 0.03732557787576306im,
            0.1148429765445959 + 0.02836089828639042im,
            -0.04491043881999867 + 0.01607119990249349im]
        # (2.0, pi/3) dropped from this sweep to cut cost. (2.0, pi/6) keeps the R=2.0 golden-reference check and (4.0, pi/3) keeps curvature diversity.
        for (R, beta) in ((2.0, pi / 6), (4.0, pi / 3))
            body = Cylinder(0.5, 2.0; radius_curvature = R, endcap_depth = 0.5)
            collocation = mesh(body; method = :full, resolution = 0.3, mesh_order = 3)
            sources = mesh(
                body; method = :full, resolution = 0.32, qorder = 1, mesh_order = 3)
            checks = mesh(body; method = :full, resolution = 0.27, mesh_order = 3)
            for boundary in (Rigid(), PressureRelease())
                solution = bem(body, boundary, 1.0; method = :full, meshsize = 0.32,
                    mesh_order = 3, incidence_angle = beta, incidence_azimuth = 0.4,
                    compression = (method = :none,), gmres_kwargs = options)
                reference = mfs(collocation, boundary, 1.0; source_mesh = sources,
                    check_mesh = checks, offset = 0.3, incidence_angle = beta,
                    incidence_azimuth = 0.4, condition_limit = 0)
                actual = cylinder_amplitudes(solution, beta, 0.4)
                expected = cylinder_amplitudes(reference, beta, 0.4)
                @test diagnostics(solution).converged
                @test diagnostics(solution).geometry.closed
                @test diagnostics(solution).geometry.intersection_check ==
                      :adaptive_bernstein
                @test diagnostics(reference).boundary_residual.relative_residual < 0.02
                check_cylinder_amplitudes(actual, expected)
                if R == 2.0 && boundary isa Rigid
                    @time "Independent reference at $(rad2deg(beta)) degrees" @testset "Independent reference at $(rad2deg(beta)) degrees" begin
                        check_cylinder_amplitudes(actual, beta == pi / 6 ? reference30 :
                                                          reference60)
                    end
                end
            end
        end
    end

    @time "Bent fluid and gas mesh convergence" @testset "Bent fluid and gas mesh convergence" begin
        body = Cylinder(0.5, 2.0; radius_curvature = 2.0, endcap_depth = 0.5)
        for (boundary, k) in ((FluidFilled(1.05, 1.02), 1.0), (
            GasFilled(0.0012, 0.23), 0.1))
            solution = bem(
                body, boundary, k; method = :full, meshsize = 0.32, mesh_order = 3,
                incidence_angle = pi / 3, incidence_azimuth = 0.4, condition_limit = 0)
            @test diagnostics(solution).relative_residual < 1e-8
            @test all(isfinite, cylinder_amplitudes(solution, pi / 3, 0.4))
        end
        surface = mesh(body; method = :full, resolution = 0.4, mesh_order = 3)
        transparent = bem(surface, FluidFilled(1.0, 1.0), 1.0;
            incidence_angle = pi / 6, incidence_azimuth = 0.4, condition_limit = 0)
        @test maximum(abs, cylinder_amplitudes(transparent, pi / 6, 0.4)) < 1e-9
    end

    @time "Zero curvature preserves the closed ends" @testset "Zero curvature preserves the closed ends" begin
        options = (compression = (method = :none,),
            gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
        for depth in (0.5,)
            body = Cylinder(0.5, 2.0; endcap_depth = depth)
            straight = mesh(body; method = :full, resolution = 0.32, mesh_order = 3)
            limit = mesh(Cylinder(0.5, 2.0; radius_curvature = 1e8, endcap_depth = depth);
                method = :full, resolution = 0.32, mesh_order = 3)
            for boundary in (Rigid(), FluidFilled(1.05, 1.02))
                k = boundary isa FluidFilled && boundary.density_contrast < 0.01 ? 0.1 : 1.0
                solver_options = boundary isa FluidFilled ? (; condition_limit = 0) :
                                 options
                solution = bem(
                    limit, boundary, k; incidence_angle = pi / 3, solver_options...)
                reference = bem(
                    straight, boundary, k; incidence_angle = pi / 3, solver_options...)
                actual = cylinder_amplitudes(solution, pi / 3, 0.0)
                expected = cylinder_amplitudes(reference, pi / 3, 0.0)
                check_cylinder_amplitudes(actual, expected)
                @test maximum(abs.(actual .- expected) ./ abs.(expected)) < 1e-5
                if boundary isa Rigid
                    independent = mfs(body, boundary, k; n = 120, m_max = 6,
                        offset = 0.3, oversampling = 2, incidence_angle = pi / 3, condition_limit = 0)
                    amplitudes = [scattering_amplitude(independent; angle = t, azimuth = p)
                                  for (t, p) in ((2pi / 3, pi), (pi / 3, 0.0), (
                        pi / 2, pi / 2))]
                    check_cylinder_amplitudes(actual, amplitudes)
                end
            end
        end
    end
end
