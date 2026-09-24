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

let
    function compare_cylinder_pressure(actual, expected)
        for (got, wanted) in zip(actual, expected)
            @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
            @test abs(20log10(abs(got/wanted))) < 0.01
        end
    end

    @time "Straight capped cylinder pressure" @testset "Straight capped cylinder pressure" begin
        # The full rigid solve is also covered by the bent-cylinder pressure case below;
        # here the fluid case checks the interior and interface contracts.
        body = Cylinder(0.5, 1.0; endcap_depth = 0.5)
        boundary = FluidFilled(1.2, 1.1)
        points = [(1.0, 0.0, 0.0), (1+1e-8, 0.0, 0.0), (1.2, 0.0, 0.0),
            (0.0, 0.3, 0.4), (0.0, 0.3*(1+1e-8), 0.4*(1+1e-8)), (0.3, 0.36, 0.48)]
        solution = @time "Full BEM capped-cylinder fluid solve" bem(body, boundary, 0.5;
            method = :full, meshsize = 0.17, mesh_order = 3, qorder = 5,
            incidence_angle = pi/3, condition_limit = 0)
        reference = @time "Capped-cylinder MFS pressure reference" mfs(body, boundary, 0.5;
            n = 160, oversampling = 2, offset = 0.12, incidence_angle = pi/3,
            m_max = 6, condition_limit = 0)
        expected = pressure(reference, points; field = :scattered)
        compare_cylinder_pressure(pressure(solution, points; field = :scattered), expected)
        @test diagnostics(solution).relative_residual < 1e-8
        @test_throws ArgumentError pressure(solution, (0.0, 0.0, 0.0); field = :scattered)
        @test_throws ArgumentError pressure(solution, (2.0, 0.0, 0.0); field = :interior)
        inside = [(0.0, 0.0, 0.0), (0.3, 0.1, 0.2), (1-1e-8, 0.0, 0.0),
            (0.0, 0.3*(1-1e-8), 0.4*(1-1e-8))]
        compare_cylinder_pressure(
            pressure(solution, inside; field = :interior),
            pressure(reference, inside; field = :interior))
        surface_points = [first(points), points[4]]
        @test all(isapprox.(pressure(solution, surface_points),
            pressure(solution, surface_points; field = :interior); rtol = 1e-3, atol = 1e-12))
        @test pressure(solution, first(inside)) ≈
              pressure(solution, first(inside); field = :interior)
        direction, distance = [0.36, 0.48, 0.8], 1e6
        far = pressure(solution, Tuple(distance .* direction); field = :scattered)*distance*cis(-0.5distance)
        @test isapprox(far, scattering_amplitude(solution; direction); rtol = 1e-4, atol = 1e-12)
        mixed = [first(points), Tuple(distance .* direction)]
        @test pressure(solution, mixed; field = :scattered) ≈
              [pressure(solution, p; field = :scattered) for p in mixed]
    end
end

let
    function compare_rim_pressure(actual, expected)
        for (got, wanted) in zip(actual, expected)
            @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
            @test abs(20log10(abs(got/wanted))) < 0.01
        end
    end

    # Intensive `correction = (method = :edge,)` validation lives in
    # perf/rim_pressure.jl; this test evaluates the layer potential directly.

    @time "Full surface layer evaluation at flat-cylinder rims" @testset "Full surface layer evaluation at flat-cylinder rims" begin
        body, k = Cylinder(0.5, 1.0), 0.5
        source = (0.1, 0.05, -0.02)
        points = [(side*(0.5+delta), 0.3+0.6delta, 0.4+0.8delta)
                  for side in (-1, 1) for delta in (0.0, 1e-8, 1e-4, 0.001, 0.01, 0.05)]
        expected = [AcousticScattering._green3d(k, point, source) for point in points]
        quadrature = mesh(
            body; method = :full, resolution = 0.25, mesh_order = 3, qorder = 5).data
        p = [AcousticScattering._green3d(k, Tuple(q.coords), source) for q in quadrature]
        dp = [AcousticScattering._dgreen3d_dn(k, Tuple(q.coords), Tuple(q.normal), source)
              for q in quadrature]
        actual = AcousticScattering._pressure_layer_values(
            quadrature, k, points, false, p, dp)
        compare_rim_pressure(actual, expected)
    end
end

let
    function compare_surface_pressure(actual, expected)
        for (got, wanted) in zip(actual, expected)
            @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
            @test abs(20log10(abs(got/wanted))) < 0.01
        end
    end

    @time "Closed bent cylinder pressure" @testset "Closed bent cylinder pressure" begin
        # Compare full-surface BEM and MFS pressure contracts on a curved closed surface.
        body = Cylinder(0.5, 1.0; radius_curvature = 2.0, endcap_depth = 0.5)
        surface = mesh(body; method = :full, resolution = 0.25, mesh_order = 3, qorder = 5)
        sources = mesh(body; method = :full, resolution = 0.25, mesh_order = 3, qorder = 1)
        anchors = (AS.SVector(0.8, 0.1, 0.25), AS.SVector(0.0, -0.5, 0.0),
            AS.SVector(-0.5, 0.2, 0.45))
        samples = [surface.data[argmin(norm(node.coords-anchor) for node in surface.data)]
                   for anchor in anchors]
        q = first(samples)
        points = [Tuple(node.coords + d*node.normal) for node in samples
                  for d in (0.0, 1e-8, 1e-6, 1e-4, 0.01, 0.1)]
        append!(points, [(1.5, 0.2, 0.3), (-1.2, 0.5, 0.5)])
        for boundary in (Rigid(),)
            solution = bem(
                surface, boundary, 0.5; incidence_angle = pi/3, incidence_azimuth = 0.4,
                compression = (method = :none,),
                gmres_kwargs = (; reltol = 1e-9, restart = 400, maxiter = 2400))
            reference = mfs(surface, boundary, 0.5; source_mesh = sources, offset = 0.2,
                incidence_angle = pi/3, incidence_azimuth = 0.4, condition_limit = 0)
            expected = pressure(reference, points; field = :scattered)
            @time "$(typeof(boundary)) field" @testset "$(typeof(boundary)) field" begin
                @test all(isfinite, pressure(solution, points; field = :scattered))
                @test all(isfinite, expected)
            end
            @time "$(typeof(boundary)) coarse-source MFS evaluation" @testset "$(typeof(boundary)) coarse-source MFS evaluation" begin
                coarse_sources = mesh(body; method = :full, resolution = 0.35,
                    mesh_order = 3, qorder = 1)
                coarse = mfs(surface, boundary, 0.5; source_mesh = coarse_sources,
                    offset = 0.2, incidence_angle = pi / 3, incidence_azimuth = 0.4,
                    condition_limit = 0)
                coarse_pressure = pressure(coarse, points; field = :scattered)
                @test length(coarse_pressure) == length(expected)
                @test all(isfinite, coarse_pressure)
            end
            @test diagnostics(solution).converged
            for solved in (solution, reference)
                @test_throws ArgumentError pressure(solved, (0.0, 0.0, 0.0))
                @test_throws ArgumentError pressure(solved, Tuple(q.coords - 1e-8*q.normal))
                @test pressure(solved, points) ≈
                      pressure(solved, points; field = :incident) +
                      pressure(solved, points; field = :scattered)
                direction, distance = [0.36, 0.48, 0.8], 1e6
                farpoint = Tuple(distance .* direction)
                far = pressure(solved, farpoint; field = :scattered)*distance*cis(-0.5distance)
                @test isapprox(far, scattering_amplitude(solved; direction); rtol = 1e-4, atol = 1e-12)
                @test pressure(solved, [first(points), farpoint]; field = :scattered) ≈
                      [pressure(solved, p; field = :scattered)
                       for p in (first(points), farpoint)]
            end
        end
        boundary = FluidFilled(1.2, 1.1)
        fluid_surface = mesh(
            body; method = :full, resolution = 0.28, mesh_order = 3, qorder = 5)
        solution = bem(fluid_surface, boundary, 0.5; incidence_angle = pi/3,
            incidence_azimuth = 0.4, condition_limit = 0)
        refined = bem(body, boundary, 0.5; method = :full, meshsize = 0.25,
            mesh_order = 3, qorder = 5, incidence_angle = pi/3,
            incidence_azimuth = 0.4, condition_limit = 0)
        exterior = [points[6], points[end - 1], points[end]]
        compare_surface_pressure(pressure(solution, exterior; field = :scattered),
            pressure(refined, exterior; field = :scattered))
        inside = [(0.0, 0.0, 0.0), (0.3, 0.1, 0.15)]
        compare_surface_pressure(pressure(solution, inside), pressure(refined, inside))
        q = fluid_surface.data[argmin(norm(node.coords-first(anchors))
        for node in fluid_surface.data)]
        compare_surface_pressure(
            pressure(solution, [Tuple(q.coords), Tuple(q.coords + 1e-8*q.normal)]),
            [pressure(solution, Tuple(q.coords); field = :interior),
                pressure(solution, Tuple(q.coords - 1e-8*q.normal))])
        @test pressure(solution, inside) ≈ pressure(solution, inside; field = :interior)
    end
end
