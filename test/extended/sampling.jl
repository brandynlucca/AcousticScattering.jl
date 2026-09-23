using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Reusable fluid incidence sweeps" @testset "Reusable fluid incidence sweeps" begin
        outer = mesh(; semiaxes = (1.0, 0.7, 0.8), center = (0.2, -0.1, 0.15),
            resolution = 0.9, qorder = 4)
        inner = mesh(; semiaxes = (0.25, 0.16, 0.18), center = (0.3, -0.06, 0.1),
            resolution = 0.9, qorder = 4)
        surfaces = [outer, inner]
        materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
        angles = [0.6, pi / 2, 0.6]
        k, azimuth = 0.8, 0.4
        options = (;
            formulation = :muller, equilibrate = true, condition_limit = 0, incidence_azimuth = azimuth)
        single = incidence_angle_sweep(outer, materials[1], k, angles; options...)
        fresh_single = incidence_angle_sweep(angles) do incidence_angle
            bem(outer, materials[1], k; incidence_angle, options...)
        end
        @test single.amplitudes ≈ fresh_single.amplitudes rtol = 1e-11
        @test single.target_strength ≈ fresh_single.target_strength atol = 1e-10
        @test single.labels == ["Scattered field"]
        @test single.angles == angles
        @test single.amplitudes[1] == single.amplitudes[end]

        reused = incidence_angle_sweep(surfaces, materials, k, angles;
            components = true, labels = ["body", "inclusion"], options...)
        fresh = incidence_angle_sweep(angles) do incidence_angle
            components(bem(surfaces, materials, k; incidence_angle, options...);
                labels = ["body", "inclusion"])
        end
        @test reused.amplitudes ≈ fresh.amplitudes rtol = 1e-11
        @test reused.target_strength ≈ fresh.target_strength atol = 1e-10
        @test reused.labels == fresh.labels
        @test size(reused.amplitudes) == (3, 4)
        @test reused.amplitudes[:, 4] ≈ reused.amplitudes[:, 2] + reused.amplitudes[:, 3]
        @test reused.amplitudes[1, :] == reused.amplitudes[end, :]

        let (surface, material, wavenumber) = (inner, materials[2], k)
            sweep = incidence_angle_sweep(surface, material, wavenumber, [0.7])
            direct = bem(surface, material, wavenumber; incidence_angle = 0.7)
            @test sweep.amplitudes[1] ≈ scattering_amplitude(direct) rtol = 1e-11
        end
        siblings = [outer,
            mesh(; semiaxes = (0.25, 0.16, 0.18), center = (2.0, 0.3, 0.0),
                resolution = 0.9, qorder = 4)]
        separate = incidence_angle_sweep(siblings, materials, k, angles; parents = [0, 0])
        direct = incidence_angle_sweep(angles) do incidence_angle
            bem(siblings, materials, k; parents = [0, 0], incidence_angle)
        end
        @test separate.amplitudes ≈ direct.amplitudes rtol = 1e-11
        @test separate.labels == ["Scattered field"]

        @test_throws ArgumentError incidence_angle_sweep(outer, materials[1], k, Float64[])
        @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, [NaN])
        @test_throws ArgumentError incidence_angle_sweep(outer, materials[1], k, angles;
            incidence_azimuth = Inf)
        @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
            condition_limit = -1)
        @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
            components = true, labels = ["one"])
        @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
            labels = ["body", "inclusion"])
        @test_throws ArgumentError incidence_angle_sweep(surfaces, materials[1:1], k, angles)
        @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
            parents = [0, 2])
        @test_throws ArgumentError incidence_angle_sweep(outer, materials[1], 0.0, angles)
    end
end

let
    @time "Rigid and pressure-release incidence sweeps" @testset "Rigid and pressure-release incidence sweeps" begin
        surface = mesh(
            Sphere(1.0); method = :full, resolution = 0.4, mesh_order = 3, qorder = 4)
        angles = [pi / 6, pi / 2, pi / 6]
        gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200)
        for boundary in (Rigid(),),
            compression in ((method = :hmatrix, tol = 1e-7),),
            (formulation, k) in ((:cbie, 0.8),)
            options = (; formulation, compression, gmres_kwargs, incidence_azimuth = 0.4)
            reused = incidence_angle_sweep(surface, boundary, k, angles; options...)
            fresh = incidence_angle_sweep(angles) do incidence_angle
                solution = bem(surface, boundary, k; incidence_angle, options...)
                @test diagnostics(solution).converged
                @test diagnostics(solution).relative_residual < 2e-9
                solution
            end
            @test reused.amplitudes ≈ fresh.amplitudes rtol = 1e-10
            @test reused.target_strength ≈ fresh.target_strength atol = 1e-9
            @test reused.amplitudes[1] ≈ reused.amplitudes[end] rtol = 1e-12
            @test reused.angles == angles
            @test reused.labels == ["Scattered field"]
            reference = scattering_amplitude(modal(Sphere(1.0), boundary, k))
            @test maximum(abs.(reused.amplitudes .- reference)) / abs(reference) < 0.01
            @test maximum(abs.(reused.target_strength .- target_strength(reference))) < 0.1
        end
        for boundary in (Rigid(), PressureRelease())
            @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 1.0, Float64[])
            @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 1.0, [NaN])
            @test_throws ArgumentError incidence_angle_sweep(
                surface, boundary, 1.0, angles;
                incidence_azimuth = Inf)
            @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 0.0, angles)
            @test_throws ArgumentError incidence_angle_sweep(
                surface, boundary, 1.0, angles;
                formulation = :unknown)
            @test_throws ArgumentError bem(surface, boundary, 1.0; incidence_angle = NaN)
        end
    end

    @time "Bent-surface incidence sweeps" @testset "Bent-surface incidence sweeps" begin
        body = Cylinder(0.5, 2.0; radius_curvature = 2.0, endcap_depth = 0.5)
        surface = mesh(body; method = :full, resolution = 0.32, mesh_order = 3, qorder = 4)
        collocation = mesh(body; method = :full, resolution = 0.3, mesh_order = 3)
        sources = mesh(body; method = :full, resolution = 0.32, mesh_order = 3, qorder = 1)
        options = (;
            incidence_azimuth = 0.4, compression = (method = :hmatrix, tol = 1e-7),
            gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
        angles = [pi / 6, pi / 3]
        for boundary in (Rigid(),)
            reused = incidence_angle_sweep(surface, boundary, 1.0, angles; options...)
            fresh = incidence_angle_sweep(
                a -> bem(surface, boundary, 1.0;
                    incidence_angle = a, options...), angles)
            independent = incidence_angle_sweep(angles) do incidence_angle
                mfs(collocation, boundary, 1.0; source_mesh = sources, offset = 0.3,
                    incidence_angle, incidence_azimuth = 0.4, condition_limit = 0)
            end
            @test reused.amplitudes ≈ fresh.amplitudes rtol = 1e-10
            @test maximum(abs.(reused.target_strength - independent.target_strength)) < 0.1
            @test maximum(abs.((reused.amplitudes - independent.amplitudes) ./
                               independent.amplitudes)) < 0.01
            if boundary isa Rigid
                reference = [-0.115044319980511 + 0.02223869525583432im,
                    -0.2798784736915524 - 0.03732557787576306im]
                @test maximum(abs.((reused.amplitudes - reference) ./ reference)) < 0.01
                @test maximum(abs.(reused.target_strength - target_strength.(reference))) <
                      0.1
            end
        end
    end
end

let
    @time "Fourier matching: incidence_angle_sweep reuses the transition operator" @testset "Fourier matching: incidence_angle_sweep reuses the transition operator" begin
        a = 2.0
        k = 1.3
        body = Irregular(a, Float64[], Float64[])
        angles = [pi / 6, pi / 3, pi / 2, 2pi / 3]
        # This checks transition reuse, not truncation convergence; the public API test above
        # exercises the default order against the exact sphere solution.
        m_max = n_max = 6
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02))
            sweep = incidence_angle_sweep(body, boundary, k, angles;
                mapping_order = 4, continuation_steps = 1, m_max, n_max)
            for (i, angle) in enumerate(angles)
                individual = fourier(body, boundary, k; incidence_angle = angle,
                    mapping_order = 4, continuation_steps = 1, m_max, n_max)
                @test sweep.target_strength[i] ≈ target_strength(individual) atol = 1e-10
                @test sweep.amplitudes[i] ≈ scattering_amplitude(individual) atol = 1e-10
            end
        end
        @test_throws ArgumentError incidence_angle_sweep(body, Rigid(), k, Float64[])
    end
end

let
    @time "Visualization sampling (frequency/incidence-angle sweeps)" @testset "Visualization sampling (frequency/incidence-angle sweeps)" begin
        a = 0.01
        c_water = 1477.4
        sphere = AS.Sphere(a)

        @time "frequency_sweep: shape, k conversion, endpoint agreement" @testset "frequency_sweep: shape, k conversion, endpoint agreement" begin
            freqs = 20e3:10e3:60e3
            sweep = AS.frequency_sweep(k -> AS.modal(sphere, AS.Rigid(), k), freqs, c_water)
            @test sweep.frequencies == collect(freqs)
            @test length(sweep.k) == length(freqs)
            @test length(sweep.target_strength) == length(freqs)
            @test sweep.k ≈ 2pi .* collect(freqs) ./ c_water

            single = AS.frequency_sweep(k -> AS.modal(sphere, AS.Rigid(), k), [38000.0], c_water)
            k_direct = 2pi * 38000.0 / c_water
            ts_direct = AS.target_strength(AS.modal(sphere, AS.Rigid(), k_direct))
            @test single.target_strength[1] == ts_direct
        end

        @time "incidence_angle_sweep: shape, endpoint agreement" @testset "incidence_angle_sweep: shape, endpoint agreement" begin
            body = AS.Spheroid(0.02, 0.01)
            angles = [0.0, pi / 4, pi / 2]
            solve(angle) = AS.modal(body, AS.Rigid(), 1.0;
                incidence_angle = angle, m_max = 2, n_max = 3)
            sweep = AS.incidence_angle_sweep(solve, angles)
            @test sweep.angles == angles
            @test length(sweep.amplitudes) == length(angles)
            @test sweep.target_strength ≈ AS.target_strength.(sweep.amplitudes)
            @test first(sweep.amplitudes) ≈ AS.scattering_amplitude(solve(first(angles)))
            @test last(sweep.amplitudes) ≈ AS.scattering_amplitude(solve(last(angles)))
        end
    end

    @time "Visualization sampling (bistatic sweep/map)" @testset "Visualization sampling (bistatic sweep/map)" begin
        a = 0.01
        c_water = 1477.4
        k = 2pi * 38000.0 / c_water
        sphere = AS.Sphere(a)
        bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)

        @time "bistatic_sweep: shape, endpoint agreement, azimuthal periodicity" @testset "bistatic_sweep: shape, endpoint agreement, azimuthal periodicity" begin
            angles = 0:(pi / 4):pi
            sweep = AS.bistatic_sweep(bem_sol, angles)
            @test sweep.angles == collect(angles)
            @test length(sweep.target_strength) == length(angles)

            single = AS.bistatic_sweep(bem_sol, [pi]; azimuth = 0.3)
            ts_direct = AS.target_strength(bem_sol; angle = pi, azimuth = 0.3)
            @test single.target_strength[1] == ts_direct

            sweep_0 = AS.bistatic_sweep(bem_sol, [pi / 2]; azimuth = 0.0)
            sweep_2pi = AS.bistatic_sweep(bem_sol, [pi / 2]; azimuth = 2pi)
            @test sweep_0.target_strength[1] ≈ sweep_2pi.target_strength[1] atol = 1e-8
        end

        @time "bistatic_sweep/BistaticSweep records the solution's actual incidence angle" @testset "bistatic_sweep/BistaticSweep records the solution's actual incidence angle" begin
            β = deg2rad(30.0)
            spheroid = AS.Spheroid(0.02, 0.01)
            sol = AS.bem(spheroid, AS.Rigid(), k; incidence_angle = β, m_max = 8, n = 16)
            sweep = AS.bistatic_sweep(sol, 0:(pi / 4):pi; azimuth = pi)
            @test sweep.incidence_angle == β
        end

        @time "bistatic_map: shape, endpoint agreement" @testset "bistatic_map: shape, endpoint agreement" begin
            thetas = 0:(pi / 4):pi
            phis = 0:(pi / 2):(2pi)
            map_result = AS.bistatic_map(bem_sol, thetas, phis)
            @test size(map_result.target_strength) == (length(thetas), length(phis))
            @test map_result.target_strength[2, 3] ==
                  AS.target_strength(bem_sol; angle = thetas[2], azimuth = phis[3])
        end

        @time "axisymmetric vs full-BEM cross-validation (coordinate convention correctness)" @testset "axisymmetric vs full-BEM cross-validation (coordinate convention correctness)" begin
            full_sol = AS.bem(sphere, AS.Rigid(), k; method = :full,
                meshsize = AS.bem3d_elements_per_wavelength(k))
            thetas = [0.0, pi / 2, pi]
            phis = [0.0, pi / 2]
            axi_map = AS.bistatic_map(bem_sol, thetas, phis)
            full_map = AS.bistatic_map(full_sol, thetas, phis)
            @test all(abs.(axi_map.target_strength .- full_map.target_strength) .< 0.5)
            @test diagnostics(full_sol).converged
            @test scattering_amplitude(full_sol) ≈
                  scattering_amplitude(modal(sphere, Rigid(), k)) rtol = 0.07
        end
    end
end

let
    @time "Elastic and layered radial FEM frequency sweeps" @testset "Elastic and layered radial FEM frequency sweeps" begin
        body = Sphere(0.01)
        for boundary in (SolidElastic(2.7, 4.0, 2.0),
            Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8),
            Shelled(FluidLayer(1.04, 1.04), VacuumInterior(), 0.8),
            Shelled(FluidLayer(1.04, 1.04), FluidInterior(0.0012, 0.23), 0.8))
            solve = k -> fem(body, boundary, k; n_elements = 400, m_max = 10)
            sweep = frequency_sweep(solve, [12000.0, 38000.0], 1500.0)
            @test sweep.amplitudes == scattering_amplitude.(solve.(sweep.k))
            reference = scattering_amplitude.([modal(body, boundary, k; m_max = 10)
                                               for k in sweep.k])
            @test sweep.amplitudes ≈ reference rtol = 0.001
            @test sweep.target_strength == target_strength.(sweep.amplitudes)
        end
    end
end

let
    @time "Radial FEM complex frequency sweeps" @testset "Radial FEM complex frequency sweeps" begin
        body, boundary = Sphere(0.01), Rigid()
        solve = k -> fem(body, boundary, k; order = 2, n_elements = 40)
        sweep = frequency_sweep(solve, [12000.0, 38000.0, 80000.0], 1500.0)
        references = scattering_amplitude.([modal(body, boundary, k) for k in sweep.k])
        @test sweep.amplitudes == scattering_amplitude.(solve.(sweep.k))
        @test sweep.amplitudes ≈ references rtol = 1e-5
        @test sweep.target_strength == target_strength.(sweep.amplitudes)
    end
end

let
    @time "Ellipsoid construction and complex sweeps" @testset "Ellipsoid construction and complex sweeps" begin
        surface = mesh(; semiaxes = (0.3, 0.4, 0.6), center = (0.1, -0.2, 0.3),
            rotation = (axis = (0, 1, 0), angle = pi / 2), resolution = 0.8)
        nodes = surface.body.nodes
        radii = ((nodes[1, :] .- 0.1) ./ 0.6) .^ 2 .+
                ((nodes[2, :] .+ 0.2) ./ 0.4) .^ 2 .+
                ((nodes[3, :] .- 0.3) ./ 0.3) .^ 2
        @test maximum(abs.(radii .- 1)) < 1e-10
        @test_throws ArgumentError mesh(; semiaxes = (1, 0, 1))
        @test_throws ArgumentError mesh(; semiaxes = (1, 1, 1), tip_ratio = 0)
        @test_throws ArgumentError mesh(; semiaxes = (1, 1, 1),
            rotation = (axis = (0, 0, 0), angle = 1))

        solve = k -> modal(Sphere(0.1), Rigid(), k)
        sweep = frequency_sweep(solve, [100.0, 200.0], 1500.0)
        @test sweep.amplitudes == scattering_amplitude.(solve.(sweep.k))
        @test sweep.target_strength == target_strength.(sweep.amplitudes)
        radial = frequency_sweep(k -> fem(Sphere(0.1), Rigid(), k), [100.0], 1500.0)
        @test radial.amplitudes[1] ==
              scattering_amplitude(fem(Sphere(0.1), Rigid(), only(radial.k)))
        scalar_solve = k -> fem(Sphere(0.1), Rigid(), k;
            method = :meridian, n_r = 3, n_theta = 8, l_max = 3)
        scalar = frequency_sweep(scalar_solve, [100.0], 1500.0)
        @test scalar.amplitudes === nothing
        @test scalar.target_strength[1] == target_strength(scalar_solve(only(scalar.k)))
        @test_throws ArgumentError frequency_sweep(solve, Float64[], 1500.0)
        @test_throws ArgumentError frequency_sweep(solve, [100.0], 0.0)

        surfaces = [mesh(; semiaxes = (r, r, r), resolution = 0.9, qorder = 4)
                    for r in (1.0, 0.3)]
        materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
        sol = bem(surfaces, materials, 0.8; incidence_angle = 0.7, incidence_azimuth = 0.4)
        comparison = components(sol; labels = ["flesh", "bladder"])
        @test comparison.coupled === sol
        @test comparison.labels ==
              ["Coupled", "Isolated flesh", "Isolated bladder", "Coherent sum"]
        @test_throws ArgumentError components(sol; labels = ["one"])
        pattern = bistatic_sweep(comparison, [0.3, 1.2, 4.1]; azimuth = 0.6)
        @test pattern.incidence_azimuth == 0.4
        @test size(pattern.amplitudes) == (3, 4)
        @test pattern.amplitudes[:, 4] ≈ pattern.amplitudes[:, 2] + pattern.amplitudes[:, 3]
        direct = bem(
            surfaces[2], materials[2], 0.8; incidence_angle = 0.7, incidence_azimuth = 0.4)
        for (i, theta) in enumerate(pattern.angles)
            direction = [cos(theta), sin(theta) * cos(0.6), sin(theta) * sin(0.6)]
            @test pattern.amplitudes[i, 1] ≈ scattering_amplitude(sol; direction)
            @test pattern.amplitudes[i, 3] ≈ scattering_amplitude(direct; direction)
        end
        spectrum = frequency_sweep(_ -> comparison, [0.8 * 1500 / (2pi)], 1500.0)
        tilt = incidence_angle_sweep(_ -> comparison, [0.7])
        @test size(spectrum.amplitudes) == (1, 4)
        @test spectrum.amplitudes == tilt.amplitudes
        @test spectrum.amplitudes[1, 1] ≈ scattering_amplitude(sol)
        single_pattern = bistatic_sweep(sol, pattern.angles; azimuth = 0.6)
        @test single_pattern.amplitudes ≈ pattern.amplitudes[:, 1]
    end
end
