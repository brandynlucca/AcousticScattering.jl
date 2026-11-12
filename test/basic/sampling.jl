using AcousticScattering
using Test

@testset "Sampling" begin
    sphere = Sphere(0.01)
    frequencies = [20_000.0, 30_000.0]
    sweep = frequency_sweep(k -> modal(sphere, Rigid(), k; m_max = 3),
        frequencies, 1500.0)
    @test sweep isa AcousticScattering.FrequencySweep
    @test sweep.frequencies == frequencies
    @test sweep.k ≈ 2pi .* frequencies ./ 1500.0
    @test size(sweep.amplitudes) == (2,)
    @test sweep.target_strength ≈ target_strength.(sweep.amplitudes)

    body = Cylinder(0.1, 0.5)
    angles = [pi / 4, pi / 2]
    aspect = incidence_angle_sweep(
        angle -> modal(body, Rigid(), 1.0; incidence_angle = angle, m_max = 2),
        angles)
    @test aspect isa AcousticScattering.IncidenceAngleSweep
    @test aspect.angles == angles
    @test length(aspect.amplitudes) == length(angles)
    @test aspect.target_strength ≈ target_strength.(aspect.amplitudes)

    irregular = Irregular(1.0, Float64[], Float64[])
    fourier_angles = [pi / 3, pi / 2]
    fourier_aspect = incidence_angle_sweep(irregular, PressureRelease(), 0.5,
        fourier_angles; mapping_order = 2, continuation_steps = 1,
        m_max = 2, n_max = 2)
    @test fourier_aspect.angles == fourier_angles
    @test length(fourier_aspect.amplitudes) == 2
    @test fourier_aspect.target_strength ≈
          target_strength.(fourier_aspect.amplitudes)

    solution = mfs(Sphere(1.0), Rigid(), 0.5; incidence_angle = 0.0,
        n = 12, offset = 0.2, condition_limit = 0)
    cut = bistatic_sweep(solution, [0.0, pi / 2, pi])
    @test cut isa AcousticScattering.BistaticSweep
    @test length(cut.amplitudes) == 3
    @test cut.target_strength ≈ target_strength.(cut.amplitudes)
    grid = bistatic_map(solution, [0.0, pi / 2], [0.0, pi])
    @test grid isa AcousticScattering.BistaticMap
    @test size(grid.target_strength) == (2, 2)
    @test grid.thetas == [0.0, pi / 2]

    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    surface = mesh(nodes, triangles; qorder = 2)
    coupled = bem([surface], [FluidFilled(1.2, 1.1)], 0.3;
        condition_limit = 0)
    comparison = components(coupled; labels = ["inclusion"])
    @test comparison isa AcousticScattering.ComponentComparison
    @test length(comparison.isolated) == 1
    @test comparison.labels == ["Coupled", "Isolated inclusion", "Coherent sum"]
    @test_throws ArgumentError components(coupled; labels = ["too", "many"])
    component_cut = bistatic_sweep(comparison, [0.0, pi])
    @test size(component_cut.amplitudes) == (2, 3)
    @test component_cut.labels == comparison.labels

    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        Float64[], 1500.0)
    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        [20_000.0], 0.0)
    @test_throws ArgumentError incidence_angle_sweep(angle -> modal(body, Rigid(), 1.0),
        Float64[])
    @test_throws ArgumentError bistatic_sweep(solution, Float64[])
    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        [-1.0], 1500.0)
    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        [NaN], 1500.0)
    @test_throws ArgumentError incidence_angle_sweep(angle -> modal(body, Rigid(), 1.0),
        [NaN])
    @test_throws ArgumentError incidence_angle_sweep(irregular, PressureRelease(), 0.5,
        Float64[])
    @test_throws ArgumentError bistatic_sweep(solution, [NaN])
    @test_throws ArgumentError bistatic_sweep(solution, [0.0]; azimuth = Inf)
    @test_throws ArgumentError incidence_angle_sweep([surface], [FluidFilled(1.2, 1.1)],
        0.3, [0.0]; labels = ["inclusion"])
end

@testset "Sweep constructors and mesh sweeps" begin
    @test AcousticScattering.FrequencySweep([1.0, 2.0], [1.0, 2.0], [-10.0, -11.0]).labels ==
          ["Scattered field"]
    @test AcousticScattering.IncidenceAngleSweep([0.0, 1.0], [-10.0, -11.0]).amplitudes ===
          nothing
    @test AcousticScattering.BistaticSweep([0.0, 1.0], 0.0, [-10.0, -11.0], 0.3).incidence_azimuth ==
          0.0

    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    tetra = mesh(nodes, triangles; qorder = 2)
    angles = [0.0, 0.5]
    rigid = incidence_angle_sweep(tetra, Rigid(), 0.3, angles;
        compression = (method = :none,))
    @test length(rigid.target_strength) == 2 && all(isfinite, rigid.target_strength)
    fluid = incidence_angle_sweep(tetra, FluidFilled(1.2, 1.1), 0.3, angles)
    @test length(fluid.amplitudes) == 2 && all(isfinite, fluid.target_strength)
    coupled = incidence_angle_sweep([tetra], [FluidFilled(1.2, 1.1)], 0.3, angles;
        components = true, labels = ["body"])
    @test size(coupled.amplitudes) == (2, 3)
    @test coupled.labels == ["Coupled", "Isolated body", "Coherent sum"]
    @test_throws ArgumentError incidence_angle_sweep(
        [tetra], [FluidFilled(1.2, 1.1)], 0.3, angles; labels = ["body"])

    scalar = frequency_sweep(
        k -> fem(Cylinder(0.05, 0.1), Rigid(), k; incidence_angle = 0.0,
            n_r = 6, n_theta = 12, m_max = 0),
        [1000.0, 2000.0], 1477.4)
    @test scalar.amplitudes isa Vector{ComplexF64} && all(isfinite, scalar.amplitudes) &&
          all(isfinite, scalar.target_strength)
    @test scalar.target_strength ≈ target_strength.(scalar.amplitudes)
end

@testset "Scoped exact operator reuse for curved BEM and coupled FEM" begin
    AS = AcousticScattering
    @test AS._numerical_reuse_context() === nothing
    @test_throws ErrorException AS._with_numerical_reuse() do
        error("intentional callback failure")
    end
    @test AS._numerical_reuse_context() === nothing
    cache = AS._NumericalReuse()
    cache.limit = 2Base.summarysize(zeros(8))
    first_value = AS._reuse_numerical(()->zeros(8), cache, :first)
    @test AS._reuse_numerical(()->error("unexpected rebuild"), cache, :first) ===
          first_value
    AS._reuse_numerical(()->ones(8), cache, :second)
    AS._reuse_numerical(()->ones(8), cache, :third)
    @test !haskey(cache.entries, :first)
    @test cache.bytes <= cache.limit
    AS._reuse_numerical(()->ones(100), cache, :oversize)
    @test !haskey(cache.entries, :oversize)
    cache = AS._NumericalReuse()
    cache.limit = 4Base.summarysize(zeros(8))
    AS._reuse_numerical(()->zeros(8), cache, (:shared, 1))
    AS._reuse_numerical(()->zeros(8), cache, (:fembem_operator, 1))
    AS._reuse_numerical(()->zeros(8), cache, (:fembem_operator, 2))
    @test haskey(cache.entries, (:shared, 1))
    @test !haskey(cache.entries, (:fembem_operator, 1))
    @test haskey(cache.entries, (:fembem_operator, 2))
    @test cache.bytes<=cache.limit
    cache = AS._NumericalReuse()
    cache.limit = 2Base.summarysize(zeros(8))
    for m in 0:4
        AS._reuse_numerical(()->zeros(8), cache, (:fembem_angular, m))
    end
    @test haskey(cache.entries, (:fembem_angular, 0))
    @test haskey(cache.entries, (:fembem_angular, 1))
    @test !haskey(cache.entries, (:fembem_angular, 4))
    AS._reuse_numerical(()->zeros(8), cache, (:fembem_operator, 0))
    @test haskey(cache.entries, (:fembem_angular, 0))
    adaptive_cache=AS._NumericalReuse(; retain_prefix = false)
    adaptive_cache.limit=2Base.summarysize(zeros(8))
    for m in 0:4
        AS._reuse_numerical(()->zeros(8), adaptive_cache, (:fembem_radial_inverse, m))
    end
    @test !haskey(adaptive_cache.entries, (:fembem_radial_inverse, 0))
    @test haskey(adaptive_cache.entries, (:fembem_radial_inverse, 3))
    @test haskey(adaptive_cache.entries, (:fembem_radial_inverse, 4))
    @test adaptive_cache.bytes<=adaptive_cache.limit

    body, k = Spheroid(0.02, 0.015), 100.0
    angles = [0.0, 0.6, pi/2, -0.6, 0.6]
    for bc in (Rigid(), PressureRelease(), FluidFilled(1.2, 0.85), FluidFilled(0.0012, 0.23))
        solve = if bc isa FluidFilled
            beta -> coupled(
                body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem),
                n_r = 3, n_theta = 16, n_boundary = 12, order = 4, boundary_order = 4,
                m_max = 4, incidence_angle = beta, adaptive = false, mode_threads = 2, rtol = 1e-8)
        else
            beta -> bem(body, bc, k; n = 12, boundary_order = 4, m_max = 4,
                incidence_angle = beta, adaptive = false, mode_threads = 2, rtol = 1e-8)
        end
        direct = solve.(angles)
        visited = Float64[]
        cached = Any[]
        sweep = incidence_angle_sweep(angles) do beta
            push!(visited, beta)
            solution = solve(beta)
            push!(cached, solution)
            solution
        end
        @test visited == angles
        @test sweep.amplitudes≈scattering_amplitude.(direct) rtol=1e-11 atol=1e-14
        @test AS._numerical_reuse_context() === nothing
        for (old, new) in zip(direct, cached)
            @test scattering_amplitude(new; angle = 0.35, azimuth = 0.4)≈
            scattering_amplitude(old; angle = 0.35, azimuth = 0.4) rtol=1e-11 atol=1e-14
            @test length(diagnostics(new).systems) == length(diagnostics(old).systems)
            @test diagnostics(new).relative_residual < 1e-7
        end
    end
    # Callback parameters need not be constant. Cache keys must isolate their
    # frequency, geometry, material, quadrature and mesh choices.
    variants = ((body, k, 12, 0), (Spheroid(0.021, 0.015), k, 12, 0),
        (body, 1.1k, 12, 0), (body, k, 16, 0), (body, k, 12, 2))
    varied = i -> bem(variants[Int(i)][1], Rigid(), variants[Int(i)][2];
        n = variants[Int(i)][3], quadrature_extra = variants[Int(i)][4],
        boundary_order = 3, m_max = 2, incidence_angle = 0.6, adaptive = false)
    expected = scattering_amplitude.(varied.(1:5))
    @test incidence_angle_sweep(varied, collect(1:5)).amplitudes ≈ expected rtol=1e-12
    for bc in (Rigid(), FluidFilled(1.2, 0.85))
        solve = if bc isa FluidFilled
            beta -> coupled(
                body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem),
                n_r = 2, n_theta = 12, n_boundary = 8, order = 4, boundary_order = 4, m_max = 4,
                incidence_angle = beta, rtol = 1e-8, target_tol = 1e-4, mode_threads = 2)
        else
            beta -> bem(body, bc, k; n = 8, boundary_order = 4, m_max = 4,
                incidence_angle = beta, rtol = 1e-8, target_tol = 1e-4, adaptive = true, mode_threads = 2)
        end
        angles = [0.6, pi/2, -0.6]
        direct = solve.(angles)
        cached = Any[]
        incidence_angle_sweep(angles) do beta
            solution = solve(beta)
            push!(cached, solution)
            solution
        end
        for (old, new) in zip(direct, cached)
            @test scattering_amplitude(new) ≈ scattering_amplitude(old) rtol=1e-11
            @test diagnostics(new).refinement.converged ==
                  diagnostics(old).refinement.converged
            @test diagnostics(new).refinement.mode_refinements ==
                  diagnostics(old).refinement.mode_refinements
        end
    end
    # Changing the reflection decomposition must not reuse a differently sized
    # coupled Schur operator, even when the physical mesh counts match.
    AS._with_numerical_reuse() do
        options = (; half = false, incidence_angle = 0.6, m_max = 2, mode_threads = 2,
            rtol = 1e-8, chief_points = 8, reports = nothing)
        split = AS._spheroid_fembem_level(body, FluidFilled(1.2, 0.85), k, 3, 16, 12, 4, 4;
            options..., split_symmetry = true)
        full = AS._spheroid_fembem_level(body, FluidFilled(1.2, 0.85), k, 3, 16, 12, 4, 4;
            options..., split_symmetry = false)
        for (a, b) in zip(split, full)
            @test a.pressure≈b.pressure rtol=1e-9 atol=1e-12
            @test a.derivative≈b.derivative rtol=1e-9 atol=1e-10
        end
    end
end

module BEMFrequencySweepIntegrationChecks
using AcousticScattering, LinearAlgebra, Test
const AS = AcousticScattering

@testset "Public frozen-basis frequency sweep" begin
    surface = mesh(
        Sphere(1.0); method = :full, resolution = 0.8, mesh_order = 2, qorder = 4)
    material = FluidFilled(1.05, 1.02)
    speed = 1500.0
    frequencies = [0.6, 0.7, 1.2, 0.7] .* speed ./ (2π)
    training_frequencies = [0.6, 0.8] .* speed ./ (2π)
    angles = (; incidence_angle = 0.7, incidence_azimuth = 0.3)
    direct = frequency_sweep(
        k -> bem(surface, material, k;
            formulation = :cbie, condition_limit = 0, angles...),
        frequencies, speed)
    reduced = frequency_sweep(surface, material, frequencies, speed;
        training_frequencies, angles..., return_diagnostics = true)
    @test reduced.sweep.frequencies == frequencies
    @test reduced.sweep.k ≈ [0.6, 0.7, 1.2, 0.7]
    @test reduced.sweep.amplitudes≈direct.amplitudes rtol=1e-8 atol=1e-12
    @test reduced.sweep.target_strength ≈ direct.target_strength atol=1e-7
    report = reduced.diagnostics
    @test report.basis_rank == 2
    @test report.basis_bytes == 16*2length(surface.data)*report.basis_rank
    @test first(report.queries).reduced
    @test any(q -> !q.reduced, report.queries)
    @test all(q -> max(q.residual.original, q.residual.scaled) <= 1e-10, report.queries)
    @test length(report.training) == 2
    # Empty storage forces independently validated full solves at every query.
    fallback = frequency_sweep(surface, material, frequencies, speed;
        training_frequencies, maxbytes = 0, angles..., return_diagnostics = true)
    @test fallback.diagnostics.basis_rank == 0
    @test all(q -> !q.reduced && q.reason === :empty, fallback.diagnostics.queries)
    @test fallback.sweep.amplitudes ≈ direct.amplitudes rtol=1e-12
    plain = frequency_sweep(surface, material, frequencies[1:1], speed;
        training_frequencies, capacity = 0, angles...)
    @test plain isa AS.FrequencySweep
    @test plain.amplitudes ≈ direct.amplitudes[1:1] rtol=1e-12
    for bad in (Float64[], [NaN], [-1.0])
        @test_throws ArgumentError frequency_sweep(
            surface, material, bad, speed; training_frequencies)
        @test_throws ArgumentError frequency_sweep(surface, material, frequencies, speed;
            training_frequencies = bad)
    end
    for options in ((capacity = -1,), (maxbytes = -1,), (tolerance = NaN,),
        (rank_tolerance = 0.0,), (incidence_angle = Inf,))
        @test_throws ArgumentError frequency_sweep(surface, material, frequencies, speed;
            training_frequencies, options...)
    end
    @test_throws ArgumentError frequency_sweep(
        surface, material, frequencies, 0.0; training_frequencies)
end
end
