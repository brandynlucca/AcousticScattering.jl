using AcousticScattering, Test, SparseArrays, LinearAlgebra

@testset "Banded angular FEM preserves the generalized eigenproblem" begin
    AS = AcousticScattering
    bc = FluidFilled(1.2, 0.85)
    for p in (2, 5, 8), half in (false, true), m in (0, 3), parity in (0, -1)
        mesh = AS._spheroid_fembem_confocal(bc, 0.07, 0.01, 3, 8, p;
            half, assemble_bulk = false, quadrature_extra = 2)
        k = 600.0
        n = mesh.nt*p+1
        A, B = zeros(n, n), zeros(n, n)
        for (e, t) in enumerate(mesh.angular_factors)
            ix = (e-1)*p .+ (1:(p + 1))
            A[ix, ix] += t.stiffness+m^2*t.azimuthal-(k/bc.soundspeed_contrast)^2*t.wave
            B[ix, ix] += t.mass
        end
        free = (m > 0 ? 2 : 1):(parity == -1 || (!half && m > 0) ? n - 1 : n)
        result = AS._spheroid_fembem_angular(mesh, k, m, parity)
        E = result.vectors[free, :]
        eigenvalues = eigvals(Symmetric(A[free, free]), Symmetric(B[free, free]))
        @test result.values ≈ eigenvalues rtol=1e-12
        @test transpose(E)*B[free, free]*E ≈ I atol=2e-12
        residual = A[free, free]*E-B[free, free]*E*Diagonal(result.values)
        @test norm(residual) <= 2e-13*norm(A)*norm(E)
        @test all(iszero, result.vectors[setdiff(1:n, free), :])
    end
end

@testset "Explicit interface coupling and quadrature refinement" begin
    body, bc, k = Spheroid(0.02, 0.015), FluidFilled(1.2, 0.85), 100.0
    pairing = Coupling(interior = :fem, exterior = :bem)
    @test_throws UndefKeywordError coupled(body, bc, k)
    @test_throws ArgumentError fem(body, bc, k; method = :fem_bem)
    @test_throws ArgumentError coupled(body, bc, k; coupling = pairing, method = :fem_bem)
    @test_throws ArgumentError coupled(body, bc, k;
        coupling = Coupling(interior = :bem, exterior = :fem))
    @test_throws ArgumentError coupled(
        body, bc, k; coupling = pairing, quadrature_extra = -1)
    @test_throws ArgumentError fem(body, bc, k; order = 4, quadrature_extra = -1)
    @test_throws ArgumentError bem(body, Rigid(), k; quadrature_extra = -1)
    controls = (; incidence_angle = 0.6, m_max = 5, adaptive = false, mode_threads = 2)
    values = ComplexF64[]
    for extra in (0, 4, 8)
        sol = coupled(body, bc, k; coupling = pairing, controls...,
            order = 4, boundary_order = 5, n_r = 3, n_theta = 24,
            n_boundary = 16, rtol = 1e-9, quadrature_extra = extra)
        @test sol isa CoupledSolution
        @test sol.coupling === pairing
        @test diagnostics(sol).discretization === :coupled
        @test diagnostics(sol).solver_options.quadrature_extra == extra
        observations = [0.0, 0.7, pi/2]
        sweep = bistatic_sweep(sol, observations; azimuth = 0.4)
        @test sweep.amplitudes == [scattering_amplitude(sol; angle, azimuth = 0.4)
               for angle in observations]
        push!(values, scattering_amplitude(sol))
    end
    @test isapprox(values[2], values[3]; rtol = 1e-7, atol = 1e-11)
    reference = fem(body, bc, k; controls..., order = 6, n_r = 8,
        n_theta = 32, l_max = 14, retain_fields = false, quadrature_extra = 6)
    @test isapprox(values[3], scattering_amplitude(reference); rtol = 3e-6, atol = 1e-11)
    for boundary in (Rigid(), PressureRelease())
        a = bem(body, boundary, k; controls..., n = 16, boundary_order = 5,
            rtol = 1e-9, quadrature_extra = 4)
        b = bem(body, boundary, k; controls..., n = 16, boundary_order = 5,
            rtol = 1e-10, quadrature_extra = 8)
        @test isapprox(scattering_amplitude(a), scattering_amplitude(b); rtol = 1e-7)
        @test diagnostics(b).solver_options.quadrature_extra == 8
    end
end

@testset "Modal mesh-change TS bound" begin
    AS=AcousticScattering
    for amplitude in (1.0+0im, 0.4+0.9im, -0.7-0.2im)
        changes=ComplexF64[0.003 + 0.008im, -0.005 + 0.002im, 0.004 - 0.006im]
        estimate=AS._spheroid_hp_ts_change([amplitude], changes)
        for signs in Iterators.product((-1, 1), (-1, 1), (-1, 1))
            perturbed=amplitude+sum(signs[i]*changes[i] for i in eachindex(changes))
            actual=abs(20log10(abs(perturbed)/abs(amplitude)))
            @test actual<=estimate.change_db+1e-13
        end
    end
    phase=AS._spheroid_hp_ts_change([1.0+0im], [0.01im])
    @test phase.change_db ≈ 10log10(1+0.01^2) atol=1e-14
    @test AS._spheroid_hp_ts_change([0.0im], [0.0im]).change_db==0.0
    @test isinf(AS._spheroid_hp_ts_change([0.0im], [0.01im]).change_db)
end

@testset "Independent FEM-BEM refinement directions" begin
    AS = AcousticScattering
    @test AS._spheroid_fembem_refinement_limits(3)==(3, 3, 3)
    @test AS._spheroid_fembem_refinement_limits((radial = 2, angular = 3, boundary = 5))==(
        2, 3, 5)
    @test AS._spheroid_fembem_refinement_limits((boundary = 5, radial = 2, angular = 3))==(
        2, 3, 5)
    for invalid in (0, -1, 1.5, (2, 3, 4), (radial = 1, angular = 0, boundary = 4),
        (radial = 2, boundary = 4), (radial = 1, angular = 2, boundary = 3, extra = 4))
        @test_throws ArgumentError AS._spheroid_fembem_refinement_limits(invalid)
    end
    level=(2, 3, 4)
    # Separate coordinate checks expose cancellation between refinement errors.
    f(r, t, b) = complex(2.0^-r+0.1*2.0^-t, 0.01*2.0^-b)+0.3im*2.0^(-r-t)
    samples=Dict((r, t, b)=>f(r, t, b) for r in 1:2 for t in 2:3 for b in 3:4)
    delta=AS._spheroid_fembem_directional_changes(samples, level)
    @test delta[1]≈f(2, 3, 4)-f(1, 3, 4)
    @test delta[2]≈f(2, 3, 4)-f(2, 2, 4)
    @test delta[3]≈f(2, 3, 4)-f(2, 3, 3)
    @test sum(delta)≈f(2, 3, 4)-f(1, 2, 3)
    @test !iszero(delta[4])
    @test AS._spheroid_fembem_mark_directions([1.0, 0.01, 0.001, 0.0], level, (5, 5, 5))==(
        3, 3, 4)
    @test AS._spheroid_fembem_mark_directions([1.0, 0.01, 0.001, 0.0], level, (2, 5, 5))==level
    @test AS._spheroid_fembem_mark_directions([1.0, 0.9, 0.001, 0.0], level, (2, 5, 5))==(
        2, 4, 4)
    @test AS._spheroid_fembem_mark_directions([0.0, 0.0, 0.0, 1.0], level, (5, 5, 5))==(
        3, 4, 5)
    @test AS._spheroid_fembem_mark_directions(zeros(4), level, (5, 5, 5))==level
    @test AS._spheroid_fembem_mark_directions(ones(4), level, level)==level
end

@testset "FEM-BEM operator groups preserve existing spans" begin
    AS = AcousticScattering
    target_modes = vcat(filter(!=(12), collect(0:31)), 33, 50)
    cases = [
        (Int[], UnitRange{Int}[], UnitRange{Int}[]),
        ([7], [7:7], [7:7]),
        (collect(0:15), [0:15], [0:15]),
        (collect(0:31), [0:15, 16:31], [0:31]),
        (collect(0:32), [0:15, 16:31, 32:32], [0:31, 32:32]),
        (collect(5:54), [5:20, 21:36, 37:52, 53:54], [5:36, 37:54]),
        (collect(32:95), [32:47, 48:63, 64:79, 80:95], [32:63, 64:95]),
        ([0, 15, 17, 32], [0:15, 17:32], [0:15, 17:32]),
        ([3, 19, 35], [3:3, 19:19, 35:35], [3:3, 19:19, 35:35]),
        (target_modes, [0:15, 16:31, 33:33, 50:50], [0:31, 33:33, 50:50])]
    for (modes, original, merged) in cases, boundary_nodes in (1, 296, 400, 512, 513, 536, 720)
        saved_modes = copy(modes)
        groups = AS._spheroid_fembem_operator_groups(modes, boundary_nodes)
        expected = boundary_nodes<=512 ? merged : original
        @test groups == expected
        @test modes == saved_modes
        @test all(span -> length(span)<=(boundary_nodes<=512 ? 32 : 16), groups)
        covered = collect(Iterators.flatten(groups))
        # Preserve gaps between independently requested mode spans.
        @test covered == collect(Iterators.flatten(original))
        @test all(m -> m in covered, modes)
        @test length(covered) == length(unique(covered))
    end
    @test AS._spheroid_fembem_operator_groups(0:31, 512) == [0:31]
    @test AS._spheroid_fembem_operator_groups(0:31, 513) == [0:15, 16:31]
    @test 12 in only(AS._spheroid_fembem_operator_groups(target_modes[1:31], 512))
    target_groups = AS._spheroid_fembem_operator_groups(target_modes, 512)
    @test length.(target_groups) == [32, 1, 1]
    @test all(m -> all(span -> !(m in span), target_groups), 34:49)
end

@testset "Wider FEM-BEM operator spans preserve boundary matrices" begin
    AS = AcousticScattering
    boundary = AS._spheroid_curved_boundary(0.1, 0.1/1.001, 2, 8; half = true)
    controls = (; rtol = 1e-6, split = true, chief_points = 2)
    for (k, modes) in ((40.0, 0:31), (2pi*285000/1477.3, 176:207))
        wide = AS._spheroid_curved_operators(
            boundary, k, modes, Val(32), Val(8); controls...)
        left = AS._spheroid_curved_operators(
            boundary, k, first(modes):(first(modes)+15), Val(16), Val(8); controls...)
        right = AS._spheroid_curved_operators(
            boundary, k, (first(modes)+16):last(modes), Val(16), Val(8); controls...)
        for component in eachindex(wide)
            reference = vcat(left[component], right[component])
            @test length(wide[component]) == length(modes)
            for (actual, expected) in zip(wide[component], reference)
                @test size(actual) == (length(boundary.chi)+2, length(boundary.chi))
                @test all(isfinite, actual)
                @test norm(actual-expected) <= 2e-7norm(expected)
            end
        end
    end
    # A partial final span exercises padding without adding physical modes.
    partial = AS._spheroid_curved_operators(
        boundary, 40.0, 3:21, Val(32), Val(8); controls...)
    left = AS._spheroid_curved_operators(
        boundary, 40.0, 3:18, Val(16), Val(8); controls...)
    right = AS._spheroid_curved_operators(
        boundary, 40.0, 19:21, Val(4), Val(8); controls...)
    for component in eachindex(partial)
        reference = vcat(left[component], right[component])
        @test length(partial[component]) == 19
        for (actual, expected) in zip(partial[component], reference)
            @test norm(actual-expected) <= 2e-7norm(expected)
        end
    end
end

@testset "FEM-BEM complete Fourier prefixes respect storage and reuse limits" begin
    AS = AcousticScattering
    for nodes in (296, 400, 512, 513)
        baseline = AS._spheroid_fembem_operator_groups(0:333, nodes)
        expanded = AS._spheroid_fembem_operator_groups(0:341, nodes)
        @test collect(Iterators.flatten(baseline)) == collect(0:333)
        @test collect(Iterators.flatten(expanded)) == collect(0:341)
        @test baseline[1:end-1] == expanded[1:length(baseline)-1]
        for span in baseline
            if length(span)>32
                @test length(span) in (64, 128)
                @test mod(first(span), length(span)) == 0
                @test 64nodes*(nodes+8)*length(span) <= 768*1024^2
            end
        end
    end
    @test AS._spheroid_fembem_operator_groups(0:333, 296) ==
          [0:127, 128:255, 256:319, 320:333]
    @test AS._spheroid_fembem_operator_groups(0:333, 400) ==
          [0:63, 64:127, 128:191, 192:255, 256:319, 320:333]
    @test maximum(length, AS._spheroid_fembem_operator_groups(0:333, 512)) == 32
    @test maximum(length, AS._spheroid_fembem_operator_groups(0:333, 513)) == 16
    sparse = vcat(collect(0:63), collect(80:127), [145, 161])
    groups = AS._spheroid_fembem_operator_groups(sparse, 296)
    @test groups == [0:63, 80:111, 112:127, 145:145, 161:161]
    @test collect(Iterators.flatten(groups)) == sparse

    boundary = AS._spheroid_curved_boundary(0.1, 0.1/1.001, 2, 8; half = true)
    controls = (;rtol = 1e-6, split = true, chief_points = 2)
    for modes in (0:127, 192:319)
        wide = AS._spheroid_curved_operators(boundary, 2pi*285000/1477.3,
            modes, Val(128), Val(8); controls...)
        parts = [AS._spheroid_curved_operators(boundary, 2pi*285000/1477.3,
                     start:start+31, Val(32), Val(8); controls...)
                 for start in first(modes):32:last(modes)]
        for component in eachindex(wide)
            reference = vcat((part[component] for part in parts)...)
            @test length(wide[component]) == length(modes)
            for (actual, expected) in zip(wide[component], reference)
                @test size(actual) == size(expected)
                @test all(isfinite, actual)
                @test norm(actual-expected) <= 2e-7norm(expected)
            end
        end
    end
end

@testset "Wider FEM-BEM groups exclude retained operator copies" begin
    AS = AcousticScattering
    original = [start:min(start+31, 127) for start in 0:32:127]
    @test AS._spheroid_fembem_operator_groups(0:127, 296; allow_wide = false) == original
    body = Spheroid(0.1, 0.09)
    boundary = GasFilled(0.00129, 0.23)
    function setup(; chief_points = 8, wide_operators = true)
        AS._spheroid_fembem_level_setup(body, boundary, 40.0, 2, 4, 4, 2, 1;
            half = true, incidence_angle = pi/2, m_max = 127, mode_threads = 1,
            rtol = 1e-6, chief_points, reports = nothing, modes = 0:127,
            interior_field = :total, wide_operators)
    end
    @test setup().groups == [0:127]
    @test setup(; chief_points = 9).groups == original
    @test setup(; wide_operators = false).groups == original
    cached = AS._with_numerical_reuse(; retain_prefix = false) do
        setup().groups
    end
    @test cached == original
end

@testset "Higher-order spheroid condensation and reflection" begin
    AS = AcousticScattering
    for boundary in (Rigid(), PressureRelease(), GasFilled(0.00129, 0.23))
        meshes = [AS._spheroid_hp_geometry(boundary, 0.07, 0.01, 0.084,
                      3, half ? 6 : 12, 3; condensed, half)
                  for (condensed, half) in ((false, false), (true, false), (true, true))]
        for m in 0:2
            coefficients = [first(AS._spheroid_hp_mode(mesh, boundary, 30.0, m;
                                l_max = 12)) for mesh in meshes]
            @test coefficients[2]≈coefficients[1] rtol=2e-9 atol=1e-12
            @test coefficients[3]≈coefficients[1] rtol=2e-9 atol=1e-12
        end
    end
end

@testset "Higher-order fluid condensation at oblique incidence" begin
    AS = AcousticScattering
    # Compare Schur buffers and incident lifting against uncondensed equations.
    for order in (2, 4, 6), boundary in (FluidFilled(1.2, 0.8), GasFilled(0.00129, 0.23))

        meshes = [AS._spheroid_hp_geometry(boundary, 0.02, 0.015, 0.03,
                      3, 12, order; condensed) for condensed in (false, true)]
        for m in (0, 2)
            direct, condensed = [AS._spheroid_hp_mode(mesh, boundary, 40.0, m, pi / 3;
                                     l_max = 10, retain_fields = true) for mesh in meshes]
            @test condensed[1]≈direct[1] rtol=2e-9 atol=1e-12
            @test condensed[2].exterior≈direct[2].exterior rtol=2e-9 atol=1e-11
            @test condensed[2].interior≈direct[2].interior rtol=2e-9 atol=1e-11
        end
    end
end

@testset "Higher-order spheroid spherical limit" begin
    # A near-sphere reference checks normalization, pressure, rotations and radiation.
    a = 0.02
    k = 100.0
    body = Spheroid(a * (1 + 1e-10), a)
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 0.8),
        GasFilled(0.00129, 0.23))
        reference = modal(Sphere(a), boundary, k)
        for beta in (0.0, pi / 3, pi / 2, -pi / 3)
            @testset "$(typeof(boundary)), incidence=$beta" begin
                solution = fem(body, boundary, k; order = 6, n_r = 6, n_theta = 24,
                    m_max = 10, l_max = 16, incidence_angle = beta)
                @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 2e-5
                for point in ((0.021, 0.003, 0.001), (-0.021, 0.003, 0.001),
                    (0.04, -0.03, 0.02))
                    rotated = (point[1] * cos(beta) + point[2] * sin(beta),
                        -point[1] * sin(beta) + point[2] * cos(beta), point[3])
                    @test pressure(solution, point)≈pressure(reference, rotated) rtol=2e-5 atol=1e-8
                end
                if boundary isa FluidFilled
                    for point in ((0.0, 0.0, 0.0), (0.006, 0.003, 0.002))
                        rotated = (point[1] * cos(beta) + point[2] * sin(beta),
                            -point[1] * sin(beta) + point[2] * cos(beta), point[3])
                        @test pressure(solution, point; field = :interior)≈
                        pressure(reference, rotated; field = :interior) rtol=2e-5 atol=1e-8
                    end
                end
            end
        end
    end
end

@testset "Meridian high-order angular normalization" begin
    # Evaluate high-order unnormalized polynomials independently.
    setprecision(BigFloat, 256) do
        for (m, l_max) in ((0, 12), (3, 14), (97, 113), (175, 206))
            for x in (-0.7, 0.0, 0.3, 1.0)
                actual = AcousticScattering._meridian_legendre_values!(
                    zeros(l_max - m + 1), m, x)
                expected = [Float64(sqrt(BigFloat(2l + 1) / 2 *
                                         factorial(big(l - m)) / factorial(big(l + m))) *
                                    AcousticScattering.legendre_p(l, m, BigFloat(x)))
                            for l in m:l_max]
                @test all(isfinite, actual)
                @test actual≈expected rtol=2e-12 atol=2e-14
            end
        end
    end
end

@testset "Meridian prescribed-value elimination" begin
    matrix = ComplexF64[4 1+im 0 2; 1+im 5 3 0; 0 3 6 1; 2 0 1 7]
    rhs = ComplexF64[1, 2, 3, 4]
    fixed, free = [1, 4], [2, 3]
    values = ComplexF64[0.5 + 0.2im, 0]
    reference = matrix[free, free] \ (rhs[free] - matrix[free, fixed] * values)
    K, b = sparse(matrix), copy(rhs)
    # A shared corner is prescribed twice. The last value takes precedence.
    AcousticScattering._meridian_fem_dirichlet!(K, b, [1, 4, 4], [values[1], 1, 0])
    solution = K \ b
    @test solution[free] ≈ reference
    @test solution[fixed] ≈ values
    @test issymmetric(K)
    @test all(!iszero, nonzeros(K))
end

# Frozen mode-assembly regressions supplement independent physical checks.
const MERIDIAN_REFACTOR_AMPLITUDES = [
    [complex(0.00038669466408967, 0.001104526889109793),
        complex(0.0022748124814603934, 0.0013599000088818922),
        complex(-0.0044503814911359485, 0.0007181957747036869),
        complex(-0.30326321478600743, 0.9160578695193952),
        complex(-0.2738913371331802, 0.9138713054210372)],
    [complex(-0.009508987144451902, 0.010859822208669061),
        complex(-0.013342540040117428, 0.01161193581092228),
        complex(0.0036977403448248828, 0.0054205086780022204),
        complex(-0.10652791752287018, 0.4472650879892518),
        complex(-0.07750751567571568, 0.4957446118270481)],
    [complex(-0.00022670096035826308, 2.0318817498575186e-05),
        complex(-0.0001844256686873787, 1.5198297000900062e-05),
        complex(-0.00031233908540749663, 2.4022973994208442e-05),
        complex(-0.19058188798177644, 0.9723642035837718),
        complex(-0.189978256333379, 0.9739474306827851)],
    [complex(-0.001713643545281735, 0.0028465829921995736),
        complex(0.0037567060670989612, 0.003862331954702374),
        complex(-0.0028219942187343236, 0.002831576824682387),
        complex(-0.32837332986501166, 0.7589414542538354),
        complex(-0.32191745115607684, 0.774688416545024)],
    [complex(-0.011177611644957073, 0.016744058502547312),
        complex(-0.019665368872401206, 0.02009516104714747),
        complex(0.007881011357628153, -0.0005010082200819829),
        complex(-0.05357881539584128, 0.2034716026358392),
        complex(-0.06471462204022851, 0.31402337146848047)],
    [complex(-0.0006372985649624455, 4.900349995902152e-05),
        complex(-0.0005507018073877548, 4.866105924467838e-05),
        complex(-0.00034194541980158316, 4.230956655247037e-05),
        complex(-0.17844286678876575, 0.9582627712576819),
        complex(-0.17830124706008205, 0.9658970850351157)],
    [complex(-0.00014265650440377858, 0.0068108855992384506),
        complex(0.005615834090936603, 0.006678824375972212),
        complex(-0.009576105478217875, 0.004601127048598547),
        complex(-0.35151118893439337, 0.654393971219185),
        complex(-0.3216365944933174, 0.7128773041543748)],
    [complex(-0.008726864095094784, 0.019600464288577615),
        complex(-0.02083237130141106, 0.024677165783369162),
        complex(0.011591204849840867, 0.003807713413646261),
        complex(-0.06273340418228793, 0.2610385775714985),
        complex(-0.019838991194041228, 0.14229268301839504)],
    [complex(-0.0007856616542897821, 7.440082537839072e-05),
        complex(-0.000730586222728813, 7.1269388030752e-05),
        complex(-0.0008498173969015799, 7.507917981583013e-05),
        complex(-0.17153288653724652, 0.9612945101026833),
        complex(-0.17057284007843615, 0.9564212459161199)]
]

@testset "Meridian assembly reference amplitudes and pressure" begin
    for (shape, body) in enumerate((
        Cylinder(0.01, 0.02), Spheroid(0.02, 0.015), Spheroid(0.015, 0.02)))
        for (bc, boundary) in enumerate((
            Rigid(), PressureRelease(), FluidFilled(1.05, 1.02)))
            sol = fem(body, boundary, 100.0; method = :meridian, R = 0.03,
                incidence_angle = pi / 4, m_max = 2, n_r = 5, n_theta = 12, l_max = 5)
            actual = ComplexF64[scattering_amplitude(sol; angle = a, azimuth = 0.3)
                                for a in (0.0, 0.7, pi)]
            append!(actual, pressure(sol, [(0.025, 0.0, 0.0), (0.0, 0.025, 0.0)]))
            @test actual≈MERIDIAN_REFACTOR_AMPLITUDES[3(shape - 1) + bc] rtol=1e-10 atol=1e-12
        end
    end
end

@testset "Prolate FEM-BEM radiation and transmission" begin
    AS = AcousticScattering
    # Exact monopole and dipole fields check jumps, reflection and normalization.
    for half in (true, false)
        be = AS._spheroid_curved_boundary(0.02, 0.015, half ? 8 : 16, 6; half)
        k = 100.0
        Ks, Vs = AS._spheroid_curved_operators(be, k, 0:1, Val(16), Val(6);
            rtol = 1e-8, chief_points = 8)
        r = hypot.(be.rho, be.z)
        G = exp.(im*k .* r) ./ r
        nr = (be.nrho .* be.rho+be.nz .* be.z) ./ r
        pressures = [G, (im*k .- inv.(r)) .* G .* be.rho ./ r]
        derivatives = [(im*k .- inv.(r)) .* G .* nr,
            (-k^2 .- 3im*k ./ r .+ 3 ./ r .^ 2) .* G .* be.rho ./ r .* nr .+
            (im*k .- inv.(r)) .* G .* be.nrho ./ r]
        for m in 0:1
            D = -Ks[m + 1]
            for i in eachindex(r)
                D[i, i] += 0.5
            end
            @test norm(D*pressures[m + 1]+Vs[m + 1]*derivatives[m + 1])/norm(pressures[m + 1]) <
                  2e-6
            field = AS._SpheroidSurfaceMode(be, pressures[m + 1], derivatives[m + 1])
            for (theta, phi) in ((0.7, 0.4), (-0.7, 0.4), (1.3, 2.1))
                exact = m==0 ? 1.0+0im : im*k*sin(theta)*cos(phi)
                @test isapprox(AS._spheroid_curved_amplitude(field, k, m, theta, phi),
                    exact; rtol = 1e-8, atol = 1e-9)
            end
        end
    end
    body, bc, k = Spheroid(0.02, 0.015), FluidFilled(1.2, 0.85), 100.0
    for beta in (pi/2, 0.0, 0.6)
        coupled_solution = coupled(
            body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem),
            incidence_angle = beta, m_max = 6,
            n_r = 3, n_theta = 24, n_boundary = 16, order = 4, adaptive = false,
            rtol = 1e-8, mode_threads = 2)
        reference = fem(body, bc, k; order = 6, incidence_angle = beta, m_max = 6,
            n_r = 8, n_theta = 32, l_max = 14, retain_fields = false)
        for (theta, phi) in ((pi-beta, pi), (0.4, 0.7), (-0.4, 0.7))
            @test isapprox(
                scattering_amplitude(coupled_solution; angle = theta, azimuth = phi),
                scattering_amplitude(reference; angle = theta, azimuth = phi); rtol = 2e-6, atol = 1e-11)
        end
        report = diagnostics(coupled_solution)
        @test report.discretization===:coupled
        @test report.relative_residual < 1e-10
        @test report.representation_residual < 1e-7
        @test report.equatorial_symmetry == (beta==pi/2)
        @test report.equatorial_decomposition == (beta==pi/2 ? :even : :even_odd)
        @test report.solver_options.retain_fields===false
        @test length(report.systems)==(beta==pi/2 ? 7 : 14)
        if beta != pi/2
            # Compare reduction against complete equations on identical discretizations.
            full = AS._spheroid_fembem_level(body, bc, k, 3, 24, 16, 4, 5;
                half = false, incidence_angle = beta, m_max = 6, mode_threads = 2,
                rtol = 1e-8, chief_points = 8, reports = AS._SolveReports(),
                split_symmetry = false)
            for (theta, phi) in ((pi-beta, pi), (0.4, 0.7), (2.3, 1.2))
                expected = sum(AS._spheroid_curved_amplitude(
                                   field, k, m-1, theta, phi)
                for (m, field) in enumerate(full))
                @test isapprox(
                    scattering_amplitude(coupled_solution; angle = theta, azimuth = phi),
                    expected; rtol = 1e-8, atol = 1e-11)
            end
            @test Set(system.parity for system in report.systems)==Set((-1, 1))
        end
        @test_throws ArgumentError pressure(coupled_solution, (0.03, 0.0, 0.0))
        @test_throws ArgumentError scattering_amplitude(coupled_solution; angle = NaN)
        if beta == 0.6
            # Disjoint mode updates preserve previously computed traces.
            subset = copy(coupled_solution.data.modes)
            AS._spheroid_fembem_level(body, bc, k, 3, 24, 16, 4, 5;
                half = false, incidence_angle = beta, m_max = 6, mode_threads = 2,
                rtol = 1e-8, chief_points = 8, reports = AS._SolveReports(),
                modes = [1, 4], fields = subset)
            for m in 0:6
                if m in (1, 4)
                    @test isapprox(
                        subset[m + 1].pressure, coupled_solution.data.modes[m + 1].pressure;
                        rtol = 1e-8, atol = 1e-11)
                else
                    @test subset[m + 1] === coupled_solution.data.modes[m + 1]
                end
            end
        end
    end
    for controls in ((; n_theta = 25, n_boundary = 16, chief_points = 8),
        (; n_theta = 24, n_boundary = 15, chief_points = 8),
        (; n_theta = 24, n_boundary = 16, chief_points = 7))
        unreduced = coupled(
            body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem),
            incidence_angle = 0.6,
            n_r = 3, order = 4, m_max = 4, adaptive = false, controls...)
        @test diagnostics(unreduced).equatorial_decomposition===:none
        @test length(diagnostics(unreduced).systems)==5
        @test isfinite(target_strength(unreduced))
    end
    # Equal fluids have no scattered field, including near the spherical limit.
    for a in (0.02, 0.01500015)
        transparent = coupled(Spheroid(a, 0.015), FluidFilled(1.0, 1.0), 100.0;
            coupling = Coupling(interior = :fem, exterior = :bem),
            n_r = 4, n_theta = 24, n_boundary = 16, order = 5,
            m_max = 5, adaptive = false, incidence_angle = 0.6, rtol = 1e-8)
        @test abs(scattering_amplitude(transparent)) < 1e-8
    end
    # CHIEF constraints preserve the sphere reference near fictitious resonances.
    near_sphere = coupled(Spheroid(1.00000001, 1.0), bc, pi;
        coupling = Coupling(interior = :fem, exterior = :bem),
        order = 6, n_r = 6, n_theta = 32, n_boundary = 24,
        m_max = 10, rtol = 1e-8, adaptive = false, mode_threads = 2)
    exact_sphere = modal(Sphere(1.0), bc, Float64(pi))
    @test isapprox(scattering_amplitude(near_sphere),
        scattering_amplitude(exact_sphere); rtol = 1e-6)
    @test diagnostics(near_sphere).representation_residual < 1e-8
    # Independent spatial refinements supply the reported TS change.
    refined = coupled(body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem),
        n_r = 2, n_theta = 12, n_boundary = 8,
        order = 4, m_max = 6, target_tol = 1e-7, rtol = 1e-8, mode_threads = 2,
        max_refinements = (radial = 4, angular = 4, boundary = 4))
    refinement = diagnostics(refined).refinement
    @test refinement.converged
    @test length(refinement.history)>=2
    @test refinement.change_db<=1e-7
    @test maximum(refinement.mode_n_boundary)>first(refinement.history).n_boundary
    @test maximum(refinement.mode_n_r)>first(refinement.history).n_r
    @test maximum(refinement.mode_n_theta)>first(refinement.history).n_theta
    @test length(refinement.mode_refinements) == 7
    @test all(1 .<= refinement.mode_refinements .<= 4)
    @test length(unique(refinement.mode_refinements)) > 1
    @test refinement.estimator===:directional_fourier_ts_mesh_change
    @test size(refinement.directional_amplitude_changes)==(7, 3)
    @test length(refinement.interaction_amplitude_changes)==7
    for axis in (:radial, :angular, :boundary)
        @test all(1 .<= getproperty(refinement.mode_refinements_by_axis, axis) .<= 4)
    end
    uniform = coupled(body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem),
        order = 4, m_max = 6,
        n_r = maximum(refinement.mode_n_r), n_theta = maximum(refinement.mode_n_theta),
        n_boundary = maximum(refinement.mode_n_boundary), adaptive = false,
        rtol = 1e-8, mode_threads = 2)
    @test abs(target_strength(uniform)-target_strength(refined)) < 0.01
    limited = @test_logs (:warn, r"FEM-BEM did not reach target_tol") coupled(
        body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem),
        order = 3, n_r = 2, n_theta = 8, n_boundary = 6, m_max = 2,
        target_tol = 1e-20, max_refinements = 1)
    @test !diagnostics(limited).refinement.converged
    @test abs(target_strength(refined)-target_strength(fem(body, bc, k;
        order = 6, n_r = 8, n_theta = 32, m_max = 6, l_max = 14, retain_fields = false))) <
          0.01
    @test_throws ArgumentError coupled(
        Spheroid(0.015, 0.02), bc, k; coupling = Coupling(interior = :fem, exterior = :bem))
    @test_throws ArgumentError coupled(body, Rigid(), k; coupling = Coupling(interior = :fem, exterior = :bem))
    for controls in ((; order = 1), (; boundary_order = 0), (; chief_points = 0), (;
        rtol = 0.0),
        (; n_r = 0), (; n_theta = 1), (; n_boundary = 1), (; m_max = -1), (;
        mode_threads = 0),
        (; retain_fields = true), (; dtn_form = :augmented), (; R = 0.05))
        @test_throws ArgumentError coupled(
            body, bc, k; coupling = Coupling(interior = :fem, exterior = :bem), controls...)
    end
end

@testset "Radial reduction preserves the complete FEM-BEM equations" begin
    AS=AcousticScattering
    body=Spheroid(0.02, 0.015)
    for bc in (FluidFilled(1.2, 0.85), FluidFilled(1.0, 1.0), GasFilled(0.00129, 0.23)),
        (beta, half, split) in ((0.6, false, true), (pi/2, true, true), (0.0, false, false))

        options=(; half, incidence_angle = beta, m_max = 2, mode_threads = 2, rtol = 1e-9,
            chief_points = 8, split_symmetry = split)
        reference=AS._spheroid_fembem_level(body, bc, 100.0, 3, 24, 16, 4, 5;
            options..., reports = AS._SolveReports(), tensor = false)
        reports=AS._SolveReports()
        radial=AS._spheroid_fembem_level(body, bc, 100.0, 3, 24, 16, 4, 5;
            options..., reports, interior_reduction = :radial)
        @test all(system->system.interior_reduction===:radial, reports.systems)
        @test all(system->system.fem_relative_residual<=1e-9, reports.systems)
        for (index, (old, new)) in enumerate(zip(reference, radial))
            @test new.pressure≈old.pressure rtol=2e-9 atol=1e-12
            @test new.derivative≈old.derivative rtol=2e-9 atol=1e-10
            for (theta, phi) in ((pi-beta, pi), (0.3, 0.8), (2.1, 1.2))
                @test AS._spheroid_curved_amplitude(new, 100.0, index-1, theta, phi)≈
                AS._spheroid_curved_amplitude(old, 100.0, index-1, theta, phi) rtol=2e-9 atol=1e-12
            end
        end
    end
    bc=FluidFilled(1.2, 0.85)
    mesh=AS._spheroid_fembem_confocal(
        bc, body.a, body.b, 3, 8, 4; half = true, assemble_bulk = false)
    be=AS._spheroid_curved_boundary(body.a, body.b, 8, 6; half = true)
    T, B=AS._spheroid_fembem_trace(mesh, be)
    K, V=AS._spheroid_curved_operators(
        be, 100.0, 0:0, Val(1), Val(6); rtol = 1e-9, chief_points = 4)
    reports=AS._SolveReports()
    fallback=AS._spheroid_fembem_tensor_mode(
        mesh, be, T, B, K[1], V[1], 100.0, 0, pi/2, reports, 1;
        interior_reduction = :radial, radial_maxiter = 1)
    @test only(reports.systems).interior_reduction===:tensor
    @test only(reports.systems).radial_attempt.reason===:iteration
    direct=AS._spheroid_fembem_tensor_mode(
        mesh, be, T, B, K[1], V[1], 100.0, 0, pi/2, AS._SolveReports(), 1;
        interior_reduction = :angular)
    @test fallback.pressure≈direct.pressure rtol=1e-12
    @test fallback.derivative≈direct.derivative rtol=1e-12
    # Compare reused radial inverses against an independent Kronecker system.
    for m in (0, 2), parity in (1, -1)

        inverse=AS._spheroid_fembem_radial_inverse(mesh, 100.0, m, parity)
        n=length(inverse.angular_free)
        radial=complex.(Matrix(inverse.Ar))
        radial[end, end]-=im*inverse.eta
        complete=kron(radial, inverse.Mt)+kron(inverse.Mr, inverse.At)
        workspace=zeros(ComplexF64, n, length(inverse.radial_free))
        for phase in (0.3, 0.8, 0.3)
            load=cis.(phase .* (1:n))
            rhs=zeros(ComplexF64, size(complete, 1))
            rhs[(end - n + 1):end]=load
            expected=reshape(complete\rhs, n, :)[:, end]
            actual=AS._spheroid_fembem_radial_apply!(zeros(ComplexF64, n), inverse, load, workspace)
            @test actual≈expected rtol=2e-11 atol=1e-12
        end
    end
    for interior_ka in (pi, 4.493409457909064)
        k=interior_ka*bc.soundspeed_contrast
        reports=AS._SolveReports()
        modes=AS._spheroid_fembem_level(Spheroid(1.00000001, 1.0), bc, k, 6, 32, 24, 6, 5;
            half = false, incidence_angle = 0.6, m_max = 10, mode_threads = 2, rtol = 1e-8,
            chief_points = 8, reports, interior_reduction = :radial)
        amplitude=sum(AS._spheroid_curved_amplitude(field, k, m-1, pi-0.6, pi)
        for (m, field) in enumerate(modes))
        @test amplitude≈scattering_amplitude(modal(Sphere(1.0), bc, k)) rtol=1e-6
        @test all(system->system.interior_reduction===:radial, reports.systems)
    end
    fresh=[AS._spheroid_fembem_level(body, bc, 100.0, 3, 24, 16, 4, 5;
               half = false, incidence_angle = beta, m_max = 2, mode_threads = 2, rtol = 1e-9,
               chief_points = 8, reports = AS._SolveReports(), interior_reduction = :radial)
           for beta in (0.6, -0.6, 0.6)]
    cached=AS._with_numerical_reuse() do
        [AS._spheroid_fembem_level(body, bc, 100.0, 3, 24, 16, 4, 5;
             half = false, incidence_angle = beta, m_max = 2, mode_threads = 2, rtol = 1e-9,
             chief_points = 8, reports = AS._SolveReports(), interior_reduction = :radial)
         for beta in (0.6, -0.6, 0.6)]
    end
    for (reference, reused) in zip(fresh, cached), (old, new) in zip(reference, reused)

        @test new.pressure≈old.pressure rtol=1e-12 atol=1e-14
        @test new.derivative≈old.derivative rtol=1e-12 atol=1e-12
    end
end

@testset "Compact confocal load geometry and reusable projection" begin
    AS = AcousticScattering
    for half in (false, true), p in (2, 4), extra in (0, 2)
        a, b, nr, nt = 0.1, 0.1/1.001, 3, 5
        bc = FluidFilled(1.24/1026.8, 345/1477.3)
        mesh = AS._spheroid_fembem_confocal(bc, a, b, nr, nt, p;
            half, assemble_bulk = false, quadrature_extra = extra)
        q, w = AS._spheroid_hp_rule(p+3+extra)
        radial_edges = collect(range(0.0, 1.0; length = nr+1))
        angular_edges = collect(range(0.0, half ? pi/2 : pi; length = nt+1))
        focal = sqrt((a-b)*(a+b))
        direct = map(1:(nr*nt)) do e
            er, et = mod1(e, nr), cld(e, nr)
            sl, sr = radial_edges[er:er+1]
            tl, tr = angular_edges[et:et+1]
            points = zeros(3, length(q)^2)
            for jt in eachindex(q), js in eachindex(q)
                r = (sl+sr)/2+(sr-sl)*q[js]/2
                theta = (tl+tr)/2+(tr-tl)*q[jt]/2
                aa = hypot(focal, b*r)
                j = (jt-1)*length(q)+js
                points[:, j] = [b*r*sin(theta), aa*cos(theta),
                    b^2*r/aa*(focal^2*sin(theta)^2+b^2*r^2)*sin(theta)*
                    w[js]*w[jt]*(sr-sl)*(tr-tl)/4]
            end
            points
        end
        buffer = similar(first(direct))
        @test length(mesh.load_points)==nr*nt
        for e in eachindex(direct)
            @test isequal(mesh.load_points[e], direct[e])
            @test AS._spheroid_fembem_load_points!(buffer, mesh.load_points, e)===buffer
            @test isequal(buffer, direct[e])
        end
        first_copy = mesh.load_points[1]
        first_copy[1] = NaN
        @test isequal(mesh.load_points[1], direct[1])
        @test_throws BoundsError mesh.load_points[0]
        @test_throws BoundsError mesh.load_points[nr*nt+1]
        @test_throws DimensionMismatch AS._spheroid_fembem_load_points!(zeros(2, 2), mesh.load_points, 1)
        be = AS._spheroid_curved_boundary(a, b, 8, 3; half)
        T, B = AS._spheroid_fembem_trace(mesh, be)
        for beta in (0.0, deg2rad(37.0), pi/2), m in (0, 3, 240)
            k = 2pi*285000/1477.3
            expected = zeros(ComplexF64, nt*p+1, nr*p+1)
            scale = k^2*(inv(bc.soundspeed_contrast^2)-1)/bc.density_contrast
            for (cell, points) in zip(mesh.cells, direct)
                samples = [scale*points[3, j]*(m==0 ? 1.0 : 2.0)*im^m*
                    AS.besselj(m, k*points[1, j]*sin(beta))*cis(k*points[2, j]*cos(beta))
                    for j in axes(points, 2)]
                local_load = mesh.load_basis*samples
                for j in eachindex(cell.ids)
                    expected[cell.ids[j]] += local_load[j]
                end
            end
            derivative = [AS._dpdn_inc_mode(m, k, beta, r, z, nx, nz)
                for (r, z, nx, nz) in zip(be.rho, be.z, be.nrho, be.nz)]
            expected[:, end] += (1-inv(bc.density_contrast))*(B*derivative)
            actual = AS._spheroid_fembem_tensor_load(mesh, be, B, k, m, beta)
            @test isequal(actual, expected)
        end
    end
    mesh = AS._spheroid_fembem_confocal(FluidFilled(1.0, 1.0), 0.1, 0.08, 2, 4, 2;
        assemble_bulk = false, assemble_load = false)
    @test isempty(mesh.load_points)
end

@testset "Real contractions preserve the complex angular load" begin
    AS = AcousticScattering
    for (rows, modes, columns) in ((1, 1, 1), (7, 3, 5), (3, 7, 2), (5, 4, 0), (0, 3, 2)),
        phase in (0.0+0im, 1.0+0im, 0.0+1im, 1.0-2im)
        E = [sin(i+2j)/(i+j) for i in 1:rows, j in 1:modes]
        load = ComplexF64[phase*(-1)^(i+j)*(i+2j)/9 for i in 1:rows, j in 1:columns]
        saved_E, saved_load = copy(E), copy(load)
        expected = setprecision(BigFloat, 192) do
            ComplexF64[ComplexF64(sum((BigFloat(E[i, j])*Complex{BigFloat}(load[i, col])
                            for i in 1:rows); init = zero(Complex{BigFloat})))
             for j in 1:modes, col in 1:columns]
        end
        actual = AS._spheroid_fembem_transform_load(E, load)
        @test size(actual)==(modes, columns)
        @test eltype(actual)===ComplexF64
        @test actual ≈ expected rtol=3e-14 atol=3e-15
        @test E==saved_E
        @test load==saved_load
    end
    E = [0.2 -0.5; 0.7 0.3; -0.4 0.9]
    load = ComplexF64[1+2im -3+4im; 5-6im 7+8im; -9+10im 11-12im]
    @test AS._spheroid_fembem_transform_load(E, load) ≈ transpose(E)*load rtol=2e-14
    @test AS._spheroid_fembem_transform_load(view(E, :, 1:1), view(load, :, 2:2)) ≈
          transpose(E[:, 1:1])*load[:, 2:2] rtol=2e-14
    for entry in (NaN, Inf, -Inf), component in (:real, :imaginary)
        nonfinite_load = zeros(ComplexF64, 3, 1)
        nonfinite_load[1] = component===:real ? complex(entry, 0.0) : complex(0.0, entry)
        expected = complex.(transpose(E)*real.(nonfinite_load), transpose(E)*imag.(nonfinite_load))
        @test isequal(AS._spheroid_fembem_transform_load(E, nonfinite_load), expected)
    end
    for entry in (NaN, Inf, -Inf)
        nonfinite_E = copy(E)
        nonfinite_E[1, 1] = entry
        zero_load = zeros(ComplexF64, 3, 1)
        expected = complex.(transpose(nonfinite_E)*real.(zero_load), transpose(nonfinite_E)*imag.(zero_load))
        @test isequal(AS._spheroid_fembem_transform_load(nonfinite_E, zero_load), expected)
    end
    @test_throws DimensionMismatch AS._spheroid_fembem_transform_load(E, zeros(ComplexF64, 2, 4))
end

@testset "Fused parity loads preserve scalar arithmetic and ownership" begin
    AS = AcousticScattering
    finite = ComplexF64[complex((-1)^(i+j)*(i+2j)/7, (2i-j)/9)
                       for i in 1:4, j in 1:3]
    special = reshape(ComplexF64[
        complex(0.0, 0.0), complex(-0.0, 0.0), complex(0.0, -0.0), complex(-0.0, -0.0),
        complex(nextfloat(0.0), -nextfloat(0.0)), complex(floatmin(Float64), -floatmin(Float64)),
        complex(Inf, 1.0), complex(-Inf, -1.0), complex(1.0, Inf), complex(-1.0, -Inf),
        complex(NaN, 2.0), complex(3.0, NaN)], 4, 3)
    for load in (finite, complex.(real.(finite)), complex.(0.0, imag.(finite)), special,
            zeros(ComplexF64, 3, 0), zeros(ComplexF64, 0, 2), view(finite, :, 1:2)),
        m in (0, 1, 24, 35)
        saved = copy(load)
        for parity in (0, 1, -1)
            expected = parity==0 ? load : (load+parity*(-1)^m*conj(load))/2
            actual = AS._spheroid_fembem_parity_load(load, m, parity)
            @test isequal(actual, expected)
            @test isequal(load, saved)
            @test size(actual)==size(load)
            @test (actual===load)==(parity==0)
        end
        even = AS._spheroid_fembem_parity_load(load, m, 1)
        odd = AS._spheroid_fembem_parity_load(load, m, -1)
        @test isequal(even, (saved+(-1)^m*conj(saved))/2)
        @test isequal(odd, (saved+(-1)*(-1)^m*conj(saved))/2)
        if !isempty(even)
            even[1] = complex(19.0, -23.0)
            @test isequal(load, saved)
            @test isequal(odd, (saved+(-1)*(-1)^m*conj(saved))/2)
        end
    end
end

@testset "Buffered radial contractions preserve array sum reductions" begin
    AS = AcousticScattering
    direct(radial, transformed, columns, rows) = ComplexF64[
        sum(view(radial, :, column) .* view(transformed, row, :))
        for (column, row) in zip(columns, rows)]
    for n in (0, 1, 2, 15, 16, 1023, 1024, 1025, 1363), real_response in (false, true)
        radial_real = [(-1)^(i+j)*(i+3j)/(i+j+1) for i in 1:n, j in 1:4]
        radial = real_response ? radial_real :
                 complex.(radial_real, [(-1)^i*(2i-j)/(i+j+1) for i in 1:n, j in 1:4])
        transformed = ComplexF64[complex((-1)^i*(i+2j)/11, (3i-j)/13)
                                 for j in 1:4, i in 1:n]
        saved_radial, saved_transformed = copy(radial), copy(transformed)
        products = fill(complex(NaN, NaN), n)
        for selected in (collect(1:4), [4, 2, 1], [3], Int[])
            expected = direct(radial, transformed, selected, selected)
            actual = AS._spheroid_fembem_contract_load!(products, radial, transformed,
                selected, selected)
            @test isequal(actual, expected)
            @test eltype(actual)===ComplexF64
            @test isequal(radial, saved_radial)
            @test isequal(transformed, saved_transformed)
            @test isequal(AS._spheroid_fembem_contract_load!(products, radial, transformed,
                selected, selected), expected)
        end
        regular = [4, 1, 3]
        compact = radial[:, regular]
        @test isequal(AS._spheroid_fembem_contract_load!(products, compact, transformed,
            eachindex(regular), regular), direct(compact, transformed, eachindex(regular), regular))
        @test isempty(AS._spheroid_fembem_contract_load!(products, compact[:, 1:0],
            transformed, 1:0, Int[]))
    end
    values = [0.0, -0.0, nextfloat(0.0), -nextfloat(0.0), 1e100, -1e100, Inf, -Inf, NaN]
    for real_response in (false, true)
        radial = real_response ? reshape(copy(values), :, 1) :
                 reshape(complex.(values, reverse(values)), :, 1)
        transformed = reshape(complex.(reverse(values), values), 1, :)
        products = zeros(ComplexF64, length(values))
        @test isequal(AS._spheroid_fembem_contract_load!(products, radial, transformed,
            1:1, 1:1), direct(radial, transformed, 1:1, 1:1))
    end
    radial = [(-1)^i*(i+j)/7 for i in 1:1363, j in 1:3]
    transformed = ComplexF64[complex((i+j)/11, (-1)^i*(i-j)/13) for j in 1:3, i in 1:1363]
    expected = direct(radial, transformed, [3, 1], [2, 3])
    tasks = [Threads.@spawn begin
                 products = zeros(ComplexF64, size(radial, 1))
                 AS._spheroid_fembem_contract_load!(products, radial, transformed, [3, 1], [2, 3])
             end for _ in 1:4]
    @test all(value->isequal(value, expected), fetch.(tasks))
    @test_throws DimensionMismatch AS._spheroid_fembem_contract_load!(ComplexF64[],
        radial, transformed, 1:1, 1:1)
    @test_throws DimensionMismatch AS._spheroid_fembem_contract_load!(zeros(ComplexF64, 1363),
        radial, transformed, 1:1, 1:2)
    @test_throws BoundsError AS._spheroid_fembem_contract_load!(zeros(ComplexF64, 1363),
        radial, transformed, [4], [1])
end

@testset "Dense complex-real Schur products" begin
    AS = AcousticScattering
    for (rows, inner, columns) in ((1, 1, 1), (7, 3, 5), (3, 7, 2),
            (0, 3, 2), (5, 3, 0), (5, 0, 4), (0, 0, 0)),
        phase in (1.0+0im, 0.0+1im, 1.0-2im)
        A = ComplexF64[phase*(-1)^(i+j)*(2i-j)/7 for i in 1:rows, j in 1:inner]
        B = Float64[(-1)^(i+j)*(i+3j)/11 for i in 1:inner, j in 1:columns]
        saved_A, saved_B = copy(A), copy(B)
        expected = setprecision(BigFloat, 192) do
            ComplexF64[ComplexF64(sum((Complex{BigFloat}(A[i, k])*BigFloat(B[k, j])
                            for k in 1:inner); init = zero(Complex{BigFloat})))
             for i in 1:rows, j in 1:columns]
        end
        actual = AS._spheroid_fembem_mul(A, B)
        @test size(actual)==(rows, columns)
        @test eltype(actual)===ComplexF64
        @test isapprox(actual, expected; rtol = 3e-14, atol = 3e-15)
        @test A==saved_A
        @test B==saved_B
    end
    A = ComplexF64[1+2im -3+4im 5-6im; -7+8im 9-10im -11-12im]
    B = [0.2 -0.5; 0.7 0.3; -0.4 0.9]
    # Unsupported storage and element combinations retain ordinary dispatch.
    for (left, right) in ((real.(A), B), (A, complex.(B)),
            (view(A, :, :), view(B, :, :)), (A, sparse(B)),
            (transpose(copy(transpose(A))), B), (ComplexF32.(A), Float32.(B)))
        @test AS._spheroid_fembem_mul(left, right)==left*right
    end
    @test_throws DimensionMismatch AS._spheroid_fembem_mul(A, zeros(2, 4))
    @test_throws DimensionMismatch AS._spheroid_fembem_mul(zeros(ComplexF64, 0, 3), zeros(2, 0))

    # Independent block algebra checks both contractions and complex fallback.
    for (nt, nb) in ((4, 8), (12, 2)), m in (0, 1)
        mesh = AS._spheroid_fembem_confocal(
            FluidFilled(1.2, 0.85), 0.02, 0.015, 2, nt, 2;
            half = true, assemble_bulk = false, assemble_load = false)
        be = AS._spheroid_curved_boundary(0.02, 0.015, nb, 2; half = true)
        T, B = AS._spheroid_fembem_trace(mesh, be)
        nbe = length(be.chi)
        K = ComplexF64[0.01cis(i+2j)/(i+j) for i in 1:nbe, j in 1:nbe]
        V = Matrix{ComplexF64}(I, nbe, nbe)
        cache = AS._NumericalReuse()
        initial = AS._spheroid_fembem_tensor_operator(
            mesh, be, T, B, K, V, 100.0, m, 1; reuse = cache)
        eta = mesh.a*mesh.b*100.0/(mesh.density_contrast*mesh.soundspeed_contrast)
        # A prescribed retained endpoint isolates augmented-block arithmetic.
        initial.radial[end, 1] = -0.9im/eta
        for fallback in (false, true)
            if fallback
                initial.radial[2, last(initial.regular)] += 1.0
            end
            op = AS._spheroid_fembem_tensor_operator(
                mesh, be, T, B, K, V, 100.0, m, 1; reuse = cache)
            @test 1 in op.retained
            @test (op.regular_radial===nothing)==fallback
            @test (op.regular_exterior===nothing)==(nt==12)
            expected = setprecision(BigFloat, 192) do
                D = Complex{BigFloat}.(op.D)
                P = BigFloat.(op.pressure_map)
                F = BigFloat.(op.flux_map)
                response = Complex{BigFloat}.(op.response)
                schur = Complex{BigFloat}.(V)+D*(P[:, op.regular]*Diagonal(response)*F[op.regular, :])
                lower = -Diagonal(Complex{BigFloat}.(op.beta_response[op.retained]))*F[op.retained, :]
                ComplexF64.([schur D*P[:, op.retained];
                    lower Matrix(Diagonal(Complex{BigFloat}.(op.alpha[op.retained])))])
            end
            @test isapprox(op.matrix, expected; rtol = 3e-13, atol = 3e-15)
            rhs = ComplexF64[cis(j)/(j+1) for j in axes(expected, 1)]
            @test isapprox(op.factor\rhs, expected\rhs; rtol = 5e-12, atol = 5e-14)
        end
    end
end

@testset "Endpoint radial responses preserve full-column validation" begin
    AS = AcousticScattering
    function compare_endpoints(base, mass, p, values, eta)
        radial = Matrix{ComplexF64}(undef, size(base, 2), length(values))
        band = similar(base)
        for j in eachindex(values)
            AS._spheroid_fembem_radial_map!(view(radial, :, j), band, base, mass, p, values[j])
        end
        beta_response = vec(radial[end, :])
        alpha = 1 .- im*eta .* beta_response
        retained = findall(j -> abs(alpha[j])<=0.25eta*abs(beta_response[j]), eachindex(alpha))
        regular = setdiff(eachindex(alpha), retained)
        real_radial = AS._spheroid_fembem_real_radial(
            radial, alpha, base, mass, p, values, regular, eta)
        expected = real_radial===nothing ? beta_response[regular] ./ alpha[regular] :
                   vec(real_radial[end, :])
        saved_base, saved_mass = copy(base), copy(mass)
        actual = AS._spheroid_fembem_radial_endpoints(base, mass, p, values, eta)
        @test actual.beta_response==beta_response
        @test actual.alpha==alpha
        @test actual.retained==retained
        @test actual.regular==regular
        @test actual.real_regular==(real_radial!==nothing)
        @test actual.response==expected
        @test eltype(actual.response)===eltype(expected)
        @test base==saved_base
        @test mass==saved_mass
        @test actual.beta_response!==actual.alpha
        snapshot = deepcopy(actual)
        repeated = AS._spheroid_fembem_radial_endpoints(base, mass, p, reverse(values), eta)
        @test actual==snapshot
        @test repeated.beta_response!==actual.beta_response
        @test repeated.response!==actual.response
        return actual
    end
    stiffness = [2.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 3.0]
    for scale in (1e-8, 1.0, 1e8), density in (1.0, 0.0012), m in (0, 1)
        factor = (; stiffness = scale*stiffness, mass = scale*Matrix(Diagonal([0.5, 1.0, 1.5])),
            wave = zeros(3, 3), azimuthal = zeros(3, 3))
        eta = 2scale/density
        base, mass = AS._spheroid_fembem_radial_pencil([factor], 2, m, 1.0, density, eta)
        actual = compare_endpoints(base, mass, 2, [0.0, 0.2, -0.3], eta)
        @test actual.real_regular
        @test isempty(actual.retained)
    end
    factor = (; stiffness, mass = Matrix(Diagonal([0.5, 1.0, 1.5])),
        wave = zeros(3, 3), azimuthal = zeros(3, 3))
    base, mass = AS._spheroid_fembem_radial_pencil([factor], 2, 0, 1.0, 1.0, 2.0)
    # A solvable complex interior pencil must retain the complete complex path.
    base[5, 1] += 1e-4im
    fallback = compare_endpoints(base, mass, 2, [0.0, 0.2, -0.3], 2.0)
    @test !fallback.real_regular
    @test eltype(fallback.response)===ComplexF64
    pole_factor = (; stiffness = Matrix(Diagonal([2.0, 1.0, 1e-12])),
        mass = zeros(3, 3), wave = zeros(3, 3), azimuthal = zeros(3, 3))
    base, mass = AS._spheroid_fembem_radial_pencil([pole_factor], 2, 0, 1.0, 1.0, 2.0)
    pole = compare_endpoints(base, mass, 2, [0.0], 2.0)
    @test pole.retained==[1]
    @test isempty(pole.regular) && isempty(pole.response)
    empty = compare_endpoints(base, mass, 2, Float64[], 2.0)
    @test isempty(empty.alpha) && isempty(empty.retained) && isempty(empty.regular)
    for coupling in (-3.14159e-303, -2.71828e-308, -9.999e-312)
        factor = (; stiffness = [1e5 coupling; coupling 1e4], mass = zeros(2, 2),
            wave = zeros(2, 2), azimuthal = zeros(2, 2))
        base, mass = AS._spheroid_fembem_radial_pencil([factor], 1, 0, 1.0, 1.0, 3e4)
        @test compare_endpoints(base, mass, 1, [0.0], 3e4).real_regular
    end
end

@testset "Endpoint operators preserve total fields and formulation cache isolation" begin
    AS = AcousticScattering
    for (nt, nb) in ((4, 8), (12, 2)), m in (0, 1), parity in (1, -1)
        mesh = AS._spheroid_fembem_confocal(
            FluidFilled(1.2, 0.85), 0.02, 0.015, 2, nt, 2;
            half = true, assemble_bulk = false, assemble_load = false)
        be = AS._spheroid_curved_boundary(0.02, 0.015, nb, 2; half = true)
        T, B = AS._spheroid_fembem_trace(mesh, be)
        nbe = length(be.chi)
        K = ComplexF64[0.01cis(i+2j)/(i+j) for i in 1:nbe, j in 1:nbe]
        V = Matrix{ComplexF64}(I, nbe, nbe)
        full = AS._spheroid_fembem_tensor_operator(mesh, be, T, B, K, V, 100.0, m, parity)
        streamed = AS._spheroid_fembem_tensor_operator(
            mesh, be, T, B, K, V, 100.0, m, parity; boundary_only = true)
        @test streamed.radial===nothing && streamed.regular_radial===nothing
        @test streamed.real_regular==full.real_regular
        for name in (:alpha, :beta_response, :retained, :regular, :response,
                :pressure_map, :flux_map, :regular_exterior, :matrix)
            @test isequal(getproperty(streamed, name), getproperty(full, name))
        end
        @test Base.summarysize(streamed)<Base.summarysize(full)
        if nt==4 && m==0 && parity==1
            factors = [merge(factor, (; stiffness = complex.(factor.stiffness)))
                       for factor in mesh.radial_factors]
            factors[1].stiffness[2, 2] += 1e-4im
            fallback_mesh = merge(mesh, (; radial_factors = factors))
            full_fallback = AS._spheroid_fembem_tensor_operator(
                fallback_mesh, be, T, B, K, V, 100.0, m, parity)
            streamed_fallback = AS._spheroid_fembem_tensor_operator(
                fallback_mesh, be, T, B, K, V, 100.0, m, parity; boundary_only = true)
            @test !full_fallback.real_regular && !streamed_fallback.real_regular
            @test full_fallback.regular_radial===nothing
            @test streamed_fallback.radial===nothing
            for name in (:alpha, :beta_response, :retained, :regular, :response, :matrix)
                @test isequal(getproperty(streamed_fallback, name), getproperty(full_fallback, name))
            end
        end
        beta = 0.61
        incident = [AS._p_inc_mode(m, 100.0, beta, r, z) for (r, z) in zip(be.rho, be.z)]
        derivative = [AS._dpdn_inc_mode(m, 100.0, beta, r, z, nx, nz)
                      for (r, z, nx, nz) in zip(be.rho, be.z, be.nrho, be.nz)]
        incident = (incident+parity*(-1)^m*conj(incident))/2
        derivative = (derivative+parity*(-1)^m*conj(derivative))/2
        # Full-column endpoint elimination supplies an independent boundary RHS.
        forcing = full.beta_response .* (full.flux_map*derivative)
        particular = forcing[full.regular] ./ full.alpha[full.regular]
        rhs = vcat(full.D*(incident-full.pressure_map[:, full.regular]*particular),
            forcing[full.retained])
        solution = full.matrix\rhs
        coefficients = zeros(ComplexF64, length(full.alpha))
        coefficients[full.regular] = particular+full.response .* (full.flux_map[full.regular, :]*solution[1:nbe])
        coefficients[full.retained] = solution[nbe+1:end]
        expected_pressure = full.pressure_map*coefficients-incident
        field = AS._spheroid_fembem_tensor_mode(
            mesh, be, T, B, K, V, 100.0, m, beta, AS._SolveReports(), 1;
            parity, interior_reduction = :angular, interior_field = :total)
        @test isapprox(field.pressure, expected_pressure; rtol = 2e-12, atol = 2e-14)
        @test isapprox(field.derivative, solution[1:nbe]; rtol = 2e-12, atol = 2e-13)
        load = ComplexF64[cis(i+2j)/(i+j) for i in 1:(mesh.nt*mesh.p+1), j in 1:(mesh.nr*mesh.p+1)]
        saved_load = copy(load)
        fresh = Dict(formulation => AS._spheroid_fembem_tensor_mode(
            mesh, be, T, B, K, V, 100.0, m, beta, AS._SolveReports(), 1;
            parity, load, interior_reduction = :angular, interior_field = formulation)
                     for formulation in (:total, :scattered))
        for order in ((:total, :scattered, :total), (:scattered, :total, :scattered))
            cache = AS._NumericalReuse()
            for formulation in order
                cached = AS._spheroid_fembem_tensor_mode(
                    mesh, be, T, B, K, V, 100.0, m, beta, AS._SolveReports(), 1;
                    parity, load, reuse = cache, reuse_key = :endpoint_regression,
                    exterior_key = :endpoint_regression, interior_reduction = :angular,
                    interior_field = formulation)
                @test cached.pressure==fresh[formulation].pressure
                @test cached.derivative==fresh[formulation].derivative
            end
            total_op = cache.entries[(:fembem_operator, :endpoint_regression, m, parity==-1, true)][1]
            scattered_op = cache.entries[(:fembem_operator, :endpoint_regression, m, parity==-1, false)][1]
            @test total_op.radial===nothing
            @test scattered_op.radial isa Matrix{ComplexF64}
            @test total_op.E===scattered_op.E
            radial_keys = filter(key -> first(key)===:fembem_radial, collect(keys(cache.entries)))
            @test Set(last.(radial_keys))==Set((false, true))
        end
        @test load==saved_load
    end
end

@testset "Real regular radial Robin responses" begin
    AS = AcousticScattering
    stiffness = [2.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 3.0]
    radial_mass = Matrix(Diagonal([0.5, 1.0, 1.5]))
    values = [0.0, 0.2, -0.3]
    load = ComplexF64[0.2, 0.3im, 0.4]
    for scale in (1e-8, 1.0, 1e8), density in (1.0, 0.0012), m in (0, 1)
        factor = (; stiffness = scale*stiffness, mass = scale*radial_mass,
            wave = zeros(3, 3), azimuthal = zeros(3, 3))
        eta = 2scale/density
        base, mass = AS._spheroid_fembem_radial_pencil([factor], 2, m, 1.0, density, eta)
        band = similar(base)
        radial = zeros(ComplexF64, 3, length(values))
        for j in eachindex(values)
            AS._spheroid_fembem_radial_map!(view(radial, :, j), band, base, mass, 2, values[j])
        end
        alpha = 1 .- im*eta .* radial[end, :]
        regular = eachindex(values)
        actual = AS._spheroid_fembem_real_radial(radial, alpha, base, mass, 2, values, regular, eta)
        @test actual!==nothing
        actual===nothing && continue
        @test eltype(actual)===Float64
        for j in regular
            # Dense high-precision solves check scaling and the constrained node.
            reference = setprecision(BigFloat, 192) do
                A = (BigFloat.(factor.stiffness)+BigFloat(values[j])*BigFloat.(factor.mass))/BigFloat(density)
                if m>0
                    A[1, :] .= 0
                    A[:, 1] .= 0
                    A[1, 1] = 1
                end
                Float64.(A \ BigFloat[0, 0, 1])
            end
            @test actual[:, j] ≈ reference rtol=2e-13
            @test alpha[j]*actual[:, j] ≈ radial[:, j] rtol=2e-13
            @test sum(actual[:, j] .* load) ≈ sum(radial[:, j] .* load)/alpha[j] rtol=2e-13
            @test actual[end, j] ≈ radial[end, j]/alpha[j] rtol=2e-13 atol=1e-13*maximum(abs, reference)
            m==0 || @test iszero(actual[1, j])
        end
        # The m=1, lambda=0 pencil has an exactly zero endpoint Green value.
        m==0 || @test abs(actual[end, 1])<=1e-14*maximum(abs, actual[:, 1])
        pressure = [1.0 0.2 -0.3; 0.4 0.7 0.5]
        flux = [0.6 -0.2; 0.3 0.5; -0.4 0.8]
        exterior = ComplexF64[1+0.2im 0.3-0.1im; -0.2+0.4im 0.8+0.1im]
        original = (exterior*pressure .* transpose(radial[end, :] ./ alpha))*flux
        reconstructed = exterior*((pressure .* transpose(actual[end, :]))*flux)
        @test reconstructed ≈ original rtol=3e-13
    end

    factor = (; stiffness, mass = radial_mass, wave = zeros(3, 3), azimuthal = zeros(3, 3))
    base, mass = AS._spheroid_fembem_radial_pencil([factor], 2, 0, 1.0, 1.0, 2.0)
    radial = reshape((ComplexF64.(stiffness)+Diagonal(ComplexF64[0, 0, 2im])) \ ComplexF64[0, 0, 1], 3, 1)
    alpha = [1-2im*radial[end]]
    check(response, coefficient = alpha, pencil = base, shifts = [0.0], indices = [1], eta = 2.0) =
        AS._spheroid_fembem_real_radial(response, coefficient, pencil, mass, 2, shifts, indices, eta)
    @test check(radial)!==nothing
    for invalid in (NaN, Inf, -1.0)
        @test check(radial, alpha, base, [0.0], [1], invalid)===nothing
    end
    @test check(radial, [0.0+0im])===nothing
    @test check(radial, [NaN+0im])===nothing
    @test check(radial, alpha, base, [Inf])===nothing
    @test check(radial, alpha, base, [0.0], [0])===nothing
    @test check(radial, alpha, base, [0.0], [2])===nothing
    invalid = copy(radial)
    invalid[2] += 1e-4im
    @test check(invalid)===nothing
    invalid = copy(radial)
    invalid[2] += alpha[1]*1e-4
    @test check(invalid)===nothing
    invalid = copy(radial)
    invalid[2] = Inf
    @test check(invalid)===nothing
    invalid_base = copy(base)
    invalid_base[5, 1] += 1e-4im
    @test check(radial, alpha, invalid_base)===nothing
    @test size(check(radial, alpha, base, [0.0], Int[]))==(3, 0)

    # A near-pole endpoint remains in the original complex augmented system.
    pole_factor = (; stiffness = Matrix(Diagonal([2.0, 1.0, 1e-12])),
        mass = zeros(3, 3), wave = zeros(3, 3), azimuthal = zeros(3, 3))
    pole_base, pole_mass = AS._spheroid_fembem_radial_pencil([pole_factor], 2, 0, 1.0, 1.0, 2.0)
    pole_radial = zeros(ComplexF64, 3, 1)
    AS._spheroid_fembem_radial_map!(view(pole_radial, :, 1), similar(pole_base), pole_base, pole_mass, 2, 0.0)
    pole_alpha = [1-2im*pole_radial[end]]
    @test abs(pole_alpha[1])<=0.5abs(pole_radial[end])
    @test AS._spheroid_fembem_real_radial(
        pole_radial, pole_alpha, pole_base, pole_mass, 2, [0.0], [1], 2.0)===nothing
end

@testset "Real radial residuals include gradual-underflow quantization" begin
    AS = AcousticScattering
    relative_failures = 0
    for coupling in (-3.14159e-303, -2.71828e-308, -9.999e-312)
        stiffness = [1e5 coupling; coupling 1e4]
        factor = (; stiffness, mass = zeros(2, 2), wave = zeros(2, 2), azimuthal = zeros(2, 2))
        eta = 3e4
        base, mass = AS._spheroid_fembem_radial_pencil([factor], 1, 0, 1.0, 1.0, eta)
        radial = zeros(ComplexF64, 2, 1)
        AS._spheroid_fembem_radial_map!(view(radial, :, 1), similar(base), base, mass, 1, 0.0)
        alpha = [1-im*eta*radial[end]]
        actual = AS._spheroid_fembem_real_radial(radial, alpha, base, mass, 1, [0.0], [1], eta)
        @test actual!==nothing
        actual===nothing && continue
        @test issubnormal(actual[1])
        setprecision(BigFloat, 256) do
            reference = BigFloat.(stiffness) \ BigFloat[0, 1]
            difference = BigFloat.(actual[:, 1])-reference
            @test norm(difference)<=big"2e-13"*norm(reference)
            # Resolve normalized subnormal responses against a high-precision solve.
            @test abs(difference[1])<=64BigFloat(nextfloat(0.0))/BigFloat(real(alpha[1]))
        end
        terms = stiffness[1, :] .* actual[:, 1]
        relative_failures += abs(sum(terms))>128eps(Float64)*3sum(abs, terms)
        # Reject ordinary residual errors and excessive subnormal corruption.
        for perturbation in (1e-200, 1e-318)
            corrupted = copy(radial)
            corrupted[1] += alpha[1]*perturbation
            rejected = AS._spheroid_fembem_real_radial(
                corrupted, alpha, base, mass, 1, [0.0], [1], eta; partition = true)
            @test isempty(rejected.regular)
            @test rejected.reasons==[:radial_residual]
        end
    end
    @test relative_failures>=2
end

@testset "Partitioned real responses preserve complex augmented equations" begin
    AS = AcousticScattering
    stiffness = [2.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 3.0]
    factor = (; stiffness, mass = Matrix(Diagonal([0.5, 1.0, 1.5])),
        wave = zeros(3, 3), azimuthal = zeros(3, 3))
    values, eta = [0.0, 0.2, -0.3], 2.0
    base, mass = AS._spheroid_fembem_radial_pencil([factor], 2, 0, 1.0, 1.0, eta)
    radial = zeros(ComplexF64, 3, 3)
    for j in eachindex(values)
        AS._spheroid_fembem_radial_map!(view(radial, :, j), similar(base), base, mass, 2, values[j])
    end
    alpha = 1 .- im*eta .* radial[end, :]
    partition = AS._spheroid_fembem_real_radial(
        radial, alpha, base, mass, 2, values, 1:3, eta; partition = true)
    @test partition.regular==[1, 2, 3]
    @test isempty(partition.failed)
    @test isempty(partition.reasons)
    @test partition.radial ≈ AS._spheroid_fembem_real_radial(radial, alpha, base, mass, 2, values, 1:3, eta)

    # Preserve complex rejected columns while compacting accepted responses.
    radial[2, 2] += 1e-5im
    saved_radial, saved_alpha = copy(radial), copy(alpha)
    @test AS._spheroid_fembem_real_radial(radial, alpha, base, mass, 2, values, 1:3, eta)===nothing
    pressure = [1.0 0.2 -0.3; 0.4 0.7 0.5]
    flux = [0.6 -0.2; 0.3 0.5; -0.4 0.8]
    exterior = ComplexF64[1+0.2im 0.3-0.1im; -0.2+0.4im 0.8+0.1im]
    single = ComplexF64[2+0.3im 0.1+0.2im; -0.3+0.1im 1.5-0.2im]
    load = ComplexF64[complex(sin(i+j), cos(i+2j)) for i in 1:3, j in 1:3]
    incident = ComplexF64[0.2+0.3im, -0.1+0.4im]
    derivative = ComplexF64[0.3-0.2im, 0.5+0.1im]
    beta_response = vec(radial[end, :])
    full_matrix = [single exterior*pressure;
                   -Diagonal(beta_response)*flux Matrix(Diagonal(alpha))]
    for original_retained in (Int[], [3]), interior_field in (:scattered, :total)
        original_regular = setdiff(1:3, original_retained)
        partition = AS._spheroid_fembem_real_radial(
            radial, alpha, base, mass, 2, values, original_regular, eta; partition = true)
        @test partition.regular==setdiff(original_regular, [2])
        @test partition.failed==[2]
        @test partition.reasons==[:reconstruction]
        regular = partition.regular
        retained = sort(vcat(original_retained, partition.failed))
        response = vec(partition.radial[end, :])
        boundary_forcing = flux*derivative
        forcing = interior_field===:scattered ?
                  [sum(radial[:, j] .* load[:, j]) for j in 1:3] :
                  beta_response .* boundary_forcing
        particular = interior_field===:scattered ?
                     [sum(partition.radial[:, index] .* load[:, j])
                      for (index, j) in enumerate(regular)] :
                     response .* boundary_forcing[regular]
        source = interior_field===:scattered ? zeros(ComplexF64, 2) : exterior*incident
        reference = full_matrix \ vcat(source, forcing)
        schur = single+exterior*((pressure[:, regular] .* transpose(response))*flux[regular, :])
        matrix = [schur exterior*pressure[:, retained];
                  -Diagonal(beta_response[retained])*flux[retained, :] Matrix(Diagonal(alpha[retained]))]
        solution = matrix \ vcat(source-exterior*(pressure[:, regular]*particular), forcing[retained])
        coefficients = zeros(ComplexF64, 3)
        coefficients[regular] = particular+response .* (flux[regular, :]*solution[1:2])
        coefficients[retained] = solution[3:end]
        actual = vcat(solution[1:2], coefficients)
        @test actual ≈ reference rtol=2e-13 atol=2e-14
        @test full_matrix*actual ≈ vcat(source, forcing) rtol=2e-13 atol=2e-14
    end
    @test radial==saved_radial
    @test alpha==saved_alpha
    rejected = AS._spheroid_fembem_real_radial(
        radial, zeros(ComplexF64, 3), base, mass, 2, values, 1:3, eta; partition = true)
    @test isempty(rejected.regular)
    @test size(rejected.radial)==(3, 0)
    @test rejected.failed==[1, 2, 3]
    @test rejected.reasons==fill(:denominator, 3)
end

@testset "Tensor FEM-BEM elimination" begin
    AS = AcousticScattering
    body = Spheroid(0.02, 0.015)
    # Compare independent eliminations across materials, domains and FE/BE sizes.
    for (bc, beta, split) in ((FluidFilled(1.2, 0.85), 0.6, true),
            (FluidFilled(1.2, 1.0), 0.6, true), (FluidFilled(1.0, 0.85), 0.6, true),
            (FluidFilled(0.0012, 0.23), 0.6, true),
            (FluidFilled(1.2, 0.85), pi/2, true), (FluidFilled(1.2, 0.85), 0.0, false)),
        (nt, nb) in ((24, 16), (48, 12))

        options = (;
            half = beta==pi/2, incidence_angle = beta, m_max = 4, mode_threads = 2,
            rtol = 1e-8, chief_points = 8, split_symmetry = split)
        original = AS._spheroid_fembem_level(body, bc, 100.0, 3, nt, nb, 4, 5;
            options..., tensor = false, reports = AS._SolveReports())
        reduced = AS._spheroid_fembem_level(body, bc, 100.0, 3, nt, nb, 4, 5;
            options..., tensor = true, reports = AS._SolveReports())
        for (old, new) in zip(original, reduced)
            @test isapprox(new.pressure, old.pressure; rtol = 1e-8, atol = 1e-12)
            @test isapprox(new.derivative, old.derivative; rtol = 1e-8, atol = 1e-10)
        end
    end
    # A singular local interior block must not create an artificial pole.
    stiffness = [2.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 3.0]
    factor = (; stiffness, mass = Matrix(Diagonal([0.5, 1.0, 1.5])),
        wave = zeros(3, 3), azimuthal = zeros(3, 3))
    load = ComplexF64[0.2, 0.3im, 0.4]
    for m in (0, 1)
        base, mass = AS._spheroid_fembem_radial_pencil([factor], 2, m, 1.0, 1.0, 2.0)
        band, response = similar(base), zeros(ComplexF64, 3)
        # Revisit singular shifts after factorization reuse with independent loads.
        for lambda in (0.0, 0.2, -0.3, 0.0)
            A = ComplexF64.(stiffness+lambda*factor.mass)
            rhs = hcat(ComplexF64[0, 0, 1], load)
            if m>0
                A[1, :] .= 0
                A[:, 1] .= 0
                A[1, 1] = 1
                rhs[1, :] .= 0
            end
            A[end, end] += 2im
            AS._spheroid_fembem_radial_map!(response, band, base, mass, 2, lambda)
            actual = [response[end], sum(response .* load)]
            @test isapprox(actual, (A \ rhs)[end, :]; rtol = 1e-13)
        end
    end
    # Interior Dirichlet and Neumann poles remain regular transmission problems.
    bc = FluidFilled(1.2, 0.85)
    for interior_ka in (pi, 4.493409457909064)
        k = interior_ka*bc.soundspeed_contrast
        numerical = coupled(Spheroid(1.00000001, 1.0), bc, k;
            coupling = Coupling(interior = :fem, exterior = :bem),
            order = 6, n_r = 6, n_theta = 32, n_boundary = 24, m_max = 10, rtol = 1e-8,
            adaptive = false, incidence_angle = 0.6, mode_threads = 2)
        exact = modal(Sphere(1.0), bc, k)
        @test isapprox(scattering_amplitude(numerical), scattering_amplitude(exact); rtol = 1e-6)
        @test diagnostics(numerical).relative_residual<1e-10
    end
end
