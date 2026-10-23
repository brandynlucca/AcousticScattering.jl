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
    # Coordinate checks separate anisotropic errors and expose mixed terms
    # that could otherwise cancel in a simultaneous mesh change.
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
    # Uncondensed equations independently check the reused Schur buffers and
    # incident lifting across cells both on and away from the fluid interface.
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
    # An independently solved sphere checks wave normalization, the incident
    # lifting, pressure recovery, rotations and outgoing continuation. The
    # spheroid constructor excludes exactly equal semiaxes.
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
    # Independent high-precision evaluation of the unnormalized polynomials.
    # These orders occur for a 70 by 10 mm spheroid at 100 and 200 kHz.
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

# Recorded before merging the cylinder/spheroid mode assemblies on 2026-10-02.
# These detect refactor drift; independent physical checks remain in the extended suite.
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
    # Exact outgoing monopole and Cartesian dipole check the jump, outward
    # derivative, Fourier normalization, reflection and far-field phase.
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
            # Compare the decomposition against the unreduced equations with
            # the same partitions, modes, quadrature and CHIEF locations.
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
            # Updating disjoint modes preserves the other traces and agrees
            # with assembling the complete Fourier range on the same mesh.
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
    # Near a spherical exterior Dirichlet irregular frequency, auxiliary
    # interior constraints retain agreement with the exact fluid sphere.
    near_sphere = coupled(Spheroid(1.00000001, 1.0), bc, pi;
        coupling = Coupling(interior = :fem, exterior = :bem),
        order = 6, n_r = 6, n_theta = 32, n_boundary = 24,
        m_max = 10, rtol = 1e-8, adaptive = false, mode_threads = 2)
    exact_sphere = modal(Sphere(1.0), bc, Float64(pi))
    @test isapprox(scattering_amplitude(near_sphere),
        scattering_amplitude(exact_sphere); rtol = 1e-6)
    @test diagnostics(near_sphere).representation_residual < 1e-8
    # Independent spatial changes and their interaction supply a reported TS
    # change without a spheroidal modal answer.
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
    # An independently assembled Kronecker matrix checks the boundary inverse,
    # with unrelated loads and repeated application through the same workspace.
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

@testset "Tensor FEM-BEM elimination" begin
    AS = AcousticScattering
    body = Spheroid(0.02, 0.015)
    # Compare independent elimination orders of the same discrete weak form,
    # including density-only and speed-only scattering, unreduced domains and
    # both contraction orders for the independently sized FE/BE interfaces.
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
        # Reuse the factorization storage at different shifts, then revisit the
        # singular local block. Compare to two independent full-system loads.
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
    # Interior spherical Dirichlet and Neumann frequencies remain regular
    # transmission problems. Both are independent of exterior CHIEF checks.
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
