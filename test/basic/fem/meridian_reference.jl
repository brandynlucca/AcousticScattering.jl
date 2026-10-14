using AcousticScattering, Test, SparseArrays, LinearAlgebra

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
