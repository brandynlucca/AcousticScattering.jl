using AcousticScattering
using Test
using LinearAlgebra: BLAS

const AS = AcousticScattering

BLAS.set_num_threads(1)

@time @testset "Fourier matching: sphere limit" begin
    a = 2.0
    profile = AS.Irregular(a, Float64[], Float64[])
    mapping = AS.solve_mapping(profile, 4; continuation_steps = 1)

    @test all(iszero, mapping.delta_c)
    @test all(iszero, mapping.delta_s)
    @test mapping.c[1] ≈ a atol = 1e-10
    @test all(c -> abs(c) < 1e-10, mapping.c[2:end])
    @test AS.is_admissible(mapping)

    for w in range(0, π; length = 41)
        @test AS.mapping_theta(mapping, w) ≈ w atol = 1e-12
        g, f = AS.mapping_surface(mapping, w)
        @test g ≈ a * cos(w) atol = 1e-10
        @test f ≈ a * sin(w) atol = 1e-10
    end
end

@time @testset "Fourier matching: prolate spheroid, ellipse-equation check" begin
    a, b = 3.0, 1.0
    order = 32
    Rtheta(theta) = 1 / sqrt((cos(theta) / a)^2 + (sin(theta) / b)^2)
    profile = AS.Irregular(Rtheta, order; npoints = 800)
    mapping = AS.solve_mapping(profile, order; continuation_steps = 16)

    @test AS.is_admissible(mapping)
    for w in range(0, π; length = 73)
        g, f = AS.mapping_surface(mapping, w)
        # Independent check: the reconstructed (g,f) must satisfy the exact ellipse equation, without reference to theta(w) or the profile fit at all.
        @test (g / a)^2 + (f / b)^2 ≈ 1 atol = 1e-4
    end
end

@time @testset "Fourier matching: mapping order convergence" begin
    a, b = 3.0, 1.0
    Rtheta(theta) = 1 / sqrt((cos(theta) / a)^2 + (sin(theta) / b)^2)

    function ellipse_residual(order)
        profile = AS.Irregular(Rtheta, order; npoints = 800)
        mapping = AS.solve_mapping(profile, order; continuation_steps = 16)
        return maximum(range(0, π; length = 37)) do w
            g, f = AS.mapping_surface(mapping, w)
            abs((g / a)^2 + (f / b)^2 - 1)
        end
    end

    residuals = [ellipse_residual(order) for order in (8, 16, 24, 32)]
    # Monotonic, and each doubling of order should cut the residual by well over an order of magnitude, per both source papers; a plateauing or non-monotonic sequence would indicate a bug.
    @test issorted(residuals; rev = true)
    @test residuals[end] < residuals[1] / 100
end

@time @testset "Fourier matching: noncanonical bumpy profile" begin
    a = 1.0
    Rtheta(theta) = a * (1 + 0.12 * cos(3theta) - 0.05 * sin(2theta))
    order = 24
    profile = AS.Irregular(Rtheta, order; npoints = 400)
    mapping = AS.solve_mapping(profile, order; continuation_steps = 12)

    @test AS.is_admissible(mapping)
    # Self-consistency: the mapped radial distance from the origin at w must match the supplied profile evaluated at theta(w) (Eq. (28)'s relationship, checked pointwise).
    maxerr = maximum(range(0, π; length = 73)) do w
        g, f = AS.mapping_surface(mapping, w)
        theta = AS.mapping_theta(mapping, w)
        abs(hypot(g, f) - Rtheta(theta))
    end
    @test maxerr < 1e-3

    # Convergence: more modes should reduce the same self-consistency residual.
    function self_consistency_residual(ord)
        p = AS.Irregular(Rtheta, ord; npoints = 400)
        m = AS.solve_mapping(p, ord; continuation_steps = 12)
        return maximum(range(0, π; length = 37)) do w
            g, f = AS.mapping_surface(m, w)
            abs(hypot(g, f) - Rtheta(AS.mapping_theta(m, w)))
        end
    end
    coarse = self_consistency_residual(8)
    fine = self_consistency_residual(24)
    @test fine < coarse / 20
end

@time @testset "Fourier matching: input validation and admissibility rejection" begin
    @test_throws ArgumentError AS.Irregular(1.0, [1.0], Float64[])
    @test_throws ArgumentError AS.Irregular(-1.0, Float64[], Float64[])
    @test_throws ArgumentError AS.Irregular(theta -> 1.0, -1)
    @test_throws ArgumentError AS.solve_mapping(AS.Irregular(1.0, Float64[], Float64[]), 0)
    @test_throws ArgumentError AS.solve_mapping(
        AS.Irregular(1.0, Float64[], Float64[]), 4; continuation_steps = 0)

    # Directly construct a mapping whose Jacobian is known analytically to vanish (c_(-1)=1, c_1=1, else 0, so dG/drho|_{u=0}=2im*sin(w)=0 at w=0,pi), to test is_admissible in isolation from solve_mapping.
    dummy_profile = AS.Irregular(1.0, Float64[], Float64[])
    inadmissible = AS.ConformalMapping(
        dummy_profile, Float64[], Float64[], ComplexF64[1.0, 0.0, 1.0])
    @test AS.mapping_jacobian_squared(inadmissible, 0.0) == 0
    @test !AS.is_admissible(inadmissible)
end
