using AcousticScattering
using LinearAlgebra
using Test

@testset "Fluid interface principal-part inverse" begin
    AS = AcousticScattering
    S = ComplexF64[2 0.2im 0.1; 0.3 1.5 0.2; 0.1im 0.1 1.2]
    n = 6
    groups = ([4, 1, 6], [2, 5, 3])
    rows, cols = collect(range(0.3, 1.2; length = 2n)),
    collect(range(2.0, 0.8; length = 2n))
    rhs = ComplexF64[sin(j)+im*cos(2j) for j in 1:2n]
    original = copy(rhs)
    for (re, ri) in ((1.0, 0.0012), (0.0012, 1.0), (1.04, 0.8), (1.0, 1.0)),
        equilibrate in (false, true)

        r, c = equilibrate ? (rows, cols) : (ones(2n), ones(2n))
        interfaces = [(; indices, S, factor = lu(S),
                          density_exterior = a, density_interior = b)
                      for (indices, a, b) in ((groups[1], re, ri), (groups[2], 0.7, 1.2))]
        P = AS._fluid_interface_preconditioner(interfaces, r, c)
        # Independently assemble and factor the full pressure/flux block system.
        M = Matrix{ComplexF64}(I, 2n, 2n)
        for interface in interfaces
            (; indices, density_exterior, density_interior) = interface
            M[indices, indices .+ n] = (density_exterior-density_interior)*S
            M[indices .+ n, indices] = (1/density_exterior-1/density_interior)/4*inv(S)
        end
        expected = c .* (M \ (r .* rhs))
        out = similar(rhs)
        @test ldiv!(out, P, rhs) === out
        @test out ≈ expected rtol = 2e-12
        @test rhs == original
        @test ldiv!(P, copy(rhs)) ≈ expected rtol = 2e-12
        @test norm((M*(out ./ c)) ./ r-rhs)/norm(rhs) < 2e-11
        @test ldiv!(P, zeros(ComplexF64, 2n)) == zeros(ComplexF64, 2n)
        @test ldiv!(P, copy(rhs)) ≈ expected rtol = 2e-12
    end
    # Equal-density interfaces are exactly identity in physical coordinates and
    # must not invoke either operator. Extreme densities must not square overflow.
    for (re, ri) in ((1.0, 1.0), (1e200, 1e200), (1e200, 0.5e200), (1e-200, 0.5e-200))
        equal = re == ri
        interfaces = [(; indices = 1:n, S = equal ? nothing : Matrix{ComplexF64}(I, n, n),
            factor = equal ? nothing : lu(Matrix{ComplexF64}(I, n, n)),
            density_exterior = re, density_interior = ri)]
        P = AS._fluid_interface_preconditioner(interfaces, rows, cols)
        result = ldiv!(P, copy(rhs))
        @test all(isfinite, result)
        equal && @test result ≈ rows .* cols .* rhs
    end
end
