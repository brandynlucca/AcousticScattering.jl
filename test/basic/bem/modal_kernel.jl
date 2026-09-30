using AcousticScattering
using LinearAlgebra
using StaticArrays
using Test

@testset "Split BEM ring kernels against independent high-precision integrals" begin
    AS = AcousticScattering
    function reference(k, rho, rho2, z2, normal, projection, modes, ::Val{N}) where {N}
        setprecision(128) do
            kb, r, r2, z, n, p = BigFloat.((k, rho, rho2, z2, normal, projection))
            scale = hypot(r-r2, z)/sqrt(r*r2)
            upper = asinh(big(pi)/scale)
            points = collect(range(zero(scale), upper; length = max(40, 2last(modes)+2)))
            value, error = AS.quadgk(points; rtol = big"1e-19", atol = big"1e-23", maxevals = 100000) do tau
                phi = scale*sinh(tau)
                kv = AS._ring_KV(kb, r, zero(r), r2, z, n, p, phi)
                SVector{2N}(ntuple(2N) do i
                    c = mod1(i, N)
                    c <= length(modes) ?
                    kv[i<=N ? 1 : 2]*cos(BigFloat(modes[c])*phi)*scale*cosh(tau) :
                    zero(kb)*im
                end)
            end
            @test error <= max(big"1e-23", big"1e-19"*norm(value))
            return ComplexF64.(2value)
        end
    end
    accepted = 0
    for geometry in ((1.0, 1.0, 1e-8, 0.7, 0.0),
            (1.0, 1.0-1e-6, 0.0, 1.0, 1e-6),
            (1.0, 0.99, 0.03, -0.7, 0.006),
            (1.0, 1.0, 0.1, 0.7, 0.03),
            (1.0, 1.0, 0.5, 0.7, 0.15)),
        ka in (0.0, 0.1, 4.0, 8.0), modes in (0:0, 2:4, 32:39, 56:63)
        rho, rho2, z2, normal, projection = geometry
        N = length(modes)==1 ? 1 : length(modes)<=4 ? 4 : 8
        width = Val(N)
        scratch = AS._ring_split_workspace(modes, width)
        expected = reference(ka, geometry..., modes, width)
        for scale in (1e-3, 1.0, 1e3)
            accepted_case, actual = AS._ring_split_KV(ka/scale, rho*scale, 0.0, rho2*scale,
                z2*scale, normal, projection*scale, modes, 1e-6, width, scratch)
            accepted_case || continue
            accepted += 1
            scaled = SVector{2N}(ntuple(i -> expected[i]/(i<=N ? scale^2 : scale), Val(2N)))
            @test norm(actual-scaled) <= 0.01max(AS._QUAD_ATOL, 1e-6norm(scaled))
            @test all(isfinite, actual)
            for i in (length(modes) + 1):N
                @test actual[i] == actual[N + i] == 0
            end
        end
    end
    @test accepted >= 120
end

@testset "Split ring eligibility and fallback preserve adaptive evaluation" begin
    AS = AcousticScattering
    for (k, rho, rho2, z, modes, rtol) in (
        (4.0, 0.0, 0.1, 0.3, 0:7, 1e-6),
        (4.0, 1.0, 1.0, 1.0, 0:7, 1e-6),
        (40.0, 1.0, 1.0, 0.01, 0:7, 1e-6),
        (4.0, 1.0, 1.0, 0.01, 120:127, 1e-6),
        (4.0, 1.0, 1.0, 1e-13, 0:7, 1e-6),
        (4.0, 1.0, 1.0, 0.03, 32:39, 1e-14))
        workspace = AS._bem_quadrature_workspace(modes, Val(8))
        @test !first(AS._ring_split_KV(k, rho, 0.0, rho2, z, 0.7, 0.01,
            modes, rtol, Val(8), workspace.modal))
        geometry = AS._ring_geometry(rho, 0.0, rho2, z, 0.7, 0.01)
        f = phi -> AS._ring_KV_modes(k, geometry, phi, first(modes), length(modes), Val(8))
        expected = 2AS._azimuthal_peak_quadrature(f, last(modes), rho, 0.0, rho2, z,
            rtol, AS._QUAD_ATOL/2, workspace)
        actual = AS._azimuthal_KV_modes(k, rho, 0.0, rho2, z, 0.7, 0.01,
            modes, false, nothing, rtol, Val(8), workspace)
        @test actual == expected
    end
    modes = 0:7
    scratch = AS._ring_split_workspace(modes, Val(8))
    args = (4.0, 1.0, 0.0, 1.0, 0.03, 0.7, 0.0, modes, 1e-6, Val(8))
    @test first(AS._ring_split_KV(args..., scratch))
    # A corrupted coarse rule forces the quadrature-resolution safeguard.
    scratch.coarse.weights .*= 2
    @test !first(AS._ring_split_KV(args..., scratch))
    @test AS._ring_split_workspace(65:65, Val(1)) === nothing
    @test AS._ring_split_workspace(-1:-1, Val(1)) === nothing
    @test !first(AS._ring_split_KV(args..., nothing))
end

@testset "Split remainder formulas at the Taylor transition" begin
    AS = AcousticScattering
    for x in (-16.0, -1.0, -0.999, -1e-8, 0.0, 1e-8, 0.999, 1.0, 16.0)
        expected = setprecision(256) do
            b = BigFloat(x)
            (complex(cos(b)+b*sin(b)-1-b^2/2+b^4/8, sin(b)-b*cos(b)),
                complex(cos(b)-1+b^2/2-b^4/24, sin(b)))
        end
        actual = AS._ring_split_remainder(x)
        for i in 1:2, component in (real, imag)

            @test component(actual[i])≈component(expected[i]) rtol=5e-13 atol=1e-58
        end
    end
end
