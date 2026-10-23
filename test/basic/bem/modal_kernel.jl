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
    convolution_accepted = 0
    for geometry in ((1.0, 1.0, 1e-6, 0.7, 0.0),
            (1.0, 0.99, 0.03, -0.7, 0.006),
            (1.0, 1.0, 0.3, 0.7, 0.03)),
        ka in (12.0, 43.0, 60.0), modes in (0:15, 16:31, 48:63, 96:111)
        rho, rho2, z2, normal, projection = geometry
        expected = reference(ka, geometry..., modes, Val(16))
        scratch = AS._ring_convolution_workspace(modes, ka)
        for scale in (1e-3, 1.0, 1e3)
            accepted_case, actual = AS._ring_convolution_KV(ka/scale, rho*scale, 0.0,
                rho2*scale, z2*scale, normal, projection*scale, modes, 1e-6, Val(16), scratch)
            accepted_case || continue
            convolution_accepted += 1
            scaled = SVector{32}(ntuple(i->expected[i]/(i<=16 ? scale^2 : scale), Val(32)))
            @test norm(actual-scaled) <= 0.01max(AS._QUAD_ATOL, 1e-6norm(scaled))
            @test all(isfinite, actual)
        end
    end
    @test convolution_accepted >= 20
end

@testset "Convolution ring guards and quadrature rejection" begin
    AS = AcousticScattering
    for (modes, ka) in ((0:0, 40.0), (0:15, 0.0), (0:15, 65.0), (120:135, 40.0))
        @test AS._ring_convolution_workspace(modes, ka) === nothing
    end
    modes, width = 0:15, Val(16)
    workspace = AS._bem_quadrature_workspace(modes, width; maxlambda = 40.0)
    args = (40.0, 1.0, 0.0, 1.0, 0.001, 0.7, 0.0, modes)
    @test first(AS._ring_convolution_KV(args..., 1e-6, width, workspace.convolution))
    @test !first(AS._ring_convolution_KV(args..., 1e-12, width, workspace.convolution))
    workspace.convolution.coarse.transform .*= 2
    @test !first(AS._ring_convolution_KV(args..., 1e-6, width, workspace.convolution))
    geometry = AS._ring_geometry(1.0, 0.0, 1.0, 0.001, 0.7, 0.0)
    f = phi -> AS._ring_KV_modes(40.0, geometry, phi, 0, 16, width)
    expected = 2AS._azimuthal_peak_quadrature(f, 15, 1.0, 0.0, 1.0, 0.001,
        1e-6, AS._QUAD_ATOL/2, workspace)
    actual = AS._azimuthal_KV_modes(args..., false, nothing, 1e-6, width, workspace)
    @test actual == expected
    partial = 3:8
    workspace = AS._bem_quadrature_workspace(partial, Val(8); maxlambda = 12.0)
    accepted, result = AS._ring_convolution_KV(12.0, 1.0, 0.0, 1.0, 0.001,
        0.7, 0.0, partial, 1e-6, Val(8), workspace.convolution)
    @test accepted
    @test all(iszero, result[[7, 8, 15, 16]])
    integrand = phi->AS._ring_KV_modes(12.0, geometry, phi, 3, 6, Val(8))
    reference = 2AS._azimuthal_peak_quadrature(integrand, 8, 1.0, 0.0, 1.0, 0.001,
        1e-10, AS._QUAD_ATOL/2, workspace)
    @test result ≈ reference rtol=1e-10
end

@testset "High-frequency curved operators against tighter direct quadrature" begin
    AS = AcousticScattering
    boundary = AS._spheroid_curved_boundary(0.02, 0.01, 4, 5; half = true)
    # The tighter run falls outside the convolution eligibility and recomputes
    # the original azimuthal integrals. Compare complete boundary and CHIEF rows.
    accelerated = AS._spheroid_curved_operators(boundary, 1200.0, 0:7, Val(8), Val(5);
        rtol = 1e-6, split = true, chief_points = 2)
    direct = AS._spheroid_curved_operators(boundary, 1200.0, 0:7, Val(8), Val(5);
        rtol = 1e-10, split = true, chief_points = 2)
    for (left, right) in zip(accelerated, direct), (a, b) in zip(left, right)

        @test norm(a-b) <= 2e-7norm(b)
    end
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
