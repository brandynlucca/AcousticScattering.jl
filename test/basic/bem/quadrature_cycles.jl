using AcousticScattering
using LinearAlgebra
using StaticArrays
using Test

@testset "BEM full-cycle starts against tightly resolved ring integrals" begin
    AS = AcousticScattering
    function precise_reference(
            k, rho, rho2, z2, normal, projection, modes, ::Val{N}) where {N}
        setprecision(BigFloat, 128) do
            kb, rb, r2b, zb, nb, pb = BigFloat.((k, rho, rho2, z2, normal, projection))
            scale = clamp(hypot(rb-r2b, zb)/sqrt(rb*r2b), big"1e-12", big"1e6")
            points = collect(range(zero(BigFloat), asinh(big(pi)/scale);
                length = max(40, 4last(modes)+2)))
            AS.quadgk(points; rtol = big"1e-18", atol = big"1e-18", maxevals = 100000) do tau
                phi = min(scale*sinh(tau), big(pi))
                kv = AS._ring_KV(kb, rb, zero(BigFloat), r2b, zb, nb, pb, phi)
                # Direct high-precision cosines independently check the production recurrence.
                SVector{2N}(ntuple(2N) do i
                    index = mod1(i, N)
                    index <= length(modes) ?
                    kv[i <= N ? 1 : 2]*cos(BigFloat(modes[index])*phi)*(scale*cosh(tau)) :
                    zero(Complex{BigFloat})
                end)
            end
        end
    end
    for scale in (0.01, 1.0, 100.0),
        geometry in ((1.0, 0.99, 0.03, 0.7, 0.006),
            (1.0, 1.0-1e-6, 0.0, 1.0, 1e-6),
            (1e-5, 2e-5, 0.1, 1.0, 0.03),
            (1.0, 0.3, 0.5, 0.4, 0.05)),
        k0 in (0.1, 4.0, 40.0), modes in (0:0, 1:1, 2:4, 8:12, 32:39, 120:127)
        rho, rho2, z2, normal, projection = geometry
        rho *= scale
        rho2 *= scale
        z2 *= scale
        projection *= scale
        k = k0/scale
        width = length(modes)==1 ? Val(1) : length(modes)<=4 ? Val(4) : Val(8)
        integrand = phi -> AS._ring_KV_modes(k, rho, 0.0, rho2, z2, normal,
            projection, phi, first(modes), length(modes), width)
        peak_scale = clamp(hypot(rho-rho2, z2)/sqrt(rho*rho2),
            AS._RING_PEAK_FLOOR, AS._RING_PEAK_CEILING)
        upper = asinh(pi/peak_scale)
        # Uniform transformed-coordinate splits differ from either production grid.
        points = collect(range(0.0, upper; length = max(40, 4last(modes)+2)))
        reference, error = AS.quadgk(points; rtol = 1e-11, atol = 1e-13, maxevals = 100000) do tau
            integrand(min(peak_scale*sinh(tau), pi))*(peak_scale*cosh(tau))
        end
        # Strong cancellation can reach the Float64 quadrature roundoff floor.
        # Resolve those references at higher precision rather than relax the check.
        if error > max(1e-13, 1e-11norm(reference))
            reference, error = precise_reference(
                k, rho, rho2, z2, normal, projection, modes, width)
        end
        @test error <= max(1e-13, 1e-11norm(reference))
        workspace = AS._bem_quadrature_workspace(modes, width)
        actual = AS._azimuthal_peak_quadrature(integrand, last(modes), rho, 0.0,
            rho2, z2, 1e-7, AS._QUAD_ATOL/2, workspace)
        if last(modes)>1 && length(workspace.breaks)==last(modes)+1
            # The former capped rule can itself remain unresolved. Its fallback
            # must preserve that result, not claim a stronger convergence guarantee.
            previous = AS._azimuthal_peak_quadrature(integrand, last(modes), rho,
                0.0, rho2, z2, 1e-7, AS._QUAD_ATOL/2)
            @test actual == previous
        else
            @test norm(actual-reference) <= 5max(AS._QUAD_ATOL/2, 1e-7norm(reference))
        end
    end
end

@testset "BEM full-cycle evaluation count and capped fallback" begin
    AS = AcousticScattering
    calls = Ref(0)
    integrand = phi -> begin
        calls[] += 1
        AS._ring_KV_modes(4.0, 1.0, 0.0, 0.96, 0.02, 0.8, 0.03, phi, 32, 8, Val(8))
    end
    args = (39, 1.0, 0.0, 0.96, 0.02, 1e-6, AS._QUAD_ATOL/2)
    previous = AS._azimuthal_peak_quadrature(integrand, args...)
    old_calls = calls[]
    calls[] = 0
    actual = AS._azimuthal_peak_quadrature(integrand, args...,
        AS._bem_quadrature_workspace(32:39, Val(8)))
    @test calls[] < old_calls
    @test actual≈previous rtol=1e-8 atol=1e-11

    # Deliberately unresolved oscillation must retry the exact previous start.
    workspace = AS._bem_quadrature_workspace(3:3, Val(1))
    oscillatory = phi -> begin
        calls[] += 1
        SVector(cis(2000phi), cis(1999phi))
    end
    args = (3, 1.0, 0.0, 0.5, 1.0, 1e-10, 1e-12)
    calls[] = 0
    previous = AS._azimuthal_peak_quadrature(oscillatory, args...)
    old_calls = calls[]
    calls[] = 0
    actual = AS._azimuthal_peak_quadrature(oscillatory, args..., workspace)
    @test calls[] > old_calls
    @test length(workspace.breaks) == 4
    @test actual≈previous rtol=1e-13 atol=1e-13
    simple = phi -> SVector(1.0+0im, 0.0+0im)
    actual = AS._azimuthal_peak_quadrature(simple, args..., workspace)
    @test actual ≈ SVector(pi+0im, 0.0+0im) rtol=1e-10
    @test length(workspace.breaks) == 3
end
