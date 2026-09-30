using AcousticScattering
using Test
using QuadGK: quadgk

@testset "MFS fused and selective assembly" begin
    AS = AcousticScattering
    # Independent scalar integration, with tighter targets and a larger budget.
    function reference(k, p, rho, z, m, derivative)
        scale = clamp(hypot(p.rhom-rho, p.zm-z)/sqrt(p.rhom*rho), 1e-12, 1e6)
        upper = asinh(pi/scale)
        breaks = collect(range(0, upper; length = max(4m+2, 40)))
        kernel = derivative ?
                 phi -> AS._ring_dGdn_field(k, p.rhom, p.zm, p.nrho, p.nz, rho, z, phi) :
                 phi -> AS._ring_G(k, p.rhom, p.zm, rho, z, phi)
        value, error = quadgk(
            t -> kernel(min(scale*sinh(t), pi))*cos(m*min(scale*sinh(t), pi))*scale*cosh(t),
            breaks...; rtol = 1e-11, atol = 1e-12, maxevals = 100000)
        @test error <= max(1e-12, 1e-11*abs(value))
        return 2value
    end
    for radius in (0.01, 1.0, 100.0), m in (0, 1, 8, 24)

        mesh = AS._axisymmetric_mesh(Sphere(radius), 12)
        p = AS.panels(mesh)[4]
        k = 2.0/radius
        workspace = AS._mfs_quadrature_workspace(m)
        for offset in (1e-6radius, 0.001radius, 0.2radius, -0.2radius)
            rho, z = p.rhom-offset*p.nrho, p.zm-offset*p.nz
            expected = (reference(k, p, rho, z, m, false), reference(k, p, rho, z, m, true))
            both = AS._mfs_pair_integrals(Val(:both), k, p, rho, z, m, 1e-6, workspace)
            pressure = AS._mfs_pair_integrals(
                Val(:pressure), k, p, rho, z, m, 1e-6, workspace)
            derivative = AS._mfs_pair_integrals(
                Val(:derivative), k, p, rho, z, m, 1e-6, workspace)
            @test pressure[2] === nothing
            @test derivative[1] === nothing
            for (got, want) in zip(both, expected)
                @test got≈want rtol=2e-6 atol=2e-10
            end
            @test pressure[1]≈expected[1] rtol=2e-6 atol=2e-10
            @test derivative[2]≈expected[2] rtol=2e-6 atol=2e-10
        end
    end
    # Axis source, nearly vanishing derivative and strongly cancelling high modes.
    mesh = AS._axisymmetric_mesh(Sphere(1.0), 12)
    # Oscillatory pressure cancellation exercises the pressure retry, complementing
    # the derivative retries at small offsets and large geometric scales above.
    p = AS.panels(mesh)[4]
    got = AS._mfs_pair_integrals(Val(:both), 100.0, p, 1.5, 0.0, 0, 1e-6,
        AS._mfs_quadrature_workspace(0))
    @test got[1]≈reference(100.0, p, 1.5, 0.0, 0, false) rtol=2e-6 atol=2e-10
    @test got[2]≈reference(100.0, p, 1.5, 0.0, 0, true) rtol=2e-6 atol=2e-10
    p = AS.panels(mesh)[6]
    @test_throws ArgumentError AS._mfs_pair_integrals(Val(:both), 2.0, p,
        p.rhom-0.01, p.zm-0.02, 8, 0.0, AS._mfs_quadrature_workspace(8); atol = 1e-30)
    for (rho, z) in ((0.0, 0.1), (p.rhom, p.zm+0.5)), m in (0, 12, 40)

        workspace=AS._mfs_quadrature_workspace(m)
        got=AS._mfs_pair_integrals(Val(:both), 0.1, p, rho, z, m, 1e-7, workspace)
        @test got[1]≈reference(0.1, p, rho, z, m, false) rtol=2e-7 atol=2e-10
        @test got[2]≈reference(0.1, p, rho, z, m, true) rtol=2e-7 atol=2e-10
    end
    for body in (Sphere(1.0), Spheroid(1.4, 0.6), Cylinder(0.5, 1.0)), m in (0, 3)

        mesh = AS._axisymmetric_mesh(body, 12)
        rho, z = AS.mfs_source_points(mesh, 0.15)
        both = AS.assemble_mfs_operators(mesh, 0.7, rho, z; m, threaded = false)
        parallel = AS.assemble_mfs_operators(mesh, 0.7, rho, z; m, threaded = true)
        @test both[1] == parallel[1]
        @test both[2] == parallel[2]
        for (which, index) in ((Val(:pressure), 1), (Val(:derivative), 2))
            selected=AS._assemble_mfs_operators(
                which, mesh, 0.7, rho, z; m, threaded = true)
            @test selected[index]≈both[index] rtol=2e-6 atol=2e-10
            @test selected[3 - index] === nothing
        end
        concurrent=fetch.([Threads.@spawn(AS.assemble_mfs_operators(mesh, 0.7, rho, z; m))
                           for _ in 1:3])
        @test all(result -> result[1] == both[1] && result[2] == both[2], concurrent)
        # Generic-Real fallback keeps the existing interface and storage type.
        generic=AS.assemble_mfs_operators(mesh, 1, rho, z; m, threaded = false)
        @test eltype(generic[1]) == ComplexF64
        @test generic[1][1, 1] == AS._azimuthal_G(1, AS.panels(mesh)[1].rhom,
            AS.panels(mesh)[1].zm, rho[1], z[1]; m)
    end
end
