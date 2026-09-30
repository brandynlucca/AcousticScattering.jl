using AcousticScattering
using LinearAlgebra
using Test

@testset "Precomputed BEM ring geometry and reciprocal distance" begin
    AS = AcousticScattering
    for scale in (1e-13, 1e-12, 1e-6, 1.0, 1e6, 1e150),
        coordinates in ((1.0, 0.99, 0.03, 0.7, 0.006),
            (1.0, 1.0-1e-10, 0.0, 1.0, 1e-10),
            (0.0, 0.5, 0.2, 0.4, -0.1),
            (1e-6, 2e-6, 0.1, 1.0, 0.03)),
        ka in (0.0, 0.1, 4.0, 40.0), phi in (0.0, 1e-15, 0.2, pi-1e-12, pi, 2pi)
        rho, rho2, z2, normal, projection = coordinates
        rho *= scale
        rho2 *= scale
        z2 *= scale
        projection *= scale
        k = ka/scale
        geometry = AS._ring_geometry(rho, 0.0, rho2, z2, normal, projection)
        half_sine = sin(phi/2)
        @test hypot(geometry.delta_rho, geometry.delta_z, geometry.chord*half_sine) ==
              hypot(rho-rho2, -z2, 2sqrt(rho*rho2)*half_sine)
        reference = AS._ring_KV(k, rho, 0.0, rho2, z2, normal, projection, phi)
        actual = AS._ring_KV_precomputed(k, geometry, half_sine)
        @test all(isfinite, actual)
        for i in 1:2
            @test actual[i]≈reference[i] rtol=5e-14 atol=0
        end
    end
end

@testset "Cached BEM sine tables and complete ring integration" begin
    AS = AcousticScattering
    ps = AS.panels(AS.spheroid_mesh(1.6, 0.7, 24))
    for modes in (0:0, 2:4, 8:12, 32:39), k in (0.7, 4.0, 40.0)

        width = length(modes)==1 ? Val(1) : length(modes)<=4 ? Val(4) : Val(8)
        rules = AS._mode_rules(k, ps, modes, width)
        for (angles, weights, table, half_sines) in values(rules)
            @test length(angles)==length(weights)==length(table)==length(half_sines)
            @test half_sines == sin.(angles ./ 2)
        end
        for (rho, rho2, z2) in ((1.0, 0.99, 0.03), (0.1, 0.2, 1.0))
            geometry = AS._ring_geometry(rho, 0.0, rho2, z2, 0.7, 0.03)
            old_node = phi -> AS._ring_KV_modes(k, rho, 0.0, rho2, z2, 0.7,
                0.03, phi, first(modes), length(modes), width)
            new_node = phi -> AS._ring_KV_modes(
                k, geometry, phi, first(modes), length(modes), width)
            for phi in (0.0, 1e-12, 0.3, pi)
                @test new_node(phi)≈old_node(phi) rtol=5e-14 atol=0
            end
            workspace = AS._bem_quadrature_workspace(modes, width)
            reference = AS._azimuthal_peak_quadrature(old_node, last(modes), rho, 0.0,
                rho2, z2, 1e-8, AS._QUAD_ATOL/2, workspace)
            actual = AS._azimuthal_peak_quadrature(new_node, last(modes), rho, 0.0,
                rho2, z2, 1e-8, AS._QUAD_ATOL/2, workspace)
            @test actual≈reference rtol=1e-10 atol=1e-11
        end
    end
end
