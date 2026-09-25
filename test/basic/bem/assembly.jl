using AcousticScattering
using Test

const AS = AcousticScattering

@testset "Axisymmetric mode-shared assembly" begin
    mesh = AS.spheroid_mesh(1.5, 1.0, 10)
    function generic(k, m)
        return invoke(AS.assemble_cbie_operators, Tuple{AS.MeridianMesh, Real}, mesh, k; m)
    end
    for (k, modes) in ((0.7, 0:0), (1.3, 0:2), (2.1, 3:7))
        Ks, Vs, ps = AS.assemble_cbie_operators_modes(mesh, k, modes)
        @test length(Ks) == length(Vs) == length(modes)
        @test length(ps) == 10
        for (index, m) in enumerate(modes)
            K, V, _ = generic(k, m)
            @test maximum(abs, Ks[index] .- K) < 1e-5 * maximum(abs, K)
            @test maximum(abs, Vs[index] .- V) < 1e-5 * maximum(abs, V)
        end
    end
    K, V, _ = AS.assemble_cbie_operators(mesh, 0.7; m = 2)
    Ks, Vs, _ = AS.assemble_cbie_operators_modes(mesh, 0.7, 2:2)
    @test K == Ks[1]
    @test V == Vs[1]
    @test_throws ArgumentError AS.assemble_cbie_operators_modes(mesh, 1.0, 0:8)
    @test_throws ArgumentError AS.assemble_cbie_operators_modes(mesh, 1.0, 1:0)
end

@testset "Fixed azimuthal rules requested from several threads" begin
    orders = 300:340
    rules = Vector{Any}(undef, length(orders))
    Threads.@threads for i in eachindex(orders)
        rules[i] = AS._azimuthal_fixed_rule(orders[i])
    end
    for (i, order) in enumerate(orders)
        @test rules[i] == AS._azimuthal_fixed_rule(order)
        @test sum(rules[i][2]) ≈ 2π
    end
end

@testset "Dynamic row scheduling visits every row once" begin
    for n in (0, 1, 2, 7, 200)
        hits = zeros(Int, n)
        AS._foreach_row(n) do i
            hits[i] += 1
        end
        @test all(==(1), hits)
    end
    @test_throws Exception AS._foreach_row(20) do i
        i == 5 && error("row failure")
    end
end

@testset "Distance-based panel node counts" begin
    panel_length = 0.1
    counts = [AS._meridian_node_count(r * panel_length, panel_length, 1.0)
              for r in (0.6, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0)]
    @test issorted(counts; rev = true)
    @test all(AS._MERIDIAN_MIN_ORDER .<= counts .<= AS._MERIDIAN_FIXED_ORDER)
    @test counts[1] == AS._MERIDIAN_FIXED_ORDER
    @test counts[end] == AS._MERIDIAN_MIN_ORDER
    @test AS._meridian_node_count(5.0, 0.1, 400.0) == AS._MERIDIAN_FIXED_ORDER
end

@testset "Far-pair panel quadrature against adaptive integration" begin
    k = 2.0
    ps = AS.panels(AS.sphere_mesh(1.0, 24))
    for m in (0, 3), (i, j) in ((1, 5), (3, 20), (10, 12), (12, 24), (7, 8), (2, 23))

        pj = ps[j]
        xρ, xz = ps[i].rhom, ps[i].zm
        AS._azimuthal_is_far(xρ, xz, pj) || continue
        projection = pj.nrho * (xρ - pj.rhom) + pj.nz * (xz - pj.zm)
        function reference(kernel)
            return AS.quadgk(0.0, 1.0; rtol = 1e-12) do s
                ρ2, z2 = AS._panel_point(pj, s)
                return kernel(ρ2, z2) * ρ2 * pj.L
            end[1]
        end
        single = reference((ρ2, z2) -> AS._azimuthal_G(
            k, xρ, xz, ρ2, z2; m, rtol = 1e-12, far = true))
        double = reference((ρ2, z2) -> AS._azimuthal_dGdn(
            k, xρ, xz, ρ2, z2, pj.nrho, pj.nz; m, rtol = 1e-12, far = true,
            meridian_projection = projection))
        @test AS._pair_V(k, xρ, xz, pj, false, 1e-6; m) ≈ single rtol = 1e-8
        @test AS._pair_K(k, xρ, xz, pj, false, 1e-6; m) ≈ double rtol = 1e-8
    end
end

@testset "Oblique solve across the mode chunk boundary" begin
    ka = 1.0
    reference = scattering_amplitude(modal(Sphere(1.0), Rigid(), ka))
    solution = bem(Sphere(1.0), Rigid(), ka;
        method = :axisymmetric, incidence_angle = pi / 3, n = 40, m_max = 9)
    amplitude = scattering_amplitude(solution)
    @test abs(target_strength(amplitude) - target_strength(reference)) < 0.1
    @test abs(amplitude - reference) / abs(reference) < 0.01
end
