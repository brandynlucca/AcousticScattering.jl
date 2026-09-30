using AcousticScattering
using Test

@testset "BEM cosine recurrence and padded modes" begin
    AS = AcousticScattering
    for width in (1, 4, 8), count in 1:width, m0 in (0, 17, 128), x in (0.0, 1e-12, 0.2, pi)
        actual = AS._cos_multiples(x, m0, count, Val(width))
        expected = [i <= count ? cos((m0 + i - 1) * x) : 0.0 for i in 1:width]
        @test actual ≈ expected atol = 3e-13
    end
end

@testset "BEM nested quadrature workspace agrees with fresh integrals" begin
    AS = AcousticScattering
    for scale in (1e-3, 1.0),
        body in (Sphere(scale), Spheroid(1.6scale, 0.7scale), Cylinder(scale, 3scale)),
        modes in (0:0, 1:1, 2:4, 8:12, 32:39)
        ps = AS.panels(AS._axisymmetric_mesh(body, 20))
        k, rtol = 2.0/scale, 1e-6
        width = length(modes) == 1 ? Val(1) : length(modes) <= 4 ? Val(4) : Val(8)
        rules = AS._mode_rules(k, ps, modes, width)
        workspace = AS._bem_quadrature_workspace(modes, width)
        @test workspace.azimuthal !== workspace.meridian
        # Reuse after unrelated panels, including self, adjacent, far, and near-axis pairs.
        for (i, j) in ((1, 1), (10, 10), (10, 11), (1, 2), (3, 18), (10, 10))
            args = (k, ps[i].rhom, ps[i].zm, ps[j], i == j, rtol, modes, rules, width)
            expected = AS._pair_KV_modes(args...)
            actual = AS._pair_KV_modes(args..., workspace)
            # Different initial grids need not agree to roundoff; this remains
            # much tighter than the requested 1e-6 quadrature tolerance.
            @test actual≈expected rtol=1e-10 atol=1e-11
        end
    end
end

@testset "BEM worker scratch is task-local" begin
    AS = AcousticScattering
    for n in (0, 1, 2, 31)
        hits = zeros(Int, n)
        correct_owner = zeros(Bool, n)
        isolated_buffer = zeros(Bool, n)
        owners = fill!(Vector{Union{Nothing, Task}}(undef, n), nothing)
        AS._foreach_row(n, () -> (; owner = current_task(), buffer = [0])) do i, workspace
            correct_owner[i] = workspace.owner === current_task()
            workspace.buffer[1] = i
            yield()
            isolated_buffer[i] = workspace.buffer[1] == i
            hits[i] += 1
            owners[i] = current_task()
        end
        @test all(==(1), hits)
        @test all(correct_owner)
        @test all(isolated_buffer)
        @test length(unique(owners)) <= min(n, Threads.nthreads())
    end
    mesh = AS.spheroid_mesh(1.6, 0.7, 12)
    cases = ((0.7, 0:0), (2.0, 8:12), (4.0, 32:39))
    expected = [AS.assemble_cbie_operators_modes(mesh, k, modes) for (k, modes) in cases]
    jobs = map(cases) do (k, modes)
        Threads.@spawn AS.assemble_cbie_operators_modes(mesh, k, modes)
    end
    for (job, reference) in zip(jobs, expected)
        actual = fetch(job)
        @test actual[1] == reference[1]
        @test actual[2] == reference[2]
    end
end
