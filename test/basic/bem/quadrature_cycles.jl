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
        # Resolve cancellation-sensitive references at higher precision.
        if error > max(1e-13, 1e-11norm(reference))
            reference, error = precise_reference(
                k, rho, rho2, z2, normal, projection, modes, width)
        end
        @test error <= max(1e-13, 1e-11norm(reference))
        workspace = AS._bem_quadrature_workspace(modes, width)
        actual = AS._azimuthal_peak_quadrature(integrand, last(modes), rho, 0.0,
            rho2, z2, 1e-7, AS._QUAD_ATOL/2, workspace)
        if last(modes)>1 && length(workspace.breaks)==last(modes)+1
            # Preserve the original capped result without implying convergence.
            previous = AS._azimuthal_peak_quadrature(integrand, last(modes), rho,
                0.0, rho2, z2, 1e-7, AS._QUAD_ATOL/2)
            @test actual == previous
        else
            @test norm(actual-reference) <= 5max(AS._QUAD_ATOL/2, 1e-7norm(reference))
        end
    end
end

@testset "BEM row priming preserves task ownership and row coverage" begin
    AS = AcousticScattering
    for compile_serially in (nothing, false, true), n in (0, 1, 2, 7, 31)
        options = compile_serially === nothing ? (;) : (; compile_serially)
        factories = Threads.Atomic{Int}(0)
        visits = Threads.Atomic{Int}(0)
        prime_finished = Threads.Atomic{Bool}(false)
        hits = zeros(Int, n)
        values = zeros(Int, n)
        sequence = zeros(Int, n)
        correct_owner = zeros(Bool, n)
        isolated_buffer = zeros(Bool, n)
        correct_order = fill(true, n)
        primed = compile_serially !== false && Threads.nthreads() > 1 && n > 1
        make_workspace = () -> begin
            Threads.atomic_add!(factories, 1)
            (; owner = current_task(), buffer = [0])
        end
        AS._foreach_row(n, make_workspace; options...) do i, workspace
            position = Threads.atomic_add!(visits, 1) + 1
            sequence[position] = i
            correct_owner[i] = workspace.owner === current_task()
            workspace.buffer[1] = i
            yield()
            isolated_buffer[i] = workspace.buffer[1] == i
            if primed
                if i == cld(n, 2)
                    prime_finished[] = true
                else
                    correct_order[i] = prime_finished[]
                end
            end
            hits[i] += 1
            values[i] = i^2
        end
        @test hits == ones(Int, n)
        @test values == [i^2 for i in 1:n]
        @test all(correct_owner) && all(isolated_buffer) && all(correct_order)
        @test factories[] == (n < 2 ? 1 : min(n, Threads.nthreads()))
        @test visits[] == n
        primed && @test first(sequence) == cld(n, 2)

        fill!(hits, 0)
        visits[] = 0
        AS._foreach_row(n; options...) do i
            position = Threads.atomic_add!(visits, 1) + 1
            sequence[position] = i
            hits[i] += 1
        end
        @test hits == ones(Int, n)
        primed && @test first(sequence) == cld(n, 2)
    end
    # Priming failures must reach the caller without waiting for unreleased work.
    @test_throws Exception AS._foreach_row(7, () -> nothing; compile_serially = true) do i, _
        i == 4 && error("priming row failure")
    end
    @test_throws Exception AS._foreach_row(
            (i, workspace) -> nothing, 7, () -> error("workspace failure");
            compile_serially = true)
    @test_throws Exception AS._foreach_row(7, () -> nothing; compile_serially = true) do i, _
        i == 1 && error("remaining row failure")
    end
end

@testset "Budget-limited high modes retain accuracy and avoid duplicate work" begin
    AS = AcousticScattering
    for m in (134, 191, 251, 259, 511)
        calls = Ref(0)
        integrand = phi -> begin
            calls[] += 1
            AS._ring_KV_modes(88.0, 1.0, 0.0, 0.96, 0.02, 0.8, 0.03,
                phi, m-7, 8, Val(8))
        end
        args = (m, 1.0, 0.0, 0.96, 0.02, 1e-6, AS._QUAD_ATOL/2)
        previous = AS._azimuthal_peak_quadrature(integrand, args...)
        previous_calls = calls[]
        calls[] = 0
        actual = AS._azimuthal_peak_quadrature(integrand, args...,
            AS._bem_quadrature_workspace(m-7:m, Val(8)))
        @test actual == previous
        @test calls[] == previous_calls == 15m
        scale = hypot(0.04, 0.02)/sqrt(0.96)
        points = collect(range(0.0, asinh(pi/scale); length = 4m+2))
        reference, error = AS.quadgk(points; rtol = 1e-11, atol = 1e-13,
                maxevals = 100000) do tau
            integrand(min(scale*sinh(tau), pi))*(scale*cosh(tau))
        end
        if error > max(1e-13, 1e-11norm(reference))
            reference, error = setprecision(BigFloat, 128) do
                k, r, r2, z, n, p = BigFloat.((88.0, 1.0, 0.96, 0.02, 0.8, 0.03))
                scale = hypot(r-r2, z)/sqrt(r*r2)
                points = collect(range(zero(scale), asinh(big(pi)/scale); length = 4m+2))
                AS.quadgk(points; rtol = big"1e-18", atol = big"1e-18",
                        maxevals = 100000) do tau
                    phi = min(scale*sinh(tau), big(pi))
                    kv = AS._ring_KV(k, r, zero(r), r2, z, n, p, phi)
                    SVector{16}(ntuple(i -> kv[i<=8 ? 1 : 2]*
                        cos(BigFloat(m-8+mod1(i, 8))*phi)*scale*cosh(tau), 16))
                end
            end
        end
        @test error <= max(1e-13, 1e-11norm(reference))
        @test norm(actual-reference) <= 5max(AS._QUAD_ATOL/2, 1e-6norm(reference))
    end
    # Preserve the capped half-cycle fallback when both estimates fail.
    calls = Ref(0)
    oscillatory = phi -> begin
        calls[] += 1
        SVector(cis(20000phi), cis(19999phi))
    end
    args = (251, 1.0, 0.0, 0.5, 1.0, 1e-12, 1e-14)
    previous = AS._azimuthal_peak_quadrature(oscillatory, args...)
    previous_calls = calls[]
    calls[] = 0
    actual = AS._azimuthal_peak_quadrature(oscillatory, args...,
        AS._bem_quadrature_workspace(251:251, Val(1)))
    @test actual == previous
    @test previous_calls < calls[] < 2previous_calls
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

@testset "Reusable initial intervals preserve QuadGK evaluations and error estimates" begin
    AS = AcousticScattering
    workspace = AS._bem_quadrature_workspace(0:251, Val(1))
    initial = workspace.azimuthal_initial
    calls = Float64[]
    integrand = tau -> begin
        push!(calls, tau)
        SVector(complex(exp(-4000(tau-0.37)^2)), cis(71tau))
    end
    # Compare reused segments against independently constructed initial grids.
    for intervals in (1, 126, 251, 2, 39, 1)
        points = collect(range(0.0, 1.0; length = intervals+1))
        empty!(calls)
        expected = AS.quadgk(integrand, points; rtol = 1e-10, atol = 1e-12,
            maxevals = AS._AZIMUTHAL_MAXEVALS)
        expected_calls = copy(calls)
        empty!(calls)
        actual = AS._azimuthal_quadgk(integrand, points, 1e-10, 1e-12, workspace)
        @test actual == expected
        @test calls == expected_calls
        @test workspace.azimuthal_initial === initial
        @test length(initial) == intervals
        @test [(segment.a, segment.b) for segment in initial] ==
              collect(zip(points[1:end-1], points[2:end]))
        @test all(segment -> segment.I===nothing && segment.E===nothing, initial)
        intervals==1 && @test length(workspace.azimuthal)>length(initial)
    end
    # Restart from the new grid after localized adaptive subdivision.
    points = [0.0, 0.25, 1.0]
    simple = tau -> SVector(1.0+0im, tau+0im)
    expected = AS.quadgk(simple, points; rtol = 1e-10, atol = 1e-12,
        maxevals = AS._AZIMUTHAL_MAXEVALS)
    @test AS._azimuthal_quadgk(simple, points, 1e-10, 1e-12, workspace) == expected
    @test length(workspace.azimuthal) == length(initial) == 2
    @test !Base.mightalias(initial, workspace.azimuthal)

    workspaces = [AS._bem_quadrature_workspace(0:251, Val(1)) for _ in 1:4]
    points = collect(range(0.0, 1.0; length = 252))
    functions = [tau -> SVector(cis((31+j)*tau), complex(tau^2)) for j in 1:4]
    expected = [AS.quadgk(f, points; rtol = 1e-10, atol = 1e-12,
                maxevals = AS._AZIMUTHAL_MAXEVALS) for f in functions]
    tasks = [Threads.@spawn AS._azimuthal_quadgk(
                 functions[j], points, 1e-10, 1e-12, workspaces[j]) for j in 1:4]
    @test fetch.(tasks) == expected
    @test all(!Base.mightalias(workspaces[i].azimuthal_initial,
                  workspaces[j].azimuthal_initial) for i in 1:4 for j in 1:4 if i!=j)
end
