using AcousticScattering
using LinearAlgebra
using StaticArrays
using Test

# Routing faults remain separate from the independent numerical fixtures.
mutable struct _RingRecurrenceRouteProbe{W}
    inner::W
    calls::Vector{Symbol}
    stops::Vector{Int}
    direct_states::Vector{Tuple{Int,Float64}}
    direct_reason::Int
    decay_reason::Int
end

function _RingRecurrenceRouteProbe(inner; direct_reason=0, decay_reason=0)
    _RingRecurrenceRouteProbe(inner, Symbol[], Int[], Tuple{Int,Float64}[],
        direct_reason, decay_reason)
end

Base.getproperty(probe::_RingRecurrenceRouteProbe, name::Symbol) =
    hasfield(typeof(probe), name) ? getfield(probe, name) :
    getproperty(getfield(probe, :inner), name)
Base.setproperty!(probe::_RingRecurrenceRouteProbe, name::Symbol, value) =
    hasfield(typeof(probe), name) ? setfield!(probe, name, value) :
    setproperty!(getfield(probe, :inner), name, value)

function AcousticScattering._ring_recurrence_seed!(output, geometry, mode,
        refined, probe::_RingRecurrenceRouteProbe)
    AcousticScattering._ring_recurrence_seed!(
        output, geometry, mode, refined, probe.inner)
end

function AcousticScattering._ring_recurrence_direct!(
        probe::_RingRecurrenceRouteProbe, geometry, rtol, mode)::Int
    push!(probe.calls, :direct)
    push!(probe.direct_states, (probe.extensions, probe.last_residual))
    fill!(probe.answer, 17.0+0im)
    probe.direct_reason
end

function AcousticScattering._ring_recurrence_checked_decay!(
        probe::_RingRecurrenceRouteProbe, geometry, rtol, stop)::Int
    push!(probe.calls, :decay)
    push!(probe.stops, stop)
    probe.extensions = 2
    probe.last_residual = 0.125
    fill!(probe.answer, 29.0+0im)
    probe.decay_reason
end

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
    for (lambda, m0) in ((88.0, 192), (88.0, 240), (88.0, 352), (88.0, 496), (160.0, 128))
        rho = 0.07/1.01
        d = 1e-6
        modes = m0:(m0+15)
        k = lambda/rho
        geometry = (rho, rho, d*rho, 0.7, -d^2*rho/2)
        expected = reference(k, geometry..., modes, Val(16))
        scratch = AS._ring_convolution_workspace(modes, lambda)
        accepted_case, actual = AS._ring_convolution_KV(k, rho, 0.0, rho,
            d*rho, 0.7, -d^2*rho/2, modes, 1e-6, Val(16), scratch)
        @test accepted_case
        for component in (1:16, 17:32)
            @test norm(actual[component]-expected[component]) <=
                  0.05max(AS._QUAD_ATOL, 1e-6norm(expected[component]))
        end
    end
end

@testset "Convolution ring guards and quadrature rejection" begin
    AS = AcousticScattering
    for (modes, ka) in ((0:0, 8.0), (0:15, 0.0), (0:15, 161.0), (498:513, 40.0))
        @test AS._ring_convolution_workspace(modes, ka) === nothing
    end
    modes, width = 0:15, Val(16)
    workspace = AS._bem_quadrature_workspace(modes, width; maxlambda = 40.0)
    args = (40.0, 1.0, 0.0, 1.0, 0.001, 0.7, 0.0, modes)
    @test first(AS._ring_convolution_KV(args..., 1e-6, width, workspace.convolution))
    @test !first(AS._ring_convolution_KV(args..., 1e-12, width, workspace.convolution))
    workspace.convolution.coarse.half_sines .*= 0.9
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
    # Compare tighter direct boundary and CHIEF operators.
    accelerated = AS._spheroid_curved_operators(boundary, 1200.0, 0:7, Val(8), Val(5);
        rtol = 1e-6, split = true, chief_points = 2)
    direct = AS._spheroid_curved_operators(boundary, 1200.0, 0:7, Val(8), Val(5);
        rtol = 1e-10, split = true, chief_points = 2)
    for (left, right) in zip(accelerated, direct), (a, b) in zip(left, right)

        @test norm(a-b) <= 2e-7norm(b)
    end
end

@testset "Extended-frequency curved operators against direct quadrature" begin
    AS = AcousticScattering
    boundary = AS._spheroid_curved_boundary(0.07, 0.07/1.01, 4, 6; half = true)
    k = 88/boundary.b
    for modes in (0:7, 192:199)
        accelerated = AS._spheroid_curved_operators(boundary, k, modes, Val(8), Val(6);
            rtol = 1e-6, split = true, chief_points = 2)
        direct = AS._spheroid_curved_operators(boundary, k, modes, Val(8), Val(6);
            rtol = 1e-9, split = true, chief_points = 2, quadrature_extra = 4)
        for (left, right) in zip(accelerated, direct), (a, b) in zip(left, right)
            @test norm(a-b) <= 2e-7norm(b)
        end
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

@testset "Convolution DCT matches endpoint-weighted cosine sums" begin
    AS = AcousticScattering
    # Rounding can otherwise collapse both independent quadrature grids.
    for (modes, lambda) in ((208:223, 88.14), (496:511, 160.0), (497:512, 160.0))
        scratch = AS._ring_convolution_workspace(modes, lambda)
        @test length(scratch.fine.half_sines)>length(scratch.coarse.half_sines)
        @test length(scratch.coarse.half_sines)-1 >=
              max(2scratch.coarse.order, scratch.coarse.order+last(modes)+8)
    end
    for (order, modes) in ((33, 0:7), (112, 24:35), (112, 240:251), (184, 496:511))
        rule = AS._ring_convolution_rule(order, modes)
        points = size(rule.samples, 1)-1
        # Include endpoints, low modes, and the Nyquist component independently.
        for c in 1:4, j in 0:points
            rule.samples[j+1, c] = cospi(c*j/points) + 0.3cospi(j) +
                                   sin(0.17j*c) + (j==0 ? 0.7 : 0.0)
        end
        samples = copy(rule.samples)
        weights = fill(inv(Float64(points)), points+1)
        weights[[1, end]] ./= 2
        dense(indices, columns) = [sum(weights[j+1]*cospi(m*j/points)*samples[j+1, c]
            for j in 0:points) for m in indices, c in columns]
        coefficients = dense(0:order, 1:2)
        imaginary = dense(modes, 3:4)
        AS._ring_convolution_transform!(rule, modes)
        @test rule.coefficients ≈ coefficients rtol=1e-12 atol=5e-14
        @test rule.imaginary ≈ imaginary rtol=1e-12 atol=5e-14
    end
end

@testset "Convolution plans preserve repeated and parallel ring evaluation" begin
    AS = AcousticScattering
    rho, d = 0.07/1.01, 1e-6
    for modes in (0:15, 240:251, 497:512)
        scratch = AS._ring_convolution_workspace(modes, 88.0)
        evaluate(k, work) = AS._ring_convolution_KV(k, rho, 0.0, rho, d*rho,
            0.7, -d^2*rho/2, modes, 1e-6, Val(16), work)
        accepted, positive = evaluate(88/rho, scratch)
        @test accepted
        for sign in (-1, 1, -1, 1)
            accepted, actual = evaluate(sign*88/rho, scratch)
            @test accepted
            @test actual ≈ (sign==1 ? positive : conj(positive)) rtol=2e-13 atol=1e-14
        end
        tasks = [Threads.@spawn evaluate(88/rho,
            AS._ring_convolution_workspace(modes, 88.0)) for _ in 1:4]
        for task in tasks
            accepted, actual = fetch(task)
            @test accepted
            @test actual == positive
        end
    end
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


@testset "Contour recurrence kernels and independent numerical checks" begin
    AS = AcousticScattering
    function recurrence_reference(k, rho, rho2, z2, normal, projection, modes, ::Val{N}) where {N}
        setprecision(128) do
            kb, r, r2, z, n, p = BigFloat.((k, rho, rho2, z2, normal, projection))
            indices = BigFloat.(modes)
            scale = hypot(r-r2, z)/sqrt(r*r2)
            points = collect(range(zero(scale), asinh(big(pi)/scale);
                length = max(40, 2maximum(modes)+2)))
            value, error = AS.quadgk(points;
                rtol = big"1e-14", atol = big"1e-20", maxevals = 400000) do tau
                phi = scale*sinh(tau)
                kv = AS._ring_KV(kb, r, zero(r), r2, z, n, p, phi)
                SVector{2N}(ntuple(2N) do i
                    c = mod1(i, N)
                    c<=length(indices) ?
                    kv[i<=N ? 1 : 2]*cos(indices[c]*phi)*scale*cosh(tau) :
                    zero(Complex{BigFloat})
                end)
            end
            @test error <= max(big"1e-20", big"1e-14"*norm(value))
            ComplexF64.(2value)
        end
    end

    @testset "Singleton convolution and padded short spans" begin
        rho = 0.1/1.001
        lambda = (2pi*285000/1477.3)*rho
        cases = ((0, 1, 1, 12.0, 0.001), (33, 1, 1, lambda, 1e-6),
            (50, 1, 1, lambda, 1e-6), (512, 1, 1, 88.0, 1e-6),
            (50, 1, 4, lambda, 1e-6), (33, 2, 4, lambda, 1e-6),
            (48, 3, 4, lambda, 1e-6), (33, 4, 4, lambda, 1e-6),
            (48, 5, 8, lambda, 1e-6))
        for (m0, count, N, acoustic_size, d) in cases
            modes, k = m0:(m0+count-1), acoustic_size/rho
            workspace = AS._ring_convolution_workspace(modes, acoustic_size)
            @test workspace !== nothing
            args = (k, rho, 0.0, rho, d*rho, 0.7, -d^2*rho/2,
                modes, 1e-6, Val(N), workspace)
            accepted, value = AS._ring_convolution_KV(args...)
            @test accepted
            reference = recurrence_reference(k, rho, rho, d*rho, 0.7,
                -d^2*rho/2, modes, Val(N))
            for indices in (1:count, (N+1):(N+count))
                @test norm(value[indices]-reference[indices]) <=
                      0.05max(AS._QUAD_ATOL, 1e-6norm(reference[indices]))
            end
            @test all(iszero, value[(count+1):N])
            @test all(iszero, value[(N+count+1):(2N)])
            if m0==33 && count==1
                negative = AS._ring_convolution_KV(-k, args[2:end]...)
                @test first(negative) && last(negative)==conj(value)
                @test (@inferred AS._ring_convolution_KV(args...)) == (accepted, value)
            end
        end
        # Adjacent separations exercise the two moment propagation branches.
        modes, acoustic_size = 33:33, 12.0
        workspace = AS._ring_convolution_workspace(modes, acoustic_size)
        threshold = 2/(length(workspace.q)-1)
        for factor in (1-1e-6, 1+1e-6)
            d, k = threshold*factor, acoustic_size/rho
            @test ((length(workspace.q)-1)*d<=2) == (factor<1)
            accepted, value = AS._ring_convolution_KV(k, rho, 0.0, rho,
                d*rho, 0.0, 0.0, modes, 1e-6, Val(1), workspace)
            @test accepted && iszero(value[1])
            reference = recurrence_reference(k, rho, rho, d*rho,
                0.0, 0.0, modes, Val(1))
            @test norm(value-reference) <= 0.05max(AS._QUAD_ATOL, 1e-6norm(reference))
        end
        modes, width, k = 33:33, Val(1), lambda/rho
        workspace = AS._bem_quadrature_workspace(modes, width; maxlambda = lambda)
        args = (k, rho, 0.0, rho, 1e-6rho, 0.7, -1e-12rho/2, modes)
        @test !first(AS._ring_convolution_KV(args..., 1e-12, width, workspace.convolution))
        workspace.convolution.coarse.half_sines .*= 0.9
        @test !first(AS._ring_convolution_KV(args..., 1e-6, width, workspace.convolution))
        geometry = AS._ring_geometry(args[2:7]...)
        integrand = phi -> AS._ring_KV_modes(k, geometry, phi, 33, 1, width)
        expected = 2AS._azimuthal_peak_quadrature(integrand, 33, rho, 0.0,
            rho, 1e-6rho, 1e-6, AS._QUAD_ATOL/2, workspace)
        actual = AS._azimuthal_KV_modes(args..., false, nothing, 1e-6, width, workspace)
        @test isequal(actual, expected)
        @test AS._ring_convolution_workspace(1:0, lambda) === nothing
    end

    @testset "Bounded decay rescue retains independent accuracy" begin
        rho = 0.1/1.001
        for (m0, count, N, d) in ((256, 64, 64, 0.01), (256, 64, 64, 0.03),
                (192, 128, 128, 0.01), (320, 14, 16, 0.0095),
                (320, 14, 16, 0.03), (320, 22, 32, 0.01))
            modes, k = m0:(m0+count-1), 121.1/rho
            workspace = AS._ring_recurrence_workspace(modes, Val(N))
            args = (k, rho, 0.0, rho, d*rho, 0.7, -d^2*rho/2)
            geometry = AS._ring_recurrence_geometry(args...)
            @test AS._ring_recurrence_seed!(workspace.low, geometry, 1, true, workspace)
            @test AS._ring_recurrence_seed!(workspace.lowcheck, geometry, 1, false, workspace)
            @test AS._ring_recurrence_seed_check(geometry, workspace.low, workspace.lowcheck)
            low_evaluations = workspace.seed_evaluations
            @test AS._ring_recurrence_direct!(workspace, geometry, 1e-6, last(modes)) in (4, 5, 7)
            accepted, value = AS._ring_recurrence_KV(args..., modes, 1e-6, Val(N), workspace)
            @test accepted
            @test workspace.extensions >= 2
            # Preferred decay skips high seeds while retaining independent checks.
            @test workspace.seed_evaluations == low_evaluations
            @test all(iszero, value[count+1:N])
            @test all(iszero, value[N+count+1:2N])
            selected = unique(round.(Int, range(first(modes), last(modes); length = 8)))
            reference = recurrence_reference(k, rho, rho, d*rho, 0.7,
                -d^2*rho/2, selected, Val(8))
            indices = selected .- first(modes) .+ 1
            for (actual_indices, reference_indices) in ((indices, 1:8), (indices .+ N, 9:16))
                @test norm(value[actual_indices]-reference[reference_indices]) <=
                      0.05max(AS._QUAD_ATOL, 1e-6norm(reference[reference_indices]))
            end
            negative = AS._ring_recurrence_KV(-k, args[2:end]..., modes, 1e-6, Val(N), workspace)
            @test first(negative) && last(negative)==conj(value)
            @test AS._ring_recurrence_KV(args..., modes, 1e-6, Val(N), workspace) ==
                  (accepted, value)
        end
        # A successful direct solve retains the original operation sequence.
        modes, k, d = 128:135, 88.0/rho, 0.01
        workspace = AS._ring_recurrence_workspace(modes, Val(8))
        args = (k, rho, 0.0, rho, d*rho, 0.7, -d^2*rho/2)
        geometry = AS._ring_recurrence_geometry(args...)
        @test AS._ring_recurrence_seed!(workspace.low, geometry, 1, true, workspace)
        @test AS._ring_recurrence_seed!(workspace.lowcheck, geometry, 1, false, workspace)
        @test AS._ring_recurrence_seed_check(geometry, workspace.low, workspace.lowcheck)
        @test AS._ring_recurrence_direct!(workspace, geometry, 1e-6, last(modes)) == 0
        direct = SVector{16, ComplexF64}(workspace.answer)
        accepted, value = AS._ring_recurrence_KV(args..., modes, 1e-6, Val(8), workspace)
        @test accepted && isequal(value, direct)
        @test workspace.extensions == 0

        for (m0, N, d) in ((256, 64, 0.003), (176, 16, 0.03), (320, 32, 0.001))
            modes, k = m0:(m0+N-1), 121.1/rho
            workspace = AS._ring_recurrence_workspace(modes, Val(N))
            accepted, value = AS._ring_recurrence_KV(k, rho, 0.0, rho,
                d*rho, 0.7, -d^2*rho/2, modes, 1e-6, Val(N), workspace)
            @test !accepted && iszero(value)
            @test workspace.last_reason == :seed
            @test workspace.extensions == 0
        end
        # Integer ceilings require a tighter admission bound than 16capacity/25.
        for (modes, width, capacity, d) in ((320:325, 16, 8192, 0.01),
                (320:333, 16, 8192, 0.009), (320:333, 16, 1024, 0.0095))
            workspace = AS._ring_recurrence_workspace(modes, Val(width); capacity)
            accepted, value = AS._ring_recurrence_KV(121.1/rho, rho, 0.0, rho,
                d*rho, 0.7, -d^2*rho/2, modes, 1e-6, Val(width), workspace)
            @test !accepted && iszero(value)
            @test workspace.extensions == 0
        end
        for capacity in (166, 8192)
            limit = fld(4fld(4capacity, 5), 5)
            @test cld(5cld(5limit, 4), 4) <= capacity
            @test cld(5cld(5(limit+1), 4), 4) > capacity
            for width in (8, 64, 128)
                admitted = min(64width, limit)
                @test cld(5cld(5admitted, 4), 4) <= capacity
            end
        end
    end

    @testset "Preferred decay routing and exact work boundaries" begin
        rho, k = 0.1/1.001, 121.1/(0.1/1.001)
        geometry(d) = AS._ring_recurrence_geometry(
            k, rho, 0.0, rho, d*rho, 0.7, -d^2*rho/2)
        function remote_stop(g)
            transition = g.kappa*g.alpha /
                sqrt(2*(1+sqrt((1-g.alpha)*(1+g.alpha))))
            transition + 36/(2asinh(g.bm/sqrt(2)))
        end
        modes = 320:333
        for (decay_reason, direct_reason, expected_reason, expected_calls) in (
                (0, 0, 0, [:decay]),
                (5, 0, 0, [:decay, :direct]),
                (6, 4, 6, [:decay, :direct]),
                (7, 5, 7, [:decay, :direct]),
                (8, 7, 8, [:decay, :direct]),
                (5, 8, 8, [:decay, :direct]))
            probe = _RingRecurrenceRouteProbe(
                AS._ring_recurrence_workspace(modes, Val(16));
                direct_reason, decay_reason)
            @test AS._ring_recurrence_candidate!(probe, geometry(0.03), 1e-6) ==
                  expected_reason
            @test probe.calls == expected_calls
            @test length(probe.stops) == 1
            @test all(==((0, 0.0)), probe.direct_states)
            if expected_reason==0
                expected = decay_reason==0 ? 29.0+0im : 17.0+0im
                @test all(==(expected), probe.answer)
            end
        end

        # Supported direct spans remain outside the preferred-decay scope.
        for (span, width, d) in ((128:135, 8, 0.1),
                (320:326, 16, 0.045), (120:135, 16, 0.1))
            probe = _RingRecurrenceRouteProbe(
                AS._ring_recurrence_workspace(span, Val(width)))
            @test AS._ring_recurrence_candidate!(probe, geometry(d), 1e-6) == 0
            @test probe.calls == [:direct]
            @test isempty(probe.stops)
        end

        # Locate adjacent Float64 separations on opposite sides of each limit.
        for (span, width, capacity, limit, lower, upper) in (
                (320:333, 16, 8192, 3996, 0.009, 0.0095),
                (256:319, 64, 1500, 960, 0.04, 0.05),
                (192:319, 128, 8192, 3828, 0.009, 0.011))
            @test remote_stop(geometry(lower)) > limit
            @test remote_stop(geometry(upper)) <= limit
            while nextfloat(lower)<upper
                middle = lower+(upper-lower)/2
                if remote_stop(geometry(middle))<=limit
                    upper = middle
                else
                    lower = middle
                end
            end
            @test nextfloat(lower) == upper
            for (d, expected_calls) in ((lower, [:direct]), (upper, [:decay]))
                probe = _RingRecurrenceRouteProbe(
                    AS._ring_recurrence_workspace(span, Val(width); capacity))
                @test AS._ring_recurrence_candidate!(probe, geometry(d), 1e-6) == 0
                @test probe.calls == expected_calls
                if expected_calls==[:decay]
                    @test only(probe.stops) == limit
                    @test cld(5cld(5only(probe.stops),4),4) <= capacity
                else
                    @test isempty(probe.stops)
                end
            end
        end

        # Invalid low seeds cannot reach either numerical route.
        probe = _RingRecurrenceRouteProbe(
            AS._ring_recurrence_workspace(modes, Val(16)))
        probe.lowellipsefine = ([NaN], [1.0])
        @test AS._ring_recurrence_candidate!(probe, geometry(0.03), 1e-6) == 4
        @test isempty(probe.calls)
    end

    @testset "Supported generalized quadrature tables" begin
        counts = (67, 76, 78, 80, 80, 80, 80, 80, 79, 79)
        @test length(AS._RING_RECURRENCE_TABLES) == 10
        for (index, (nodes, weights)) in enumerate(AS._RING_RECURRENCE_TABLES)
            @test length(nodes) == length(weights) == counts[index]
            @test all(isfinite, nodes) && all(isfinite, weights)
            @test all(t -> 0<t<1, nodes) && issorted(nodes)
            @test abs(sum(weights)-1) < 5e-13
            @test AS._ring_recurrence_table_index(ldexp(1.0, -5index+2), 1.0) == index
        end
        @test AS._ring_recurrence_table_index(2.0, 1.0) == 0
        @test AS._ring_recurrence_table_index(ldexp(1.0, -50), 1.0) == 10
        @test AS._ring_recurrence_table_index(prevfloat(ldexp(1.0, -50)), 1.0) == -1
        @test AS._ring_recurrence_table_index(0.0, 1.0) == -1
        @test AS._ring_recurrence_table_index(NaN, 1.0) == -1
        # Sinh coordinates and beta^j scaling resolve the independent reference.
        setprecision(128) do
            for index in 1:10, (power, derivative) in ((0, 0), (0, 4), (7, 4), (63, 0))
                beta = BigFloat(ldexp(1.0, -5index+2))
                root = sqrt(beta)
                endpoints = collect(range(zero(beta), asinh(inv(root)); length = 13))
                reference, error = AS.quadgk(endpoints;
                    rtol = big"1e-23", atol = zero(beta), maxevals = 20000) do u
                    v = sinh(u)
                    (root*v)^power*cosh(u)/((v*v-im)^derivative*sqrt(v*v-2im))
                end
                @test error <= big"1e-23"*abs(reference)
                nodes, weights = AS._RING_RECURRENCE_TABLES[index]
                actual = sum(BigFloat(w)*BigFloat(t)^power*
                             (beta/(BigFloat(t)^2-im*beta))^derivative/
                             sqrt(BigFloat(t)^2-2im*beta) for (t, w) in zip(nodes, weights))
                @test abs(actual-reference) <= big"2e-11"*abs(reference)
            end
        end
    end

    @testset "Native pivoted band LU against an independently assembled dense system" begin
        workspace = AS._ring_recurrence_workspace(0:15, Val(16); capacity = 128)
        geometry = (; alpha = 0.73, kappa = 4.25)
        # Reconstruct polynomial coefficients independently from the normalized band.
        for N in (12, 80, 12, 80)
            matrix = zeros(Float64, N-3, N+1)
            phase = geometry.alpha^2*geometry.kappa^2
            for m in 2:N-2
                coefficients = (phase*(m+1), -4geometry.alpha*(2m-1)*(m^2-1),
                    16m*(m^2-1)-2m*phase, -4geometry.alpha*(2m+1)*(m^2-1), phase*(m-1))
                for offset in -2:2
                    matrix[m-1, m+offset+1] = coefficients[offset+3]/(16m*(m^2-1))
                end
            end
            known = ComplexF64[complex(sin(0.31j), cos(0.19j)) for j in 1:N+1]
            forcing = zeros(ComplexF64, workspace.capacity+2)
            forcing[3:N-1] = matrix*known
            boundaries = (known[1], known[2], known[N], known[N+1])
            rhs = forcing[3:N-1]-matrix[:, [1, 2, N, N+1]]*collect(boundaries)
            dense = matrix[:, 3:N-1]\rhs
            @test AS._ring_recurrence_factor!(workspace, geometry, N)
            @test AS._ring_recurrence_solve!(workspace.G, workspace, boundaries, forcing)
            actual = copy(workspace.G[3:N-1])
            @test norm(actual-dense) <= 2e-11norm(dense)
            residual = norm(matrix*workspace.G[1:N+1]-forcing[3:N-1])
            scale = norm(matrix)*norm(workspace.G[1:N+1])+norm(forcing[3:N-1])
            @test residual <= 2e-14scale
            @test workspace.G[[1, 2, N, N+1]] == collect(boundaries)
            @test workspace.G[N+2] == 0
            workspace.G[4] += 1e-3
            @test !AS._ring_recurrence_band_residual(workspace, workspace.G, forcing)
            @test AS._ring_recurrence_solve!(workspace.G, workspace, boundaries, forcing)
            @test workspace.G[3:N-1] == actual
        end
        @test !AS._ring_recurrence_factor!(workspace, geometry, 4)
        @test !AS._ring_recurrence_factor!(workspace, geometry, workspace.capacity+1)
    end

    @testset "Pivoted band LU preserves active storage and rejects singular bands" begin
        capacity = 96
        band = Matrix{Float64}(undef, 7, capacity)
        pivots = Vector{Int}(undef, capacity)
        rhs = Matrix{Float64}(undef, capacity+3, 2)
        for n in (9, 77, 9, 77), force_pivots in (false, true)
            matrix = zeros(Float64, n, n)
            for j in 1:n, i in max(1, j-2):min(n, j+2)
                matrix[i, j] = i==j ? (force_pivots && isodd(i) ? 0.001 : 4.0) :
                               i==j+1 ? 1.2+0.02sin(0.31j) :
                               i==j-1 ? 1.1+0.02cos(0.23j) : 0.015sin(0.17i+0.11j)
            end
            known = [sin(0.31i+0.23j)+cos(0.13i-0.17j) for i in 1:n, j in 1:2]
            original_rhs = matrix*known
            expected = matrix\original_rhs
            fill!(band, -13.25)
            fill!(pivots, -77)
            fill!(rhs, NaN)
            band[:, 1:n] .= 0.0
            for j in 1:n, i in max(1, j-2):min(n, j+2)
                band[5+i-j, j] = matrix[i, j]
            end
            rhs[1:n, :] .= original_rhs
            @test AS._ring_recurrence_lu_factor!(band, pivots, n)
            @test all(j -> j<=pivots[j]<=min(j+2, n), 1:n)
            force_pivots && (@test any(j -> pivots[j]!=j, 1:n))
            @test all(==(-13.25), @view(band[:, n+1:end]))
            @test all(==(-77), @view(pivots[n+1:end]))
            @test AS._ring_recurrence_lu_solve!(band, pivots, rhs, n)
            actual = copy(rhs[1:n, :])
            @test norm(actual-expected) <= 2e-12norm(expected)
            @test norm(actual-known) <= 2e-12norm(known)
            residual = norm(matrix*actual-original_rhs)
            scale = norm(matrix)*norm(actual)+norm(original_rhs)
            @test residual <= 2e-14scale
            @test all(isnan, @view(rhs[n+1:end, :]))
            rhs[1:n, :] .= original_rhs
            @test AS._ring_recurrence_lu_solve!(band, pivots, rhs, n)
            @test rhs[1:n, :] == actual
            @test all(isnan, @view(rhs[n+1:end, :]))
        end
        fill!(band, 0.0)
        @test !AS._ring_recurrence_lu_factor!(band, pivots, 9)
        band[5, 1] = NaN
        @test !AS._ring_recurrence_lu_factor!(band, pivots, 9)
    end

    @testset "Incremental recurrence bands preserve complete factorization" begin
        extended = AS._ring_recurrence_workspace(0:15, Val(16); capacity = 256)
        complete = AS._ring_recurrence_workspace(0:15, Val(16); capacity = 256)
        boundaries = (1.0+0.2im, -0.3+0.7im, 0.02-0.01im, -0.01+0.03im)
        pivot_count = 0
        for geometry in ((; alpha = 0.73, kappa = 4.25),
                (; alpha = 0.999999, kappa = 171.0),
                (; alpha = 0.1, kappa = 300.0), (; alpha = 0.995, kappa = 25.0))
            for sizes in ((5, 6, 7, 8, 9, 10, 12, 16, 21, 27, 34, 43, 54, 68, 85, 107),
                    (12, 15, 19, 24, 80, 81, 200))
                @test AS._ring_recurrence_factor!(extended, geometry, first(sizes))
                for (index, N) in enumerate(sizes)
                    index==1 || (@test AS._ring_recurrence_extend!(extended, geometry, N))
                    @test AS._ring_recurrence_factor!(complete, geometry, N)
                    n = N-3
                    @test extended.band[:, 1:n] == complete.band[:, 1:n]
                    @test extended.pivots[1:n] == complete.pivots[1:n]
                    @test extended.coefficients[:, 1:n] == complete.coefficients[:, 1:n]
                    @test extended.scales[1:n] == complete.scales[1:n]
                    pivot_count += count(j -> extended.pivots[j]!=j, 1:n)
                    accepted = AS._ring_recurrence_solve!(extended.G, extended, boundaries)
                    @test AS._ring_recurrence_solve!(complete.G, complete, boundaries) == accepted
                    @test extended.G[1:N+2] == complete.G[1:N+2]
                    @test extended.last_residual == complete.last_residual
                    forcing = ComplexF64[complex(sin(0.31j), cos(0.19j))
                                         for j in 1:extended.capacity+2]
                    accepted = AS._ring_recurrence_solve!(extended.G, extended, boundaries, forcing)
                    @test AS._ring_recurrence_solve!(complete.G, complete, boundaries, forcing) == accepted
                    @test extended.G[1:N+2] == complete.G[1:N+2]
                end
            end
        end
        @test pivot_count > 0
        @test !AS._ring_recurrence_extend!(extended, (; alpha = 0.73, kappa = 4.25), extended.N)
        @test !AS._ring_recurrence_extend!(extended, (; alpha = 0.73, kappa = 4.25), 257)
        extended.checkpoint.n = 0
        @test !AS._ring_recurrence_extend!(extended, (; alpha = 0.73, kappa = 4.25), 250)
        geometry = (; alpha = 0.73, kappa = 4.25)
        @test AS._ring_recurrence_factor!(extended, geometry, 12)
        extended.checkpoint.band[5, 1] = NaN
        @test !AS._ring_recurrence_extend!(extended, geometry, 16)
        @test extended.checkpoint.n == 0
        @test !AS._ring_recurrence_extend!(extended, geometry, 20)
        @test AS._ring_recurrence_factor!(extended, geometry, 12)
        @test AS._ring_recurrence_extend!(extended, geometry, 16)
    end

    @testset "Unequal radii and padded mode spans" begin
        cases = ((; lambda = 88.0, m0 = 128, ratio = 0.99, dz = 0.01, normal = -0.7, projection = 0.006),
            (; lambda = -88.0, m0 = 128, ratio = 0.99, dz = 0.01, normal = 0.4, projection = -0.009),
            (; lambda = 121.1, m0 = 176, ratio = 0.999, dz = 0.01, normal = 0.0, projection = 0.002),
            (; lambda = 121.1, m0 = 176, ratio = 0.999, dz = 0.01, normal = -0.3, projection = -0.004))
        for spec in cases
            rho = 0.1/1.001
            rho2, z2 = spec.ratio*rho, spec.dz*rho
            k, projection = spec.lambda/rho, spec.projection*rho
            modes = spec.m0:(spec.m0+5)
            workspace = AS._ring_recurrence_workspace(modes, Val(8))
            arguments = (k, rho, 0.0, rho2, z2, spec.normal, projection, modes, 1e-6, Val(8), workspace)
            accepted, value = AS._ring_recurrence_KV(arguments...)
            @test accepted
            reference = recurrence_reference(k, rho, rho2, z2, spec.normal, projection, modes, Val(8))
            for indices in (1:6, 9:14)
                @test norm(value[indices]-reference[indices]) <=
                      0.05max(AS._QUAD_ATOL, 1e-6norm(reference[indices]))
            end
            @test value[7] == value[8] == value[15] == value[16] == 0
            @test (@inferred AS._ring_recurrence_KV(arguments...)) == (accepted, value)
            @test workspace.answer !== workspace.coarse
            @test workspace.low !== workspace.lowcheck
            @test workspace.high !== workspace.highcheck
            @test workspace.G !== workspace.A && workspace.G !== workspace.H
        end
    end

    @testset "Wide recurrence and independent selected Fourier modes" begin
        rho, d = 0.1/1.001, 1e-6
        k, modes = 121.1/rho, 192:319
        workspace = AS._ring_recurrence_workspace(modes, Val(128))
        arguments = (k, rho, 0.0, rho, d*rho, 0.7, -d^2*rho/2, modes, 1e-6, Val(128), workspace)
        accepted, value = AS._ring_recurrence_KV(arguments...)
        @test accepted
        selected = [192, 193, 223, 255, 287, 303, 318, 319]
        reference = recurrence_reference(k, rho, rho, d*rho, 0.7, -d^2*rho/2, selected, Val(8))
        indices = selected .- first(modes) .+ 1
        for (actual_indices, reference_indices) in ((indices, 1:8), (indices .+ 128, 9:16))
            @test norm(value[actual_indices]-reference[reference_indices]) <=
                  0.05max(AS._QUAD_ATOL, 1e-6norm(reference[reference_indices]))
        end
        @test AS._ring_recurrence_KV(arguments...) == (accepted, value)
        negative = AS._ring_recurrence_KV(-k, arguments[2:end]...)
        @test first(negative)
        @test last(negative) == conj(value)
        @test value == last(AS._ring_recurrence_KV(arguments...))
        tasks = [Threads.@spawn begin
            own = AS._ring_recurrence_workspace(modes, Val(128))
            AS._ring_recurrence_KV(arguments[1:end-1]..., own)
        end for _ in 1:4]
        for task in tasks
            @test fetch(task) == (accepted, value)
        end
    end

    @testset "Rejected recurrence preserves adaptive fallback" begin
        modes, rho = 176:181, 0.1/1.001
        k = 121.1/rho
        base = merge(AS._bem_quadrature_workspace(modes, Val(8)), (; modal = nothing, convolution = nothing))
        for reason in (:tolerance, :seed, :tail)
            workspace = AS._ring_recurrence_workspace(modes, Val(8);
                capacity = reason==:tail ? 200 : 8192)
            reason==:seed && (workspace.lowellipsefine = ([NaN], [1.0]))
            z2 = (reason==:tail ? 0.2 : 0.01)*rho
            rtol = reason==:tolerance ? 1e-8 : 1e-6
            arguments = (k, rho, 0.0, rho, z2, 0.7, -0.5z2^2/rho, modes)
            accepted, value = AS._ring_recurrence_KV(arguments..., rtol, Val(8), workspace)
            @test !accepted && iszero(value)
            if reason!=:tolerance
                @test workspace.last_reason == reason
            end
            original = AS._azimuthal_KV_modes(arguments..., false, nothing, rtol, Val(8), base)
            candidate = AS._azimuthal_KV_modes(arguments..., false, nothing, rtol, Val(8),
                merge(base, (; recurrence = workspace)))
            @test isequal(candidate, original)
        end
        workspace = AS._ring_recurrence_workspace(modes, Val(8))
        for (kk, rr, r2, zz, normal, projection, rtol) in (
                (0.0, rho, rho, 0.01rho, 0.7, 0.0, 1e-6),
                (Inf, rho, rho, 0.01rho, 0.7, 0.0, 1e-6),
                (k, 0.0, rho, 0.01rho, 0.7, 0.0, 1e-6),
                (k, rho, rho, 0.0, 0.7, 0.0, 1e-6),
                (k, rho, rho, rho, 0.7, 0.0, 1e-6),
                (161/rho, rho, rho, 0.01rho, 0.7, 0.0, 1e-6),
                (k, rho, rho, 0.01rho, 0.7, 0.0, NaN))
            accepted, value = AS._ring_recurrence_KV(
                kk, rr, 0.0, r2, zz, normal, projection, modes, rtol, Val(8), workspace)
            @test !accepted && iszero(value)
        end
        @test AS._ring_recurrence_workspace(0:0, Val(1)) === nothing
        @test AS._ring_recurrence_workspace(513:520, Val(8)) === nothing
        @test AS._ring_recurrence_workspace(1:0, Val(8)) === nothing
        @test_throws ArgumentError AS._ring_recurrence_workspace(modes, Val(8); capacity = 100)
        @test !first(AS._ring_recurrence_KV(k, rho, 0.0, rho, 0.01rho,
            0.7, 0.0, modes, 1e-6, Val(8), nothing))
        @test !first(AS._ring_recurrence_KV(k, rho, 0.0, rho, 0.01rho,
            0.7, 0.0, modes, 1e-6, Val(4), workspace))
        @test !AS._ring_recurrence_seed_check((; R = 1.0, distance = 1.0, A = 1.0),
            ones(ComplexF64, 2, 3), zeros(ComplexF64, 2, 3))
        fill!(workspace.answer, 1)
        copyto!(workspace.coarse, workspace.answer)
        @test AS._ring_recurrence_output_check(workspace, 1e-6)
        workspace.coarse[1] += 1e-3
        @test !AS._ring_recurrence_output_check(workspace, 1e-6)
    end

    @testset "Analytic ring-tail bounds at physical scales" begin
        rho = 0.1/1.001
        for (m0, d) in ((320, 0.3), (320, 0.5), (240, 0.5))
            modes, k = m0:(m0+3), 121.1/rho
            normal, projection = 0.7, -0.5d^2*rho
            logbound = AS._ring_fourier_tail_logbound(k, rho, 0.0, rho, d*rho, normal, projection, modes)
            @test logbound <= log(0.01AS._QUAD_ATOL)
            reference = recurrence_reference(k, rho, rho, d*rho, normal, projection, modes, Val(4))
            # Direct reference quadrature has a separate absolute resolution limit.
            @test norm(reference) <= max(exp(logbound), 1e-20)
            @test AS._ring_fourier_tail_logbound(-k, rho, 0.0, rho, d*rho, normal, projection, modes) == logbound
        end
        modes = 320:323
        for (k, r, r2, z2, normal, projection) in (
                (NaN, rho, rho, 0.03, 0.7, 0.0),
                (1.0, 0.0, rho, 0.03, 0.7, 0.0),
                (1.0, rho, rho, 0.0, 0.7, 0.0),
                (1.0, rho, rho, 1e200, 0.7, 0.0),
                (1.0, 1e-200, 1e-200, 0.03, 0.7, 0.0),
                (1.0, 1e100, 1e100, 1.0, 1e-300, 0.0))
            @test AS._ring_fourier_tail_logbound(k, r, 0.0, r2, z2, normal, projection, modes) == Inf
        end
        @test AS._ring_fourier_tail_logbound(1.0, rho, 0.0, rho, 0.03, 0.7, 0.0, 1:0) == Inf
        @test AS._ring_fourier_tail_logbound(1.0, rho, 0.0, rho, 0.03, 0.7, 0.0, -1:3) == Inf
        @test AS._ring_fourier_tail_logbound(1.0, rho, 0.0, rho, 0.03, 0.7, 0.0, 510:514) == Inf
    end
end
