_sample_amplitudes(sol::AbstractSolution; kwargs...) = scattering_amplitude(sol; kwargs...)
_sample_amplitudes(sol::FEMSolution{_ScalarFEMData}; kwargs...) = nothing
function _sample_amplitudes(result::ComponentComparison; kwargs...)
    _comparison_amplitudes(result; kwargs...)
end
_sample_labels(::AbstractSolution) = ["Scattered field"]
_sample_labels(result::ComponentComparison) = result.labels

function _resolve_sweep(solve, xs)
    isempty(xs) && throw(ArgumentError("a sweep needs at least one sample"))
    all(isfinite, xs) || throw(ArgumentError("sweep coordinates must be finite"))
    samples = map(xs) do x
        solution = solve(Float64(x))
        amplitude = _sample_amplitudes(solution)
        ts = amplitude === nothing ? target_strength(solution) : target_strength.(amplitude)
        (; amplitude, ts, labels = _sample_labels(solution))
    end
    labels = first(samples).labels
    all(s -> s.labels == labels, samples) ||
        throw(ArgumentError("each sample must have the same response labels"))
    scalar = first(samples).ts isa Real
    all(s -> (s.ts isa Real) == scalar, samples) ||
        throw(ArgumentError("cannot mix single solutions and comparisons in a sweep"))
    amplitudes = if all(s -> s.amplitude === nothing, samples)
        nothing
    elseif any(s -> s.amplitude === nothing, samples)
        throw(ArgumentError("cannot mix results with and without complex amplitudes"))
    elseif scalar
        ComplexF64[s.amplitude for s in samples]
    else
        permutedims(reduce(hcat, [s.amplitude for s in samples]))
    end
    ts = scalar ? Float64[s.ts for s in samples] :
         permutedims(reduce(hcat, [s.ts for s in samples]))
    return ts, amplitudes, labels
end

"""
    BistaticMap

Target strength in dB re 1 m² on a `(thetas, phis)` observation grid in radians.
`target_strength[i,j]` corresponds to `thetas[i]`, `phis[j]`.
"""
struct BistaticMap
    thetas::Vector{Float64}
    phis::Vector{Float64}
    target_strength::Matrix{Float64}
end

"""
    bistatic_map(solution, thetas, phis)

Sample target strength on a polar/azimuthal observation grid in radians without
re-solving. Supported for axisymmetric and full-3D BEM/MFS, coupled-region BEM, shell FEM, volume
FEM and [`free_surface`](@ref) solutions. See [`bistatic_sweep`](@ref) for the angle convention,
which differs between the axisymmetric/full-BEM and volume FEM/free-surface families.
"""
function bistatic_map(sol, thetas::AbstractVector{<:Real}, phis::AbstractVector{<:Real})
    ts = [target_strength(_bistatic_amplitudes(sol, t, p)) for t in thetas, p in phis]
    return BistaticMap(Float64.(thetas), Float64.(phis), ts)
end

"""
    FrequencySweep

Frequency samples in Hz, exterior `k` in inverse meters, `target_strength` in dB re
1 m², and complex `amplitudes` in meters. Arrays have samples along the first dimension;
component comparisons have one column per entry in `labels`. `amplitudes` is `nothing`
for scalar-only FEM results. Phase, when available, is `angle.(sweep.amplitudes)`.
"""
struct FrequencySweep{T, A}
    frequencies::Vector{Float64}
    k::Vector{Float64}
    target_strength::T
    amplitudes::A
    labels::Vector{String}
end

function FrequencySweep(frequencies, k, ts)
    FrequencySweep(frequencies, k, ts, nothing, ["Scattered field"])
end

"""
    frequency_sweep(solve, frequencies, sound_speed)

Call `solve(k)` once per frequency in Hz, with `k=2π*frequency/sound_speed` and exterior
`sound_speed` in m/s. Retain target strength and complex amplitude when available.
The callback may return a solution or a [`components`](@ref) comparison. Dense solution
state is discarded after sampling. Plot the result directly with `plot(sweep)` or
`plot(sweep; quantity=:phase)`; multiple sweeps can share a comparison figure.

The result contains `frequencies` in Hz, exterior `k` in rad/m, `target_strength`
in dB re 1 m², complex `amplitudes` in meters, and `labels`. Arrays have samples
along the first dimension, with one column per label for component comparisons.
`amplitudes` is `nothing` for scalar-only FEM results. When available, phase is
`angle.(sweep.amplitudes)` in radians.

# Examples
```julia
sweep = frequency_sweep(k -> modal(Sphere(0.01), Rigid(), k), 10e3:1e3:100e3, 1477.4)
```
"""
function frequency_sweep(solve::Function, frequencies::AbstractVector{<:Real}, sound_speed::Real)
    isfinite(sound_speed) && sound_speed > 0 ||
        throw(ArgumentError("sound_speed must be finite and positive"))
    all(>(0), frequencies) || throw(ArgumentError("frequencies must be positive"))
    k = 2π .* Float64.(frequencies) ./ Float64(sound_speed)
    ts, amplitudes, labels = _resolve_sweep(solve, k)
    return FrequencySweep(Float64.(frequencies), k, ts, amplitudes, labels)
end

# Frozen snapshot spaces with current-equation acceptance and full-solve fallback.
module FrequencyReduction
using LinearAlgebra

mutable struct SnapshotSpace
    vectors::Matrix{ComplexF64}
    count::Int
end

function SnapshotSpace(n::Integer; capacity::Integer = 16, maxbytes::Integer = 64*1024^2)
    n > 0 || throw(ArgumentError("positive state dimension required"))
    capacity >= 0 && maxbytes >= 0 || throw(ArgumentError("negative storage bound"))
    count = min(capacity, n, maxbytes ÷ sizeof(ComplexF64) ÷ n)
    return SnapshotSpace(zeros(ComplexF64, n, count), 0)
end

function add_snapshot!(space, x; tolerance = 1e-12)
    length(x) == size(space.vectors, 1) || throw(DimensionMismatch("snapshot dimension"))
    all(isfinite, x) || throw(ArgumentError("nonfinite snapshot"))
    isfinite(tolerance) && 0 < tolerance < 1 ||
        throw(ArgumentError("invalid rank tolerance"))
    space.count == size(space.vectors, 2) && return false
    magnitude = norm(x)
    isfinite(magnitude) || throw(ArgumentError("snapshot norm overflow"))
    iszero(magnitude) && return false
    v = x ./ magnitude
    for _ in 1:2, j in 1:space.count

        q = view(space.vectors, :, j)
        v .-= dot(q, v) .* q
    end
    remainder = norm(v)
    remainder > tolerance || return false
    space.count += 1
    space.vectors[:, space.count] .= v ./ remainder
    return true
end

freeze(space) = copy(space.vectors[:, 1:space.count])

function relative_residual(A, x, b, rows)
    residual = A*x-b
    relative(r, rhs) = iszero(norm(rhs)) ? (iszero(norm(r)) ? 0.0 : Inf) : norm(r)/norm(rhs)
    return (; original = relative(residual, b),
        scaled = relative(residual ./ rows, b ./ rows))
end

function solve_reduced(A::Matrix{ComplexF64}, b::Vector{ComplexF64}, V::Matrix{ComplexF64};
        tolerance = 1e-10, rank_tolerance = 1e-12, fallback = (A, b) -> A\b)
    n, m = size(A)
    n == m && n > 0 && length(b) == n && size(V, 1) == n ||
        throw(DimensionMismatch("matrix, forcing and basis dimensions must agree"))
    all(isfinite, A) && all(isfinite, b) && all(isfinite, V) ||
        throw(ArgumentError("nonfinite system or basis"))
    isfinite(tolerance) && 0 < tolerance < 1 ||
        throw(ArgumentError("invalid residual tolerance"))
    isfinite(rank_tolerance) && 0 < rank_tolerance < 1 ||
        throw(ArgumentError("invalid rank tolerance"))
    rows = vec(maximum(abs, A; dims = 2))
    rows[iszero.(rows)] .= 1
    all(isfinite, rows) || throw(ArgumentError("row norm overflow"))
    reason = size(V, 2) == 0 ? :empty : :residual
    x = zeros(ComplexF64, n)
    rank = 0
    if !iszero(norm(b)) && size(V, 2) > 0
        # Recompute images at every frequency. Never reuse A(k_train)*V here.
        images = (A*V) ./ rows
        if all(isfinite, images)
            factor = qr(images, ColumnNorm())
            diagonal = abs.(diag(factor.R))
            cutoff = rank_tolerance * maximum(diagonal; init = 0.0)
            rank = count(>(cutoff), diagonal)
            if rank == size(V, 2)
                x = V * (factor \ (b ./ rows))
            else
                reason = :rank
            end
        else
            reason = :nonfinite
        end
    end
    projected = relative_residual(A, x, b, rows)
    accepted = all(isfinite, x) && max(projected.original, projected.scaled) <= tolerance
    if accepted
        return (; x, reduced = true, reason = :accepted, rank, projected,
            residual = projected)
    end
    all(isfinite, x) || (reason = :nonfinite)
    x = fallback(A, b)
    residual = relative_residual(A, x, b, rows)
    all(isfinite, x) && max(residual.original, residual.scaled) <= tolerance ||
        error("full solve failed current-equation residual acceptance")
    return (; x, reduced = false, reason, rank, projected, residual)
end

end

function _frequency_fluid_equations(
        surface, material, k, incidence_angle, incidence_azimuth)
    system = _assemble_full_fluid(material, k, surface.data; formulation = :cbie)
    p, dp = _incident_traces(surface.data, k, incidence_angle, incidence_azimuth, nothing)
    return system.A, _fluid_forcing(system, p, dp)
end

function _frequency_full_solve(A, b)
    factor = _factor_fluid_system(A; condition_limit = 0)
    return _solve_fluid_system(factor, b).x
end

"""
    frequency_sweep(surface::Mesh, material::FluidFilled, frequencies, sound_speed;
                    training_frequencies, incidence_angle=π/2, incidence_azimuth=0,
                    tolerance=1e-10, rank_tolerance=1e-12, capacity=16,
                    maxbytes=64*1024^2, return_diagnostics=false)

Opt-in reduced-order frequency sweep for dense single-interface fluid/gas CBIE
with DIM correction and plane-wave incidence on a fixed full-3D mesh. Frequencies
and `training_frequencies` are in Hz, `sound_speed` in m/s, and angles in radians
from +x with azimuth from +y toward +z. Returns a sweep of
complex backscatter amplitudes and target strengths in the supplied query order.

Train a bounded orthonormal solution basis using full Float64 solves, then freeze
it before evaluating queries. Every query assembles its own equations and basis
images. Both original and row-scaled relative residuals must satisfy `tolerance`.
Rank loss or failed acceptance triggers a full Float64 solve, checked against the
same residual target. Queries never enrich the basis. `capacity` and `maxbytes`
bound snapshot coefficient storage, not total assembly or process memory.

With `return_diagnostics=true`, return `(sweep, diagnostics)` as a named tuple.
Diagnostics contain training and query reports, retained basis rank and bytes,
and settings. Query reports identify reduced acceptance or full-solve fallback.
Residual acceptance does not bound physical mesh error. Resolve the supplied mesh
over the entire frequency range. Training cost can exceed the savings for short
sweeps, and no automatic speedup is assumed. The callback overload is unchanged.
"""
function frequency_sweep(surface::Mesh{<:Inti.Quadrature}, material::FluidFilled,
        frequencies::AbstractVector{<:Real}, sound_speed::Real;
        training_frequencies::AbstractVector{<:Real},
        incidence_angle::Real = π/2, incidence_azimuth::Real = 0.0,
        tolerance::Real = 1e-10, rank_tolerance::Real = 1e-12,
        capacity::Integer = 16, maxbytes::Integer = 64*1024^2,
        return_diagnostics::Bool = false)
    isfinite(sound_speed) && sound_speed > 0 ||
        throw(ArgumentError("sound_speed must be finite and positive"))
    for values in (frequencies, training_frequencies)
        !isempty(values) && all(f -> isfinite(f) && f > 0, values) ||
            throw(ArgumentError("query and training frequencies must be nonempty, finite and positive"))
    end
    all(isfinite, (incidence_angle, incidence_azimuth)) ||
        throw(ArgumentError("incidence angles must be finite"))
    all(t -> isfinite(t) && 0 < t < 1, (tolerance, rank_tolerance)) ||
        throw(ArgumentError("residual and rank tolerances must lie in (0, 1)"))
    k = 2π .* Float64.(frequencies) ./ Float64(sound_speed)
    training_k = 2π .* Float64.(training_frequencies) ./ Float64(sound_speed)
    all(t -> isfinite(t) && t > 0, k) && all(t -> isfinite(t) && t > 0, training_k) ||
        throw(ArgumentError("converted wavenumbers must be finite and positive"))
    n = length(surface.data)
    space = FrequencyReduction.SnapshotSpace(2n; capacity, maxbytes)
    empty_basis = zeros(ComplexF64, 2n, 0)
    training = NamedTuple[]
    for (frequency, wavenumber) in zip(training_frequencies, training_k)
        A, b = _frequency_fluid_equations(surface, material, wavenumber,
            incidence_angle, incidence_azimuth)
        result = FrequencyReduction.solve_reduced(A, b, empty_basis;
            tolerance, rank_tolerance, fallback = _frequency_full_solve)
        kept = FrequencyReduction.add_snapshot!(space, result.x; tolerance = rank_tolerance)
        push!(training,
            (; frequency = Float64(frequency), k = wavenumber,
                kept, rank = space.count, result.residual))
    end
    basis = FrequencyReduction.freeze(space)
    space = nothing
    direction = -_bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    amplitudes = ComplexF64[]
    queries = NamedTuple[]
    for (frequency, wavenumber) in zip(frequencies, k)
        A, b = _frequency_fluid_equations(surface, material, wavenumber,
            incidence_angle, incidence_azimuth)
        result = FrequencyReduction.solve_reduced(A, b, basis;
            tolerance, rank_tolerance, fallback = _frequency_full_solve)
        push!(amplitudes,
            far_field(surface.data, direction, wavenumber,
                view(result.x, 1:n), view(result.x, (n + 1):(2n))))
        push!(queries,
            (; frequency = Float64(frequency), k = wavenumber,
                result.reduced, result.reason, result.rank, result.projected, result.residual))
    end
    sweep = FrequencySweep(Float64.(frequencies), k, target_strength.(amplitudes),
        amplitudes, ["Scattered field"])
    report = (; method = :reduced_basis, formulation = :cbie,
        compression = (method = :none,), correction = (method = :dim,),
        training, queries, basis_rank = size(basis, 2), basis_bytes = sizeof(basis),
        tolerance, rank_tolerance, capacity, maxbytes, incidence_angle, incidence_azimuth)
    return return_diagnostics ? (; sweep, diagnostics = report) : sweep
end

"""
    IncidenceAngleSweep

Incidence `angles` in radians and sampled `target_strength`, `amplitudes`, and `labels`,
with the same layout and units as [`frequency_sweep`](@ref). Observation follows
backscatter, opposite each sample's incident direction.
"""
struct IncidenceAngleSweep{T, A}
    angles::Vector{Float64}
    target_strength::T
    amplitudes::A
    labels::Vector{String}
end

function IncidenceAngleSweep(angles, ts)
    IncidenceAngleSweep(angles, ts, nothing, ["Scattered field"])
end

"""
    incidence_angle_sweep(solve, angles)

Call `solve(angle)` once per incident polar angle in radians and retain monostatic
strength and complex amplitude. The callback supplies the fixed frequency and may return
an ordinary solution or a [`components`](@ref) comparison.
"""
function incidence_angle_sweep(solve::Function, angles::AbstractVector{<:Real})
    ts, amplitudes, labels = _resolve_sweep(solve, angles)
    return IncidenceAngleSweep(Float64.(angles), ts, amplitudes, labels)
end

function _validate_incidence_sweep(angles, incidence_azimuth)
    isempty(angles) && throw(ArgumentError("a sweep needs at least one sample"))
    all(isfinite, angles) || throw(ArgumentError("sweep coordinates must be finite"))
    isfinite(incidence_azimuth) || throw(ArgumentError("incidence azimuth must be finite"))
end

function _fluid_spheroid_modal_angle_sweep(body::Spheroid, boundary::FluidFilled,
        k::Real, angles::AbstractVector{<:Real};
        incidence_azimuth::Real = 0.0,
        m_max::Integer = _default_spheroid_orders(k, body),
        n_max::Integer = _default_spheroid_orders(k, body),
        n_quad::Integer = 64, precision::Symbol = :double)
    _validate_incidence_sweep(angles, incidence_azimuth)
    m_max >= 0 && n_max >= 0 || throw(ArgumentError("modal orders must be nonnegative"))
    n_quad >= 2 || throw(ArgumentError("n_quad must be at least 2"))
    samples = Float64.(angles)
    eta_i = cos.(samples)
    eta_s = cos.(pi .- samples)
    amplitudes = zeros(ComplexF64, length(samples))
    c = k * body.q
    c_int = c / boundary.soundspeed_contrast
    nodes, weights = gauss(n_quad)
    half = (n_quad ÷ 2 + 1):n_quad
    nodes = nodes[half]
    weights = 2 .* weights[half]
    isodd(n_quad) && (weights[1] /= 2)

    for m in 0:min(m_max, n_max)
        active = [i
                  for i in eachindex(samples)
                  if m == 0 || !(abs(eta_i[i]) == 1 || abs(eta_s[i]) == 1)]
        isempty(active) && continue
        count = length(active)
        exterior = _spheroid_wavefunctions(m, n_max, c, body.xi0,
            [eta_i[active]; eta_s[active]; nodes]; spheroid = body.kind, precision)
        interior = _spheroid_wavefunctions(m, n_max, c_int, body.xi0,
            nodes; spheroid = body.kind, precision, radial_kind = 1)
        degrees = m:n_max
        Sint = collect(eachcol(interior.angular))
        R3e = [(; value = (complex(exterior.r1[j], exterior.r2[j]),),
                   derivative = (complex(exterior.dr1[j], exterior.dr2[j]),))
               for j in eachindex(degrees)]
        R1i = [(; value = (interior.r1[j],), derivative = (interior.dr1[j],))
               for j in eachindex(degrees)]
        surface = @view exterior.angular[(2 * count + 1):end, :]
        azimuth_term = cos(m * (incidence_azimuth - (incidence_azimuth + pi)))
        for (local_i, sample_i) in enumerate(active)
            Sext = [[exterior.angular[local_i, j]; surface[:, j]]
                    for j in eachindex(degrees)]
            A = if precision === :quad
                setprecision(BigFloat, 256) do
                    _solve_fluid_coupling(BigFloat, boundary.density_contrast,
                        degrees, Sext, Sint, weights, R3e, R1i)
                end
            else
                _solve_fluid_coupling(Float64, boundary.density_contrast,
                    degrees, Sext, Sint, weights, R3e, R1i)
            end
            for j in eachindex(degrees)
                amplitudes[sample_i] += neumann_factor(m) *
                                        exterior.angular[count + local_i, j] * A[j] *
                                        azimuth_term
            end
        end
    end
    amplitudes .*= -2im / k
    return IncidenceAngleSweep(samples, target_strength.(amplitudes), amplitudes,
        ["Scattered field"])
end

"""
    incidence_angle_sweep(mfs, body, boundary, k, angles; n=default, m_max=default,
        offset=0.3*characteristic_radius, oversampling=1, rtol=1e-6,
        condition_limit=512, return_diagnostics=false)

Sample axisymmetric MFS backscatter at fixed exterior wavenumber `k` (inverse meters)
for a `Sphere`, `Spheroid`, or straight `Cylinder`. Polar `angles` are radians from
body-frame +x. Supports `Rigid`, `PressureRelease`, `Impedance`, and `FluidFilled`.
Source placement, oversampling and quadrature follow [`mfs`](@ref); fluid boundaries
also accept `offset_ext` and `offset_int`, each defaulting to `offset` in meters.

Assemble and factor each Fourier mode once for all angles, including independent
boundary-check operators and condition/rank diagnostics. Only one mode's matrices
are retained at a time; coefficients and surface fields are discarded after sampling.
Nonzero angles use modes `0:m_max`; exactly zero uses only mode zero, as in `mfs`.
No setup is cached across calls. Bent cylinders and full-surface MFS are not supported.
Rigid straight cylinders with flat caps are also unsupported by axisymmetric MFS;
use `bem` for that sharp-rim geometry.

Returns sampled angles, target strengths, and amplitudes. With `return_diagnostics=true`, returns
`(; sweep, diagnostics)`, where `diagnostics[i]` contains the solve and independent
boundary residuals for `angles[i]`, in the same form as `diagnostics(mfs(...))`.
Checks are computed in either case; `condition_limit=0` skips the condition/rank SVD.
"""
function incidence_angle_sweep(::typeof(mfs), body::Union{Sphere, Spheroid, Cylinder},
        boundary::Union{Rigid, PressureRelease, Impedance, FluidFilled}, k::Real,
        angles::AbstractVector{<:Real};
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)),
        offset::Real = 0.3_characteristic_radius(body),
        offset_ext::Real = offset, offset_int::Real = offset,
        oversampling::Integer = 1, rtol::Real = 1e-6,
        condition_limit::Integer = 512, return_diagnostics::Bool = false)
    _validate_incidence_sweep(angles, 0.0)
    body isa Cylinder && _isbent(body) &&
        throw(ArgumentError("reusable MFS sweeps require a straight Cylinder"))
    _reject_rigid_flat_cylinder_mfs(body, boundary)
    oversampling >= 1 || throw(ArgumentError("mfs: oversampling must be at least 1"))
    condition_limit >= 0 || throw(ArgumentError("mfs: condition_limit must be nonnegative"))
    m_max >= 0 || throw(ArgumentError("mfs: m_max must be nonnegative"))
    if !(boundary isa FluidFilled) && (offset_ext != offset || offset_int != offset)
        throw(ArgumentError("offset_ext and offset_int require FluidFilled; use offset"))
    end
    mesh, source_mesh = _mfs_meridian_meshes(body, n, oversampling)
    check_mesh = _mfs_check_mesh(mesh)
    rho, z = mfs_source_points(source_mesh, offset_ext)
    exterior = (; rho, z)
    interior = if boundary isa FluidFilled
        rho, z = mfs_source_points(source_mesh, -offset_int)
        (; rho, z)
    else
        nothing
    end
    samples = Float64.(angles)
    amplitudes = zeros(ComplexF64, length(samples))
    reports = [_SolveReports() for _ in samples]
    last_mode = all(iszero, samples) ? 0 : m_max
    for m in 0:last_mode
        _mfs_sweep_mode!(amplitudes, reports, boundary, k, mesh, check_mesh,
            exterior, interior, samples, m; rtol, condition_limit, offset_ext, offset_int)
    end
    sweep = IncidenceAngleSweep(samples, target_strength.(amplitudes), amplitudes,
        ["Scattered field"])
    return_diagnostics || return sweep
    diagnostics = map(eachindex(samples)) do i
        incidence_angle = samples[i]
        offsets = boundary isa FluidFilled ? (; offset_ext, offset_int) : (; offset)
        modes = iszero(incidence_angle) ? (;) : (; m_max)
        _summarize_solves(reports[i]; method = :axisymmetric,
            solver_options = (; n = npanels(source_mesh), oversampling, offsets...,
                incidence_angle, modes..., rtol, condition_limit))
    end
    return (; sweep, diagnostics)
end

# Shared by the Cylinder and Sphere/Spheroid axisymmetric-BEM sweep entry points below.
function _axisymmetric_angle_sweep(
        body, boundary::Union{Rigid, PressureRelease, Impedance}, k::Real,
        angles::AbstractVector{<:Real};
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)),
        rtol::Real = 1e-5)
    _validate_incidence_sweep(angles, 0.0)
    body isa Cylinder && _isbent(body) &&
        throw(ArgumentError("a bent Cylinder requires bem(...; method=:full)"))
    m_max >= 0 || throw(ArgumentError("m_max must be nonnegative"))
    mesh = _axisymmetric_mesh(body, n)
    samples = Float64.(angles)
    amplitudes = zeros(ComplexF64, length(samples))
    errors = zeros(Float64, length(samples))
    for first_mode in 0:_MODE_CHUNK:m_max
        modes = first_mode:min(first_mode + _MODE_CHUNK - 1, m_max)
        _axisymmetric_sweep_batch!(amplitudes, errors, boundary, k, mesh, samples, modes;
            rtol, axial_only = m_max == 0)
    end
    _axisymmetric_sweep_check!(amplitudes, errors, boundary, k, mesh, samples; m_max, rtol)
    return IncidenceAngleSweep(samples, target_strength.(amplitudes), amplitudes,
        ["Scattered field"])
end

function _axisymmetric_sweep_check!(amplitudes, errors, boundary, k, mesh, samples;
        m_max, rtol)
    # Cancellation between batches can make separate relative error targets too
    # loose. Rebuild only that angle's traces, still with bounded matrix storage,
    # and use the original combined-mode integral and its per-panel tolerance.
    for i in eachindex(samples)
        if errors[i] > 1e-6 * abs(amplitudes[i])
            p, dp, ps = _oblique_solve_streamed(boundary, k, mesh, samples[i]; m_max, rtol)
            amplitudes[i] = far_field(ps, p, dp, k, pi - samples[i], pi)
        end
    end
    return nothing
end

# Complete all angles before releasing the batch's factors. Only one angle's
# traces and reports are live, independently of the total sample count.
function _axisymmetric_sweep_batch!(amplitudes, errors, boundary, k, mesh, angles, modes;
        rtol, axial_only)
    ps, factors = _oblique_mode_factors(boundary, k, mesh; m_max = last(modes), rtol, modes)
    for (i, incidence_angle) in enumerate(angles)
        reports = _SolveReports()
        p_scat_modes, dpdn_scat_modes = _oblique_solve_factored(
            boundary, k, ps, factors, incidence_angle; rtol, solve_reports = reports,
            first_mode = first(modes))
        if axial_only
            amplitudes[i] += far_field(
                ps, only(p_scat_modes), only(dpdn_scat_modes), k, pi - incidence_angle)
        else
            amplitude, error = _far_field_modes(ps, p_scat_modes, dpdn_scat_modes, k,
                pi - incidence_angle, pi, first(modes); return_error = true)
            amplitudes[i] += amplitude
            errors[i] += error
        end
    end
    return nothing
end

"""
    incidence_angle_sweep(body::Cylinder, boundary::Union{Rigid,PressureRelease,Impedance}, k, angles;
        n=default, m_max=default, rtol=1e-5)

Sample axisymmetric BEM backscatter at fixed exterior wavenumber `k` in inverse meters, for a
straight (unbent) `Cylinder`. Each Fourier mode's boundary operator depends only on the mesh,
`k` and the mode, not the incidence angle, so it is assembled and factorized once and reused
for every angle; only the incident-field right-hand side is rebuilt per angle. `m_max` is
therefore fixed for the whole sweep rather than resolved per angle as a single [`bem`](@ref)
call would. A bent `Cylinder` needs `bem(...; method=:full)` instead. Returns a
sweep result with target strengths and amplitudes.

Process all angles in batches of at most eight Fourier modes, releasing completed
matrices and factors before assembling the next batch. Accumulate backscatter
amplitudes without retaining all modes' factors or all angles' surface traces.
If the summed far-field quadrature estimates indicate cancellation between batches,
rebuild that angle's traces in bounded batches and use the combined-mode integral.
This safeguard can require additional assembly for the affected angle.
"""
function incidence_angle_sweep(
        body::Cylinder, boundary::Union{Rigid, PressureRelease, Impedance},
        k::Real, angles::AbstractVector{<:Real};
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)),
        rtol::Real = 1e-5)
    return _axisymmetric_angle_sweep(body, boundary, k, angles; n, m_max, rtol)
end

"""
    incidence_angle_sweep(surface::Mesh, boundary::Union{Rigid,PressureRelease}, k, angles; kwargs...)

Sample full-3D BEM backscatter at fixed exterior wavenumber `k` in inverse meters.
Polar `angles` are radians from +x, with fixed `incidence_azimuth=0` by default.
Reuse layer operators, their compression and the system operator within this call.
Each angle gets an independent GMRES solve with the supplied tolerances.

Accepts `formulation`, `compression`, `correction` and `gmres_kwargs` as in [`bem`](@ref).
Returns sampled amplitudes and strengths. Operators are discarded
after sampling. Subsequent calls assemble from their own inputs.
"""
function incidence_angle_sweep(surface::Mesh{<:Inti.Quadrature},
        boundary::Union{Rigid, PressureRelease}, k::Real, angles::AbstractVector{<:Real};
        incidence_azimuth::Real = 0.0, formulation::Symbol = :burton_miller,
        compression::NamedTuple = (method = :hmatrix, tol = 1e-5),
        correction::NamedTuple = (method = :dim,),
        gmres_kwargs::NamedTuple = (reltol = 1e-4, restart = 150, maxiter = 1200))
    _validate_incidence_sweep(angles, incidence_azimuth)
    system = _assemble_full_boundary(boundary, k, surface.data;
        formulation, compression, correction)
    return incidence_angle_sweep(angles) do incidence_angle
        density = Ref{Union{Nothing, Vector{ComplexF64}}}(nothing)
        p, q, quad, report = _solve_full_boundary(system;
            incidence_angle, incidence_azimuth, gmres_kwargs, _density = density)
        _full_bem_solution(surface, boundary, k, p, q, quad, report,
            incidence_angle, incidence_azimuth; density = density[])
    end
end

"""
    incidence_angle_sweep(surface::Mesh, material::FluidFilled, k, angles; kwargs...)
    incidence_angle_sweep(surfaces, materials, k, angles; components=false, labels=nothing, kwargs...)

Sample fluid/gas full-3D BEM backscatter at fixed exterior wavenumber `k` in inverse
meters. Polar `angles` are radians from +x; `incidence_azimuth=0` sweeps the xy plane.
Geometry validation, operators and factorization are reused within this call. A new
call assembles from its supplied meshes, materials, frequency and solver options.

Accepts `formulation`, `correction`, `equilibrate` and `condition_limit` as in [`bem`](@ref).
The single-interface dense overload also accepts `precision=:mixed` and
`refinement`, reusing the guarded factorization across angles. Promotion to
ComplexF64 persists for the remaining angles in that call.
Both fluid overloads also accept `compression` and `gmres_kwargs`, reusing the
compressed operators and block preconditioner across incidence angles.
For compressed solves, `recycle_dimension=8` also retains a bounded solution subspace
to initialize later angles. Set it to zero for independent GMRES starts. Retained
solution/image pairs are capped at 64 MiB; dense solves do not use this storage.
Every recycled solve checks the true scaled residual against the original right-hand
side tolerance and retries from zero if necessary. Recycling is local to this call.
Residual corrections share an attempt's `maxiter` budget; a fallback gets a fresh
budget. Iteration diagnostics include all correction and fallback work.
The multiple-interface overload also accepts `parents` and `validation`. With
`components=true`, include each interface isolated in the exterior medium and their
coherent complex sum, using the same solver options. `labels` supplies one name per
interface. Each system is sampled separately to limit retained matrix storage.

Returns complex amplitudes and target strengths in a sweep result. Dense
solution state is discarded after sampling.

# Examples
```julia
aspect = incidence_angle_sweep(surfaces, materials, k, deg2rad.([60, 90, 120]);
    parents=[0, 1], components=true, labels=["flesh", "bladder"])
```
"""
function incidence_angle_sweep(surface::Mesh{<:Inti.Quadrature}, material::FluidFilled,
        k::Real, angles::AbstractVector{<:Real}; incidence_azimuth::Real = 0.0,
        formulation::Symbol = :muller, correction::NamedTuple = (method = :dim,),
        equilibrate::Bool = true, condition_limit::Integer = 512,
        compression::NamedTuple = (method = :none,), gmres_kwargs::NamedTuple = (;),
        recycle_dimension::Integer = 8, precision::Symbol = :double,
        refinement::NamedTuple = (;))
    _validate_incidence_sweep(angles, incidence_azimuth)
    recycle_dimension >= 0 || throw(ArgumentError("recycle_dimension must be nonnegative"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    system = _assemble_full_fluid(
        material, k, surface.data; formulation, correction, compression)
    factor = _factor_full_fluid(system; equilibrate, condition_limit, gmres_kwargs,
        precision, refinement)
    factor = _fluid_sweep_factor(factor, recycle_dimension, length(angles))
    return incidence_angle_sweep(angles) do incidence_angle
        p, q, quad, report = _solve_full_fluid(system, factor;
            incidence_angle, incidence_azimuth)
        _full_bem_solution(surface, material, k, p, q, quad, report,
            incidence_angle, incidence_azimuth)
    end
end

function _region_angle_sweep(surfaces, materials, k, angles;
        incidence_azimuth, equilibrate, condition_limit, gmres_kwargs = (;),
        recycle_dimension = 8, kwargs...)
    system = _assemble_region_bem(surfaces, materials, k; kwargs...)
    factor = _factor_full_fluid(system; equilibrate, condition_limit, gmres_kwargs,
        norm_floor = eps(Float64))
    factor = _fluid_sweep_factor(factor, recycle_dimension, length(angles))
    return incidence_angle_sweep(angles) do incidence_angle
        _solve_region_bem(system, factor; incidence_angle, incidence_azimuth)
    end
end

function incidence_angle_sweep(surfaces::AbstractVector{<:Mesh},
        materials::AbstractVector{<:FluidFilled}, k::Real, angles::AbstractVector{<:Real};
        components::Bool = false, labels = nothing,
        parents::AbstractVector{<:Integer} = collect(0:(length(surfaces) - 1)),
        incidence_azimuth::Real = 0.0, formulation::Symbol = :muller,
        correction::NamedTuple = (method = :dim,), equilibrate::Bool = true,
        condition_limit::Integer = 512, validation::NamedTuple = (;),
        compression::NamedTuple = (method = :none,), gmres_kwargs::NamedTuple = (;),
        recycle_dimension::Integer = 8)
    _validate_incidence_sweep(angles, incidence_azimuth)
    recycle_dimension >= 0 || throw(ArgumentError("recycle_dimension must be nonnegative"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    components || labels === nothing ||
        throw(ArgumentError("labels require components=true"))
    response_labels = components ? _component_labels(length(surfaces), labels) : nothing
    options = (; incidence_azimuth, formulation, correction, equilibrate, condition_limit,
        compression, gmres_kwargs, recycle_dimension)
    coupled = _region_angle_sweep(
        surfaces, materials, k, angles; parents, validation, options...)
    components || return coupled
    isolated = [incidence_angle_sweep(surface, material, k, angles; options...).amplitudes
                for (surface, material) in zip(surfaces, materials)]
    amplitudes = hcat(coupled.amplitudes, isolated..., sum(isolated))
    return IncidenceAngleSweep(Float64.(angles), target_strength.(amplitudes),
        amplitudes, response_labels)
end

"""
    incidence_angle_sweep(body::Irregular, boundary::AbstractBoundaryCondition, k, angles; kwargs...)

Sample Fourier-matching backscatter at fixed exterior wavenumber `k`, reusing the conformal
mapping and the boundary-matching transition operator (see [Fourier matching](@ref
fourier-matching-theory)) across all `angles`, since neither depends on incidence angle. Each
angle then only needs a cheap incident-coefficient recompute and matrix-vector solve, not the
expensive boundary-matching quadrature that dominates a single [`fourier`](@ref) call.

Accepts `continuation_steps`, `mapping_order`, `m_max`, `n_max`, `rtol` and `maxevals` as in
[`fourier`](@ref). Returns a sweep result with amplitudes and target strengths.
"""
function incidence_angle_sweep(body::Irregular, boundary::AbstractBoundaryCondition,
        k::Real, angles::AbstractVector{<:Real};
        continuation_steps::Integer = 8, mapping_order::Integer = max(length(body.rc), 1),
        m_max::Integer = _default_mode_count(k * body.a), n_max::Integer = m_max,
        rtol::Real = 1e-6, maxevals::Integer = 1000)
    isempty(angles) && throw(ArgumentError("a sweep needs at least one sample"))
    all(isfinite, angles) || throw(ArgumentError("sweep coordinates must be finite"))
    mapping = solve_mapping(body, mapping_order; continuation_steps)
    is_admissible(mapping) || throw(ArgumentError(
        "Irregular's conformal mapping is inadmissible (Jacobian vanishes somewhere). " *
        "Try a higher mapping_order or more continuation_steps"))
    transition = _boundary_transition(mapping, k, boundary; m_max, n_max, rtol, maxevals)
    convergence = Ref(0.0)
    sweep = incidence_angle_sweep(angles) do incidence_angle
        a = _incident_coefficients(n_max, m_max, k, incidence_angle)
        b = _apply_transition(transition, a, n_max, m_max)
        b_check = _check_coefficients(transition, k, incidence_angle, n_max, m_max)
        change = _fm_convergence(b, b_check, k)
        isnan(change) || (convergence[] = max(convergence[], change))
        FMSolution(
            body, boundary, Float64(k), mapping, b, Float64(incidence_angle), b_check)
    end
    _warn_fm_convergence(convergence[])
    return sweep
end

"""
    incidence_angle_sweep(body::Spheroid, boundary::Union{SolidElastic, Shelled{ElasticLayer}}, k, angles; kwargs...)

Sample the elastic prolate spheroid or shell backscatter at fixed exterior wavenumber `k`. The
transition matrices do not depend on direction, so each azimuthal order is computed once and
reused for every angle. Accepts `m_max`, `n_max` and `check` as in [`tmatrix`](@ref), and
`method = :volume` samples the volume FEM instead. Returns a sweep result.
"""
function incidence_angle_sweep(body::Spheroid, boundary::_ElasticSpheroidBoundary,
        k::Real, angles::AbstractVector{<:Real};
        m_max::Integer = _default_elastic_orders(boundary, k, body),
        n_max::Integer = _default_elastic_orders(boundary, k, body), check::Bool = true,
        method::Symbol = :tmatrix, kwargs...)
    method === :volume && return _volume_angle_sweep(body, boundary, k, angles; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported keywords $(join(keys(kwargs), ", ")) for the transition-matrix sweep"))
    method === :tmatrix ||
        throw(ArgumentError("method must be :tmatrix or :volume, got $method"))
    _validate_incidence_sweep(angles, 0.0)
    transition = _elastic_transition_function(boundary, k, body, n_max)
    guarded = check && boundary isa Shelled && n_max > 2 && m_max > 2
    reduced = guarded ? _elastic_transition_function(boundary, k, body, n_max - 2) : nothing
    change = Ref(0.0)
    sweep = incidence_angle_sweep(angles) do incidence_angle
        amplitude = _elastic_spheroid_amplitude(transition, k, body, incidence_angle, 0.0,
            π - incidence_angle, π, m_max, n_max)
        if guarded
            coarse = _elastic_spheroid_amplitude(reduced, k, body, incidence_angle, 0.0,
                π - incidence_angle, π, m_max - 2, n_max - 2)
            change[] = max(change[], abs(amplitude - coarse) / abs(amplitude))
        end
        TMatrixSolution(body, boundary, k, amplitude)
    end
    guarded && _warn_elastic_truncation(change[])
    return sweep
end

function _volume_angle_sweep(
        body, boundary, k, angles; incidence_azimuth::Real = 0.0, kwargs...)
    _validate_incidence_sweep(angles, incidence_azimuth)
    system = _volume_system_single(body, boundary, k; kwargs...)
    return incidence_angle_sweep(angles) do incidence_angle
        FEMSolution(body, boundary, Float64(k), :volume,
            _volume_solution(system, incidence_angle, incidence_azimuth))
    end
end

"""
    incidence_angle_sweep(body::Union{Sphere,Spheroid}, boundary, k, angles; method=:volume, incidence_azimuth=0, kwargs...)

Sample full-3D volume FEM backscatter at fixed exterior wavenumber `k`. The mesh, matrix and its
factorization do not depend on the incident direction, so they are built once and each angle only
needs a new load and solve. Accepts the keywords of `fem(...; method = :volume)`, with `angles` the
polar incidence angles in radians. Returns a sweep result.

`method=:axisymmetric` instead samples axisymmetric BEM for a `Rigid`/`PressureRelease`/
`Impedance` `boundary`, reusing each Fourier mode's factorized operator across angles; see
`incidence_angle_sweep(::Cylinder, ::Union{Rigid,PressureRelease,Impedance}, ...)`. It accepts
`n`, `m_max` and `rtol` instead of the volume-FEM keywords.

`method=:modal` supports a full-coupling `FluidFilled` spheroid. It evaluates each
azimuthal order's spheroidal basis once across all angles while retaining the
existing coupling solve for each incident direction. Fix `m_max`, `n_max`, `n_quad`,
and `precision` within a sweep.
"""
function incidence_angle_sweep(body::Union{Sphere, Spheroid},
        boundary::AbstractBoundaryCondition, k::Real, angles::AbstractVector{<:Real};
        method::Symbol = :volume, kwargs...)
    if method === :modal
        body isa Spheroid && boundary isa FluidFilled && boundary.coupling === :full ||
            throw(ArgumentError("method=:modal currently requires a full-coupling FluidFilled spheroid"))
        return _fluid_spheroid_modal_angle_sweep(body, boundary, k, angles; kwargs...)
    end
    if method === :axisymmetric
        boundary isa Union{Rigid, PressureRelease, Impedance} || throw(ArgumentError(
            "incidence_angle_sweep(..., method=:axisymmetric) only supports Rigid, PressureRelease or Impedance"))
        return _axisymmetric_angle_sweep(body, boundary, k, angles; kwargs...)
    end
    method === :volume || throw(ArgumentError(
        "incidence_angle_sweep(::Union{Sphere,Spheroid}, ...) only supports method=:volume, :axisymmetric or :modal"))
    return _volume_angle_sweep(body, boundary, k, angles; kwargs...)
end

"""
    incidence_angle_sweep(bodies::AbstractVector{<:AbstractBody}, materials::AbstractVector, k, angles; kwargs...)

Volume FEM backscatter of coupled fluid and elastic regions, as in
`fem(bodies, materials, k; method = :volume)`, with one mesh and factorization for all `angles`.
"""
function incidence_angle_sweep(bodies::AbstractVector{<:AbstractBody},
        materials::AbstractVector, k::Real,
        angles::AbstractVector{<:Real}; parents::AbstractVector{<:Integer} = collect(
            0:(length(bodies) - 1)),
        centers = [zeros(3) for _ in bodies],
        orientations = [[0.0, 0.0, 1.0] for _ in bodies], incidence_azimuth::Real = 0.0,
        kwargs...)
    _validate_incidence_sweep(angles, incidence_azimuth)
    all(b -> b isa Union{Sphere, Spheroid}, bodies) || throw(ArgumentError(
        "region bodies must be Sphere or Spheroid"))
    all(m -> m isa _VolumeRegionMaterial, materials) || throw(ArgumentError(
        "region materials must be FluidFilled, SpatialFluid, SolidElastic, ViscoelasticSolid or ViscousLayer"))
    system = _volume_system_regions(
        bodies, materials, k; parents, centers, orientations, kwargs...)
    geometry = _VolumeRegionGeometry(collect(AbstractBody, bodies),
        [Float64.(c) for c in centers], [Float64.(o) for o in orientations],
        collect(Int, parents))
    boundary = _VolumeRegionMaterials(collect(Any, materials))
    return incidence_angle_sweep(angles) do incidence_angle
        FEMSolution(geometry, boundary, Float64(k), :volume,
            _volume_solution(system, incidence_angle, incidence_azimuth))
    end
end

const _BistaticAngleAzimuthSolution = Union{
    BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData},
    FEMSolution{_ShellFEMSurfaceData}, FEMSolution{_CylinderMeridianFEMData},
    FEMSolution{_SpheroidMeridianFEMData}, FEMSolution{_VolumeFEMData}, FreeSurfaceSolution}
const _BistaticDirectionSolution = Union{BEMSolution{_FullBEMSurfaceData},
    BEMSolution{_RegionBEMData}, MFSSolution{_FullMFSSurfaceData}}

_bistatic_incidence(sol::ComponentComparison) = _bistatic_incidence(sol.coupled)
_bistatic_incidence(sol::FreeSurfaceSolution) = (sol.incidence_angle, sol.incidence_azimuth)
function _bistatic_incidence(sol)
    data = sol.data
    return (data.incidence_angle,
        hasproperty(data, :incidence_azimuth) ? data.incidence_azimuth : 0.0)
end

"""
    BistaticSweep

Observation `angles` and fixed `azimuth` in radians, sampled `target_strength` and
complex `amplitudes`, and solve-time `incidence_angle` and `incidence_azimuth` in radians.
Arrays use the same layout and units as [`frequency_sweep`](@ref).
"""
struct BistaticSweep{T, A}
    angles::Vector{Float64}
    azimuth::Float64
    target_strength::T
    incidence_angle::Float64
    incidence_azimuth::Float64
    amplitudes::A
    labels::Vector{String}
end

function BistaticSweep(angles, azimuth, ts, incidence)
    BistaticSweep(angles, azimuth, ts, incidence, 0.0, nothing, ["Scattered field"])
end

function _bistatic_amplitudes(sol::_BistaticAngleAzimuthSolution, angle, azimuth)
    scattering_amplitude(sol; angle, azimuth)
end
function _bistatic_amplitudes(sol::Union{_BistaticDirectionSolution, ComponentComparison}, angle, azimuth)
    _sample_amplitudes(sol; direction = _bem3d_incidence_direction(angle, azimuth))
end

"""
    bistatic_sweep(solution, angles; azimuth=0)

Sample an observation cut from retained BEM, MFS, meridian/shell FEM or volume FEM surface/field data,
including full-3D, coupled-region, [`free_surface`](@ref) solutions or [`components`](@ref)
comparisons. No re-solves. For axisymmetric BEM/MFS, meridian/shell FEM and full/coupled-region BEM, angles
are radians from +x toward the azimuthal direction, with π/2 along +y and 3π/2 along -y at azimuth
zero. Volume FEM and `free_surface` results instead use their own `scattering_amplitude` convention,
radians from +z with azimuth measured in the xy-plane from +x. Retain both target strength and
complex amplitude.
"""
function bistatic_sweep(
        sol::Union{_BistaticAngleAzimuthSolution, _BistaticDirectionSolution,
            ComponentComparison},
        angles::AbstractVector{<:Real};
        azimuth::Real = 0.0)
    !isempty(angles) && all(isfinite, angles) && isfinite(azimuth) ||
        throw(ArgumentError("supply finite observation angles and azimuth"))
    values = [_bistatic_amplitudes(sol, a, azimuth) for a in angles]
    amplitudes = first(values) isa Number ? ComplexF64.(values) :
                 permutedims(reduce(hcat, values))
    incidence_angle, incidence_azimuth = _bistatic_incidence(sol)
    return BistaticSweep(Float64.(angles), Float64(azimuth), target_strength.(amplitudes),
        incidence_angle, incidence_azimuth, amplitudes, _sample_labels(sol))
end
