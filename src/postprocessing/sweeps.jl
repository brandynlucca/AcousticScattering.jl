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
re-solving. Supported for axisymmetric and full-3D BEM/MFS, coupled-region BEM,
and shell FEM solutions with retained surface data.
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

"""
    IncidenceAngleSweep

Incidence `angles` in radians and sampled `target_strength`, `amplitudes`, and `labels`,
with the same layout and units as [`FrequencySweep`](@ref). Observation follows
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

"""
    incidence_angle_sweep(surface::Mesh, boundary::Union{Rigid,PressureRelease}, k, angles; kwargs...)

Sample full-3D BEM backscatter at fixed exterior wavenumber `k` in inverse meters.
Polar `angles` are radians from +x, with fixed `incidence_azimuth=0` by default.
Reuse layer operators, their compression and the system operator within this call.
Each angle gets an independent GMRES solve with the supplied tolerances.

Accepts `formulation`, `compression`, `correction` and `gmres_kwargs` as in [`bem`](@ref).
Returns an `IncidenceAngleSweep` retaining amplitudes and strengths. Operators are discarded
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
The multiple-interface overload also accepts `parents` and `validation`. With
`components=true`, include each interface isolated in the exterior medium and their
coherent complex sum, using the same solver options. `labels` supplies one name per
interface. Each system is sampled separately to limit retained matrix storage.

Returns an `IncidenceAngleSweep` containing complex amplitudes and target strengths. Dense
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
        equilibrate::Bool = true, condition_limit::Integer = 512)
    _validate_incidence_sweep(angles, incidence_azimuth)
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    system = _assemble_full_fluid(material, k, surface.data; formulation, correction)
    factor = _factor_fluid_system(system.A; equilibrate, condition_limit)
    return incidence_angle_sweep(angles) do incidence_angle
        p, q, quad, report = _solve_full_fluid(system, factor;
            incidence_angle, incidence_azimuth)
        _full_bem_solution(surface, material, k, p, q, quad, report,
            incidence_angle, incidence_azimuth)
    end
end

function _region_angle_sweep(surfaces, materials, k, angles;
        incidence_azimuth, equilibrate, condition_limit, kwargs...)
    system = _assemble_region_bem(surfaces, materials, k; kwargs...)
    factor = _factor_fluid_system(system.A; equilibrate, condition_limit,
        norm_floor = eps(Float64))
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
        condition_limit::Integer = 512, validation::NamedTuple = (;))
    _validate_incidence_sweep(angles, incidence_azimuth)
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    components || labels === nothing ||
        throw(ArgumentError("labels require components=true"))
    response_labels = components ? _component_labels(length(surfaces), labels) : nothing
    options = (; incidence_azimuth, formulation, correction, equilibrate, condition_limit)
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
[`fourier`](@ref). Returns an `IncidenceAngleSweep`.
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
reused for every angle. Accepts `m_max`, `n_max` and `check` as in [`form_function`](@ref).
"""
function incidence_angle_sweep(body::Spheroid, boundary::_ElasticSpheroidBoundary,
        k::Real, angles::AbstractVector{<:Real};
        m_max::Integer = _default_elastic_orders(boundary, k, body),
        n_max::Integer = _default_elastic_orders(boundary, k, body), check::Bool = true)
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
        ModalSolution(body, boundary, k, amplitude)
    end
    guarded && _warn_elastic_truncation(change[])
    return sweep
end

const _BistaticAngleAzimuthSolution = Union{
    BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData},
    FEMSolution{_ShellFEMSurfaceData}}
const _BistaticDirectionSolution = Union{BEMSolution{_FullBEMSurfaceData},
    BEMSolution{_RegionBEMData}, MFSSolution{_FullMFSSurfaceData}}

"""
    BistaticSweep

Observation `angles` and fixed `azimuth` in radians, sampled `target_strength` and
complex `amplitudes`, and solve-time `incidence_angle` and `incidence_azimuth` in radians.
Arrays use the same layout and units as [`FrequencySweep`](@ref).
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

Sample an observation cut from retained BEM, MFS or shell FEM surface data, including
full-3D and coupled-region solutions or [`components`](@ref) comparisons. No re-solves.
Angles are radians from +x toward the azimuthal direction; at azimuth zero, π/2 points
along +y and 3π/2 along -y. Retain both target strength and complex amplitude.
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
    data = sol isa ComponentComparison ? sol.coupled.data : sol.data
    incidence_azimuth = hasproperty(data, :incidence_azimuth) ? data.incidence_azimuth : 0.0
    return BistaticSweep(Float64.(angles), Float64(azimuth), target_strength.(amplitudes),
        data.incidence_angle, incidence_azimuth, amplitudes, _sample_labels(sol))
end
