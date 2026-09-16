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

Frequency samples in Hz, exterior `k` in inverse metres, `target_strength` in dB re
1 m², and complex `amplitudes` in metres. Arrays have samples along the first dimension;
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
