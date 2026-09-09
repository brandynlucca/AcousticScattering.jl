function _resolve_sweep(solve::Function, xs::AbstractVector{<:Real})
    ts = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ts[i] = target_strength(solve(Float64(x)))
    end
    return ts
end

"""
    FrequencySweep

Target strength [dB re 1 m²] as a function of frequency, one solve per
point. `frequencies` [Hz], `k` [rad/m] (`k = 2π*frequencies/sound_speed`),
`target_strength` [dB re 1 m²].
"""
struct FrequencySweep
    frequencies::Vector{Float64}
    k::Vector{Float64}
    target_strength::Vector{Float64}
end

"""
    frequency_sweep(solve::Function, frequencies::AbstractVector{<:Real}, sound_speed::Real)

Re-solve `solve(k)` at each of `frequencies` [Hz], converting to
wavenumber via `sound_speed` [m/s], and collect `target_strength`.
`solve` is a caller-supplied closure returning an [`AbstractSolution`](@ref)
for a given wavenumber `k`, e.g. `k -> bem(Sphere(0.01), Rigid(), k)`, so
this works for any of `modal`, `kirchhoff`, `bem`, `mfs`, or `fem` (with
their other arguments captured by the closure).

# Examples
```julia
sweep = frequency_sweep(k -> modal(Sphere(0.01), Rigid(), k), 10e3:1e3:100e3, 1477.4)
```
"""
function frequency_sweep(
        solve::Function, frequencies::AbstractVector{<:Real}, sound_speed::Real)
    k = 2π .* Float64.(frequencies) ./ Float64(sound_speed)
    ts = _resolve_sweep(solve, k)
    return FrequencySweep(Float64.(frequencies), k, ts)
end

"""
    IncidenceAngleSweep

Target strength [dB re 1 m²] as a function of incidence angle [rad], one
solve per point, since the incidence angle changes the linear system
being solved rather than being a post-solve evaluation.
"""
struct IncidenceAngleSweep
    angles::Vector{Float64}
    target_strength::Vector{Float64}
end

"""
    incidence_angle_sweep(solve::Function, angles::AbstractVector{<:Real})

Re-solve `solve(incidence_angle)` at each of `angles` [rad] and collect
`target_strength`. `solve` is a caller-supplied closure returning an
[`AbstractSolution`](@ref) for a given incidence angle, e.g.
`angle -> bem(body, Rigid(), k; incidence_angle=angle)`.

# Examples
```julia
sweep = incidence_angle_sweep(
    angle -> modal(Sphere(0.01), Rigid(), k; incidence_angle=angle), 0:0.1:pi)
```
"""
function incidence_angle_sweep(solve::Function, angles::AbstractVector{<:Real})
    ts = _resolve_sweep(solve, angles)
    return IncidenceAngleSweep(Float64.(angles), ts)
end

const _BistaticAngleAzimuthSolution = Union{
    BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData},
    FEMSolution{_ShellFEMSurfaceData}}

"""
    BistaticSweep

Target strength [dB re 1 m²] vs. observation polar angle `angles` [rad] at
a fixed observation `azimuth` [rad], from an already-solved bistatic-
capable solution, no re-solve. `incidence_angle` [rad] is the solution's
own solve-time incidence angle.
"""
struct BistaticSweep
    angles::Vector{Float64}
    azimuth::Float64
    target_strength::Vector{Float64}
    incidence_angle::Float64
end

"""
    bistatic_sweep(sol::AbstractSolution, angles::AbstractVector{<:Real}; azimuth=0.0)

Evaluate `target_strength(sol; angle, azimuth)` at each of `angles` [rad]
without re-solving. Supported for axisymmetric [`bem`](@ref)/[`mfs`](@ref)
results and shell [`fem`](@ref) results, since only those store the
per-mode surface data this needs; other solution types have no reusable
bistatic surface data and error via the same `target_strength` methods
they already define.
"""
function bistatic_sweep(
        sol::_BistaticAngleAzimuthSolution, angles::AbstractVector{<:Real}; azimuth::Real = 0.0)
    ts = [target_strength(sol; angle = a, azimuth = azimuth) for a in angles]
    return BistaticSweep(Float64.(angles), Float64(azimuth), ts, sol.data.incidence_angle)
end

"""
    BistaticMap

Target strength [dB re 1 m²] over a `(theta, phi)` observation grid, an
`length(thetas) × length(phis)` matrix with `target_strength[i, j]` the
value at `(thetas[i], phis[j])`.
"""
struct BistaticMap
    thetas::Vector{Float64}
    phis::Vector{Float64}
    target_strength::Matrix{Float64}
end

"""
    bistatic_map(sol::AbstractSolution, thetas::AbstractVector{<:Real}, phis::AbstractVector{<:Real})

Evaluate target strength over the full `(theta, phi)` observation grid
without re-solving. Supported for axisymmetric [`bem`](@ref)/[`mfs`](@ref)
results, shell [`fem`](@ref) results, and full 3D [`bem`](@ref) results
(`method=:full`); other solution types have no reusable bistatic surface
data.
"""
function bistatic_map(sol::_BistaticAngleAzimuthSolution,
        thetas::AbstractVector{<:Real}, phis::AbstractVector{<:Real})
    ts = [target_strength(sol; angle = t, azimuth = p) for t in thetas, p in phis]
    return BistaticMap(Float64.(thetas), Float64.(phis), ts)
end

function bistatic_map(sol::BEMSolution{_FullBEMSurfaceData},
        thetas::AbstractVector{<:Real}, phis::AbstractVector{<:Real})
    ts = [target_strength(sol; direction = _bem3d_incidence_direction(t, p))
          for t in thetas, p in phis]
    return BistaticMap(Float64.(thetas), Float64.(phis), ts)
end
