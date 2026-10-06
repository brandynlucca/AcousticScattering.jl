"""
    target_strength(f)

Convert a scattering amplitude/length `f` [m] to target strength in dB
re 1 m². Solver-specific `target_strength(scatterer, ...)` methods build
`f` and dispatch here for the final conversion.
"""
target_strength(f::Number) = 20 * log10(abs(f))

"""
    backscattering_cross_section(f)

σ_bs = |f|² [m²].
"""
backscattering_cross_section(f::Number) = abs2(f)

"""
    target_strength(solution::AbstractSolution; kwargs...)

Return target strength in dB re 1 m². Where complex amplitude is retained, this is
`20 * log10(abs(scattering_amplitude(solution; kwargs...)))`, with the same observation
keywords and backscatter defaults. Scalar-only FEM paths return their stored target strength.
Observation support follows the underlying amplitude method; sphere radial/meridian
and scalar-only FEM paths reject observation keywords.
"""
function target_strength(sol::AbstractSolution; kwargs...)
    return target_strength(scattering_amplitude(sol; kwargs...))
end
