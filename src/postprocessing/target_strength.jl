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
