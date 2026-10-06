# Bent-cylinder modal series (BCMS). Applies a curvature-modified
# equivalent coherent length correction on top of the straight-cylinder finite-cylinder modal series.

using QuadGK: quadgk

"""
    equivalent_length_fresnel(k1, length, radius_curvature)

Curvature-modified equivalent coherent length `L_ebc(k1)` [m, complex] for
a uniformly bent finite cylinder of the given straight-line `length` [m]
and (constant) `radius_curvature` [m], using Stanton's (1989) Fresnel-integral
reduction, used by [`bcms_target_strength`](@ref):

```math
L_{ebc}(k_1) = \\int_{-L/2}^{L/2} \\exp\\!\\left[i \\frac{8k_1 z_{max}}{L^2} x^2\\right] dx,
\\qquad z_{max} = \\rho_c\\left(1 - \\cos\\frac{L}{2\\rho_c}\\right).
```

`radius_curvature = Inf` (straight cylinder) returns `length` exactly, not merely in a numerical limit, since evaluating the formula literally at
infinite curvature is a `0×Inf` indeterminate form (`z_max → Inf×0`).
"""
function equivalent_length_fresnel(k1::Real, length::Real, radius_curvature::Real)
    isinf(radius_curvature) && return complex(length)
    γ_max = length / (2radius_curvature)
    z_max = radius_curvature * (1 - cos(γ_max))
    coeff = 8k1 * z_max / length^2
    val, _ = quadgk(x -> cis(coeff * x^2), -length / 2, length / 2; rtol = 1e-10)
    return val
end

"""
    bcms_target_strength(boundary::Union{Rigid,PressureRelease,FluidFilled}, k, radius, length; aspect_angle=π/2, radius_curvature=Inf, m_max=default)

Bent-cylinder modal series (BCMS) target strength [dB re 1 m²]: the
finite-length approximation in [`form_function`](@ref) for the straight-cylinder
cross-sectional physics, combined with the [`equivalent_length_fresnel`](@ref)
bent-axis coherence correction (Stanton 1988, 1989; Stanton, Chu, Wiebe & Clay 1993).
The bent-axis correction is derived near broadside
incidence, and a warning is raised if `aspect_angle` departs from `π/2` by
more than 10° while bent (`radius_curvature` finite), since the correction
is not expected to hold accurately away from broadside.
"""
function bcms_target_strength(boundary::Union{Rigid, PressureRelease, FluidFilled},
        k::Real, radius::Real, length::Real;
        aspect_angle::Real = π / 2, radius_curvature::Real = Inf,
        m_max::Integer = _default_mode_count(k * sin(aspect_angle) * radius))
    f_straight = form_function(
        boundary, k, radius, length; aspect_angle = aspect_angle, m_max = m_max)

    if isinf(radius_curvature)
        return target_strength(f_straight)
    end
    if abs(aspect_angle - π / 2) > π / 18
        @warn "BCMS bent-cylinder correction is intended for broadside or near-broadside incidence."
    end
    Lebc = equivalent_length_fresnel(k, length, radius_curvature)
    return target_strength(Lebc * f_straight / length)
end
