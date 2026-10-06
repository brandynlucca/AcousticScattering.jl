"""
    bent_cylinder_kirchhoff_form_function(boundary::AbstractBoundaryCondition, k, radius, length, radius_curvature; aspect_angle=π/2)

*Exact* Kirchhoff (physical-optics) backscattering amplitude [m] of a
uniformly bent finite cylinder, a circular tube of `radius` [m] swept
along a circular arc of length `length` [m] and constant
`radius_curvature` [m], at incidence `aspect_angle` [rad] measured from
the chord direction, *in the plane of the bend* (`0` = end-on, `π/2` =
broadside/near-broadside; matching [`bcms_target_strength`](@ref)'s own
angle convention, which this is meant to independently cross-check at
high `ka`). No end caps: matches the "no cap scattering" scope of the
straight-cylinder modal series [`form_function`](@ref)`(::Union{Rigid,
PressureRelease,FluidFilled}, k, radius, length; ...)` this is meant to
complement, not a separate limitation introduced here.

Derivation: parametrize the centerline `r_c(s) = ρc(sin γ, 1-cos γ, 0)`,
`γ = s/ρc`, `s ∈ [-L/2, L/2]` (reducing to a straight axis along `x` as
`ρc → ∞`), with local circular cross-section `r(s,φ) = r_c(s) +
radius·(cosφ, 0, 0)`-rotated-into the plane normal to the local tangent.
Writing incidence `k̂ᵢ = (cosβ, sinβ, 0)` (in-plane, matching the bend's
own plane), the illuminated-region and phase coefficients collapse to a
single combination `A(s) = sin(β-γ)` (illumination `A·cosφ > 0`, handled
by the same [`_spheroid_phi_illuminated`](@ref) single-lobe logic the
spheroid Kirchhoff integral uses, with the constant term `B = 0`
identically here since incidence lies entirely in the bend's own plane)
and phase `2k[D(s)cosφ + C(s)]` with `D = -radius·A(s)`. Unlike the
spheroid case, the surface Jacobian itself depends on `φ` here (a genuine
torus-like effect: `dS = radius·(1 + (radius/ρc)cosφ)\\,ds\\,dφ`, the
outer edge of a bend sweeping more arc length than the inner edge), so the
inner `φ`-integral picks up a `cos²φ` term beyond the plain `cosφ` term
spheroid's own derivation reduces to a closed form, both are evaluated
by the same adaptive `QuadGK` strategy, with no departure from that
established pattern.

The weight in the integral is `-dot(n,d)` on the source-facing surface
`dot(n,d) < 0`. The phase is `exp(2im*k*dot(d,r))`, with prefactor
`-im*k*Rc/(2pi)`, consistent with `exp(-im*omega*t)` and outgoing waves.

As the curvature radius increases, the amplitude approaches the lateral-surface term
of the straight finite-cylinder Kirchhoff result. End-cap scattering is excluded.
"""
function bent_cylinder_kirchhoff_form_function(
        boundary::AbstractBoundaryCondition, k::Real, radius::Real, length::Real,
        radius_curvature::Real; aspect_angle::Real = π / 2)
    Rc = reflection_coefficient(boundary)
    a = radius
    ρc = radius_curvature
    β = aspect_angle
    cβ, sβ = cos(β), sin(β)

    function s_integrand(s::Real)
        γ = s / ρc
        A = sin(β - γ)
        bounds = _spheroid_phi_illuminated(A, 0.0)
        bounds === nothing && return 0.0 + 0.0im
        D = -a * A
        C = ρc * (cβ * sin(γ) + sβ * (1 - cos(γ)))
        lo, hi = bounds
        val, _ = quadgk(
            φ -> (A * cos(φ) + (A * a / ρc) * cos(φ)^2) * cis(2k * (D * cos(φ) + C)),
            lo, hi; rtol = 1e-10)
        return a * val
    end
    outer, _ = quadgk(s_integrand, -length / 2, length / 2; rtol = 1e-10)
    return -im * Rc * (k / (2π)) * outer
end

"""
    bent_cylinder_kirchhoff_target_strength(boundary, k, radius, length, radius_curvature; aspect_angle=π/2)

Target strength [dB re 1 m²] from [`bent_cylinder_kirchhoff_form_function`](@ref).
"""
function bent_cylinder_kirchhoff_target_strength(
        boundary::AbstractBoundaryCondition, k::Real, radius::Real, length::Real,
        radius_curvature::Real; aspect_angle::Real = π / 2)
    return target_strength(bent_cylinder_kirchhoff_form_function(
        boundary, k, radius, length, radius_curvature; aspect_angle = aspect_angle))
end
