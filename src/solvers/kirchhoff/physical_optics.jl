# Kirchhoff (physical-optics) high-frequency baseline: single-highlight backscattering amplitude
# f = Rc * sqrt(R1*R2) / 2. KRM baselines not yet implemented.

using QuadGK: quadgk

"""
    reflection_coefficient(boundary)

Plane-wave normal-incidence reflection coefficient used by the Kirchhoff
(physical-optics) high-frequency approximation: `+1` rigid, `-1`
pressure-release, `(gh-1)/(gh+1)` fluid/gas-filled with impedance contrast
`gh = (density_contrast * soundspeed_contrast)`.
"""
reflection_coefficient(::Rigid) = 1.0
reflection_coefficient(::PressureRelease) = -1.0
function reflection_coefficient(bc::FluidFilled)
    gh = bc.density_contrast * bc.soundspeed_contrast
    return (gh - 1) / (gh + 1)
end

"""
    kirchhoff_form_function(boundary, R1, R2)

Single-highlight Kirchhoff (physical-optics) backscattering amplitude [m]
of a smooth convex body with principal radii of curvature `R1`, `R2` [m]
at the specular point, valid in the high-frequency limit (ka ≫ 1):
f = Rc * sqrt(R1*R2) / 2.
"""
function kirchhoff_form_function(boundary::AbstractBoundaryCondition, R1::Real, R2::Real)
    return reflection_coefficient(boundary) * sqrt(R1 * R2) / 2
end

"""
    kirchhoff_target_strength(boundary::AbstractBoundaryCondition, k, a)

Kirchhoff physical-optics target strength [dB re 1 m^2] for a sphere of
radius `a` [m] and wavenumber `k` [rad/m]. The exact evaluation of the
physical-optics integral is

    f = Rc*a*(exp(-2im*k*a)/2 - im*(exp(-2im*k*a)-1)/(4k*a)).

With time convention `exp(-im*omega*t)`, incident pressure `exp(im*k*dot(d,x))`
and outgoing scattered pressure `f*exp(im*k*r)/r`, the backscatter integral is
`f = im*k*Rc/(2pi) * integral(dot(n,d)*exp(2im*k*dot(d,x)), dot(n,d) < 0)`.
The outward normal selects the source-facing hemisphere. The magnitude tends
to `abs(Rc)*a/2` at high frequency. Exact integration does not make the
physical-optics approximation exact at low frequency. See Jech et al. (2015).
"""
function kirchhoff_target_strength(boundary::AbstractBoundaryCondition, k::Real, a::Real)
    x = k * a
    Rc = reflection_coefficient(boundary)
    f = Rc * a * (cis(-2x) / 2 - im * (cis(-2x) - 1) / (4x))
    return target_strength(f)
end

"""
    reflection_coefficient(boundary::Shelled{FluidLayer}, k, a)

Plane-wave normal-incidence reflection coefficient off a *fluid* shell of
finite thickness (see [`FluidLayer`](@ref)), unlike [`reflection_coefficient`](@ref)'s
other methods, this one is frequency- and thickness-dependent, since a
finite layer's reflection is an interference of the front- and
back-surface reflections, not a single-interface impedance ratio):

    Rc = (R₁₂ + R₂₃·e^{2ik₂d}) / (1 + R₁₂R₂₃·e^{2ik₂d})

(Brekhovskikh, *Waves in Layered Media*) where `R₁₂ = (gh_shell-1)/(gh_shell+1)`
is the exterior→shell interface reflection coefficient (`gh_shell` from the
[`FluidLayer`](@ref) material), `R₂₃` is the
shell's *inner*-surface reflection coefficient (`-1`, i.e. pressure-release,
for a [`VacuumInterior`](@ref); `(gh_int-gh_shell)/(gh_int+gh_shell)` for
a [`FluidInterior`](@ref)), `k₂ = k/soundspeed_contrast` is the shell's own
wavenumber, and `d = a*(1-radius_ratio)` is the shell thickness. Reduces
exactly to the plain [`PressureRelease`](@ref)/[`FluidFilled`](@ref)
`reflection_coefficient` as `radius_ratio → 1` (vanishing shell, `d → 0`).
"""
function reflection_coefficient(bc::Shelled{FluidLayer, VacuumInterior}, k::Real, a::Real)
    return _fluid_shell_reflection_coefficient(
        bc.material.density_contrast, bc.material.soundspeed_contrast,
        -1.0, bc.radius_ratio, k, a)
end

function reflection_coefficient(bc::Shelled{FluidLayer, FluidInterior}, k::Real, a::Real)
    gh_int = bc.interior.density_contrast * bc.interior.soundspeed_contrast
    gh_shell = bc.material.density_contrast * bc.material.soundspeed_contrast
    R23 = (gh_int - gh_shell) / (gh_int + gh_shell)
    return _fluid_shell_reflection_coefficient(
        bc.material.density_contrast, bc.material.soundspeed_contrast,
        R23, bc.radius_ratio, k, a)
end

function _fluid_shell_reflection_coefficient(
        shell_density_contrast::Real, shell_soundspeed_contrast::Real,
        R23::Real, radius_ratio::Real, k::Real, a::Real)
    gh_shell = shell_density_contrast * shell_soundspeed_contrast
    R12 = (gh_shell - 1) / (gh_shell + 1)
    k2 = k / shell_soundspeed_contrast
    d = a * (1 - radius_ratio)
    phase = cis(2k2 * d)
    return (R12 + R23 * phase) / (1 + R12 * R23 * phase)
end

"""
    kirchhoff_target_strength(boundary::Shelled{FluidLayer}, k, a)

*Exact* Kirchhoff (physical-optics) target strength [dB re 1 m²] of a
fluid-shelled sphere, same closed-form physical-optics surface integral
as [`kirchhoff_target_strength`](@ref)`(::AbstractBoundaryCondition, k, a)`, using the
layer-interference reflection coefficient above in place of a
single-interface constant.
"""
function kirchhoff_target_strength(
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}}, k::Real, a::Real)
    x = k * a
    Rc = reflection_coefficient(boundary, k, a)
    f = Rc * a * (cis(-2x) / 2 - im * (cis(-2x) - 1) / (4x))
    return target_strength(f)
end

"""
    kirchhoff_form_function(boundary, k, radius, length; angle=π/2)

Kirchhoff physical-optics backscattering amplitude [m] of a finite cylinder,
including its lateral surface and illuminated flat endcap. `angle` [rad]
is measured from the symmetry axis (`0` end-on, `pi/2` broadside).

Uses `exp(-im*omega*t)` with incident propagation toward the target and
outgoing amplitude `f*exp(im*k*r)/r`. Integrating the source-facing surface
`dot(n,d) < 0` gives a lateral Bessel series times the axial sinc factor,
and an endcap disk integral. In the symmetric positive-hemisphere
parametrization, the phase is `exp(-2im*k*dot(d,x))` and the prefactor is
`-im*k*Rc/(2pi)`. At end-on incidence the result is
`-im*Rc*k*radius^2/2*exp(-im*k*length)`. At broadside the endcap term vanishes.
The series evaluates physical optics, not full Helmholtz scattering.
"""
function kirchhoff_form_function(
        boundary::AbstractBoundaryCondition, k::Real, radius::Real, length::Real; angle::Real = π /
                                                                                                2)
    Rc = reflection_coefficient(boundary)
    sb, cb = sincos(angle)
    x = 2 * k * radius * sb
    total = 2 * besselj(0, x) - im * π * besselj(1, x)
    for n in 1:60
        term = besselj(2n, x) / (4n^2 - 1)
        total -= 4 * term
        abs(term) < 1e-15 * abs(total) && break
    end
    f_lateral = Rc * (k * radius * length) / (2π) * sb * total * sinc(k * length * cb / π)
    besselj1x = x == 0 ? 0.5 : besselj(1, x) / x
    f_cap = Rc * abs(cb) * cis(-k * length * abs(cb)) * k * radius^2 * besselj1x
    return -im * (f_lateral + f_cap)
end

"""
    kirchhoff_target_strength(boundary, k, radius, length; angle=π/2)

Target strength [dB re 1 m²] from [`kirchhoff_form_function`](@ref)'s
exact finite-cylinder Kirchhoff amplitude at incidence `angle` [rad] from
the axis of symmetry (`0` = end-on, `π/2` = broadside, the default).
"""
function kirchhoff_target_strength(
        boundary::AbstractBoundaryCondition, k::Real, radius::Real, length::Real; angle::Real = π /
                                                                                                2)
    return target_strength(kirchhoff_form_function(
        boundary, k, radius, length; angle = angle))
end

"""
    principal_curvatures(body::Spheroid, angle)

Principal radii of curvature `(R_meridional, R_parallel)` [m] at the point
on `body`'s surface whose outward normal makes `angle` [rad] with the axis
of symmetry (`angle = 0` at the pole/end-on point, `π/2` at the equator).
Used by [`kirchhoff_form_function`](@ref) for the specular-point Kirchhoff
approximation (Meusnier's theorem for a surface of revolution).
"""
function principal_curvatures(body::Spheroid, angle::Real)
    a, b = body.a, body.b
    N = 1 / sqrt(sin(angle)^2 / a^2 + cos(angle)^2 / b^2)
    R_meridional = N^3 / (a * b)
    R_parallel = b * N / a
    return (R_meridional, R_parallel)
end

"""
    _spheroid_phi_illuminated(A, B)

For fixed polar angle `θ`, the spheroid surface's illumination condition
`A·cos(φ) + B > 0` (see [`kirchhoff_form_function`](@ref)`(::AbstractBoundaryCondition,
k, ::Spheroid)`) is a single-lobe condition in `φ` since it depends on `φ`
only through `cos(φ)`: fully lit (`(-π,π]`), fully dark (`nothing`), or the
symmetric arc `(-φ₀,φ₀)` with `φ₀ = acos(-B/A)`.
"""
function _spheroid_phi_illuminated(A::Real, B::Real)
    abs(A) < 1e-14 && return B > 0 ? (-π, π) : nothing
    t = -B / A
    t <= -1 && return (-π, π)
    t >= 1 && return nothing
    return (-acos(t), acos(t))
end

"""
    kirchhoff_form_function(boundary::AbstractBoundaryCondition, k, body::Spheroid; angle=π/2)

Kirchhoff physical-optics backscattering amplitude [m] of a spheroid at
polar incidence `angle` [rad] from its symmetry axis. Surface coordinates
are `(a*cos(theta), b*sin(theta)*cos(phi), b*sin(theta)*sin(phi))`.

For incident propagation `d` and outward normal `n`, integrate the
source-facing part `dot(n,d) < 0` with prefactor `im*k*Rc/(2pi)` and phase
`exp(2im*k*dot(d,x))`. Central symmetry permits integration over `dot(n,d) > 0`
instead, with phase `exp(-2im*k*dot(d,x))` and prefactor `-im*k*Rc/(2pi)`.
The implementation uses that parametrization, analytic illumination arcs
and adaptive polar/azimuthal quadrature. The result follows the package's
`exp(-im*omega*t)` and outgoing `f*exp(im*k*r)/r` convention.

This evaluates the physical-optics integral, not exact wave scattering.
For spheroid approximation-error studies, see Jech et al. (2015).
"""
function kirchhoff_form_function(
        boundary::AbstractBoundaryCondition, k::Real, body::Spheroid; angle::Real = π /
                                                                                    2)
    Rc = reflection_coefficient(boundary)
    a, b = body.a, body.b
    sb, cb = sincos(angle)
    function theta_integrand(θ::Real)
        st, ct = sincos(θ)
        A = a * st * sb
        B = b * ct * cb
        bounds = _spheroid_phi_illuminated(A, B)
        bounds === nothing && return 0.0 + 0.0im
        D = b * st * sb
        C = a * ct * cb
        lo, hi = bounds
        val, _ = quadgk(φ -> (A * cos(φ) + B) * cis(-2k * (D * cos(φ) + C)), lo, hi; rtol = 1e-10)
        return b * st * val
    end
    outer, _ = quadgk(theta_integrand, 0, π; rtol = 1e-10)
    return -im * Rc * (k / (2π)) * outer
end

"""
    kirchhoff_target_strength(boundary::AbstractBoundaryCondition, k, body::Spheroid; angle=π/2)

Target strength [dB re 1 m²] from [`kirchhoff_form_function`](@ref)'s exact
finite-spheroid Kirchhoff amplitude at incidence `angle` [rad] from the
axis of symmetry (`0` = end-on, `π/2` = broadside, the default).
"""
function kirchhoff_target_strength(
        boundary::AbstractBoundaryCondition, k::Real, body::Spheroid; angle::Real = π /
                                                                                    2)
    return target_strength(kirchhoff_form_function(boundary, k, body; angle = angle))
end
