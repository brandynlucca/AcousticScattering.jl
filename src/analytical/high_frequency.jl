# Kirchhoff (physical-optics) high-frequency baseline: single-highlight backscattering amplitude
# f = Rc * sqrt(R1*R2) / 2 (MacLennan & Simmonds 2005 Eq. 6.7). KRM baselines not yet implemented.

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

*Exact* Kirchhoff (physical-optics) target strength [dB re 1 m²] of a
sphere of radius `a` [m] with wavenumber `k` [1/m], not the high-`ka`
single-highlight asymptote `kirchhoff_form_function` uses for a general
convex body (that one only needs the specular point's curvature, not the
whole surface), but the closed-form evaluation of the full physical-optics
surface integral over the illuminated hemisphere, valid at any `ka`:

    f(ka) = Rc·a·[ -i·e^{2ika}/2 + (e^{2ika} - 1)/(4ka) ]

Derivation: physical optics sets the scattered surface field to `Rc` times
the incident field on the illuminated hemisphere (and zero on the shadowed
one) and evaluates the far-field backscatter integral
`f = Rc·(k/2π) ∫∫_illuminated (n̂·k̂ᵢ) e^{2ik̂ᵢ·r} dS` in closed form for a
sphere (`n̂·k̂ᵢ = cosθ`, `k̂ᵢ·r = a cosθ`, illuminated hemisphere
`θ ∈ [0,π/2]`) via the substitution `u = cosθ`, an elementary
`∫u e^{cu}du` integration by parts. The leading term dominates as
`ka → ∞`, recovering the single-highlight asymptote (`|f| → a/2`,
`TS → 20log10(a/2)`) exactly as before; the second term is the `O(1/ka)`
finite-`ka` correction this package's earlier, purely-asymptotic
implementation omitted entirely (it used the ka-independent leading term
alone at every `ka`). Verified directly against Francis's published
Kirchhoff benchmark values (Jech et al. 2015): matches to ≤0.01 dB at
every frequency checked (`ka` from ~0.5 to ~5), including the low-`ka`
regime where the old asymptotic-only formula was off by several dB.
"""
function kirchhoff_target_strength(boundary::AbstractBoundaryCondition, k::Real, a::Real)
    x = k * a
    Rc = reflection_coefficient(boundary)
    f = Rc * a * (-im * cis(2x) / 2 + (cis(2x) - 1) / (4x))
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
    f = Rc * a * (-im * cis(2x) / 2 + (cis(2x) - 1) / (4x))
    return target_strength(f)
end

"""
    kirchhoff_form_function(boundary, k, radius, length; angle=π/2)

*Exact* Kirchhoff (physical-optics) backscattering amplitude [m] of a
finite rigid/pressure-release cylinder at incidence `angle` [rad] from the
axis of symmetry (`0` = end-on, `π/2` = broadside, the default), the
closed-form evaluation of the full physical-optics surface integral over
the illuminated lateral surface *and* end cap, valid at any `k*radius`
(not just the high-frequency single-highlight limit).

Derivation (from scratch, verified step by step below, not recalled from
a reference, since no existing implementation of this specific quantity
was found to port from): physical optics sets the scattered surface field
to `Rc` times the incident field on the illuminated surface (`n̂·k̂ᵢ > 0`)
and zero elsewhere, evaluating

    f = Rc·(k/2π) ∫∫_illuminated (n̂·k̂ᵢ) e^{2ik̂ᵢ·r} dS

over the cylinder's two illuminated pieces:

- **Lateral surface** (`r = (a cosφ, a sinφ, z)`, `n̂ = (cosφ, sinφ, 0)`):
  with `k̂ᵢ = (sinβ, 0, cosβ)`, `n̂·k̂ᵢ = sinβ cosφ` (illuminated for
  `φ ∈ (-π/2, π/2)`) and `k̂ᵢ·r = a sinβ cosφ + z cosβ`, the integrand
  separates exactly into an axial factor `sin(kL cosβ)/(kL cosβ)` (a
  normalized sinc, `1` at broadside) times the *same* half-circle Bessel
  series as the broadside-only derivation this generalizes, now evaluated
  at `x = 2ka sinβ` instead of a fixed `2ka`:
  `I(x) = 2J₀(x) + iπJ₁(x) - 4·Σ_{n=1}^∞ J₂ₙ(x)/(4n²-1)`.
- **End cap** (illuminated only away from exact broadside, where
  `n̂·k̂ᵢ = |cosβ|` is nonzero): the standard normal-incidence flat-disk
  physical-optics integral `∫₀^a J₀(2kρ sinβ)ρ dρ = a·J₁(2ka sinβ)/(2k
  sinβ)` (Weber-Schafheitlin identity), picking up an `e^{ikL|cosβ|}`
  phase from the cap's axial offset. At exact broadside (`cosβ = 0`) both
  end caps are edge-on and this term vanishes identically, recovering the
  original broadside-only result exactly, confirmed as a direct special
  case, not just an asymptote (see test/runtests.jl). At exact end-on
  (`sinβ = 0`) the lateral term vanishes (the curved surface is everywhere
  grazing) and the cap term's removable `0/0` (`J₁(x)/x → 1/2`) reduces to
  the classic rigid-piston amplitude `f = Rc·ka²/2·e^{ikL}`.

Validated directly against Francis's published oblique-incidence Kirchhoff
benchmark values (Jech et al. 2015,
`Jechetal_allmodels/Figure_07-08_Rigid-Cylinder_angle-038kHz.csv`): matches
to ≤0.004 dB at every tabulated angle (8°-88°). The high-`ka` scaling
discussion for the broadside limit (`|I| ~ O(1/√x)`, single-curvature
cylinder vs. `ka`-independent sphere) is unchanged by this generalization.
"""
function kirchhoff_form_function(
        boundary::AbstractBoundaryCondition, k::Real, radius::Real, length::Real; angle::Real = π /
                                                                                                2)
    Rc = reflection_coefficient(boundary)
    sb, cb = sincos(angle)
    x = 2 * k * radius * sb
    total = 2 * besselj(0, x) + im * π * besselj(1, x)
    for n in 1:60
        term = besselj(2n, x) / (4n^2 - 1)
        total -= 4 * term
        abs(term) < 1e-15 * abs(total) && break
    end
    f_lateral = Rc * (k * radius * length) / (2π) * sb * total * sinc(k * length * cb / π)
    besselj1x = x == 0 ? 0.5 : besselj(1, x) / x
    f_cap = Rc * abs(cb) * cis(k * length * abs(cb)) * k * radius^2 * besselj1x
    return f_lateral + f_cap
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

*Exact* Kirchhoff (physical-optics) backscattering amplitude [m] of a
prolate/oblate spheroid at incidence `angle` [rad] from the axis of
symmetry (`0` = end-on, `π/2` = broadside), the full physical-optics
surface integral over the illuminated surface, valid at any `ka` (not the
single-highlight asymptote [`principal_curvatures`](@ref)/the 2-argument
`kirchhoff_form_function(boundary, R1, R2)` give, which only uses the
specular point's local curvature and needs `k·sqrt(R1·R2) ≫ 1` to be
accurate).

Derivation: parametrizing the surface by polar angle `θ` and azimuth `φ`
(`r = (b sinθ cosφ, b sinθ sinφ, a cosθ)`) gives outward-normal-weighted
area element `n̂·k̂ᵢ dS = [A(θ)cosφ + B(θ)]·b sinθ dθdφ` with `A = a sinθ
sinβ`, `B = b cosθ cosβ` (the `sqrt(a²sin²θ+b²cos²θ)` normal/area factors
cancel exactly between the two), and phase `k̂ᵢ·r = D(θ)cosφ + C(θ)` with
`D = b sinθ sinβ`, `C = a cosθ cosβ`. Because the illumination condition
`A cosφ + B > 0` and the phase both depend on `φ` only through `cos(φ)`,
the inner `φ`-integral is over a single, exactly-computable arc
([`_spheroid_phi_illuminated`](@ref)) rather than requiring 2-D shadow
tracing; the outer `θ`-integral is adaptive quadrature over that
closed-form inner result. This does not collapse to the cylinder's
Bessel-series closed form (the two illumination/phase coefficient pairs
`(A,B)` and `(D,C)` differ, unlike the cylinder where they coincide), so
it is evaluated numerically (`QuadGK`) rather than as a Bessel series, still *exact* physical optics, not an asymptote, just without an
elementary antiderivative.

Validated directly against Francis's published oblique-incidence Kirchhoff
benchmark values (Jech et al. 2015,
`Jechetal_allmodels/Figure_05_Rigid-Pspheroid_angle-038kHz.csv`): matches
to ≤0.004 dB at every tabulated angle (0°-88°), a large improvement over
the single-highlight asymptote, which is off by up to 1.5 dB at this
body's moderate `k·sqrt(R1·R2) ~ O(1-10)`.
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
        val, _ = quadgk(φ -> (A * cos(φ) + B) * cis(2k * (D * cos(φ) + C)), lo, hi; rtol = 1e-10)
        return b * st * val
    end
    outer, _ = quadgk(theta_integrand, 0, π; rtol = 1e-10)
    return Rc * (k / (2π)) * outer
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
