# Bent-cylinder modal series (BCMS), Stanton (1988, 1989). Applies a curvature-modified
# equivalent coherent length correction on top of cylinder_modal.jl's finite-cylinder modal series.

using QuadGK: quadgk

"""
    equivalent_length_fresnel(k1, length, radius_curvature)

Curvature-modified equivalent coherent length `L_ebc(k1)` [m, complex] for
a uniformly bent finite cylinder of the given straight-line `length` [m]
and (constant) `radius_curvature` [m], Stanton (1989)'s Fresnel-integral
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
*exact* finite-cylinder modal series this package already implements
(`cylinder_modal.jl`'s [`form_function`](@ref)) for the straight-cylinder
cross-sectional physics, combined with the [`equivalent_length_fresnel`](@ref)
bent-axis coherence correction, Stanton (1988, 1989); Stanton, Chu, Wiebe
& Clay (1993). The bent-axis correction itself is derived near broadside
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

Derivation: parametrize the centerline `r_c(s) = ρc(sin γ, 0, 1-cos γ)`,
`γ = s/ρc`, `s ∈ [-L/2, L/2]` (reducing to a straight axis along `x` as
`ρc → ∞`), with local circular cross-section `r(s,φ) = r_c(s) +
radius·(cosφ, 0, 0)`-rotated-into the plane normal to the local tangent.
Writing incidence `k̂ᵢ = (cosβ, 0, sinβ)` (in-plane, matching the bend's
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

`radius_curvature = Inf` reduces to (numerically) the *lateral* portion of
the existing straight finite-cylinder Kirchhoff formula, `high_frequency.jl`'s
`kirchhoff_form_function(boundary, k, radius, length; angle)`'s `f_lateral`
term, confirmed directly in `test/runtests.jl`, since this function has
no separate straight-line closed form of its own to fall back on.

This exact-limit check is what caught two real sign errors during
derivation, neither obvious from inspection alone: the by-hand Frenet-frame
algebra for "outward" first produced an *inward* normal (amplitude matched
the reference exactly in magnitude but with the opposite overall sign at
every angle tested, invisible in target strength, since `|-f| = |f|`, but
wrong), and after fixing that, the phase exponent's sign was still the
complex conjugate of the reference at every angle (matching real parts,
negated imaginary parts), a `e^{+i(...)}` vs. `e^{-i(...)}` convention
mismatch, fixed by negating the phase argument. Both were confirmed fixed
by the relative error shrinking from `O(1)` to `O(10^{-6})` at
`radius_curvature = 10^6·length` and `O(10^{-9})` at `10^8·length`, genuine convergence to the reference, not a coincidence at one scale.
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
            φ -> (A * cos(φ) + (A * a / ρc) * cos(φ)^2) * cis(-2k * (D * cos(φ) + C)),
            lo, hi; rtol = 1e-10)
        return a * val
    end
    outer, _ = quadgk(s_integrand, -length / 2, length / 2; rtol = 1e-10)
    return Rc * (k / (2π)) * outer
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

# --- Bent-cylinder MFS: no symmetry axis, so works directly in 3D Cartesian coordinates ---

function _bent_cylinder_point(s::Real, φ::Real, radius::Real, radius_curvature::Real)
    γ = s / radius_curvature
    sγ, cγ = sincos(γ)
    cφ, sφ = sincos(φ)
    x = (radius_curvature + radius * cφ) * sγ
    y = radius * sφ
    z = radius_curvature * (1 - cγ) - radius * cφ * cγ
    return (x, y, z)
end

function _bent_cylinder_normal(s::Real, φ::Real, radius_curvature::Real)
    γ = s / radius_curvature
    sγ, cγ = sincos(γ)
    cφ, sφ = sincos(φ)
    # Outward normal = (r(s,φ) - r_c(s))/radius, r_c the centerline (`_bent_cylinder_point` at radius=0).
    return (cφ * sγ, sφ, -cφ * cγ)
end

"""
    bent_cylinder_mfs_points(radius, length, radius_curvature, n_s, n_φ)

Collocation points, outward normals, and surface-element areas for a
uniformly bent finite cylinder (see [`bent_cylinder_kirchhoff_form_function`](@ref)
for the geometry), on an `n_s × n_φ` grid over `s ∈ [-length/2, length/2]`
(midpoint rule) and `φ ∈ [0, 2π)` (uniform, periodic). Returns
`(points, normals, areas)`, each a `Vector` of length `n_s*n_φ`.
"""
function bent_cylinder_mfs_points(
        radius::Real, length::Real, radius_curvature::Real, n_s::Integer, n_φ::Integer)
    ds = length / n_s
    dφ = 2π / n_φ
    points = Vector{NTuple{3, Float64}}(undef, n_s * n_φ)
    normals = Vector{NTuple{3, Float64}}(undef, n_s * n_φ)
    areas = Vector{Float64}(undef, n_s * n_φ)
    idx = 1
    for i in 1:n_s
        s = -length / 2 + (i - 0.5) * ds
        for j in 1:n_φ
            φ = (j - 0.5) * dφ
            points[idx] = _bent_cylinder_point(s, φ, radius, radius_curvature)
            normals[idx] = _bent_cylinder_normal(s, φ, radius_curvature)
            areas[idx] = radius * (1 + (radius / radius_curvature) * cos(φ)) * ds * dφ
            idx += 1
        end
    end
    return points, normals, areas
end

_dot3(a::NTuple{3, <:Real}, b::NTuple{3, <:Real}) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
_sub3(a::NTuple{3, <:Real}, b::NTuple{3, <:Real}) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
_norm3(a::NTuple{3, <:Real}) = sqrt(_dot3(a, a))

function _green3d(k::Real, x::NTuple{3, <:Real}, y::NTuple{3, <:Real})
    cis(k * _norm3(_sub3(x, y))) / (4π * _norm3(_sub3(x, y)))
end
function _dgreen3d_dn(k::Real, x::NTuple{3, <:Real}, nx::NTuple{3, <:Real}, y::NTuple{
        3, <:Real})
    r = _norm3(_sub3(x, y))
    G = cis(k * r) / (4π * r)
    return (im * k - 1 / r) * G * _dot3(nx, _sub3(x, y)) / r
end

"""
    solve_bent_cylinder_mfs(boundary::Union{Rigid,PressureRelease}, k, radius, length, radius_curvature; aspect_angle=π/2, offset, n_s=20, n_φ=16)

3D (non-axisymmetric) MFS solution for a unit-amplitude plane wave
`e^{ik k̂ᵢ·x}`, `k̂ᵢ = (cos(aspect_angle), 0, sin(aspect_angle))` (in the
bend's own plane, matching [`bent_cylinder_kirchhoff_form_function`](@ref)'s
convention), scattering off a uniformly bent finite cylinder. One source
per collocation point (see [`bent_cylinder_mfs_points`](@ref)), offset
`offset` [m] inward along that point's own normal, the same simplest
MFS scheme [`solve_axial_mfs`](@ref) uses for canonical axisymmetric
bodies, here without any azimuthal reduction since none is available.
Returns `(p_scat, dpdn_scat, points, normals, areas)`.

**Fixed bug, previously left broken and undocumented in tests**:
`_bent_cylinder_normal` returned the *inward* normal (confirmed
algebraically, `(r(s,φ) - r_c(s))/radius` gives `(cosφ·sinγ, sinφ,
-cosφ·cosγ)`, the exact negative of what the function returned), so
sources meant to sit inside the body were instead placed outside it. This
produced a *stable but wrong* answer (a discrepancy that did not shrink
with mesh resolution, ~20-26 dB off the exact modal series regardless of
`n_s`/`n_φ`) rather than an obvious failure, consistent with solving a
nearby, still-well-posed problem with sources on the wrong side, not with
under-resolution. After the fix: `PressureRelease` converges to <0.02 dB
of the exact straight-limit modal series by `n_s=30,n_φ=24` at either
tested angle; `Rigid` converges too, but more slowly and needs a larger
offset to reach the same accuracy, e.g. at oblique incidence
(`aspect_angle=1.2`), the gap shrinks monotonically with *both* levers
independently (resolution: -1.10 dB at `n_s=40` → -0.92 dB at `n_s=100`,
fixed `offset=0.3·radius`; offset: -1.59 dB at `offset=0.1·radius` →
-0.35 dB at `offset=0.7·radius`, fixed `n_s=100,n_φ=80`), genuine, if
slower, convergence, matching this package's own established pattern that
Neumann (`Rigid`) MFS problems converge less tightly than Dirichlet
(`PressureRelease`) ones (see e.g. the axisymmetric sphere oblique MFS
testset), not a second bug.
"""
function solve_bent_cylinder_mfs(
        boundary::Union{Rigid, PressureRelease}, k::Real, radius::Real, length::Real,
        radius_curvature::Real; aspect_angle::Real = π / 2, offset::Real, n_s::Integer = 20, n_φ::Integer = 16)
    points, normals, areas = bent_cylinder_mfs_points(
        radius, length, radius_curvature, n_s, n_φ)
    n = Base.length(points)
    sources = [_sub3(points[i], offset .* normals[i]) for i in 1:n]

    k̂ᵢ = (cos(aspect_angle), 0.0, sin(aspect_angle))
    p_inc = [cis(k * _dot3(k̂ᵢ, points[i])) for i in 1:n]
    dpdn_inc = [im * k * _dot3(k̂ᵢ, normals[i]) * p_inc[i] for i in 1:n]

    P = [_green3d(k, points[i], sources[j]) for i in 1:n, j in 1:n]
    V = [_dgreen3d_dn(k, points[i], normals[i], sources[j]) for i in 1:n, j in 1:n]

    A = boundary isa PressureRelease ? P \ (-p_inc) : V \ (-dpdn_inc)
    p_scat = P * A
    dpdn_scat = V * A
    return p_scat, dpdn_scat, points, normals, areas
end

"""
    bent_cylinder_mfs_target_strength(boundary, k, radius, length, radius_curvature; aspect_angle=π/2, offset, n_s=20, n_φ=16)

Monostatic backscatter target strength [dB re 1 m²] from
[`solve_bent_cylinder_mfs`](@ref), via the standard Kirchhoff-Helmholtz
far-field reduction evaluated as a direct surface sum over the same
collocation grid (consistent with the discretization already used to
solve for `p_scat`/`dpdn_scat`, rather than a separate finer quadrature):

```math
f = \\frac{1}{4\\pi}\\sum_i \\left[ik\\,(\\hat q\\cdot\\hat n_i)\\,p_i + \\partial_n p_i\\right] e^{-ik\\hat q \\cdot r_i}\\,\\Delta S_i,
\\qquad \\hat q = -\\hat k_i.
```
"""
function bent_cylinder_mfs_target_strength(
        boundary::Union{Rigid, PressureRelease}, k::Real, radius::Real, length::Real,
        radius_curvature::Real; aspect_angle::Real = π / 2, offset::Real, n_s::Integer = 20, n_φ::Integer = 16)
    p_scat, dpdn_scat, points, normals, areas = solve_bent_cylinder_mfs(
        boundary, k, radius, length, radius_curvature;
        aspect_angle = aspect_angle, offset = offset, n_s = n_s, n_φ = n_φ)
    q̂ = (-cos(aspect_angle), 0.0, -sin(aspect_angle))
    f = zero(ComplexF64)
    for i in eachindex(points)
        f += (im * k * _dot3(q̂, normals[i]) * p_scat[i] + dpdn_scat[i]) *
             cis(-k * _dot3(q̂, points[i])) * areas[i]
    end
    return target_strength(f / (4π))
end
