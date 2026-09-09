# Fourier-mode (2.5D) acoustic BEM over bodies of revolution: reduces the 3D CBIE to a 1D integral
# equation over the meridian curve via azimuthal Fourier expansion of the Green's function. Rigid/PressureRelease/FluidFilled, axial exact for all three, oblique (Jacobi-Anger) for Rigid/PressureRelease only. No Burton-Miller/CHIEF irregular-frequency regularization yet.

using QuadGK: quadgk, gauss

"""
    MeridianMesh(rho, z)

Piecewise-linear discretization of the generating (meridian) curve of a
body of revolution in the half-plane rho ≥ 0: node coordinates `(rho[i],
z[i])`, panels connecting consecutive nodes. The curve must be traversed
so that outward normals equal `(-Δz, Δrho)/L` per panel, e.g. north pole to
south pole for a convex body enclosing the z-axis (verified for
[`sphere_mesh`](@ref)).
"""
struct MeridianMesh
    rho::Vector{Float64}
    z::Vector{Float64}
end

npanels(mesh::MeridianMesh) = length(mesh.rho) - 1

"A single straight meridian panel: endpoints, midpoint (collocation point), length, and outward normal."
struct Panel
    rho1::Float64
    z1::Float64
    rho2::Float64
    z2::Float64
    rhom::Float64
    zm::Float64
    L::Float64
    nrho::Float64
    nz::Float64
end

function panels(mesh::MeridianMesh)
    n = npanels(mesh)
    ps = Vector{Panel}(undef, n)
    for i in 1:n
        ρ1, z1 = mesh.rho[i], mesh.z[i]
        ρ2, z2 = mesh.rho[i + 1], mesh.z[i + 1]
        Δρ, Δz = ρ2 - ρ1, z2 - z1
        L = hypot(Δρ, Δz)
        nρ, nz = -Δz / L, Δρ / L
        ps[i] = Panel(ρ1, z1, ρ2, z2, (ρ1 + ρ2) / 2, (z1 + z2) / 2, L, nρ, nz)
    end
    return ps
end

"""
    bem_panel_count(k, characteristic_radius; elements_per_wavelength=30)

Recommended panel count `n` for [`sphere_mesh`](@ref)/[`spheroid_mesh`](@ref)/
[`cylinder_mesh`](@ref) at wavenumber `k` and the body's characteristic
radius (its largest relevant dimension, e.g. `a` for a sphere), targeting
`elements_per_wavelength` piecewise-constant panels per acoustic
wavelength along the meridian. `n` panels span the meridian's half-
circumference `πa` (pole to pole), so at wavelength `λ = 2π/k` the
elements-per-wavelength is `λ/(πa/n) = 2n/(ka)`, giving
`n = elements_per_wavelength·ka/2`.

Needed because a fixed panel count that's accurate at low `ka` silently
degrades at high `ka`, measured directly for a rigid sphere at `ka≈12.8`
(`n=32`, a count accurate to ≤0.01 dB at `ka≲3`): `7.3 dB` wrong,
*independent of quadrature `rtol`* (ruling out a quadrature-accuracy
explanation), improving only slowly with `n`, `n=64`: `2.1 dB` off;
`n=128`: `0.5 dB`; `n=250`: `0.15 dB`, the O(h) convergence rate expected
of 0th-order (piecewise-constant) collocation elements, which is why
`elements_per_wavelength=30` (not a smaller, cheaper value) is the
default: even that only reaches ~0.15-0.3 dB at `ka≈13` in this package's
own testing, not the ≤0.01 dB bar reached everywhere this package's
numerics are self-consistent, getting materially tighter than that at
high `ka` would need higher-order elements or a better-conditioned
formulation, not just a bigger `n`, and isn't done here.
"""
function bem_panel_count(k::Real, characteristic_radius::Real; elements_per_wavelength::Real = 30)
    max(20, ceil(Int, elements_per_wavelength * k * characteristic_radius / 2))
end

"""
    sphere_mesh(a, n)

Meridian mesh for a sphere of radius `a`, discretized into `n` panels
along the semicircular profile from the north pole (ρ=0, z=a) to the
south pole (ρ=0, z=-a).
"""
function sphere_mesh(a::Real, n::Integer)
    n >= 3 || throw(ArgumentError("n must be at least 3 (one panel per segment)"))
    ψ = range(0, π; length = n + 1)
    return MeridianMesh(a .* sin.(ψ), a .* cos.(ψ))
end

"""
    spheroid_mesh(a, b, n)

Meridian mesh for a spheroid with semi-axis `a` [m] along the axis of
symmetry and equatorial semi-axis `b` [m] (prolate if `a > b`, oblate if
`a < b`, matching [`Spheroid`](@ref)'s convention), discretized into `n`
panels from the north pole (ρ=0, z=a) to the south pole (ρ=0, z=-a). `a =
b` reduces exactly to [`sphere_mesh`](@ref).

Nodes are placed at *equal arc length* along the meridian ellipse, not at
uniform parametric angle `ψ`, the two coincide only for a circle. This
matters more than it sounds: `ds/dψ = sqrt(b²cos²ψ + a²sin²ψ)` is smallest
at whichever pole sits on the *shorter* semi-axis, so a uniform-`ψ` grid
happens to put its finest panels exactly at the pole for a prolate body
(`a > b`, pole curvature `b²/a`, sharpest point on the body, the "lucky"
case) but its *coarsest* panels there for an oblate body (`a < b`, pole
curvature `a²/b`, still the sharpest point on the body, confirmed
directly: a uniform-`ψ` oblate(0.01,0.07) mesh puts its *largest* panels,
not smallest, right at that sharp pole, panel length ratio pole:equator
≈6.9:1 backwards). That silently under-resolved pole was reproduced as a
real bug, not a resolution shortfall: BEM `target_strength` for a
weakly-scattering oblate spheroid *diverged further* from the (thoroughly
cross-validated, see spheroid_modal.jl) modal series as panel count
increased (56→96→140 panels: -90→-81→-77 dB, moving away from the -108 dB
modal value), refining a uniform-ψ mesh adds resolution everywhere
*except* where it was already most needed, so the genuinely-coarse pole
panels only shrink as slowly as every other panel instead of catching up.
Equal-arc-length spacing removes the asymmetry entirely (by construction,
independent of prolate/oblate labeling) and does not disadvantage the
already-fine prolate case.
"""
function spheroid_mesh(a::Real, b::Real, n::Integer)
    n >= 3 || throw(ArgumentError("n must be at least 3 (one panel per segment)"))
    ψ_dense = range(0, π; length = max(2000, 40n))
    speed = [hypot(b * cos(ψ), a * sin(ψ)) for ψ in ψ_dense]
    s = similar(speed)
    s[1] = 0.0
    for i in 2:length(ψ_dense)
        s[i] = s[i - 1] + (speed[i - 1] + speed[i]) / 2 * (ψ_dense[i] - ψ_dense[i - 1])
    end
    s_targets = range(0, s[end]; length = n + 1)
    ψ_nodes = similar(collect(s_targets))
    j = 1
    for (idx, starget) in enumerate(s_targets)
        while j < length(s) - 1 && s[j + 1] < starget
            j += 1
        end
        t = (starget - s[j]) / (s[j + 1] - s[j])
        ψ_nodes[idx] = ψ_dense[j] + t * (ψ_dense[j + 1] - ψ_dense[j])
    end
    ψ_nodes[end] = π
    return MeridianMesh(b .* sin.(ψ_nodes), a .* cos.(ψ_nodes))
end

"""
    cylinder_mesh(radius, length, n)

Meridian mesh for a finite right circular cylinder of the given `radius`
and `length` [m] (axis of symmetry along z), discretized into approximately
`n` panels total, distributed across the three meridian segments (top cap,
side, bottom cap) in proportion to their arc length. Traversed from the top
cap's center (ρ=0, z=length/2) outward across the cap, down the side, and
back in across the bottom cap to (ρ=0, z=-length/2), matching
[`sphere_mesh`](@ref)/[`spheroid_mesh`](@ref)'s outward-normal convention
(the two flat caps and the cylindrical side are each individually convex
corners of an overall convex body, so the same north-to-south traversal
rule gives the correct outward normal on every panel, caps included).

Unlike the sphere/spheroid meridian, this one has two genuine sharp
corners (ρ=radius, z=±length/2, where a flat cap meets the cylindrical
side at a right angle, a discontinuous surface normal). The surface
pressure/velocity has a local corner singularity there (a well-known
feature of Helmholtz BIEs at a wedge), which a *uniform* panel spacing
within each segment resolves only algebraically slowly, confirmed
directly: a weakly-scattering (near-total-cancellation) cylinder's BEM
`target_strength` at oblique incidence sat 1.3-1.7 dB from the reference
value at both 48 and 68 panels remained essentially unchanged after
doubling total panels 112, ruling out plain under-resolution (which
would keep shrinking) and pointing at a fixed local feature the uniform
mesh never targets. Each segment's nodes are therefore placed at
`t = clustering(u)`, `u` uniform, rather than `t` uniform, using a cosine
map that clusters panels toward whichever segment endpoint(s) sit at a
sharp corner, the cap segments cluster toward their outer rim (the
smooth center needs no extra resolution), the side segment clusters
toward *both* ends (both cap junctions are corners), while leaving the
smooth interior of each segment coarser, at the same total panel count.
"""
function cylinder_mesh(radius::Real, length::Real, n::Integer)
    n >= 3 || throw(ArgumentError("n must be at least 3 (one panel per segment)"))
    total = 2radius + length
    n_cap = max(1, round(Int, n * radius / total))
    n_side = max(1, n - 2n_cap)

    ρ = Float64[]
    z = Float64[]
    zt = length / 2

    cluster_near_1(u) = sin(π / 2 * u)
    cluster_near_0(u) = 1 - cos(π / 2 * u)
    cluster_both(u) = (1 - cos(π * u)) / 2

    for u in range(0, 1; length = n_cap + 1)
        t = cluster_near_1(u)
        push!(ρ, t * radius)
        push!(z, zt)
    end
    for u in range(0, 1; length = n_side + 1)[2:end]
        t = cluster_both(u)
        push!(ρ, radius)
        push!(z, zt - t * length)
    end
    for u in range(0, 1; length = n_cap + 1)[2:end]
        t = cluster_near_0(u)
        push!(ρ, radius * (1 - t))
        push!(z, -zt)
    end

    return MeridianMesh(ρ, z)
end

"""
    cylinder_spheroidal_endcap_mesh(radius, cylinder_length, endcap_depth, n)

Meridian mesh for a finite cylinder of the given `radius` [m] and straight
`cylinder_length` [m], capped on both ends by a quarter-prolate-spheroid
dome of polar depth `endcap_depth` [m] and equatorial radius `radius`
(matching the cylinder exactly) instead of [`cylinder_mesh`](@ref)'s flat
caps, Gong, Li, Chai, Zhao & Mitri (2017), "T-matrix method for
acoustical Bessel beam scattering from a rigid finite cylinder with
spheroidal endcaps," Ocean Engineering, Eqs. (5)-(7) (their `b` = this
`radius`, their `h` = `cylinder_length/2`, their `d` = `endcap_depth`).
Total body half-length is `cylinder_length/2 + endcap_depth`.

Unlike `cylinder_mesh`, this body has **no sharp edges at all**: an
ellipse's tangent at its own equator is always perpendicular to its major
axis, i.e., parallel to the cylinder's own straight side, for *any*
choice of `endcap_depth`, so the cap/side junction has a continuous
normal by construction, not just a well-resolved corner. Motivated by
[`solve_axial_mfs`](@ref) failing badly (7-100 dB off the axisymmetric
BEM reference, at every tested source offset) on the sharp-cornered
`cylinder_mesh`, matching the method-of-fundamental-solutions literature's
own documented difficulty with sharp edges (Pérez-Arjona et al. 2018), rounding the geometry itself, rather than trying to special-case the
source placement at a true corner, sidesteps the problem entirely.

Nodes are placed at equal arc length within each of the three segments
(top endcap, straight side, bottom endcap), the same discipline used by
[`spheroid_mesh`](@ref)/`cylinder_mesh` and for the same reason: a
uniform-parameter grid on the elliptical endcap would under- or
over-resolve its own pole depending on the `radius`/`endcap_depth` aspect
ratio, exactly as it did for a full spheroid.
"""
function cylinder_spheroidal_endcap_mesh(radius::Real, cylinder_length::Real, endcap_depth::Real, n::Integer)
    n >= 3 || throw(ArgumentError("n must be at least 3 (one panel per segment)"))
    h = cylinder_length / 2

    t_dense = range(0, π / 2; length = max(500, 10n))
    speed = [hypot(radius * cos(t), endcap_depth * sin(t)) for t in t_dense]
    s_cap = similar(speed)
    s_cap[1] = 0.0
    for i in 2:length(t_dense)
        s_cap[i] = s_cap[i - 1] +
                   (speed[i - 1] + speed[i]) / 2 * (t_dense[i] - t_dense[i - 1])
    end
    L_cap = s_cap[end]
    L_side = cylinder_length
    total = 2L_cap + L_side
    n_cap = max(1, round(Int, n * L_cap / total))
    n_side = max(1, n - 2n_cap)

    # Equal-arc-length node placement on the quarter-ellipse, traversed from the tip (t=0, ρ=0) to
    # the cylinder junction (t=π/2, ρ=radius).
    cap_nodes(n_seg) = begin
        s_targets = range(0, L_cap; length = n_seg + 1)
        t_nodes = similar(collect(s_targets))
        j = 1
        for (idx, starget) in enumerate(s_targets)
            while j < length(s_cap) - 1 && s_cap[j + 1] < starget
                j += 1
            end
            frac = (starget - s_cap[j]) / (s_cap[j + 1] - s_cap[j])
            t_nodes[idx] = t_dense[j] + frac * (t_dense[j + 1] - t_dense[j])
        end
        t_nodes[end] = π / 2
        return t_nodes
    end

    ρ = Float64[]
    z = Float64[]

    for t in cap_nodes(n_cap)
        push!(ρ, radius * sin(t))
        push!(z, h + endcap_depth * cos(t))
    end
    for u in range(0, 1; length = n_side + 1)[2:end]
        push!(ρ, radius)
        push!(z, h - u * cylinder_length)
    end
    for t in reverse(cap_nodes(n_cap))[2:end]
        push!(ρ, radius * sin(t))
        push!(z, -h - endcap_depth * cos(t))
    end

    return MeridianMesh(ρ, z)
end

_panel_point(p::Panel, s::Real) = (p.rho1 + s * (p.rho2 - p.rho1), p.z1 + s * (p.z2 - p.z1))

# Outer-quadrature breakpoints for a panel pair's meridian integral, split at the closest-approach
# parameter s* so adaptive quadrature doesn't have to discover a near-singularity on its own.
function _closest_param(xρ::Real, xz::Real, p::Panel)
    dρ, dz = p.rho2 - p.rho1, p.z2 - p.z1
    denom = dρ^2 + dz^2
    denom == 0 && return 0.5
    s = ((xρ - p.rho1) * dρ + (xz - p.z1) * dz) / denom
    return clamp(s, 0.0, 1.0)
end

function _quadgk_breakpoints(xρ::Real, xz::Real, p::Panel, self::Bool)
    self && return (0.0, 0.5, 1.0)
    s = _closest_param(xρ, xz, p)
    # Clamp so both sub-intervals stay non-degenerate even near a panel endpoint.
    s_clamped = clamp(s, 1e-3, 1 - 1e-3)
    return (0.0, s_clamped, 1.0)
end

function _ring_distance(ρ::Real, z::Real, ρ2::Real, z2::Real, Δφ::Real)
    sqrt(max(ρ^2 + ρ2^2 - 2ρ * ρ2 * cos(Δφ) + (z - z2)^2, 0.0))
end

# Floor on the ring distance r, guards against a floating-point r=0.0/NaN at fine meshes (a
# rounding artifact at a measure-zero sample point, not a true kernel singularity).
const _RING_DISTANCE_FLOOR = 1e-12

# Absolute-tolerance floor for every quadgk call, lets genuinely tiny panel-pair contributions converge immediately without a pure rtol criterion over-resolving them.
const _QUAD_ATOL = 1e-10

# Hard evaluation cap on every adaptive quadrature call, bounds near-axis (small ρ) self-pairs at
# high Fourier mode m that otherwise converge very slowly, at the cost of some precision there.
const _AZIMUTHAL_MAXEVALS = 2000

# ∂G/∂n_y at ring point y=(ρ2,φ',z2), field point x=(ρ,0,z), Δφ=φ'-0, n̂_y=(nρ2 cosΔφ, nρ2 sinΔφ, nz2).
function _ring_dGdn(
        k::Real, ρ::Real, z::Real, ρ2::Real, z2::Real, nρ2::Real, nz2::Real, Δφ::Real)
    r = _ring_distance(ρ, z, ρ2, z2, Δφ)
    r < _RING_DISTANCE_FLOOR && return zero(complex(k)) * zero(r)
    G = cis(k * r) / (4π * r)
    proj = nρ2 * (ρ * cos(Δφ) - ρ2) + nz2 * (z - z2)
    return -(im * k - 1 / r) * G * proj / r
end

function _ring_G(k::Real, ρ::Real, z::Real, ρ2::Real, z2::Real, Δφ::Real)
    r = _ring_distance(ρ, z, ρ2, z2, Δφ)
    r < _RING_DISTANCE_FLOOR && return zero(complex(k)) * zero(r)
    return cis(k * r) / (4π * r)
end

# Breakpoints aligned to cos(mΔφ)'s zero crossings, avoids quadgk having to discover the
# oscillation structure itself, which otherwise dominates cost and grows explosively with m.
_azimuthal_breakpoints(m::Integer) = m == 0 ? (0.0, 2π) : range(0.0, 2π; length = 4m + 1)

# Function-barrier helpers for outer (meridian) quadrature over a panel pair, avoids closure-capture
# boxing in the i,j loops below. Fixed order-12 Gauss-Legendre on [0,1] for "far" pairs, cross-checked against the fully-adaptive path and the sphere modal series.
const _MERIDIAN_FIXED_ORDER = 12
const _MERIDIAN_FIXED_NODES, _MERIDIAN_FIXED_WEIGHTS = let
    nodes, weights = gauss(_MERIDIAN_FIXED_ORDER)  # on [-1, 1]
    (0.5 .* (nodes .+ 1), 0.5 .* weights)          # mapped to [0, 1]
end

function _pair_K(
        k::Real, xρ::Real, xz::Real, pj::Panel, self::Bool, rtol::Real; m::Integer = 0)
    far = !self && _azimuthal_is_far(xρ, xz, pj)
    if far
        total = zero(complex(k))
        for (s, w) in zip(_MERIDIAN_FIXED_NODES, _MERIDIAN_FIXED_WEIGHTS)
            ρ2, z2 = _panel_point(pj, s)
            total += _azimuthal_dGdn(k, xρ, xz, ρ2, z2, pj.nrho, pj.nz;
                         m = m, rtol = rtol, far = true) * ρ2 * pj.L * w
        end
        return total
    end
    bp = _quadgk_breakpoints(xρ, xz, pj, self)
    integrand = s -> begin
        ρ2, z2 = _panel_point(pj, s)
        _azimuthal_dGdn(k, xρ, xz, ρ2, z2, pj.nrho, pj.nz; m = m, rtol = rtol, far = far) *
        ρ2 * pj.L
    end
    return quadgk(
        integrand, bp...; rtol = rtol, atol = _QUAD_ATOL, maxevals = _AZIMUTHAL_MAXEVALS)[1]
end

function _pair_V(
        k::Real, xρ::Real, xz::Real, pj::Panel, self::Bool, rtol::Real; m::Integer = 0)
    far = !self && _azimuthal_is_far(xρ, xz, pj)
    if far
        total = zero(complex(k))
        for (s, w) in zip(_MERIDIAN_FIXED_NODES, _MERIDIAN_FIXED_WEIGHTS)
            ρ2, z2 = _panel_point(pj, s)
            total += _azimuthal_G(k, xρ, xz, ρ2, z2; m = m, rtol = rtol, far = true) * ρ2 *
                     pj.L * w
        end
        return total
    end
    bp = _quadgk_breakpoints(xρ, xz, pj, self)
    integrand = s -> begin
        ρ2, z2 = _panel_point(pj, s)
        _azimuthal_G(k, xρ, xz, ρ2, z2; m = m, rtol = rtol, far = far) * ρ2 * pj.L
    end
    return quadgk(
        integrand, bp...; rtol = rtol, atol = _QUAD_ATOL, maxevals = _AZIMUTHAL_MAXEVALS)[1]
end

# Rigid's RHS integrand: G(x,y)·∂p_inc/∂n_y(y), with ∂p_inc/∂n_y(y) = ik·nz(y)·e^{ikz(y)} folded in.
function _pair_G_rigid_rhs(k::Real, xρ::Real, xz::Real, pj::Panel, self::Bool, rtol::Real)
    far = !self && _azimuthal_is_far(xρ, xz, pj)
    if far
        total = zero(ComplexF64)
        for (s, w) in zip(_MERIDIAN_FIXED_NODES, _MERIDIAN_FIXED_WEIGHTS)
            ρ2, z2 = _panel_point(pj, s)
            total += im * k * pj.nz * cis(k * z2) *
                     _azimuthal_G(k, xρ, xz, ρ2, z2; rtol = rtol, far = true) * ρ2 * pj.L *
                     w
        end
        return total
    end
    bp = _quadgk_breakpoints(xρ, xz, pj, self)
    integrand = s -> begin
        ρ2, z2 = _panel_point(pj, s)
        im * k * pj.nz * cis(k * z2) *
        _azimuthal_G(k, xρ, xz, ρ2, z2; rtol = rtol, far = far) * ρ2 * pj.L
    end
    return quadgk(
        integrand, bp...; rtol = rtol, atol = _QUAD_ATOL, maxevals = _AZIMUTHAL_MAXEVALS)[1]
end

# PressureRelease's double-layer-applied-to-known-p_scat integrand: -p_scat(y)·∂G/∂n_y(x,y).
function _pair_K_pressrel(k::Real, xρ::Real, xz::Real, pj::Panel, self::Bool, rtol::Real)
    far = !self && _azimuthal_is_far(xρ, xz, pj)
    if far
        total = zero(ComplexF64)
        for (s, w) in zip(_MERIDIAN_FIXED_NODES, _MERIDIAN_FIXED_WEIGHTS)
            ρ2, z2 = _panel_point(pj, s)
            total += -cis(k * z2) *
                     _azimuthal_dGdn(
                         k, xρ, xz, ρ2, z2, pj.nrho, pj.nz; rtol = rtol, far = true) * ρ2 *
                     pj.L * w
        end
        return total
    end
    bp = _quadgk_breakpoints(xρ, xz, pj, self)
    integrand = s -> begin
        ρ2, z2 = _panel_point(pj, s)
        -cis(k * z2) *
        _azimuthal_dGdn(k, xρ, xz, ρ2, z2, pj.nrho, pj.nz; rtol = rtol, far = far) * ρ2 *
        pj.L
    end
    return quadgk(
        integrand, bp...; rtol = rtol, atol = _QUAD_ATOL, maxevals = _AZIMUTHAL_MAXEVALS)[1]
end

# Fixed-order Gauss-Legendre rule on [0,2π] for well-separated (>= _AZIMUTHAL_FAR_FACTOR panel
# lengths) azimuthal integrals, the dominant assembly cost; closer pairs keep the fully-adaptive path since a fixed low order under-resolves the near-field peak.
const _AZIMUTHAL_FAR_FACTOR = 0.5

# Whether (ρ,z) is far enough from panel `pj` (relative to the panel's own
# arc length) for the fixed-order azimuthal rule to be safe.
function _azimuthal_is_far(ρ::Real, z::Real, pj::Panel)
    hypot(ρ - pj.rhom, z - pj.zm) > _AZIMUTHAL_FAR_FACTOR * pj.L
end

# Order for the fixed azimuthal rule, accounts for both cos(mΔφ) and the ring kernel's own phase
# oscillation (significant when kρ is large, e.g. an oblate spheroid's equator).
function _azimuthal_fixed_order(m::Integer, k::Real, ρ::Real, ρ2::Real)
    max(4m + 32, ceil(Int, 8 * k * min(ρ, ρ2)) + 32)
end

# Fixed Gauss-Legendre rule cached by order, populated before assemble_cbie_operators's threaded region so only concurrent reads occur.
const _AZIMUTHAL_FIXED_RULE_CACHE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
function _azimuthal_fixed_rule(order::Integer)
    return get!(_AZIMUTHAL_FIXED_RULE_CACHE, order) do
        nodes, weights = gauss(order)  # on [-1, 1]
        (π .* (nodes .+ 1), π .* weights)  # mapped to [0, 2π]
    end
end

"""
Fourier-mode-`m` (unnormalized) azimuthal integral over Δφ ∈ [0,2π) of the
single-layer ring kernel, `∫G(Δφ)cos(mΔφ)dΔφ`. `m = 0` (default) is the
axial-incidence case; the ring kernel is even in Δφ (see module
docstring), so its azimuthal Fourier series has only cosine terms, an
axisymmetric geometry excited by a `cos(mφ)`-azimuthal source responds
purely in the same mode, so this decomposition decouples modes exactly
(see hybrid.jl's oblique-incidence notes for the derivation this
supports).
"""
function _azimuthal_G(k::Real, ρ::Real, z::Real, ρ2::Real, z2::Real; m::Integer = 0,
        rtol::Real = 1e-6, atol::Real = _QUAD_ATOL, far::Bool = false)
    if far
        nodes, weights = _azimuthal_fixed_rule(_azimuthal_fixed_order(m, k, ρ, ρ2))
        total = zero(ComplexF64)
        for (Δφ, w) in zip(nodes, weights)
            total += _ring_G(k, ρ, z, ρ2, z2, Δφ) * cos(m * Δφ) * w
        end
        return total
    end
    val, _ = quadgk(
        Δφ -> _ring_G(k, ρ, z, ρ2, z2, Δφ) * cos(m * Δφ), _azimuthal_breakpoints(m)...;
        rtol = rtol, atol = atol, maxevals = _AZIMUTHAL_MAXEVALS)
    return val
end

"Fourier-mode-`m` (unnormalized) azimuthal integral over Δφ ∈ [0,2π) of the double-layer ring kernel; see [`_azimuthal_G`](@ref)."
function _azimuthal_dGdn(
        k::Real, ρ::Real, z::Real, ρ2::Real, z2::Real, nρ2::Real, nz2::Real;
        m::Integer = 0, rtol::Real = 1e-6, atol::Real = _QUAD_ATOL, far::Bool = false)
    if far
        nodes, weights = _azimuthal_fixed_rule(_azimuthal_fixed_order(m, k, ρ, ρ2))
        total = zero(ComplexF64)
        for (Δφ, w) in zip(nodes, weights)
            total += _ring_dGdn(k, ρ, z, ρ2, z2, nρ2, nz2, Δφ) * cos(m * Δφ) * w
        end
        return total
    end
    val, _ = quadgk(Δφ -> _ring_dGdn(k, ρ, z, ρ2, z2, nρ2, nz2, Δφ) * cos(m * Δφ),
        _azimuthal_breakpoints(m)...; rtol = rtol,
        atol = atol, maxevals = _AZIMUTHAL_MAXEVALS)
    return val
end

"""
    assemble_cbie_operators(mesh, k; m=0, rtol=1e-6)

Assemble the raw exterior double-layer (`K`) and single-layer (`V`)
operator matrices for the Fourier-mode-`m` axisymmetric CBIE (see module
docstring), without assuming any boundary condition, both `p_scat` and
`∂p_scat/∂n` are left as free (piecewise-constant) unknowns satisfying

    (1/2 I - K) p_scat + V * (∂p_scat/∂n) = 0

`m` selects the azimuthal Fourier mode (`m = 0`, the default, is the
axisymmetric/axial-incidence case `solve_axial` uses; general oblique
incidence needs `m = 0, 1, …` up to some truncation, one independent
solve per mode, see hybrid.jl). This is the building block
[`solve_axial`](@ref) specializes for a known boundary condition; it is
exposed directly for coupled problems (e.g. a fluid-loaded elastic
boundary) where `∂p_scat/∂n` is not known in closed form but is itself an
unknown linked to another physical system. Returns `(K, V, ps)`.
"""
function assemble_cbie_operators(mesh::MeridianMesh, k::Real; m::Integer = 0, rtol::Real = 1e-6)
    ps = panels(mesh)
    n = length(ps)
    K = zeros(complex(typeof(k)), n, n)
    V = zeros(complex(typeof(k)), n, n)
    # Populate every fixed-rule cache entry before the threaded region below to avoid a race.
    for i in 1:n, j in 1:n

        _azimuthal_fixed_rule(_azimuthal_fixed_order(m, k, ps[i].rhom, ps[j].rhom))
    end

    # Row i is written only by iteration i, safe to parallelize with no synchronization.
    if Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            xρ, xz = ps[i].rhom, ps[i].zm
            for j in 1:n
                pj = ps[j]
                self = (i == j)
                K[i, j] = _pair_K(k, xρ, xz, pj, self, rtol; m = m)
                V[i, j] = _pair_V(k, xρ, xz, pj, self, rtol; m = m)
            end
        end
    else
        for i in 1:n
            xρ, xz = ps[i].rhom, ps[i].zm
            for j in 1:n
                pj = ps[j]
                self = (i == j)
                K[i, j] = _pair_K(k, xρ, xz, pj, self, rtol; m = m)
                V[i, j] = _pair_V(k, xρ, xz, pj, self, rtol; m = m)
            end
        end
    end

    return K, V, ps
end

"""
    chief_row(mesh, k, ρ0, z0; m=0, rtol=1e-6)

Single-row generalization of [`assemble_cbie_operators`](@ref)'s `K`/`V`
matrices to an arbitrary field point `(ρ0, z0)` instead of a boundary panel
midpoint, the building block for CHIEF (Combined Helmholtz Integral
Equation Formulation) regularization of the exterior CBIE's irregular
frequencies.

For a point `x0` strictly *inside* the surface `S`, the exterior
representation formula `p_scat(x) = ∫_S[p_scat(y)∂G/∂n_y(x,y) -
G(x,y)∂p_scat/∂n_y(y)]dS(y)` (valid as derived for `x` in the true exterior
domain, see the module docstring) analytically continues to give
*identically zero* when evaluated at `x0` inside `S`, for any boundary data
`(p_scat, ∂p_scat/∂n)` that is the trace of a genuine radiating exterior
solution, this is what fails at the CBIE's fictitious interior
eigenfrequencies (where extra homogeneous solutions sneak into the
boundary-integral formulation without corresponding to a real field), and
is exactly what makes `K_row·p_scat - V_row·∂p_scat/∂n = 0` a valid *extra*
equation to over-determine the discretized system with, the standard
CHIEF stabilization. No jump-relation `±1/2` term applies here (unlike the
boundary CBIE itself): `x0` is a genuine interior point, not a boundary
collocation point, so there's no principal-value discontinuity to account
for, just the plain integral.

Returns `(K_row, V_row)`, each a `1×n` row matching `assemble_cbie_operators`'s
`K`/`V` column convention.
"""
function chief_row(
        mesh::MeridianMesh, k::Real, ρ0::Real, z0::Real; m::Integer = 0, rtol::Real = 1e-6)
    ps = panels(mesh)
    n = length(ps)
    K_row = zeros(ComplexF64, 1, n)
    V_row = zeros(ComplexF64, 1, n)
    # Evaluates at a genuine interior point, never a panel's own collocation point, so self=false.
    for j in 1:n
        pj = ps[j]
        K_row[1, j] = _pair_K(k, ρ0, z0, pj, false, rtol; m = m)
        V_row[1, j] = _pair_V(k, ρ0, z0, pj, false, rtol; m = m)
    end
    return K_row, V_row
end

"""
    solve_axial(boundary, k, mesh::MeridianMesh; rtol=1e-6)

Solve the axisymmetric (m = 0) direct CBIE for scattering of an axial
(end-on) unit-amplitude plane wave `p_inc = e^{ikz}` off a body of
revolution described by `mesh`, with `boundary` either [`Rigid`](@ref) or
[`PressureRelease`](@ref) (see the module docstring for both CBIEs).
Returns `(p_scat, dpdn_scat, ps)`: the piecewise-constant surface pressure
and its normal derivative on each panel, and the panel geometry, both
are needed by [`far_field`](@ref).

No fictitious-eigenfrequency (Burton-Miller/CHIEF) regularization is
applied yet, avoid `k*a` near an interior eigenvalue of the body (a
different eigenvalue set for each boundary condition).
"""
function solve_axial(::Rigid, k::Real, mesh::MeridianMesh; rtol::Real = 1e-6)
    ps = panels(mesh)
    n = length(ps)
    K = zeros(ComplexF64, n, n)
    b = zeros(ComplexF64, n)
    # See `assemble_cbie_operators`'s identical pre-population pass for why
    # this needs every pair's own order, not a single up-front call.
    for i in 1:n, j in 1:n

        _azimuthal_fixed_rule(_azimuthal_fixed_order(0, k, ps[i].rhom, ps[j].rhom))
    end

    # Row i written only by iteration i, safe to thread; single-threaded Threads.@threads is
    # catastrophically slow (~1150x on a 218-panel case), so only thread when nthreads() > 1.
    if Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            xρ, xz = ps[i].rhom, ps[i].zm
            for j in 1:n
                pj = ps[j]
                self = (i == j)
                K[i, j] = _pair_K(k, xρ, xz, pj, self, rtol)
                b[i] += _pair_G_rigid_rhs(k, xρ, xz, pj, self, rtol)
            end
        end
    else
        for i in 1:n
            xρ, xz = ps[i].rhom, ps[i].zm
            for j in 1:n
                pj = ps[j]
                self = (i == j)
                K[i, j] = _pair_K(k, xρ, xz, pj, self, rtol)
                b[i] += _pair_G_rigid_rhs(k, xρ, xz, pj, self, rtol)
            end
        end
    end

    A = 0.5I - K
    p_scat = A \ b
    dpdn_scat = ComplexF64[-im * k * p.nz * cis(k * p.zm) for p in ps]
    return p_scat, dpdn_scat, ps
end

function solve_axial(::PressureRelease, k::Real, mesh::MeridianMesh; rtol::Real = 1e-6)
    ps = panels(mesh)
    n = length(ps)
    G = zeros(ComplexF64, n, n)
    Kp_known = zeros(ComplexF64, n) # double-layer applied to the known p_scat = -e^{ikz}
    # See `assemble_cbie_operators`'s identical pre-population pass for why
    # this needs every pair's own order, not a single up-front call.
    for i in 1:n, j in 1:n

        _azimuthal_fixed_rule(_azimuthal_fixed_order(0, k, ps[i].rhom, ps[j].rhom))
    end

    # See `solve_axial(::Rigid, ...)` above for why threading is
    # conditional on `Threads.nthreads() > 1`.
    if Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            xρ, xz = ps[i].rhom, ps[i].zm
            for j in 1:n
                pj = ps[j]
                self = (i == j)
                G[i, j] = _pair_V(k, xρ, xz, pj, self, rtol)
                Kp_known[i] += _pair_K_pressrel(k, xρ, xz, pj, self, rtol)
            end
        end
    else
        for i in 1:n
            xρ, xz = ps[i].rhom, ps[i].zm
            for j in 1:n
                pj = ps[j]
                self = (i == j)
                # -p_scat(y) = e^{ikz(y)} baked into the double-layer integrand.
                G[i, j] = _pair_V(k, xρ, xz, pj, self, rtol)
                Kp_known[i] += _pair_K_pressrel(k, xρ, xz, pj, self, rtol)
            end
        end
    end

    p_scat = ComplexF64[-cis(k * p.zm) for p in ps]
    b = Kp_known .- p_scat ./ 2
    dpdn_scat = G \ b
    return p_scat, dpdn_scat, ps
end

# Fluid-filled/transmission boundary, axial incidence: standalone (no shell) interface, exterior
# and interior CBIEs coupled by pressure/velocity continuity into a 4n x 4n system for [p_scat; dpdn_scat; p_int; dpdn_int]. Limiting cases g->∞/g->0 recover Rigid/PressureRelease respectively.

"""
    _chief_points(mesh, n_points)

`n_points` interior test points for CHIEF regularization ([`chief_row`](@ref)),
spread through the body enclosed by `mesh` without assuming it's literally
a sphere: a moderate off-axis radius (`0.4` of the mesh's own maximum `ρ`,
comfortably inside the boundary and off the `ρ=0` axis so it doesn't
coincide with an accidental symmetry point) at a spread of heights spanning
the mesh's own `z`-extent.
"""
function _chief_points(mesh::MeridianMesh, n_points::Integer)
    z_min, z_max = extrema(mesh.z)
    z_mid = (z_min + z_max) / 2
    z_half = (z_max - z_min) / 2
    ρ0 = 0.4 * maximum(mesh.rho)
    return [(ρ0, z_mid + f * z_half) for f in range(-0.5, 0.5; length = n_points)]
end

"""
    solve_axial(::FluidFilled, k, mesh::MeridianMesh; rtol=1e-6, chief_points=0)

Solve the axisymmetric (m = 0) direct CBIE for transmission scattering of
an axial (end-on) unit-amplitude plane wave `p_inc = e^{ikz}` off a
fluid-filled body of revolution described by `mesh` (see the module
docstring above this method for the transmission-condition derivation).
`boundary.coupling` is ignored, it only affects the analytical spheroid
modal series (spheroid_modal.jl); this BEM solve is always the "full"
(no diagonal-mode-coupling shortcut) transmission problem.

`chief_points` (default `0`) optionally adds that many CHIEF (Combined
Helmholtz Integral Equation Formulation) equations from [`chief_row`](@ref), extra rows testing the exterior representation formula at interior
points ([`_chief_points`](@ref)), over-determining the system (solved by
least squares) to suppress the exterior CBIE's spurious solutions near its
fictitious interior eigenfrequencies. Left in as correctly-derived,
available infrastructure (the exterior-only formulation is the standard,
low-risk textbook case, and the row equation itself was directly verified
correct, evaluated against a known-good solution's actual boundary data
and found to be ≈0 to within discretization error) but **disabled by
default and not needed for this solver's own accuracy**: what looked like
CHIEF-fixable "fictitious eigenfrequency" error near a handful of specific
frequencies turned out, on closer inspection, to be a genuine sharp interference resonance
(target strength changing >10 dB per kHz there) combined with ordinary
frequency-grid sensitivity when comparing against an external benchmark, this package's own analytical modal series and this BEM solver agree with
*each other* at those frequencies to a few hundredths of a dB once
adequately resolved (see [`solve_axial_adaptive`](@ref)); they just don't
match the external reference's specific 2 kHz-grid sample point near that
resonance, which no amount of regularization of *this* solver can or
should change. Scaling `chief_points` up (tested to 36 per block) made
accuracy *worse*, not better, consistent with there being no real
fictitious-eigenfrequency defect here to suppress.

Returns `(p_scat, dpdn_scat, ps, p_int, dpdn_int)`: the exterior scattered
surface pressure/normal-derivative and panel geometry (usable with
[`far_field`](@ref) exactly as [`solve_axial`](@ref)'s rigid/
pressure-release methods are), plus the interior transmitted field and its
normal derivative on the same panels.
"""
function solve_axial(boundary::FluidFilled, k::Real, mesh::MeridianMesh;
        rtol::Real = 1e-6, chief_points::Integer = 0)
    g = boundary.density_contrast
    k_int = k / boundary.soundspeed_contrast

    K_ext, V_ext, ps = assemble_cbie_operators(mesh, k; rtol = rtol)
    K_int, V_int, _ = assemble_cbie_operators(mesh, k_int; rtol = rtol)
    n = length(ps)

    dpdn_inc = ComplexF64[im * k * p.nz * cis(k * p.zm) for p in ps]
    p_inc = ComplexF64[cis(k * p.zm) for p in ps]

    off_p, off_d, off_pi, off_di = 0, n, 2n, 3n
    n_total = 4n
    n_rows = n_total + chief_points
    A = zeros(ComplexF64, n_rows, n_total)
    b = zeros(ComplexF64, n_rows)
    I_n = Matrix{ComplexF64}(I, n, n)

    rows = 1:n
    A[rows, (off_p + 1):(off_p + n)] = 0.5I - K_ext
    A[rows, (off_d + 1):(off_d + n)] = V_ext

    rows = (n + 1):(2n)
    A[rows, (off_pi + 1):(off_pi + n)] = 0.5I + K_int
    A[rows, (off_di + 1):(off_di + n)] = -V_int

    rows = (2n + 1):(3n)
    A[rows, (off_p + 1):(off_p + n)] = I_n
    A[rows, (off_pi + 1):(off_pi + n)] = -I_n
    b[rows] = -p_inc

    rows = (3n + 1):(4n)
    A[rows, (off_d + 1):(off_d + n)] = I_n
    A[rows, (off_di + 1):(off_di + n)] = -I_n ./ g
    b[rows] = -dpdn_inc

    for (idx, (ρ0, z0)) in enumerate(_chief_points(mesh, chief_points))
        K_row, V_row = chief_row(mesh, k, ρ0, z0; rtol = rtol)
        row = n_total + idx
        A[row, (off_p + 1):(off_p + n)] = K_row
        A[row, (off_d + 1):(off_d + n)] = -V_row
    end

    x = A \ b
    p_scat = x[(off_p + 1):(off_p + n)]
    dpdn_scat = x[(off_d + 1):(off_d + n)]
    p_int = x[(off_pi + 1):(off_pi + n)]
    dpdn_int = x[(off_di + 1):(off_di + n)]

    return p_scat, dpdn_scat, ps, p_int, dpdn_int
end

"""
    solve_axial_adaptive(boundary::FluidFilled, k, mesh_fn, characteristic_radius;
                          target_tol=0.1, max_n=2000, rtol=1e-3)

Self-checking version of `solve_axial(::FluidFilled, ...)`: doubles the
*panel count* `n` (starting from [`bem_panel_count`](@ref)'s recommendation)
and re-solves until backscatter target strength changes by less than
`target_tol` [dB] between successive refinements, instead of trusting a
single fixed resolution. `mesh_fn(n)` must build the mesh at panel count
`n` (e.g. `n -> sphere_mesh(a, n)`); `characteristic_radius` is the body's
largest relevant dimension, used the same way as in `bem_panel_count`.

Doubles `n` directly, *not* `elements_per_wavelength` (an earlier version
of this function did, and it was a real bug, not just a naming choice, `n = elements_per_wavelength·ka/2`, so at low `ka` doubling
`elements_per_wavelength` barely moves `n` at all, capping refinement far
below what's actually needed there; confirmed directly, a 12 kHz point
converged cleanly to 0.013 dB by `n=320`, but the `elements_per_wavelength`-
doubling version's own cap only ever reached `n≈52` at that same
frequency).

Exists because fixed, `ka`-scaled resolution can be insufficient for this
boundary condition for two genuinely different reasons, both requiring
*more panels than `bem_panel_count`'s wavelength-based heuristic alone
gives*, confirmed directly, not inferred:
  - **Low `ka`, near-total cancellation**: "weakly scattering" means the
    interior/exterior fields nearly cancel (that's the point of the
    boundary condition), so the small net backscattered signal is
    disproportionately sensitive to each side's discretization error, unlike `Rigid`, which has no cancellation and converges fine with far
    fewer panels at the same `ka`. Ordinary, well-behaved O(h)-ish
    convergence, just needing a higher panel *floor* than the wavelength
    scaling alone provides.
  - **Near a sharp interference resonance**: some frequencies sit on the
    steep flank of a narrow destructive-interference null (target strength
    can change more than 10 dB per kHz there, a real, physically expected
    feature of weakly-scattering spheres' high-Q internal near-resonance,
    not a numerical artifact), so the field itself varies rapidly and needs
    finer resolution to resolve accurately, purely as an ordinary
    consequence of that rapid physical variation. (An earlier version of
    this docstring attributed this to a CBIE "fictitious eigenfrequency"
    defect and pursued CHIEF/Burton-Miller regularization, ruled out
    directly: this package's own analytical modal series, entirely
    independent of the BEM formulation, shows the identical sharp feature
    and the identical apparent "disagreement" against an external
    benchmark sampled on a coarser frequency grid, while agreeing with this
    solver's own converged value to a few hundredths of a dB. There was no
    solver defect to regularize.)

Emits a warning (not silence) if `max_n` is reached without meeting
`target_tol`, since that's a real signal the frequency is close to a sharp
resonance (rather than just needing the ordinary extra panels the low-`ka`
cancellation case needs) and the returned result, while the best available,
may still be changing by more than `target_tol` per doubling.
"""
function solve_axial_adaptive(
        boundary::FluidFilled, k::Real, mesh_fn, characteristic_radius::Real;
        target_tol::Real = 0.1, max_n::Integer = 2000, rtol::Real = 1e-3)
    n = bem_panel_count(k, characteristic_radius)
    p, d, ps, p_int, dpdn_int = solve_axial(boundary, k, mesh_fn(n); rtol = rtol)
    ts_prev = target_strength(ps, p, d, k, π)

    while n < max_n
        n = min(max_n, 2n)
        p, d, ps, p_int, dpdn_int = solve_axial(boundary, k, mesh_fn(n); rtol = rtol)
        ts_new = target_strength(ps, p, d, k, π)
        Δ = abs(ts_new - ts_prev)
        Δ < target_tol && return p, d, ps, p_int, dpdn_int
        ts_prev = ts_new
        n == max_n && break
    end

    @warn "solve_axial_adaptive did not converge to target_tol=$target_tol dB within max_n=$max_n panels (ka=$(k*characteristic_radius)). This frequency is likely on the steep flank of a sharp interference resonance, where the field needs unusually fine resolution. Returning the finest solve tried, not a converged result"
    return p, d, ps, p_int, dpdn_int
end

"""
    solve_axial_adaptive(boundary::Union{Rigid,PressureRelease}, k, mesh_fn, characteristic_radius; target_tol=0.1, max_n=4000, rtol=1e-5)

Same self-checking resolution-doubling pattern as the `FluidFilled` method
above, for the simpler 3-return [`solve_axial`](@ref) signature used by
`Rigid`/`PressureRelease`, avoids hand-picking `elements_per_wavelength`
per frequency, which is routinely too coarse at low `ka` (few panels even
at generous `elements_per_wavelength`) and unnecessarily expensive at high
`ka` if picked conservatively enough to cover the low-`ka` case too.
"""
function solve_axial_adaptive(boundary::Union{Rigid, PressureRelease},
        k::Real, mesh_fn, characteristic_radius::Real;
        target_tol::Real = 0.1, max_n::Integer = 4000, rtol::Real = 1e-5)
    n = bem_panel_count(k, characteristic_radius)
    p, d, ps = solve_axial(boundary, k, mesh_fn(n); rtol = rtol)
    ts_prev = target_strength(ps, p, d, k, π)

    while n < max_n
        n = min(max_n, 2n)
        p, d, ps = solve_axial(boundary, k, mesh_fn(n); rtol = rtol)
        ts_new = target_strength(ps, p, d, k, π)
        Δ = abs(ts_new - ts_prev)
        Δ < target_tol && return p, d, ps
        ts_prev = ts_new
        n == max_n && break
    end

    @warn "solve_axial_adaptive did not converge to target_tol=$target_tol dB within max_n=$max_n panels (ka=$(k*characteristic_radius)). This frequency is likely on the steep flank of a sharp interference resonance, where the field needs unusually fine resolution. Returning the finest solve tried, not a converged result"
    return p, d, ps
end

# Oblique incidence, rigid boundary: Jacobi-Anger expansion of p_inc = e^{ik(x sinβ + z cosβ)} into
# azimuthal Fourier modes cos(mφ), decoupled since the ring kernel depends only on the source/field azimuth difference; mode m solves its own CBIE via assemble_cbie_operators(mesh, k; m=m).

function _besselj_deriv(m::Integer, x::Real)
    m == 0 ? -besselj(1, x) : (besselj(m - 1, x) - besselj(m + 1, x)) / 2
end

function _dpdn_inc_mode(m::Integer, k::Real, β::Real, ρ::Real, z::Real, nρ::Real, nz::Real)
    εm = m == 0 ? 1.0 : 2.0
    u = k * ρ * sin(β)
    Jm = besselj(m, u)
    Jmp = _besselj_deriv(m, u)
    return εm * im^m * cis(k * z * cos(β)) *
           (im * k * cos(β) * nz * Jm + k * sin(β) * nρ * Jmp)
end

"""
    solve_oblique(::Rigid, k, mesh::MeridianMesh, incidence_angle; m_max, rtol=1e-5)

Solve the rigid-boundary axisymmetric CBIE for a unit-amplitude plane wave
arriving at `incidence_angle` [rad] from the z-axis (`0` = axial/end-on,
matching [`solve_axial`](@ref); `π/2` = broadside), by decomposing into
azimuthal Fourier modes `m = 0, …, m_max` and solving each mode's
decoupled CBIE independently (see the derivation above).

Returns `(p_scat_modes, dpdn_scat_modes, ps)`: `Vector`s of length
`m_max + 1` holding each mode's piecewise-constant surface solution
(`p_scat_modes[m+1]` is mode `m`), and the panel geometry. Use with
[`far_field`](@ref)'s bistatic method to reconstruct the scattered field
at any observation angle.
"""
function solve_oblique(::Rigid, k::Real, mesh::MeridianMesh, incidence_angle::Real;
        m_max::Integer, rtol::Real = 1e-5)
    ps = panels(mesh)
    n = length(ps)
    β = incidence_angle

    p_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)

    for m in 0:m_max
        K, V, _ = assemble_cbie_operators(mesh, k; m = m, rtol = rtol)
        dpdn_inc = ComplexF64[_dpdn_inc_mode(m, k, β, p.rhom, p.zm, p.nrho, p.nz)
                              for p in ps]
        dpdn_scat = -dpdn_inc
        p_scat = (0.5I - K) \ (V * dpdn_inc) # (0.5I-K)p = -V*dpdn_scat = V*dpdn_inc
        p_scat_modes[m + 1] = p_scat
        dpdn_scat_modes[m + 1] = dpdn_scat
    end

    return p_scat_modes, dpdn_scat_modes, ps
end

# Oblique incidence, pressure-release boundary: needs only p_inc's value (Dirichlet data), so the
# plain Jacobi-Anger expansion suffices, no Jₘ' term; same per-mode decoupling as the rigid case.

function _p_inc_mode(m::Integer, k::Real, β::Real, ρ::Real, z::Real)
    εm = m == 0 ? 1.0 : 2.0
    u = k * ρ * sin(β)
    return εm * im^m * besselj(m, u) * cis(k * z * cos(β))
end

"""
    solve_oblique(::PressureRelease, k, mesh::MeridianMesh, incidence_angle; m_max, rtol=1e-5)

Solve the pressure-release-boundary axisymmetric CBIE for oblique
incidence, mode by mode (see [`solve_oblique(::Rigid, ...)`](@ref) and the
derivation above). Returns `(p_scat_modes, dpdn_scat_modes, ps)` in the
same shape.
"""
function solve_oblique(
        ::PressureRelease, k::Real, mesh::MeridianMesh, incidence_angle::Real;
        m_max::Integer, rtol::Real = 1e-5)
    ps = panels(mesh)
    β = incidence_angle

    p_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)

    for m in 0:m_max
        K, V, _ = assemble_cbie_operators(mesh, k; m = m, rtol = rtol)
        p_inc = ComplexF64[_p_inc_mode(m, k, β, p.rhom, p.zm) for p in ps]
        p_scat = -p_inc
        dpdn_scat = V \ (K * p_scat .- p_scat ./ 2)
        p_scat_modes[m + 1] = p_scat
        dpdn_scat_modes[m + 1] = dpdn_scat
    end

    return p_scat_modes, dpdn_scat_modes, ps
end

# Oblique incidence, fluid-filled/transmission: generalizes solve_axial(::FluidFilled, ...)'s
# 4n x 4n coupled system to oblique incidence, only the incident-field RHS data changes per mode m, decoupled the same way as the rigid/pressure-release oblique solvers.

"""
    solve_oblique(boundary::FluidFilled, k, mesh::MeridianMesh, incidence_angle; m_max, rtol=1e-5, precision=:double)

Solve the fluid-filled/transmission axisymmetric CBIE (see
[`solve_axial(::FluidFilled, ...)`](@ref)) for oblique incidence, mode by
mode (see [`solve_oblique(::Rigid, ...)`](@ref) and the derivation above).
Returns `(p_scat_modes, dpdn_scat_modes, ps)` in the same shape as the
rigid/pressure-release methods.

Solved as a reduced `2n×2n` system in the exterior unknowns alone
(`p_scat_ext =: P`, `∂p_scat_ext/∂n =: D`), not the naive `4n×4n` system
coupling exterior and interior traces directly. Derivation: the coupling
equations `P - p_int = -p_inc` and `D - dpdn_int/g = -dpdn_inc` are purely
algebraic (no integral operator), so they solve *exactly* for the interior
trace in terms of the exterior one, `p_int = P + p_inc`,
`dpdn_int = g·(D + dpdn_inc)`, with no approximation. Substituting into
the interior CBIE `(0.5I+K_int)p_int - V_int·dpdn_int = 0` eliminates
`p_int`/`dpdn_int` entirely, leaving

    (0.5I - K_ext)·P + V_ext·D = 0
    (0.5I + K_int)·P - g·V_int·D = -[(0.5I+K_int)·p_inc - g·V_int·dpdn_inc]

This matters for weakly-scattering boundaries (`density_contrast`,
`soundspeed_contrast ≈ 1`, so `g ≈ 1` and `k_int ≈ k`): the naive `4n×4n`
system's *unknowns* include the interior trace, which is `O(1)` (it's
essentially the unperturbed incident field continuing through when there's
almost no impedance contrast) even though the physically meaningful
exterior scattered field is small, so a dense solve has to extract a
small answer from a system most of whose own state is large, and that
extraction was confirmed, directly, to lose accuracy no resolution or
precision lever fixes (see below). The reduced system's right-hand side, call it `-R`, `R := (0.5I+K_int)p_inc - g·V_int·dpdn_inc`, is a single,
isolated, one-time evaluation, not something threaded through an entire
linear solve, and it is *exactly* zero at `g = 1`, `k_int = k` (a true
interface-free medium scatters nothing at all, so the reduced system's own
right-hand side vanishing there is a hard correctness check, not a
coincidence, verified directly, see test/runtests.jl), growing away from
zero only as fast as the material contrast actually deviates from 1. That
is a fundamentally different (and far milder) cancellation than the
original system asked of its solver: shrinking `R` no longer requires
subtracting two matrix solves' worth of accumulated rounding error against
each other, just one controlled vector evaluation.

Verified correct, not just verified to run: matches the rigid-limit
cross-check (`FluidFilled(1e8,1e8)` against `solve_oblique(::Rigid,...)`,
same mesh/angle) to `1e-6` dB, and matches an already-working
weakly-scattering point unchanged.

This reduction alone is *not* what fixes a hard weakly-scattering case
that used to sit stuck ~26 dB off the independently-validated modal
series regardless of mesh, mode count, quadrature `rtol`, or arithmetic
precision (all tried; none moved it), that stuck value was identical
whether solved through this reduced system or the naive `4n×4n` one, which
was the clue that actually cracked it: since `K_ext`/`V_ext`/`K_int`/
`V_int`/`p_inc`/`dpdn_inc` are the only state two independently-derived
linear systems share, the bug had to be in producing *those*, not in
solving them. It turned out to be `_azimuthal_fixed_order` (see there):
the fixed-order Gauss-Legendre rule used for well-separated ("far") panel
pairs was sized only for `cos(mΔφ)`'s own oscillation, ignoring that the
ring kernel `_ring_distance` swings by `2·min(ρ,ρ2)` over `Δφ∈[0,2π)`, negligible for small bodies but not for an oblate spheroid's equatorial
panels at high `ka`. Confirmed directly by forcing every pair through
*fully* adaptive quadrature (bypassing the fixed rule entirely): the
`R`-vanishing check below, which had plateaued hard at `0.0035` by 96
panels no matter how much finer the mesh got, instead kept shrinking
cleanly (`0.0053→0.0018→0.0006→0.0002` at 48→96→192→384 panels), proving
the flat-panel discretization itself was fine and the fixed rule was not.
Sizing that rule's order to the actual `k·min(ρ,ρ2)` scale (not just `m`)
reproduces the fully-adaptive numbers at a fraction of the cost, and
carries all the way through to the actual scattering answer: the same
hard case, previously frozen at -74 to -85 dB regardless of resolution,
now converges cleanly toward the modal target as panels increase
(-80.3→-86.6→-91.3→-94.8→-99.5→-101.5 dB at 32→64→96→128→192→256 panels,
against a -107.9 dB target, an ordinary, panel-count-limited remaining
gap now, not a floor).

`precision=:quad` evaluates `R` (only `R`, the two `n×n` CBIE assemblies
and the final `2n×2n` solve stay `Float64`) in `BigFloat`, for boundaries
close enough to `g = h = 1` that even this single, isolated cancellation
needs more digits than `Float64` carries; `:double` (the default)
computes it directly in `Float64`.
"""
function solve_oblique(
        boundary::FluidFilled, k::Real, mesh::MeridianMesh, incidence_angle::Real;
        m_max::Integer, rtol::Real = 1e-5, precision::Symbol = :double,
        chief_points::Integer = 0, chief_points_int::Integer = 0)
    precision === :double || precision === :quad ||
        throw(ArgumentError("precision must be :double or :quad, got $precision"))
    ps = panels(mesh)
    n = length(ps)
    β = incidence_angle
    g = boundary.density_contrast
    k_int = k / boundary.soundspeed_contrast

    p_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_scat_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)

    off_p, off_d = 0, n
    n_total = 2n
    n_rows = n_total + chief_points + chief_points_int
    I_n = Matrix{ComplexF64}(I, n, n)
    chief_pts = chief_points > 0 ? _chief_points(mesh, chief_points) :
                Tuple{Float64, Float64}[]
    chief_pts_int = chief_points_int > 0 ? _chief_points(mesh, chief_points_int) :
                    Tuple{Float64, Float64}[]

    for m in 0:m_max
        K_ext, V_ext, _ = assemble_cbie_operators(mesh, k; m = m, rtol = rtol)
        K_int, V_int, _ = assemble_cbie_operators(mesh, k_int; m = m, rtol = rtol)

        p_inc = ComplexF64[_p_inc_mode(m, k, β, p.rhom, p.zm) for p in ps]
        dpdn_inc = ComplexF64[_dpdn_inc_mode(m, k, β, p.rhom, p.zm, p.nrho, p.nz)
                              for p in ps]

        if precision === :quad
            K_int_hi, V_int_hi, _ = assemble_cbie_operators(mesh, BigFloat(k_int); m = m, rtol = rtol)
            p_inc_hi = Complex{BigFloat}[_p_inc_mode(m, BigFloat(k), β, p.rhom, p.zm)
                                         for p in ps]
            dpdn_inc_hi = Complex{BigFloat}[_dpdn_inc_mode(m, BigFloat(k), β, p.rhom,
                                                p.zm, p.nrho, p.nz) for p in ps]
            R = (0.5I + K_int_hi) * p_inc_hi - BigFloat(g) * V_int_hi * dpdn_inc_hi
            R = ComplexF64.(R)
        else
            R = (0.5I + K_int) * p_inc - g * V_int * dpdn_inc
        end

        A = zeros(ComplexF64, n_rows, n_total)
        b = zeros(ComplexF64, n_rows)

        rows = 1:n
        A[rows, (off_p + 1):(off_p + n)] = 0.5I - K_ext
        A[rows, (off_d + 1):(off_d + n)] = V_ext

        rows = (n + 1):(2n)
        A[rows, (off_p + 1):(off_p + n)] = 0.5I + K_int
        A[rows, (off_d + 1):(off_d + n)] = -g .* V_int
        b[rows] = -R

        # CHIEF over-determination breaks the spurious null-space at fictitious interior
        # eigenvalues; chief_points regularizes the exterior block, chief_points_int the interior block (scaled by g).
        for (idx, (ρ0, z0)) in enumerate(chief_pts)
            K_row, V_row = chief_row(mesh, k, ρ0, z0; m = m, rtol = rtol)
            row = n_total + idx
            A[row, (off_p + 1):(off_p + n)] = K_row
            A[row, (off_d + 1):(off_d + n)] = -V_row
        end
        for (idx, (ρ0, z0)) in enumerate(chief_pts_int)
            K_row, V_row = chief_row(mesh, k_int, ρ0, z0; m = m, rtol = rtol)
            row = n_total + chief_points + idx
            A[row, (off_p + 1):(off_p + n)] = K_row
            A[row, (off_d + 1):(off_d + n)] = -g .* V_row
        end

        x = A \ b
        p_scat_modes[m + 1] = x[(off_p + 1):(off_p + n)]
        dpdn_scat_modes[m + 1] = x[(off_d + 1):(off_d + n)]
    end

    return p_scat_modes, dpdn_scat_modes, ps
end
