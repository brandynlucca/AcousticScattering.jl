# Fourier-mode acoustic BEM on piecewise-linear meridians for rigid, soft and fluid boundaries.

using QuadGK: quadgk, gauss, alloc_segbuf

"""
    MeridianMesh(rho, z)

Piecewise-linear discretization of the generating (meridian) curve of a
body of revolution in the half-plane rho ≥ 0: node coordinates `(rho[i],
z[i])`, panels connecting consecutive nodes. The internal axial parameter `z`
maps to Cartesian x; rho is distance from that axis. The curve must be traversed
so that outward normals equal `(-Δz, Δrho)/L` per panel, e.g. north pole to
south pole for a convex body enclosing the symmetry axis (verified for
[`sphere_mesh`](@ref)).
"""
struct MeridianMesh
    rho::Vector{Float64}
    z::Vector{Float64}
end

function _validate_revolved_profile(x, radius, closed)
    length(x) == length(radius) >= 3 ||
        throw(ArgumentError("a meridian needs matching x/radius vectors with at least three vertices"))
    all(isfinite, x) && all(isfinite, radius) ||
        throw(ArgumentError("meridian coordinates must be finite"))
    xx, rr = Float64.(x), Float64.(radius)
    scale = max(maximum(xx) - minimum(xx), maximum(rr) - minimum(rr))
    scale > 0 || throw(ArgumentError("meridian has zero extent"))
    tolerance = 64eps(Float64) * scale
    if closed
        all(>(0), rr) ||
            throw(ArgumentError("a closed meridian loop must stay strictly away from the axis"))
        if hypot(last(xx) - first(xx), last(rr) - first(rr)) <= tolerance
            xx[end], rr[end] = xx[1], rr[1]
        else
            push!(xx, xx[1])
            push!(rr, rr[1])
        end
    else
        iszero(first(rr)) && iszero(last(rr)) && all(>(0), rr[2:(end - 1)]) ||
            throw(ArgumentError("an open meridian must meet the axis only at its two endpoints"))
    end
    # Work in local, dimensionless coordinates so tolerance is independent of units.
    vertices = [SVector((xx[i] - xx[1]) / scale, (rr[i] - rr[1]) / scale)
                for i in eachindex(xx)]
    closed || push!(vertices, first(vertices))
    count = length(vertices) - 1
    count >= 3 || throw(ArgumentError("meridian must enclose an area"))
    tol = 64eps(Float64)
    cross2(a, b) = a[1] * b[2] - a[2] * b[1]
    for i in 1:count
        a, b = vertices[i], vertices[i + 1]
        norm(b - a) > tol ||
            throw(ArgumentError("meridian has a repeated or unresolved vertex"))
        c = vertices[mod1(i + 2, count)]
        abs(cross2(b - a, c - b)) <= tol && dot(b - a, c - b) < 0 &&
            throw(ArgumentError("adjacent meridian segments overlap"))
        for j in (i + 2):count
            i == 1 && j == count && continue
            c, d = vertices[j], vertices[j + 1]
            any(
                k -> max(a[k], b[k]) < min(c[k], d[k]) - tol ||
                     max(c[k], d[k]) < min(a[k], b[k]) - tol, 1:2) && continue
            ab_c, ab_d = cross2(b - a, c - a), cross2(b - a, d - a)
            cd_a, cd_b = cross2(d - c, a - c), cross2(d - c, b - c)
            separated(u, v) = (u > tol && v > tol) || (u < -tol && v < -tol)
            separated(ab_c, ab_d) || separated(cd_a, cd_b) ||
                throw(ArgumentError("meridian segments cross, touch or have unresolved separation"))
        end
    end
    area = sum(cross2(vertices[i], vertices[i + 1]) for i in 1:count)
    abs(area) > tol || throw(ArgumentError("meridian has zero or unresolved area"))
    if area < 0
        reverse!(xx)
        reverse!(rr)
    end
    return xx, rr
end

# Preserve every supplied corner while allocating extra panels by segment length.
function _axisymmetric_mesh(body::RevolvedBody, n::Integer)
    lengths = hypot.(diff(body.x), diff(body.radius))
    n >= length(lengths) ||
        throw(ArgumentError("panel count must be at least the number of profile segments"))
    extra = (n - length(lengths)) .* lengths ./ sum(lengths)
    counts = 1 .+ floor.(Int, extra)
    remainder = n - sum(counts)
    counts[sortperm(extra .- floor.(extra); rev = true)[1:remainder]] .+= 1
    x, rho = Float64[], Float64[]
    for i in eachindex(counts), j in 0:(counts[i] - 1)

        t = j / counts[i]
        push!(x, (1 - t) * body.x[i] + t * body.x[i + 1])
        push!(rho, (1 - t) * body.radius[i] + t * body.radius[i + 1])
    end
    push!(x, last(body.x))
    push!(rho, last(body.radius))
    return MeridianMesh(rho, x)
end

function _axisymmetric_mesh(body::Torus, n::Integer)
    n >= 8 || throw(ArgumentError("a toroidal meridian needs at least eight panels"))
    t = range(0.0, 2pi; length = n + 1)
    x = body.axial_radius .* cos.(t)
    rho = body.major_radius .+ body.radial_radius .* sin.(t)
    x[end], rho[end] = x[1], rho[1]
    return MeridianMesh(rho, x)
end

function _meridian_location(mesh::MeridianMesh, point)
    x, rho = point[1], hypot(point[2], point[3])
    scale = max(maximum(abs, mesh.z), maximum(mesh.rho), abs(x), rho)
    tol = 128eps(Float64) * scale
    for panel in panels(mesh)
        t = clamp(
            ((rho - panel.rho1) * (panel.rho2 - panel.rho1) +
             (x - panel.z1) * (panel.z2 - panel.z1)) / panel.L^2,
            0.0,
            1.0)
        hypot(rho - ((1 - t) * panel.rho1 + t * panel.rho2),
            x - ((1 - t) * panel.z1 + t * panel.z2)) <= tol && return :on
    end
    inside = false
    for i in eachindex(mesh.rho)
        j = mod1(i + 1, length(mesh.rho))
        a, b = mesh.rho[i], mesh.rho[j]
        if (a > rho) != (b > rho)
            crossing = mesh.z[i] + (rho - a) * (mesh.z[j] - mesh.z[i]) / (b - a)
            x < crossing && (inside = !inside)
        end
    end
    return inside ? :inside : :outside
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

Meridian mesh for a spheroid with semi-axis `a` in m along the axis of
symmetry and equatorial semi-axis `b` in m (prolate if `a > b`, oblate if
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
and `length` in m (axis of symmetry along Cartesian x), discretized into approximately
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
feature of Helmholtz BIEs at a wedge), which uniform panel spacing
resolves only algebraically slowly. Each segment's nodes are placed at
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
caps.
Total body half-length is `cylinder_length/2 + endcap_depth`.
The geometry follows Gong, Li, Chai, Zhao & Mitri (2017), "T-matrix method for
acoustical Bessel beam scattering from a rigid finite cylinder with spheroidal endcaps,"
Eqs. (5)-(7): their `b=radius`, `h=cylinder_length/2` and `d=endcap_depth`.

Unlike `cylinder_mesh`, this body has **no sharp edges at all**: an
ellipse's tangent at its own equator is always perpendicular to its major
axis, i.e., parallel to the cylinder's own straight side, for *any*
choice of `endcap_depth`, so the cap/side junction has a continuous
normal by construction. Smooth caps allow normal-offset source placement
without the corner singularity of a flat-ended cylinder.
For MFS treatment of sharp edges, see Pérez-Arjona, Godinho & Espinosa (2018).

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
