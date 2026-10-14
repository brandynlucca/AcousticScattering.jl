# [Geometries](@id geometries)

Choose a named body, construct a meridian or elliptical-section loft, or supply
a closed triangular surface. Geometry support and material support are separate:
a mesh that works for a rigid scatterer does not automatically support elasticity.
All lengths below are in meters. The generated bodies use x for their longitudinal
or revolution axis, with y and z transverse.

## Geometry gallery

Each gallery panel shows one geometry rendered from its package mesh. Click an
image to enlarge it. Panels are independently scaled and preserve aspect ratios.
The mesh resolution is for viewing, not a scattering convergence recommendation.

```@raw html
<details><summary>Julia example</summary>
```

```@example geometries
using AcousticScattering
using CairoMakie: Figure, Axis3, plot!, save, hidedecorations!, hidespines!

function save_geometries(entries; meridian=false)
    for (name, body) in entries
        fig = Figure(; size=(570, 360))
        ring = body isa Torus || (body isa RevolvedBody && body.closed)
        ax = Axis3(fig[1, 1];
            aspect=:data, azimuth=ring ? pi/6 : -0.35pi, elevation=0.12pi,
            viewmode=:fitzoom, perspectiveness=0.15)
        surface = meridian ? mesh(body; resolution=128) :
                  mesh(body; method=:full, resolution=0.16, mesh_order=2, qorder=2)
        plot!(ax, surface)
        hidedecorations!(ax)
        hidespines!(ax)
        slug = replace(lowercase(name), r"[^a-z0-9]+" => "_")
        save("geometry_$(slug).png", fig)
    end
    return nothing
end

canonical = [
    "Sphere" => Sphere(0.7),
    "Prolate spheroid" => Spheroid(1.0, 0.4),
    "Oblate spheroid" => Spheroid(0.35, 0.85),
    "Flat-ended cylinder" => Cylinder(0.4, 1.5),
    "Capsule" => Cylinder(0.4, 1.0; endcap_depth=0.4),
    "Bent, domed cylinder" => Cylinder(0.25, 2.0;
        radius_curvature=1.2, endcap_depth=0.25),
]
save_geometries(canonical)
nothing # hide
```

```@raw html
</details>
<div class="scattering-gallery geometry-gallery">
  <figure>
```

[![Sphere surface mesh.](geometry_sphere.png)](geometry_sphere.png)

```@raw html
    <figcaption><h2>Sphere</h2><p><code>Sphere(radius)</code></p></figcaption>
  </figure>
  <figure>
```

[![Elongated prolate spheroid.](geometry_prolate_spheroid.png)](geometry_prolate_spheroid.png)

```@raw html
    <figcaption><h2>Prolate spheroid</h2><p><code>Spheroid(a, b)</code>, with <code>a &gt; b</code>.</p></figcaption>
  </figure>
  <figure>
```

[![Flattened oblate spheroid.](geometry_oblate_spheroid.png)](geometry_oblate_spheroid.png)

```@raw html
    <figcaption><h2>Oblate spheroid</h2><p><code>Spheroid(a, b)</code>, with <code>a &lt; b</code>.</p></figcaption>
  </figure>
  <figure>
```

[![Circular cylinder with flat endcaps.](geometry_flat_ended_cylinder.png)](geometry_flat_ended_cylinder.png)

```@raw html
    <figcaption><h2>Flat-ended cylinder</h2><p><code>Cylinder(r, L)</code></p></figcaption>
  </figure>
  <figure>
```

[![Cylinder with hemispherical endcaps.](geometry_capsule.png)](geometry_capsule.png)

```@raw html
    <figcaption><h2>Capsule</h2><p><code>Cylinder(r, L; endcap_depth=r)</code></p></figcaption>
  </figure>
  <figure>
```

[![Bent circular cylinder with domed ends.](geometry_bent_domed_cylinder.png)](geometry_bent_domed_cylinder.png)

```@raw html
    <figcaption><h2>Bent, domed cylinder</h2><p><code>Cylinder(r, L; radius_curvature=R, endcap_depth=d)</code></p></figcaption>
  </figure>
</div>
```

```@raw html
<details><summary>Julia example</summary>
```

```@example geometries
primitives = [
    "Triaxial ellipsoid" => Ellipsoid(1.0, 0.6, 0.35),
    "Superellipsoid" => Superellipsoid(0.9, 0.6, 0.4;
        axial_exponent=0.6, equatorial_exponent=0.6),
    "Cone" => Cone(0.6, 1.5),
    "Bicone" => Bicone(0.6, 1.8; waist=0.4),
    "Circular frustum" => Frustum(0.65, 0.3, 1.4),
    "Elliptical frustum" => Frustum((0.65, 0.35), (0.3, 0.2), 1.4),
]
save_geometries(primitives)
nothing # hide
```

```@raw html
</details>
<div class="scattering-gallery geometry-gallery">
  <figure>
```

[![Ellipsoid with three different semi-axes.](geometry_triaxial_ellipsoid.png)](geometry_triaxial_ellipsoid.png)

```@raw html
    <figcaption><h2>Triaxial ellipsoid</h2><p><code>Ellipsoid(a, b, c)</code></p></figcaption>
  </figure>
  <figure>
```

[![Superellipsoid with rounded boxlike sides.](geometry_superellipsoid.png)](geometry_superellipsoid.png)

```@raw html
    <figcaption><h2>Superellipsoid</h2><p><code>Superellipsoid(a, b, c; axial_exponent=e1, equatorial_exponent=e2)</code></p></figcaption>
  </figure>
  <figure>
```

[![Circular cone closed by a flat base.](geometry_cone.png)](geometry_cone.png)

```@raw html
    <figcaption><h2>Cone</h2><p><code>Cone(radius, length)</code></p></figcaption>
  </figure>
  <figure>
```

[![Two circular cones joined at their bases.](geometry_bicone.png)](geometry_bicone.png)

```@raw html
    <figcaption><h2>Bicone</h2><p><code>Bicone(radius, length; waist=0.4)</code></p></figcaption>
  </figure>
  <figure>
```

[![Truncated circular cone with two flat endcaps.](geometry_circular_frustum.png)](geometry_circular_frustum.png)

```@raw html
    <figcaption><h2>Circular frustum</h2><p><code>Frustum(r1, r2, length)</code></p></figcaption>
  </figure>
  <figure>
```

[![Frustum joining two elliptical sections.](geometry_elliptical_frustum.png)](geometry_elliptical_frustum.png)

```@raw html
    <figcaption><h2>Elliptical frustum</h2><p><code>Frustum((a1,b1), (a2,b2), length)</code></p></figcaption>
  </figure>
</div>
```

## Profiles, lofts and toroids

[`RevolvedBody`](@ref) accepts axial coordinates and radii describing a simple
piecewise-linear meridian. Its ordinary form has two endpoints on the axis and
positive interior radii. `closed=true` instead describes a loop away from the
axis, allowing toroids with noncircular cross-sections. Crossing or touching
profiles are rejected. Either traversal direction is accepted.

[`LoftedBody`](@ref) joins elliptical sections along a straight or bent centerline
and closes both ends with flat caps. Supply centers as a $3\times N$ matrix and
semi-axes as a $2\times N$ matrix. Section normals follow the centerline unless
`directions` is supplied. `twists` rotates each section about its normal in radians.
All semi-axes must be positive.

```@raw html
<details><summary>Julia example</summary>
```

```@example geometries
centers = [-1.0 0.0 1.0; 0.0 0.15 0.6; 0.0 0.0 0.15]
axes = [0.3 0.5 0.25; 0.2 0.3 0.15]
profiles = [
    "Circular frustum chain" => FrustumChain(
        [-1.0, -0.5, 0.2, 1.0], [0.0, 0.5, 0.35, 0.0]),
    "Elliptical frustum chain" => FrustumChain([-1.0, 0.0, 1.0], axes),
    "Bent elliptical loft" => LoftedBody(centers, axes; twists=[0.0, 0.2, 0.4]),
    "Ring torus" => Torus(0.75, 0.25),
    "Elliptical tube torus" => Torus(0.75, (0.15, 0.3)),
    "Rectangular-section toroid" => RevolvedBody(
        [-0.2, 0.2, 0.2, -0.2], [0.55, 0.55, 0.95, 0.95]; closed=true),
]
save_geometries(profiles)
nothing # hide
```

```@raw html
</details>
<div class="scattering-gallery geometry-gallery">
  <figure>
```

[![Connected circular frustums ending in two tips.](geometry_circular_frustum_chain.png)](geometry_circular_frustum_chain.png)

```@raw html
    <figcaption><h2>Circular frustum chain</h2><p><code>FrustumChain(x, radii)</code></p></figcaption>
  </figure>
  <figure>
```

[![Connected elliptical frustums with flat endcaps.](geometry_elliptical_frustum_chain.png)](geometry_elliptical_frustum_chain.png)

```@raw html
    <figcaption><h2>Elliptical frustum chain</h2><p><code>FrustumChain(x, semiaxes)</code></p></figcaption>
  </figure>
  <figure>
```

[![Bent loft with varying elliptical sections.](geometry_bent_elliptical_loft.png)](geometry_bent_elliptical_loft.png)

```@raw html
    <figcaption><h2>Bent elliptical loft</h2><p><code>LoftedBody(centers, semiaxes; twists)</code></p></figcaption>
  </figure>
  <figure>
```

[![Ring torus with a circular tube and open central hole.](geometry_ring_torus.png)](geometry_ring_torus.png)

```@raw html
    <figcaption><h2>Ring torus</h2><p><code>Torus(R, r)</code></p></figcaption>
  </figure>
  <figure>
```

[![Ring torus with an elliptical tube cross-section.](geometry_elliptical_tube_torus.png)](geometry_elliptical_tube_torus.png)

```@raw html
    <figcaption><h2>Elliptical tube torus</h2><p><code>Torus(R, (axial, radial))</code></p></figcaption>
  </figure>
  <figure>
```

[![Annular body with rectangular tube cross-section.](geometry_rectangular_section_toroid.png)](geometry_rectangular_section_toroid.png)

```@raw html
    <figcaption><h2>Rectangular-section toroid</h2><p><code>RevolvedBody(x, radii; closed=true)</code></p></figcaption>
  </figure>
</div>
```

Finite capped paraboloids, hyperboloids, ogives, waisted bodies, rounded noses
and other surfaces of revolution can use sampled profiles without another solver.
For example, the next paraboloid has $r^2=0.3(x+1)$ and a flat cap at $x=1$.
The hyperboloid has $r=0.3\sqrt{1+(x/0.6)^2}$ between two flat caps.

```@raw html
<details><summary>Julia example</summary>
```

```@example geometries
parameter = collect(range(0.0, 1.0; length=33))
paraboloid = RevolvedBody(vcat(-1 .+ 2parameter.^2, 1.0),
    vcat(sqrt(0.6) .* parameter, 0.0))
x = collect(range(-1.0, 1.0; length=41))
hyperboloid = RevolvedBody(vcat(-1.0, x, 1.0),
    vcat(0.0, 0.3sqrt.(1 .+ (x ./ 0.6).^2), 0.0))
ogive = RevolvedBody(t -> (2t-1, sqrt(1.25^2-(2t-1)^2)-0.75); n=48)
save_geometries([
    "Capped paraboloid" => paraboloid,
    "Capped hyperboloid" => hyperboloid,
    "Double circular-arc ogive" => ogive,
]; meridian=true)
nothing # hide
```

```@raw html
</details>
<div class="scattering-gallery geometry-gallery">
  <figure>
```

[![Paraboloid of revolution closed by a flat cap.](geometry_capped_paraboloid.png)](geometry_capped_paraboloid.png)

```@raw html
    <figcaption><h2>Capped paraboloid</h2><p>Sampled meridian with <code>RevolvedBody(x, radii)</code>.</p></figcaption>
  </figure>
  <figure>
```

[![Waisted hyperboloid of revolution closed at both ends.](geometry_capped_hyperboloid.png)](geometry_capped_hyperboloid.png)

```@raw html
    <figcaption><h2>Capped hyperboloid</h2><p>Sampled meridian with <code>RevolvedBody(x, radii)</code>.</p></figcaption>
  </figure>
  <figure>
```

[![Double ogive with curved sides and two pointed tips.](geometry_double_circular_arc_ogive.png)](geometry_double_circular_arc_ogive.png)

```@raw html
    <figcaption><h2>Double circular-arc ogive</h2><p><code>RevolvedBody(profile; n=48)</code></p></figcaption>
  </figure>
</div>
```

Increasing mesh resolution only subdivides the supplied profile segments.
Increase profile sampling separately to converge a curved surface. Corners are
preserved. The last gallery revolves meridian panels. Full triangular meshing
of densely sampled profiles may require smaller elements or `mesh_order=1`
near tips and closely spaced corners. Loft section placement likewise controls geometry independently of surface element size. Superellipsoid nodes lie on the prescribed surface, while the triangular interpolants approximate it. Strong curvature and cusps may need finer meshes or linear elements.

## Named constructors and solver routes

| Shape | Public constructor | Available routes |
|:--|:--|:--|
| Sphere | `Sphere(radius)` | Modal, Kirchhoff, FEM, BEM, MFS, Fourier |
| Prolate or oblate spheroid | `Spheroid(axial, radial)` | Modal, T-matrix, Kirchhoff, FEM, BEM, MFS, Fourier |
| Straight circular cylinder | `Cylinder(radius, length)` | Modal approximation, Kirchhoff, meridian FEM, BEM, MFS |
| Bent circular cylinder | `Cylinder(r, L; radius_curvature=R)` | Full BEM/MFS, lateral-surface methods and near-broadside modal approximation |
| Capsule or domed cylinder | `Cylinder(r, L; endcap_depth=d)` | Full mesh BEM/MFS, straight-cylinder MFS |
| Triaxial ellipsoid | `Ellipsoid(a, b, c)` | Full mesh BEM, MFS, Kirchhoff |
| Superellipsoid | `Superellipsoid(a, b, c; axial_exponent=e1, equatorial_exponent=e2)` | Full mesh BEM, MFS, Kirchhoff |
| Circular cone | `Cone(radius, length)` | Axisymmetric BEM and full mesh methods |
| Circular bicone | `Bicone(radius, length; waist=0.5)` | Axisymmetric BEM and full mesh methods |
| Circular frustum | `Frustum(left_radius, right_radius, length)` | Axisymmetric BEM and full mesh methods |
| Elliptical frustum | `Frustum((a1,b1), (a2,b2), length)` | Full mesh methods |
| Circular frustum chain | `FrustumChain(x, radii)` | Axisymmetric BEM and full mesh methods |
| Elliptical frustum chain | `FrustumChain(x, semiaxes)` | Full mesh methods |
| Ring torus, circular tube | `Torus(major_radius, minor_radius)` | Axisymmetric BEM and full mesh methods |
| Ring torus, elliptical tube | `Torus(R, (axial_radius, radial_radius))` | Axisymmetric BEM and full mesh methods |

Here **full mesh methods** means BEM, MFS and mesh Kirchhoff. The table lists
geometry routes, not every material combination. See [Choosing a solver](@ref
solver-selection) for canonical bodies and the material limits below for generated
surfaces. Straight-cylinder meridian BEM/FEM use flat ends. Cylinder modal models
are approximations to finite bodies, and bent-body lateral models omit endcaps.

## Other supported geometry representations

| Family | How to represent it | Practical scope |
|:--|:--|:--|
| Smooth radial bodies of revolution | `Irregular(r, order)` | Fourier matching, subject to its mapping admissibility checks |
| Sphere or prolate spheroid shell | `Shell(body, thickness)` | Dedicated structural-shell FEM, with material restrictions |
| Confocal spheroidal shell | `Spheroid(a,b)` with `Shelled(...)` | Spheroidal T-matrix, including its supported elastic layers |
| Boxes, prisms, pyramids, polyhedra | `mesh(nodes, triangles)` or a Gmsh generator | Closed surfaces with resolved edges |
| Elliptic cones, elliptic paraboloids, general quadrics | Gmsh generator or supplied triangular surface | Finite, closed bodies, including required caps |
| General toroids, handled or concave bodies | Supplied closed surface, or `RevolvedBody` where axisymmetric | Connected, non-self-intersecting boundary |
| Bent, twisted or asymmetric biological shapes | `LoftedBody` or supplied surface | Loft approximation or externally meshed geometry |
| CAD, scanned or reconstructed objects | `mesh("body.msh")` or coordinate/connectivity arrays | Must satisfy closed-surface validation |
| Multiple fluid bodies, inclusions and cavities | One mesh per interface, coupled fluid BEM with `parents` | Disjoint or strictly nested interfaces with supported fluid materials |
| Body near an infinite planar interface | `free_surface(body, boundary, k, depth)` | Sphere or spheroid coupled to its mirror image for a pressure-release surface or rigid seafloor |

See [Geometry and incidence](@ref geometry-tutorial) for mesh import, units,
labels and coupled-region examples. A single supplied mesh describes one
connected boundary. Separate region interfaces must be passed separately.

## Solver and material limits

Generated surfaces support `Rigid()`, `PressureRelease()` and homogeneous
`FluidFilled(g,h)` or `GasFilled(g,h)` through BEM and full surface MFS. Here
`g` and `h` are the interior/exterior density and sound-speed ratios. Coupled
fluid BEM additionally handles multiple lossless homogeneous regions.

```julia
body = Torus(0.025, 0.006)
solution = bem(body, PressureRelease(), 100.0; n=64, m_max=8)
surface = mesh(body; method=:full, resolution=0.003, qorder=4)
solution_3d = bem(surface, PressureRelease(), 100.0)
```

Axisymmetric BEM uses meridian panels and azimuthal modes. Full BEM uses the
closed triangular boundary. MFS requires source offsets that remain on the
correct side of the surface, which is more difficult near corners and narrow
concavities. Kirchhoff is a high-frequency approximation whose local-normal
illumination does not account for self-shadowing or multiple reflections.

FMM is an acceleration of boundary interactions, not an additional geometry
class. There is currently no public FMM BEM path. Supported hierarchical-matrix
compression can accelerate existing full BEM formulations without changing their
geometry or material scope. See [BEM and MFS](@ref boundary-theory).

## Shapes and physics needing further support

| Requested family | What is still needed |
|:--|:--|
| Unbounded cones, paraboloids or hyperboloids | A formulation with the appropriate unbounded-domain conditions. Finite capped versions use existing mesh routes |
| Open bowls, screens, plates with zero thickness | Open-surface integral equations and edge treatment. A finite-thickness closed body can use existing mesh routes |
| Horn torus | Regularization or a formulation for its pinched self-contact. `Torus` requires `major_radius > radial_radius` |
| Self-intersecting spindle-torus parameterization | Define and trim the physical outer envelope first. A simple closed envelope can use a supplied profile or mesh, but has no dedicated constructor |
| Touching bodies or intersecting interfaces | Geometry repair or Boolean union for one body, or a dedicated contact/interface formulation |
| Arbitrary elastic solids or structural shells | Additional elastic boundary solvers or general volume/structural mesh support. Acoustic surface support alone is insufficient |
| Arbitrary spatially varying interiors | General volume geometry and material-field support beyond the existing FEM geometry families |
| Specialized exact series for another shape | A separate analytic implementation and validation. Numerical mesh support does not provide an exact series |

A named constructor is a convenience, not a prerequisite for solving a valid
closed acoustic surface. Conversely, adding a constructor alone cannot supply
missing material physics or resolve a singular boundary.
