# [Geometry and materials](@id geometry-materials)

Geometry specifies shape. Boundary or material configuration specifies its acoustic response.
Keep these separate when comparing methods. See [Materials and shells](@ref materials-tutorial)
for executable constructor examples.

## Geometry and discretization

`Sphere(radius)` uses a positive radius in meters.
`Spheroid(axial_radius, equatorial_radius)` is prolate when its first radius is larger and
oblate when it is smaller. Its surface satisfies

```math
\frac{x^2+y^2}{b^2}+\frac{z^2}{a^2}=1.
```

`Cylinder(radius, length; radius_curvature = Inf, endcap_depth = 0.0)` uses meters.
A finite curvature radius selects bending only in methods that implement it. Its positive
cap-depth setting selects the smooth-cap MFS construction. It is not a universal shape option.

`Shell(body, thickness)` specifies structural shell geometry. `Shelled` specifies the material
and interior configuration.

`mesh(body; resolution)` returns `Mesh`. Axisymmetric discretization represents a meridian
curve that is revolved during integration. It is not a flat 2D obstacle. Full meshes represent
a surface via Gmsh/Inti quadrature. Resolution is panel count for `:axisymmetric`, target edge
length in meters for `:full`. Increasing panel count refines a mesh, while increasing edge length makes it coarser.

## Fluid boundaries

On a rigid surface the total normal pressure derivative is zero. On a pressure-release surface
the total pressure is zero. At a fluid interface,

```math
p_{\mathrm{ext}}=p_{\mathrm{int}},\qquad
\frac{1}{\rho_{\mathrm{ext}}}\partial_n p_{\mathrm{ext}}
=\frac{1}{\rho_{\mathrm{int}}}\partial_n p_{\mathrm{int}}.
```

Both derivatives here use the same geometrical normal. `FluidFilled(g, h)` uses
`g = density_interior / density_exterior` and
`h = sound_speed_interior / sound_speed_exterior`. `GasFilled` is its alias.

## Elastic and layered boundaries

An elastic solid supports longitudinal and shear waves. For an isotropic material,

```math
c_L^2=(\lambda+2\mu)/\rho_s,\qquad c_T^2=\mu/\rho_s.
```

Normal displacement and normal traction couple to the fluid. Tangential traction vanishes
at an inviscid-fluid interface. `SolidElastic(g, longitudinal_contrast, shear_contrast)`
describes a solid body with no cavity.

`Shelled(material, interior, radius_ratio)` combines `FluidLayer` or `ElasticLayer` with
`VacuumInterior` or `FluidInterior` where supported. All contrasts use the exterior fluid,
including the cavity contrasts. A fluid shell with vacuum interior imposes zero pressure
at its inner radius. An elastic shell with fluid interior includes elastic stresses.

`LayeredMaterial(outer, inner, interface_ratio)` represents two layers, but the implemented
viscous model specifically supports viscous outer material over an elastic wall and a fluid
core. This generic constructor does not implement arbitrary stacks. The interface ratio and
core ratio both refer to the outer body radius and must be physically ordered.

Structural shell FEM uses `Shelled(poisson, density, youngs_modulus)` with absolute units
(kg/m³ and Pa). Its fluid properties are supplied to `fem`. This is a different constructor
from a contrast-based layer model, and the supported material laws are not arbitrary callbacks.

These interface descriptions follow the equations used in the package. Sphere/shell benchmark
context and elastic references appear in [References](@ref references).
