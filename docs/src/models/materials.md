# [Geometry and materials](@id geometry-materials)

Geometry specifies shape, while materials and boundaries specify acoustic response. See [Materials and shells](@ref materials-tutorial) for constructor examples.

## Geometry and discretization

`Sphere(radius)` uses a positive radius in meters. `Spheroid(axial_radius, equatorial_radius)` is prolate when its first radius is larger and oblate when it is smaller. Its surface satisfies

```math
\frac{x^2+y^2}{b^2}+\frac{z^2}{a^2}=1.
```

`Cylinder(radius, length; radius_curvature=Inf, endcap_depth=0.0)` uses meters. Finite curvature bends the cylinder. Positive cap depth adds half-spheroid ends in full meshes and straight-cylinder MFS. Axisymmetric BEM/FEM use flat caps. Full-BEM bends lie in the `xy` plane, with midpoint tangent along `+x`.

`Shell(body, thickness)` specifies structural shell geometry. `Shelled` specifies the material and interior configuration.

`mesh(body; resolution)` returns `Mesh`. Resolution means panel count for `:axisymmetric` and target edge length in meters for `:full`. Increase panel count or decrease edge length to refine. Axisymmetric meshes represent a revolved meridian.

## Fluid boundaries

On a rigid surface the total normal pressure derivative is zero. On a pressure-release surface the total pressure is zero. At a fluid interface,

```math
p_{\mathrm{ext}}=p_{\mathrm{int}},\qquad
\frac{1}{\rho_{\mathrm{ext}}}\partial_n p_{\mathrm{ext}}
=\frac{1}{\rho_{\mathrm{int}}}\partial_n p_{\mathrm{int}}.
```

Both derivatives here use the same geometrical normal. `FluidFilled(g, h)` uses `g = density_interior / density_exterior` and
`h = sound_speed_interior / sound_speed_exterior`. `GasFilled` is its alias. 

`SpatialFluid(g, c; min_soundspeed_contrast)` accepts positive constants or position callbacks for lossless volume-FEM regions. Callbacks use global Cartesian coordinates in meters, with default symmetry axis `+z`. Moving or rotating a region does not transform its profiles. The minimum speed contrast controls mesh wavelength and must bound sampled values. See [Full 3D volume FEM](@ref).

## Elastic and layered boundaries

An elastic solid supports longitudinal and shear waves. For an isotropic material,

```math
c_L^2=(\lambda+2\mu)/\rho_s,\qquad c_T^2=\mu/\rho_s.
```

Normal displacement and normal traction couple to the fluid. Tangential traction vanishes at an inviscid-fluid interface. `SolidElastic(g, longitudinal_contrast, shear_contrast)` describes a solid body with no cavity.

`Shelled(material, interior, radius_ratio)` combines fluid or elastic layers with a fluid or vacuum core where supported. All contrasts, including core contrasts, are relative to the exterior fluid. A vacuum core imposes zero pressure on a fluid shell.

`LayeredMaterial(outer, inner, interface_ratio)` composes concentric layers. Its interface ratio refers to the enclosing layer, while the `Shelled` core ratio refers to the body. Thus `LayeredMaterial(f1, LayeredMaterial(f2, f3, 0.75), 0.8)` places interfaces at 0.8 and 0.6 of the body radius, requiring a core ratio below 0.6. Fluid interfaces match pressure and $(1/\rho)\partial_r p$. Fluid-solid interfaces match normal displacement and traction with zero tangential traction. Bonded solids match displacement and traction, using the same radial direction on both sides.

Spherical `modal` supports fluid/elastic stacks with `interior_coupling=:generalized`. Confocal spheroids use volume FEM or the bounded [mixed-layer T-matrix](@ref tmatrix-theory). The viscous monopole model supports a `ViscousLayer`, elastic wall and fluid core. Acoustic pressure is defined only in fluid regions.

Volume FEM provides displacement [m], velocity [m/s] and stress [Pa] through `solution(points; quantity=...)`. See [Material fields in volume FEM](@ref fem-material-fields) for normalization and supported regions.

Structural shell FEM uses `Shelled(poisson, density, youngs_modulus)` with absolute density [kg/m^3] and modulus [Pa]. Supply fluid properties to `fem`.

`ViscoelasticSolid`, `ViscousLayer` and `SpatialFluid` do not supply pore-pressure or temperature fields. See [Surface waves, guided waves and resonances](@ref wave-diagnostics) for physics outside these models.

See [References](@ref references) for the underlying scattering and elasticity models.
