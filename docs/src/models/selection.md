# [Choosing a solver](@id solver-selection)

Start with the least costly method that supports both your physics and your desired output.
The table lists supported combinations. All five solver families return solution objects.

| Geometry and configuration | Modal | Kirchhoff | FEM | BEM | MFS |
|:--|:--|:--|:--|:--|:--|
| Sphere, rigid/pressure-release/fluid | Yes | Yes, physical optics | Radial or axial meridian | Axisymmetric or full | Axisymmetric |
| Sphere, solid elastic | Yes | No implemented reflection model | Radial | No | No |
| Sphere, fluid shell + vacuum/fluid interior | Yes | Layer-reflection approximation | Radial | Axial two-surface | No |
| Sphere, elastic shell + fluid interior | Yes | No | Radial | No | No |
| Sphere, viscous + elastic layers + fluid interior | Monopole only | No | No | No | No |
| Prolate/oblate spheroid, rigid/pressure-release/fluid | Yes | Yes, physical optics | Meridian | Axisymmetric or full | Axisymmetric |
| Supplied closed triangular surface, rigid/pressure-release/fluid | No | No | No | Full, via `bem(mesh(...), ...)` | No |
| Nested or disjoint closed interfaces, homogeneous fluids | Spherical shell reference | No | No | Coupled full 3D, via `bem(surfaces, materials, k; parents)` | No |
| Straight cylinder, rigid/pressure-release/fluid | Finite-length approximation | Lateral surface and caps | Meridian | Axisymmetric | Axisymmetric, preferably with smooth caps |
| Straight cylinder, solid elastic/elastic shell + fluid | Finite-length approximation | No | Radial reduction | No | No |
| Bent cylinder, rigid/pressure-release | Near-broadside correction | Curved-surface integral | No curved geometry | Full 3D, closed ends | Closed surface via `mfs(mesh, ...)`; lateral only via `mfs(body, ...)` |
| Bent cylinder, fluid | Near-broadside correction | Curved-surface approximation | No curved geometry | Full 3D, closed ends | No |
| Structural `Shell` geometry, absolute elastic material | No | No | Coupled thin/general shell | Through coupled FEM | No |

For structural shell FEM, `:thin` supports prolate spheroids at axial incidence. The `:general` method
supports sphere and prolate-spheroid shell geometries and a fluid interior.

## Choosing by question

- Use sphere or spheroid modal methods as analytical references, while checking series
  convergence and special-function conditioning.
- Use Kirchhoff to explore physical-optics behavior. A numerically converged surface integral
  does not establish validity of physical optics at low frequency.
- Use FEM or BEM when checking analytical reductions, geometry discretization, or interface
  coupling. Compare against a canonical case first.
- Use MFS when source placement is well controlled and its supported geometry suits the problem.
- Use BEM surface results for repeated observation-angle queries. Supported radial FEM spheres
  also provide complex backscatter; cylinder radial and meridian FEM paths retain target
  strength only.

Finite-cylinder modal and radial reductions are not exact closed-finite-cylinder solutions.
FEM's cylinder routes do not model bend curvature. `endcap_depth` affects full-3D cylinder
meshes and straight-cylinder MFS; axisymmetric BEM/FEM use flat caps.

Full 3D volume FEM is not implemented. Full BEM accepts supplied closed triangular surfaces,
including non-axisymmetric and nonconvex bodies. Built-in shape meshing supports spheres and
spheroids and closed cylinders. Each supplied mesh must form one connected boundary.
Pass separate meshes and fluid materials to `bem(surfaces, materials, k; parents)` for
coupled regions; all contrasts refer to the unbounded exterior. Rigid/pressure-release full BEM uses Burton–Miller coupling by default;
`formulation=:cbie` selects the conventional equation. Axisymmetric fluid-transmission BEM
has a limited optional CHIEF augmentation. These treatments have distinct boundary-condition
and solver scopes; see [BEM and MFS](@ref boundary-theory).

Full-BEM fluid transmission uses a dense Müller system with row and column equilibration.
Its two traces require fewer unknowns than the four-trace `formulation=:cbie` alternative.
High contrast and sharp resonances still require mesh and quadrature convergence checks.
