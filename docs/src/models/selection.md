# [Choosing a solver](@id solver-selection)

Start with the least costly method that supports both your physics and your desired output.
This table describes current implementations, not every combination accepted by an outer Julia
method signature. All five public solver families return solution objects.

| Geometry and configuration | Modal | Kirchhoff | FEM | BEM | MFS |
|:--|:--|:--|:--|:--|:--|
| Sphere, rigid/pressure-release/fluid | Yes | Yes, physical optics | Radial or axial meridian | Axisymmetric or full | Axisymmetric |
| Sphere, solid elastic | Yes | No implemented reflection model | Radial | No | No |
| Sphere, fluid shell + vacuum/fluid interior | Yes | Layer-reflection approximation | Radial | Axial two-surface | No |
| Sphere, elastic shell + fluid interior | Yes | No | Radial | No | No |
| Sphere, viscous + elastic layers + fluid interior | Monopole only | No | No | No | No |
| Prolate/oblate spheroid, rigid/pressure-release/fluid | Yes | Yes, physical optics | Meridian | Axisymmetric or full | Axisymmetric |
| Straight cylinder, rigid/pressure-release/fluid | Finite-length approximation | Lateral surface and caps | Meridian | Axisymmetric | Axisymmetric, preferably with smooth caps |
| Straight cylinder, solid elastic/elastic shell + fluid | Finite-length approximation | No | Radial reduction | No | No |
| Bent cylinder, rigid/pressure-release | Near-broadside correction | Curved-surface integral | No curved geometry | No | 3D |
| Bent cylinder, fluid | Near-broadside correction | Curved-surface approximation | No curved geometry | No | No |
| Structural `Shell` geometry, absolute elastic material | No | No | Coupled thin/general shell | Through coupled FEM | No |

For structural shell FEM, `:thin` supports prolate spheroids at axial incidence. The `:general` method
supports sphere/spheroid shell geometries and a fluid interior. Unsupported combinations can
currently fail inside the implementation with a `MethodError`. Do not infer support from a
successful constructor.

## Choosing by question

- Use sphere or spheroid modal methods as analytical references, while checking series
  convergence and special-function conditioning.
- Use Kirchhoff to explore physical-optics behavior. A numerically converged surface integral
  does not establish validity of physical optics at low frequency.
- Use FEM or BEM when checking analytical reductions, geometry discretization, or interface
  coupling. Compare against a canonical case first.
- Use MFS when source placement is well controlled and its supported geometry suits the problem.
- Use BEM surface results for repeated observation-angle queries. Ordinary radial/meridian FEM
  currently retains only target strength.

Finite-cylinder modal and radial reductions are not exact closed-finite-cylinder solutions.
FEM's cylinder routes do not model bend curvature. `endcap_depth` affects MFS, while ordinary
BEM/FEM cylinder routes use flat caps.

Full 3D volume FEM is not implemented. Full BEM's public mesh generation is limited to sphere
and spheroid bodies. Full BEM has no Burton–Miller/CHIEF implementation. Axisymmetric
fluid-transmission BEM has a limited optional CHIEF augmentation. It is not universal
regularization for all BEM paths.
