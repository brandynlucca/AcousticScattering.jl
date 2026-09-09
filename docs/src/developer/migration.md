# [API migration](@id migration)

This guide maps older scripts to the current development API. It does not promise that removed
names still exist as deprecation aliases. The current version is `0.1.0-DEV`. Check the
installed source when using an older release or checkout.

## Solvers and outputs

| Previous pattern | Current pattern |
|:--|:--|
| Scalar `modal(...)` or `kirchhoff(...)` result | `target_strength(modal(...))` or `target_strength(kirchhoff(...))` |
| `modal(...; as = :amplitude)` | `scattering_amplitude(modal(...))` |
| Scalar `fem(...)` | `target_strength(fem(...))` |
| `shell(body, material, ...)` | `fem(body, material, ...)` with `method = :thin` or `:general` |
| `form_function(solution)` | `scattering_amplitude(solution)`, where retained by the solver |
| Abstract `Body` / `Solution` | `AbstractBody` / `AbstractSolution` |
| `MeshSolution` | `BEMSolution` or `MFSSolution`, according to solver |
| `QuadSolution` | `BEMSolution` from `method = :full` |
| `BentMFSSolution` | `MFSSolution` from a bent cylinder |
| `ShellSolution` | `FEMSolution` from structural shell FEM |

Modal and Kirchhoff solutions preserve complex amplitude. Ordinary radial/meridian FEM
preserves only target strength and rejects amplitude queries. A uniform parent type does not
imply identical retained data or post-processing capabilities.

## Shell constructors

| Previous type | Current constructor |
|:--|:--|
| `ShellSoft(g, h, ratio)` | `Shelled(FluidLayer(g, h), VacuumInterior(), ratio)` |
| `ShellFluidFilled(gs, hs, gi, hi, ratio)` | `Shelled(FluidLayer(gs, hs), FluidInterior(gi, hi), ratio)` |
| `ElasticShell(gs, cl, ct, ratio, gi, hi)` | `Shelled(ElasticLayer(gs, cl, ct), FluidInterior(gi, hi), ratio)` |
| `ShellMaterial(poisson, density, youngs_modulus)` | `Shelled(poisson, density, youngs_modulus)` |
| `ShellFEMMaterial(...)` | The same absolute-property `Shelled(...)` form |

For `ViscoelasticShell`, construct
`Shelled(LayeredMaterial(ViscousLayer(...), ElasticLayer(...), wall_ratio),
FluidInterior(...), core_ratio)`.
The exterior sound speed and two kinematic viscosities belong to `ViscousLayer`.
See [Materials and shells](@ref materials-tutorial) for a full executable example.

The geometric `Shell(body, thickness)` remains in the structural FEM interface. It has not
been replaced by `Shelled`, which describes material/boundary configuration.

## Mesh and low-level names

Use `mesh(Sphere(radius); resolution = count)` in place of `sphere_mesh(radius, count)`,
and the analogous `Spheroid` or `Cylinder` constructors for their old mesh helpers.
The result is `Mesh`. `resolution` has method-dependent units, as described in the [API reference](@ref api-reference).

`Panel`, `panels`, `npanels`, `bem_panel_count`, `reflection_coefficient`,
`principal_curvatures`, `form_function`, and `backscattering_cross_section` are no
longer exported. Do not make new application code depend on their storage or implementation
details. Mesh inspection functions currently remain qualified, unexported functions. Their
future public status is still being consolidated. Use `mesh(...; k = wavenumber)` for
automatic resolution instead of the BEM panel-count helper.

If a complex amplitude is available, its squared magnitude gives the package's cross-section
convention: `abs2(scattering_amplitude(solution))`. Reflection and curvature computations
now belong inside the selected model rather than the ordinary application workflow.
