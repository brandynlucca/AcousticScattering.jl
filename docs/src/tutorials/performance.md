# [Performance](@id performance-tutorial)

Separate compilation, solution, and post-processing timings. Establish a converged small case before increasing frequency, resolution, or sweep size.

## Compilation and threads

Start Julia with `julia --project=. -t auto` to make threads available. Actual parallelism depends on the implementation and dependency constraints. Special-function backends and shared caches are not automatically safe to call from arbitrary parallel loops.

Compare `@time` on first and second identical calls in a session. For repeated source edits, consider the [precompile opt-out](@ref getting-started). It shifts compilation cost to first use.

When BLAS and Julia threads compete, compare configurations explicitly. For example, `using LinearAlgebra: BLAS; BLAS.set_num_threads(1)` changes the entire process, so use it as a measured tuning choice. Do not assume more threads always improve performance.

## Accuracy and memory controls

For fluid/gas full-3D BEM at a fixed frequency, pass meshes directly to `incidence_angle_sweep(surface, material, k, angles)` or
`incidence_angle_sweep(surfaces, materials, k, angles; parents)`. These forms assemble and factorize once, then solve each incident forcing using that factorization. The callback form executes its callback at every angle. Add `components=true` to the multiple-interface form to compare coupled and isolated responses. Each system is sampled separately, and the result retains only amplitudes and strengths. Changing frequency, material, mesh or numerical options requires a new call and assembly. See [the fish tutorial](@ref fish-tutorial) for an example.

The single-mesh overload also accepts `Rigid()` and `PressureRelease()`. It reuses assembled operators and their compression, while each angle starts a fresh GMRES iteration with the supplied `gmres_kwargs`. See [closed bent cylinders](@ref bent-cylinder-tutorial) for a complete example. These iterative solves do not use a dense LU factorization.

| Method | Controls | Cost or limitation |
|:--|:--|:--|
| Sphere modal | `m_max` | Series convergence near resonances |
| Spheroid modal | `m_max`, `n_max`, fluid `coupling` | Special functions and dense coupling |
| Axisymmetric BEM | `n`, `m_max`, quadrature | Dense systems per mode and near-singular integration |
| Full BEM | `meshsize`, `compression`, `gmres_kwargs` | Compressed rigid/soft systems, dense fluid transmission |
| Radial FEM | Element counts, `order` where supported, `m_max` | Retains target strength only |
| Meridian FEM | Radial/angular grids and modal orders | Two-dimensional system per mode |
| MFS | `n`, `offset`, modal orders | Conditioning depends on source placement |

The documentation's reduced cases are not problem-size benchmarks. Dense matrix memory scales quadratically with unknown count. Factorization cost grows faster. Record numerical controls, hardware, Julia version, thread settings, and whether caches were warm with any benchmark.

See [Building and contributing](@ref developer-guide) for documentation execution and testing.
