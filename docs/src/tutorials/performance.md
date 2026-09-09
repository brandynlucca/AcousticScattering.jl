# [Performance](@id performance-tutorial)

Separate compilation, solution, and post-processing timings. Establish a converged small case
before increasing frequency, resolution, or sweep size.

## Compilation and threads

Start Julia with `julia --project=. -t auto` to make threads available. Actual parallelism depends
on the implementation and dependency constraints. Special-function backends and shared caches
are not automatically safe to call from arbitrary parallel loops.

Compare `@time` on first and second identical calls in a session. For repeated source edits,
consider the [precompile opt-out](@ref getting-started). It shifts compilation cost to first use.

When BLAS and Julia threads compete, compare configurations explicitly. For example,
`using LinearAlgebra: BLAS; BLAS.set_num_threads(1)` changes the entire process, so use it as a
measured tuning choice. Do not assume more threads always improve performance.

## Accuracy and memory controls

| Method | Controls | Cost or limitation |
|:--|:--|:--|
| Sphere modal | `m_max` | Series convergence near resonances |
| Spheroid modal | `m_max`, `n_max`, fluid `coupling` | Special functions and dense coupling |
| Axisymmetric BEM | `n`, `m_max`, quadrature | Dense systems per mode and near-singular integration |
| Full BEM | `meshsize`, `compression`, `gmres_kwargs` | Compressed rigid/soft systems, dense fluid transmission |
| Radial FEM | Element counts, `order` where supported, `m_max` | Retains target strength only |
| Meridian FEM | Radial/angular grids and modal orders | Two-dimensional system per mode |
| MFS | `n`, `offset`, modal orders | Conditioning depends on source placement |

The documentation's reduced cases are not problem-size benchmarks. Dense matrix memory scales
quadratically with unknown count. Factorization cost grows faster. Record numerical controls,
hardware, Julia version, thread settings, and whether caches were warm with any benchmark.

See [Building and contributing](@ref developer-guide) for documentation execution and testing.
