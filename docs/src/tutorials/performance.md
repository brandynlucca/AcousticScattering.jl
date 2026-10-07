# [Performance](@id performance-tutorial)

Choose a solver for the geometry and material, then establish numerical convergence before tuning runtime. Compare complete solves at the same accuracy, including assembly, factorization and post-processing.

## Timing and memory

Measure the first call separately from repeated calls because Julia compiles code on first use. Start Julia with `julia --project=. -t auto` to enable multiple threads. Compare thread counts on your workload. Special-function backends and shared caches may restrict concurrent calls.

If Julia threads compete with BLAS threads, test one BLAS thread:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```

Record allocation traffic separately from peak memory and retained solution storage. Dense matrix storage grows quadratically with the number of unknowns. A smaller allocation count does not necessarily reduce peak memory.

## Reuse across angles and observation directions

Use an incidence-sweep overload to reuse assembly and factorization at fixed frequency:

```julia
sweep = incidence_angle_sweep(surface, material, k, angles)
```

For coupled fluid regions, pass the meshes and materials together with `parents`. Add `components=true` to compare coupled and isolated responses. For axisymmetric MFS, use `incidence_angle_sweep(mfs, body, boundary, k, angles)`.

These overloads retain sampled amplitudes and target strengths. The callback form evaluates its solver at each angle. Changing the frequency, material, mesh or numerical options requires a new assembly.

Use `bistatic_sweep` or `bistatic_map` to sample additional observation directions from a retained solution. These queries reuse the solved field.

## BEM compression

For full rigid or pressure-release BEM and fluid Müller BEM, compare dense operators with `compression=(method=:hmatrix, tol=1e-9)`. Compression can reduce memory at larger sizes but adds setup cost. Dense solves can be faster for small systems.

Compressed fluid incidence sweeps reuse a bounded solution subspace with `recycle_dimension=8`. Set it to zero to compare independent GMRES starts. Each solve checks the current residual and retries if reuse fails. This reuse applies at fixed frequency.

Refine mesh size and quadrature independently of compression and GMRES tolerances. Small linear residuals alone do not establish physical accuracy. See [BEM and MFS](@ref boundary-theory).

## Mixed-precision fluid BEM

For dense single-interface fluid or gas BEM, `precision=:mixed` uses ComplexF32 factors with ComplexF64 residual refinement:

```julia
solution = bem(surface, material, k; formulation=:cbie,
    precision=:mixed, condition_limit=0)
report = diagnostics(solution)
report.factor_precision
report.componentwise_backward_error
```

Keep `equilibrate=true`. Both normwise and componentwise backward errors must satisfy the refinement tolerance. Conditioning or failed refinement triggers ComplexF64 factorization. `refinement=(tolerance=1e-13, maxiter=8)` changes the acceptance tolerance and correction budget. The default remains `precision=:double`.

The dense single-interface incidence sweep accepts the same options and reuses the factorization across angles. Promotion can make mixed precision slower than a direct double-precision solve, particularly near resonances.

## Reuse across frequencies

For dense single-interface fluid CBIE on a fixed mesh, supply training frequencies to build a reduced solution basis:

```julia
result = frequency_sweep(surface, material, [150.0, 200.0, 250.0], 1500.0;
    training_frequencies=[120.0, 180.0, 240.0, 300.0], return_diagnostics=true)
result.sweep.amplitudes
result.diagnostics.queries
```

Frequencies are in Hz and sound speed is in m/s. Resolve the mesh over the full frequency range. The sweep fixes the training basis, rebuilds the equations at each query, and checks original and scaled residuals. Queries that fail acceptance use a full Float64 solve.

`capacity=16` and `maxbytes=64*1024^2` limit snapshot coefficients. They do not limit total process memory. Training can cost more than it saves for a short sweep. Compare training plus query time with independent solves.
