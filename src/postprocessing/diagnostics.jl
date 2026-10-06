"""
    diagnostics(solution)

Return a named tuple of solver diagnostics for BEM, MFS, FEM, Fourier matching and
the far-field-generated T-matrix. Modal, Kirchhoff and other T-matrix paths return `nothing`.

`tmatrix(...; method=:farfield)` reports angular orders, `holdout_error`,
`fem_solves`, `fem_assemblies`, mesh sizes, domain radius, DtN order and system size.

Fourier matching reports `admissible`, `m_max`, `n_max`, `convergence`, `convergence_tolerance` and
`converged`, which is `nothing` when `n_max` is too small to reduce.

Full-3D BEM reports `converged`, `iterations`, `relative_residual`, `residual_history`,
`unknown_count`, `meshsize`, `quadrature_order`, and the solver settings used. Axisymmetric
BEM/MFS/FEM report per-system entries in `systems`. MFS systems also report source counts,
`condition_number`, `numerical_rank` and a held-out `boundary_residual`.

See [BEM and MFS](@ref boundary-theory) for field definitions and what each residual does and
does not measure.

# Example
```julia
solution = bem(Sphere(0.01), Rigid(), 100.0; method = :full, meshsize = 0.01)
diagnostics(solution).converged
```
"""
diagnostics(::AbstractSolution) = nothing
