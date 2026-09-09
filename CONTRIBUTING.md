# Contributing

Run commands from the repository root. Keep numerical changes separate from formatting changes.
Add tests for new behavior and update the relevant model page or tutorial.

## Tests

Run the complete suite, including plotting:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

For numerical groups, install the smaller test environment:

```sh
julia --project=test/core -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --check-bounds=yes --project=test/core test/runtests.jl Spheroidal
```

The numerical groups are `Analytical`, `RadialFEM`, `MeridianFEM`, `Boundary`, `Spheroidal`,
`Oblique`, and `Interfaces`. `Core` runs all seven.

Plotting groups use the full test environment:

```sh
julia --project=test -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --check-bounds=yes --project=test test/runtests.jl Plots1D
```

The plotting groups are `Plots1D`, `Plots2D`, and `Plots3D`. `All` runs every group and is the
default. The `GROUP` environment variable can select a group instead of the command-line argument.
Each group can also be run directly, for example `julia --project=test/core test/spheroidal.jl`.

CI prepares dependencies once per environment and Julia version, then runs groups in separate
jobs with bounds checking. Numerical jobs do not load CairoMakie. Routine tests disable the
optional precompile workload, which is covered separately by the installation checks:

```sh
julia --startup-file=no test/precompile.jl standard
julia --startup-file=no test/precompile.jl opt-out
```

These scripts create isolated projects. Local runs reuse the Julia depot. Installation CI uses
fresh depots. Keep Fortran-backed calculations sequential within each process until the backend
supports concurrent calls.

## Formatting

Use the same JuliaFormatter version as CI:

```julia
import Pkg
Pkg.activate(; temp = true)
Pkg.add(Pkg.PackageSpec(; name = "JuliaFormatter", version = "2.14.0"))
using JuliaFormatter: format
format(["src", "ext", "test", "docs"])
```

Pass `overwrite = false` to check formatting without modifying files.

## Documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The site is written to `docs/build`. The build runs doctests and executable examples. Local
builds do not publish. Pull requests receive a downloadable `documentation` artifact.

For publication, repository maintainers must select **Settings > Pages > Source > GitHub
Actions** and allow `main` and release tags in the `github-pages` environment. The Documentation
workflow publishes successful upstream `main` and `v*` builds. If Pages configuration fails,
correct the settings and use **Re-run failed jobs** while the build artifact is available.
No personal access token is needed for this workflow.

Run the deployment-guard checks with `julia --project=docs test/documentation.jl`.
