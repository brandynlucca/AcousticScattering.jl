# [Building and contributing](@id developer-guide)

This website uses Documenter, executable `@example` blocks, and doctests. Tutorials are the
source of their generated figures and numerical output.

## Build locally

From the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The HTML site is written to `docs/build/index.html`. The generated CSV and PNG files live
under `docs/build/tutorials`. Build failures include invalid references, doctest mismatches,
and failed executable examples. The existing docs GitHub Actions job runs the same build.

The docs environment includes CairoMakie for headless plots. A local build may take substantial
time on first use because Documenter, plotting, and numerical dependencies need compilation.
It does not require a display server.

## What the build checks

The sphere tutorials assert agreement with existing rigid/fluid regression values. The
frequency tutorial checks modal truncation. Convergence examples compare FEM and BEM with a
modal sphere. Material and geometry examples also check finite outputs. Finite-output tests
are smoke checks, not independent model validation. The full package tests remain necessary:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Examples on different pages are independent. Blocks on one page use a named Documenter session.
Copy all preceding blocks on that page when reproducing a later calculation. The structural
shell example is explicitly unexecuted because it is more expensive.

See the repository's
[contributor instructions](https://github.com/brandynlucca/AcousticScattering.jl/blob/main/CONTRIBUTING.md)
for individual test groups and their environments.

## Reference and source docstrings

The public reference is curated prose. `checkdocs = :none` disables source-docstring coverage
checking. Doctests, executable examples, and unresolved-reference errors still fail the build.
A successful website build does not establish source-docstring coverage.

Prefer interface descriptions and public accessors over undocumented solution storage.
When adding a capability, update the support table, relevant theory page, and an executable example.

## Style and citations

Follow the [SciML style guide](https://docs.sciml.ai/SciMLStyle/stable/) and the repository's
JuliaFormatter configuration. Use descriptive ASCII names in examples and semicolons before
keywords. Keep numerical algorithm changes separate from documentation changes.

CI pins JuliaFormatter directly in its formatting job. To use the same version locally, start
Julia from the repository root and run:

```julia
import Pkg
Pkg.activate(; temp = true)
Pkg.add(Pkg.PackageSpec(; name = "JuliaFormatter", version = "2.14.0"))
using JuliaFormatter: format
format(["src", "ext", "test", "docs"]; overwrite = false)
```

The check returns `true` when all files are formatted. To apply changes, call
`format(["src", "ext", "test", "docs"])`. It uses `.JuliaFormatter.toml` and leaves Markdown
prose and embedded examples unchanged. Coordinate with other contributors before formatting
files they are editing. Update the version in `.github/workflows/CI.yml` and this example
together, and review the resulting formatting changes before committing.

Review API naming, type piracy, and test organization separately from automatic formatting.

Use a persistent primary-source link near literature claims and add author–year metadata to
[References](@ref references). Distinguish model assumptions from numerical approximations.
Verify supported combinations against the implementation and tests.
