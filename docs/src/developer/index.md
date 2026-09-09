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
julia --project=. -t auto test/runtests.jl
```

Examples on different pages are independent. Blocks on one page use a named Documenter session.
Copy all preceding blocks on that page when reproducing a later calculation. The structural
shell example is explicitly unexecuted because it is more expensive.

## Reference and source docstrings

Verify the installation preference recipe with `julia docs/check_precompile.jl` from the
repository root. This now uses an empty Julia depot and a manifest-free source snapshot,
checks numerical output with the workload disabled, and re-enables it in a new process.
See [CI and installation checks](@ref ci-guide) for both installation modes and a faster,
explicitly cache-reusing option.

The public reference is curated prose. `checkdocs = :none` temporarily disables source-docstring
coverage checking because legacy docstrings still refer to removed API names and internal
fields. This does **not** disable doctests, executable examples, or unresolved-reference errors.
Migrating exported source docstrings and restoring `checkdocs = :exports` remain open tasks.
A successful website build does not establish source-docstring coverage.

Prefer interface descriptions and public accessors over undocumented solution storage.
When adding a capability, update the support table, relevant theory page, an executable example,
and the checklist. Check an item only after its stated acceptance criteria are met.

## Style and citations

Follow the [SciML style guide](https://docs.sciml.ai/SciMLStyle/stable/) and the repository's
JuliaFormatter configuration. Use descriptive ASCII names in examples and semicolons before
keywords. Keep numerical algorithm changes separate from documentation changes.

The `dev` environment pins JuliaFormatter to the same version used by CI. From the repository
root, install it and run the read-only check:

```sh
julia --project=dev -e 'using Pkg; Pkg.instantiate()'
julia --project=dev dev/format.jl --check
```

To apply formatting, run `julia --project=dev dev/format.jl` without `--check`. It formats Julia
files under `src`, `ext`, `test`, `docs`, and `dev` using `.JuliaFormatter.toml`. Markdown prose
and embedded examples are unchanged. Coordinate with other contributors before formatting
files they are editing. Upgrade the formatter pin in `dev/Project.toml` deliberately, and
review any resulting formatting changes before committing.

The formatter does not check API naming, type piracy, or test organization. Those require the
separate manual checks in the documentation and API checklist.

Use a persistent primary-source link near literature claims and add author–year metadata to
[References](@ref references). Label package-derived formulas and unverified extensions.
Do not infer support from a literature inventory, an old roadmap, or an outer dispatch signature.
