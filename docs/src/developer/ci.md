# [CI and installation checks](@id ci-guide)

Pull requests run tests, formatting checks, isolated installation checks, and the documentation
build. The dedicated Documentation workflow publishes successful builds from upstream `main`
and `v*` tags, including manual runs on those refs. Local documentation builds never deploy.

## Isolated installation checks

From the repository root, check the precompile opt-out and re-enable paths:

```sh
julia --startup-file=no test/precompile.jl
```

To check normal installation, select the standard mode:

```sh
julia --startup-file=no test/precompile.jl standard
```

The single test script creates a temporary project and a copy of the package source without
local manifests or preferences. It rejects local dependency overrides. Local runs reuse your Julia
depot. CI sets an empty depot for each matrix job and does not restore an installation cache.
System tools, network settings, and the installed Julia runtime are still used, so this is not
an isolated operating-system test.

The standard mode checks default precompilation. The opt-out mode sets the workload preference
before the first import, checks it, re-enables it, and checks again in a new Julia process.
Both modes use strict precompilation and test rigid-sphere, fluid-sphere, and spheroid outputs.
The spheroid calculation exercises the registered dependency's native backend.

Local runs test your working files, including uncommitted changes. CI tests the GitHub checkout,
so an untracked file can make a local run pass while CI fails. Include all required source files
in the package commit. These checks do not establish that a published registry version installs.

Install the [native build prerequisites](@ref getting-started) before running these commands.
The CI installation jobs intentionally do not restore a Julia cache.

### Local verification record

Before the test scripts were consolidated, both modes passed on September 9, 2026 on Windows
with Julia 1.12.1 and registered
SpheroidalWaves 0.3.0. The default, disabled, and re-enabled workload probes passed all
12 assertions. The dependency built with CMake and GNU Fortran from a MinGW-w64 toolchain.
This does not establish results for Linux or Julia 1.10 before CI runs there.

The measured strict-precompile steps took about 252 seconds for the standard case, 69 seconds
with the workload disabled, and 193 seconds after re-enabling it. These exclude earlier package
downloads and native builds. Other Julia jobs were active, so these are run logs, not a controlled
performance comparison or installation-time guarantee.

The runs emitted upstream warnings from SpheroidalWaves about backend configuration during
incremental compilation and from Inti about an unqualified `Array` constructor. Strict
precompilation and the subsequent native-backend numerical checks still passed. Retain these
warnings when reporting installation problems rather than treating this run as warning-free.

## Workflow coverage

The package workflow in `.github/workflows/CI.yml` has three jobs:

- `format` checks Julia files with the pinned SciML formatter.
- `test` runs the package tests on Julia 1.10 and the latest stable Julia, on Linux.
- `install` runs both isolated installation modes on those Julia versions.

The separate `.github/workflows/Documentation.yml` workflow has two jobs:

- `docs` builds the site, runs doctests and executable examples, and uploads the HTML as the
  `documentation` artifact. This job has read-only repository permissions.
- `docs-deploy` waits for the docs build before publishing the validated artifact. It checks
  that GitHub Pages is enabled before updating the versioned site.

Documentation deployment does not wait for the full package test and installation matrix.
Package CI remains separate and should still be required before merging code changes.

As of September 9, 2026, Julia 1.10 is both the package's minimum supported series and the
current LTS series. Review this matrix when the LTS changes, retaining minimum-version coverage.
The `1` selector tracks the current stable release.
[Julia release listings](https://julialang.org/downloads/manual-downloads/).

To preview a pull request, download the `documentation` artifact from its Actions run. Extract
it and serve that directory with a local HTTP server, since CI uses directory-style URLs.
Pull requests do not publish to `gh-pages` or receive a deployment token.

## Enable deployment on GitHub

Repository maintainers must complete these settings after the workflow is pushed:

1. In **Settings > Pages**, select **GitHub Actions** as the publishing source.
2. Configure the `github-pages` environment to permit `main` and the intended release tags.
3. Confirm repository or organization policy permits the deployment job's token permissions.
4. After the first successful run, select the appropriate format, test, installation, and docs
   checks in the protected-branch rules or ruleset.
5. Inspect the published site and a versioned release before considering deployment verified.

Once the workflow is on `main`, open **Actions > Documentation > Run workflow** and select
`main` to build and publish without making another commit. Feature-branch and fork runs can
build previews but cannot deploy. Ordinary pushes to `main` and `v*` tags trigger it automatically.

`docs/deploy.jl` uses Documenter to update the versioned `gh-pages` tree. The workflow then
packages that tree and publishes it using the GitHub Pages actions. This explicit deployment
is needed because a push made with `GITHUB_TOKEN` does not trigger a separate Pages build.
No personal token or SSH deployment key is required by this workflow.
[Documenter hosting](https://documenter.juliadocs.org/stable/man/hosting/) and
[GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
See also [GitHub token event behavior](https://docs.github.com/en/actions/concepts/security/github_token).

The first main-branch deployment publishes development documentation. Documenter's stable
version requires a suitable release tag. This workflow validates tags, but does not create
releases, register the package, or configure repository protection rules.

## Dependency updates

`.github/dependabot.yml` schedules weekly pull requests for Julia dependencies in the root,
and `docs` projects, and for GitHub Actions versions. Dependabot replaces the checklist's
original CompatHelper proposal, following the maintainers' recommendation.
[CompatHelper status](https://juliaregistries.github.io/CompatHelper.jl/dev/).

Review compatibility changes and run CI before merging. The formatter version is pinned inline
in `.github/workflows/CI.yml` and updated manually, with a review of any formatting changes.
Dependabot configuration does not establish that all declared dependency lower bounds have
been tested. That remains a separate compatibility task.

FMM3D major-version updates are temporarily ignored because Inti's current compatibility bound
requires FMM3D 1.x. The package keeps its existing `FMM3D = "1.0.1"` compat entry. Revisit the
ignore rule when Inti supports a newer major version. Do not disable the test action's
latest-version check to make an incompatible update pass.
[Inti's registered compatibility bounds](https://github.com/JuliaRegistries/General/blob/master/I/Inti/WeakCompat.toml).
