# [Building and contributing](@id developer-guide)

See the [contributor instructions](https://github.com/brandynlucca/AcousticScattering.jl/blob/main/CONTRIBUTING.md) for environment setup, documentation builds and test commands.

## Examples and validation

Documenter executes examples and doctests during the build. Invalid references, doctest mismatches and failed examples stop the build. Plotting uses CairoMakie without a display server.

Examples on different pages are independent. Copy preceding blocks from the same page when running a later example. The structural shell example is unexecuted because of its cost.

Check numerical agreement under mesh, quadrature and modal refinement. Finite outputs and small linear residuals alone do not establish accuracy. Keep independent numerical references with the tests.

## Documenting an API

Document public functions, returned fields, units and supported configurations. Use public accessors instead of internal storage. For a new capability, update solver selection, the relevant theory page and an executable example.

## Style and citations

Follow the [SciML style guide](https://docs.sciml.ai/SciMLStyle/stable/) and the repository formatter settings. Keep numerical changes separate from documentation edits.

Link scientific claims to primary sources and add them to [References](@ref references). State model assumptions and numerical limits where they affect how an API should be used.
