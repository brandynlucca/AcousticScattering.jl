using Documenter
using AcousticScattering

DocMeta.setdocmeta!(AcousticScattering, :DocTestSetup, :(using AcousticScattering);
    recursive = true)

makedocs(
    sitename = "AcousticScattering.jl",
    modules = [AcousticScattering],
    # Curated reference prose is used while legacy source docstrings are being migrated.
    # Doctests, examples, and cross-references still fail the build on errors.
    checkdocs = :none,
    doctest = true,
    warnonly = false,
    authors = "AcousticScattering contributors",
    repo = "https://github.com/brandynlucca/AcousticScattering.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/brandynlucca/AcousticScattering.jl",
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started/index.md",
        "Tutorials" => [
            "Your first frequency sweep" => "tutorials/index.md",
            "Materials and shells" => "tutorials/materials.md",
            "Geometry and incidence" => "tutorials/geometry.md",
            "Numerical convergence" => "tutorials/convergence.md",
            "Performance" => "tutorials/performance.md"
        ],
        "Models and Theory" => [
            "Conventions" => "models/index.md",
            "Choosing a solver" => "models/selection.md",
            "Geometry and materials" => "models/materials.md",
            "Modal series" => "models/modal.md",
            "Kirchhoff physical optics" => "models/kirchhoff.md",
            "BEM and MFS" => "models/boundary_methods.md",
            "FEM and shell coupling" => "models/fem.md",
            "References" => "models/references.md"
        ],
        "Visualization Gallery" => "gallery/index.md",
        "API Reference" => "api.md",
        "Developer Guide" => [
            "Building and contributing" => "developer/index.md",
            "CI and installation checks" => "developer/ci.md",
            "API migration" => "developer/migration.md"
        ]
    ]
)
