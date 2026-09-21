using Documenter
using AcousticScattering
# Initialize plotting extensions before Documenter evaluates example blocks.
using CairoMakie

# Keep curated prose while requiring a source docstring for every exported binding.
for name in names(AcousticScattering)
    name === :AcousticScattering && continue
    haskey(Base.Docs.meta(AcousticScattering), Base.Docs.Binding(AcousticScattering, name)) ||
        error("Exported name AcousticScattering.$name has no source docstring")
end

DocMeta.setdocmeta!(AcousticScattering, :DocTestSetup, :(using AcousticScattering);
    recursive = true)

makedocs(
    sitename = "AcousticScattering.jl",
    modules = [AcousticScattering],
    # The API reference uses curated prose rather than source-docstring inclusion.
    # Doctests, examples, and cross-references still fail the build on errors.
    checkdocs = :none,
    doctest = true,
    warnonly = false,
    authors = "AcousticScattering contributors",
    repo = "https://github.com/brandynlucca/AcousticScattering.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = ["assets/gallery.css", "assets/branding.css", "assets/favicon.ico"],
        sidebar_sitename = false,
        footer = "Powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl). " *
                 "[Logo attribution and license](" *
                 "https://github.com/brandynlucca/AcousticScattering.jl/blob/main/" *
                 "docs/src/assets/LICENSE).",
        repolink = "https://github.com/brandynlucca/AcousticScattering.jl",
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started/index.md",
        "Visualization Gallery" => [
            "Gallery" => "gallery/index.md",
            "Short examples" => "gallery/examples/index.md",
            "Pressure field" => "gallery/examples/pressure.md",
            "Gas-bubble resonance" => "gallery/examples/bubble.md",
            "Coated particle" => "gallery/examples/coating.md",
            "Elastic sphere" => "gallery/examples/elastic.md",
            "Nested geometry" => "gallery/examples/nested.md",
            "Bent-body views" => "gallery/examples/bent.md",
            "Interacting particles" => "gallery/examples/interacting.md",
            "Off-center inclusion" => "gallery/examples/eccentric.md",
            "Multiple gas cavities" => "gallery/examples/cavities.md",
            "Nested fluid layers" => "gallery/examples/layers.md",
            "Closed fluid cylinder" => "gallery/examples/capped.md",
            "Supplied faceted mesh" => "gallery/examples/faceted.md",
            "3D pressure slices" => "gallery/examples/field_slices.md"
        ],
        "Tutorials" => [
            "Your first frequency sweep" => "tutorials/index.md",
            "Materials and shells" => "tutorials/materials.md",
            "Reference-target models" => "tutorials/reference_targets.md",
            "Gas-bubble resonance" => "tutorials/bubble_resonance.md",
            "Coated particles" => "tutorials/coated_particles.md",
            "Geometry and incidence" => "tutorials/geometry.md",
            "Closed bent cylinders" => "tutorials/bent_cylinder.md",
            "A fish and swimbladder" => "tutorials/fish.md",
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
            "Fourier matching" => "models/fourier_matching.md",
            "References" => "models/references.md"
        ],
        "API Reference" => "api.md",
        "Developer Guide" => "developer/index.md"
    ]
)
