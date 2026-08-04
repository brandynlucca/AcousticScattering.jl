using Documenter
using AcousticScattering

makedocs(
    sitename = "AcousticScattering.jl",
    modules = [AcousticScattering],
    authors = "AcousticScattering contributors",
    repo = "https://github.com/brandynlucca/AcousticScattering.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/brandynlucca/AcousticScattering.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/brandynlucca/AcousticScattering.jl.git",
    devbranch = "main",
)
