using Documenter: deploydocs

function deployment_allowed(environment)
    event = get(environment, "GITHUB_EVENT_NAME", "")
    reference = get(environment, "GITHUB_REF", "")
    repository = get(environment, "GITHUB_REPOSITORY", "")
    return get(environment, "GITHUB_ACTIONS", "false") == "true" &&
           event in ("push", "workflow_dispatch") &&
           repository == "brandynlucca/AcousticScattering.jl" &&
           (reference == "refs/heads/main" || startswith(reference, "refs/tags/v"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    deployment_allowed(ENV) ||
        throw(ArgumentError("Documentation deployment requires an upstream main or release-tag push or manual run"))
    isfile(joinpath(@__DIR__, "build", "index.html")) ||
        throw(ArgumentError("The validated documentation artifact is missing"))
    deploydocs(;
        root = @__DIR__,
        target = "build",
        repo = "github.com/brandynlucca/AcousticScattering.jl.git",
        devbranch = "main",
        push_preview = false
    )
end
