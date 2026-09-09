using Test: @test, @testset

# Including the script defines its guard without performing a deployment.
include(joinpath(@__DIR__, "..", "docs", "deploy.jl"))

@testset "Documentation deployment guard" begin
    trusted = Dict(
        "GITHUB_ACTIONS" => "true",
        "GITHUB_EVENT_NAME" => "push",
        "GITHUB_REPOSITORY" => "brandynlucca/AcousticScattering.jl",
        "GITHUB_REF" => "refs/heads/main"
    )
    @test deployment_allowed(trusted)
    @test deployment_allowed(merge(trusted, Dict("GITHUB_REF" => "refs/tags/v0.1.0")))
    manual = merge(trusted, Dict("GITHUB_EVENT_NAME" => "workflow_dispatch"))
    @test deployment_allowed(manual)
    @test deployment_allowed(merge(manual, Dict("GITHUB_REF" => "refs/tags/v0.1.0")))
    @test !deployment_allowed(Dict{String, String}())
    for override in (
        "GITHUB_ACTIONS" => "false",
        "GITHUB_EVENT_NAME" => "pull_request",
        "GITHUB_EVENT_NAME" => "pull_request_target",
        "GITHUB_EVENT_NAME" => "schedule",
        "GITHUB_REPOSITORY" => "contributor/AcousticScattering.jl",
        "GITHUB_REF" => "refs/heads/feature",
        "GITHUB_REF" => "refs/tags/unversioned",
        "GITHUB_REF" => "refs/pull/1/merge"
    )
        @test !deployment_allowed(merge(trusted, Dict(override)))
        @test !deployment_allowed(merge(manual, Dict(override)))
    end
end
