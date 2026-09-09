# Run separately from `runtests.jl` to test preferences before importing the package.
import Pkg

mode = isempty(ARGS) ? "opt-out" : only(ARGS)
mode in ("standard", "opt-out") ||
    throw(ArgumentError("Usage: julia test/precompile.jl [standard|opt-out]"))

# Exclude local manifests and preferences, including path overrides in the developer checkout.
repository = dirname(@__DIR__)
snapshot = mktempdir(; prefix = "acoustic-precompile-")
for name in ("Project.toml", "src", "ext", "deps", "Artifacts.toml")
    source = joinpath(repository, name)
    ispath(source) && cp(source, joinpath(snapshot, name))
end
Pkg.activate(; temp = true)
withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
    Pkg.develop(Pkg.PackageSpec(; path = snapshot))
    Pkg.add(["Preferences", "PrecompileTools"])
end
all(dependency.name == "AcousticScattering" || !dependency.is_tracking_path
for dependency in values(Pkg.dependencies())) ||
    throw(ArgumentError("Unexpected local dependency override in the installation test"))

# Preferences is installed in the temporary environment before it can be imported.
using Preferences: set_preferences!

if mode == "opt-out"
    set_preferences!("AcousticScattering", "precompile_workload" => false; force = true)
end

# Each probe uses a fresh process so an already-loaded module cannot hide a preference change.
probe = raw"""
using AcousticScattering
using PrecompileTools: workload_enabled
using Test: @test, @testset

enabled = parse(Bool, only(ARGS))
@testset "Installed package (workload enabled: $enabled)" begin
    @test workload_enabled(AcousticScattering) == enabled
    wavenumber = 2pi * 38000.0 / 1477.4
    @test isapprox(target_strength(modal(Sphere(0.01), Rigid(), wavenumber)),
        -49.088291; atol = 1e-4)
    fluid = FluidFilled(1028.9 / 1026.8, 1480.3 / 1477.4)
    @test isapprox(target_strength(modal(Sphere(0.01), fluid, wavenumber)),
        -94.278687; atol = 1e-3)
    spheroid = modal(Spheroid(0.02, 0.01), Rigid(), 2pi * 12000.0 / 1477.4;
        incidence_angle = pi / 4, m_max = 6, n_max = 8)
    @test isfinite(target_strength(spheroid))
end
"""

project = dirname(Base.active_project())
states = mode == "standard" ? (true,) : (false, true)
for enabled in states
    if enabled && mode == "opt-out"
        set_preferences!("AcousticScattering", "precompile_workload" => true; force = true)
    end
    Pkg.precompile(; strict = true)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $probe $enabled`
    separator = Sys.iswindows() ? ";" : ":"
    run(addenv(command, "JULIA_LOAD_PATH" => join(["@", "@stdlib"], separator)))
end
