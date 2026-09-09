import Pkg
using UUIDs: UUID

length(ARGS) == 2 || throw(ArgumentError("Expected source snapshot and installation mode"))
snapshot, mode = ARGS
mode in ("standard", "opt-out") || throw(ArgumentError("Unknown installation mode: $mode"))
@assert LOAD_PATH == ["@", "@stdlib"]
@assert !isfile(joinpath(snapshot, "Manifest.toml"))
@assert !isfile(joinpath(snapshot, "LocalPreferences.toml"))

# Automatic precompilation is postponed so preferences can be set before the first import.
Pkg.develop(Pkg.PackageSpec(; path = snapshot))
Pkg.add(["Preferences", "PrecompileTools"])

package_uuid = UUID("37bc0722-8b3f-41d8-b4ae-17decd3486a9")
for (uuid, dependency) in Pkg.dependencies()
    if uuid != package_uuid && dependency.is_tracking_path
        error("Unexpected local dependency override: $(dependency.name)")
    end
end
println("Dependency resolution contains no local overrides outside the tested source snapshot.")
Pkg.status()

using Preferences: set_preferences!

if mode == "opt-out"
    set_preferences!("AcousticScattering", "precompile_workload" => false; force = true)
end

elapsed = @elapsed Pkg.precompile(; strict = true)
println("Precompile elapsed seconds (", mode, "): ", round(elapsed; digits = 2))
project = dirname(Base.active_project())
probe = joinpath(@__DIR__, "install_probe.jl")
enabled = mode == "standard"
run(`$(Base.julia_cmd()) --startup-file=no --project=$project $probe $enabled`)

if mode == "opt-out"
    set_preferences!("AcousticScattering", "precompile_workload" => true; force = true)
    elapsed = @elapsed Pkg.precompile(; strict = true)
    println("Re-enabled precompile elapsed seconds: ", round(elapsed; digits = 2))
    run(`$(Base.julia_cmd()) --startup-file=no --project=$project $probe true`)
end
