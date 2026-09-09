# Test a manifest-free source snapshot without changing the developer's active environment.
function check_install(arguments)
    allowed = ("--mode=standard", "--mode=opt-out", "--reuse-depot")
    all(argument -> argument in allowed, arguments) ||
        throw(ArgumentError("Usage: julia dev/check_install.jl [--mode=standard|--mode=opt-out] [--reuse-depot]"))
    modes = [replace(argument, "--mode=" => "")
             for argument in arguments
             if startswith(argument, "--mode=")]
    length(modes) <= 1 || throw(ArgumentError("Choose at most one installation mode"))
    isempty(modes) && append!(modes, ["standard", "opt-out"])

    repository = dirname(@__DIR__)
    reuse_depot = "--reuse-depot" in arguments
    separator = Sys.iswindows() ? ";" : ":"
    mktempdir(; prefix = "acoustic-install-") do workspace
        snapshot = joinpath(workspace, "source")
        mkpath(snapshot)
        for name in ("Project.toml", "src", "ext", "deps", "Artifacts.toml")
            source = joinpath(repository, name)
            ispath(source) && cp(source, joinpath(snapshot, name))
        end
        @assert !isfile(joinpath(snapshot, "Manifest.toml"))
        @assert !isfile(joinpath(snapshot, "LocalPreferences.toml"))

        for mode in modes
            project = joinpath(workspace, mode, "project")
            depot = joinpath(workspace, mode, "depot")
            mkpath(project)
            mkpath(depot)
            depot_path = reuse_depot ? join(DEPOT_PATH, separator) : depot
            println("Installation mode: ", mode)
            println("Source snapshot: ", snapshot)
            println("Dependency depot: ", reuse_depot ? "reused (not a cold-depot test)" :
                                          depot)
            worker = joinpath(@__DIR__, "install_worker.jl")
            command = `$(Base.julia_cmd()) --startup-file=no --project=$project $worker $snapshot $mode`
            run(addenv(command,
                "JULIA_DEPOT_PATH" => depot_path,
                "JULIA_LOAD_PATH" => join(["@", "@stdlib"], separator),
                "JULIA_PKG_PRECOMPILE_AUTO" => "0"))
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_install(ARGS)
end
