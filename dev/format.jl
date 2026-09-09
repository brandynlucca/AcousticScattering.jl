using JuliaFormatter: format

function main(arguments)
    all(argument -> argument == "--check", arguments) ||
        throw(ArgumentError("Usage: julia --project=dev dev/format.jl [--check]"))

    check_only = "--check" in arguments
    repository = dirname(@__DIR__)
    paths = [joinpath(repository, directory)
             for directory in ("src", "ext", "test", "docs", "dev")]
    already_formatted = format(paths; overwrite = !check_only, throw_on_error = true)

    if check_only && !already_formatted
        println(stderr, "Formatting required. Run: julia --project=dev dev/format.jl")
        return 1
    end

    println(check_only ? "SciML formatting check passed." : "SciML formatting applied.")
    return 0
end

exit(main(ARGS))
