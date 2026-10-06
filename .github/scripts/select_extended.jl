# Select affected extended test files for PRs. Unknown source paths select all.
include(joinpath(@__DIR__, "..", "..", "test", "extended", "groups.jl"))

function affected_files(path::AbstractString)
    path = replace(path, '\\' => '/')
    section(prefix) = filter(file -> startswith(file, prefix), EXTENDED_FILES)
    if path in ("Project.toml", "test/Project.toml", "test/core/Project.toml",
        "test/runtests.jl", "test/extended/runtests.jl", "test/extended/groups.jl",
        ".github/scripts/select_extended.jl", ".github/workflows/ExtendedCI.yml")
        return EXTENDED_FILES
    elseif path == "test/extended/tmatrix/spheroid_farfield_samples.txt"
        return ["tmatrix/spheroid_farfield.jl"]
    elseif startswith(path, "test/extended/")
        file = path[(length("test/extended/") + 1):end]
        return file in EXTENDED_FILES ? [file] : String[]
    elseif startswith(path, "ext/AcousticScatteringMakieExt/")
        return section("plot/")
    elseif startswith(path, "src/solvers/")
        # Include consumers of shared operators, not only the owning solver.
        family = split(path, '/')[3]
        consumers = if family == "bem"
            ("bem/", "mfs/", "fem/", "tmatrix/")
        elseif family == "fem"
            ("fem/", "tmatrix/")
        elseif family == "coupled"
            ("fem/",)
        elseif family in ("mfs", "tmatrix", "kirchhoff", "fourier")
            (family * "/",)
        else
            # Modal utilities and shared loaders/types have package-wide consumers.
            return EXTENDED_FILES
        end
        return filter(
            file -> any(prefix -> startswith(file, prefix), consumers) ||
                    file in ("output.jl", "sampling.jl", "mesh.jl") ||
                    startswith(file, "plot/"),
            EXTENDED_FILES)
    elseif startswith(path, "src/") || startswith(path, "ext/")
        # Core, geometry, numerics, field dispatch and post-processing are shared.
        # New/unrecognized paths deliberately select all tests.
        return EXTENDED_FILES
    end
    return String[]
end

function select_files(paths)
    selected = Set{String}()
    for path in paths
        union!(selected, affected_files(path))
    end
    return filter(in(selected), EXTENDED_FILES)
end

# ExtendedCI runs each entry on three operating systems. GitHub permits at most
# 256 matrix jobs; the existing runner accepts comma-separated file selections.
function matrix_batches(files; max_tasks = 85)
    max_tasks > 0 || throw(ArgumentError("Require a positive matrix task limit"))
    width = max(1, cld(length(files), max_tasks))
    return [collect(batch) for batch in Iterators.partition(files, width)]
end

function matrix_entries(files)
    task(file) = replace(replace(replace(splitext(file)[1], '/' => '-'), '_' => '-'),
        "plot-plot" => "plot-")
    return [string("{\"task\":\"", join(task.(batch), "+"),
                "\",\"file\":\"", join(batch, ','), "\"}")
            for batch in matrix_batches(files)]
end

if abspath(PROGRAM_FILE) == @__FILE__
    full = ARGS == ["All"]
    selected_files = if full
        EXTENDED_FILES
    else
        length(ARGS) == 2 || error("Usage: select_extended.jl All | BASE_SHA HEAD_SHA")
        base, head = ARGS
        all(sha -> occursin(r"^[0-9a-fA-F]{40}$", sha), (base, head)) ||
            error("Expected two full Git commit SHAs")
        paths = filter(!isempty, split(read(`git diff --name-only -z $base...$head`, String), '\0'))
        println("Changed paths: ", join(paths, ", "))
        select_files(paths)
    end
    entries = matrix_entries(selected_files)
    # A nonempty sentinel keeps fromJSON(matrix) valid for docs-only PRs.
    value = isempty(entries) ? "[{\"task\":\"none\",\"file\":\"None\"}]" :
            "[" * join(entries, ",") * "]"
    println("Extended tasks: ", value)
    open(ENV["GITHUB_OUTPUT"], "a") do output
        println(output, "selections=", value)
        println(output, "has_tests=", !isempty(selected_files))
    end
end
