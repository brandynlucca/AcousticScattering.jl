# Shared group names for the extended runner and the CI change selector.
const EXTENDED_ROOT = @__DIR__
const EXTENDED_FILES = sort!([replace(relpath(joinpath(root, file), EXTENDED_ROOT), '\\' => '/')
    for (root, _, files) in walkdir(EXTENDED_ROOT) for file in files
    if endswith(file, ".jl") && file ∉ ("runtests.jl", "groups.jl")])

function extended_group(file::AbstractString)
    parts = split(replace(file, '\\' => '/'), '/')
    if length(parts) == 1
        return splitext(parts[1])[1]
    end
    geometry = splitext(parts[2])[1]
    return parts[1] == "plot" ? "plot-" * replace(geometry, "plot" => "") :
           parts[1] * "-" * geometry
end

const EXTENDED_GROUPS = sort!(unique(extended_group.(EXTENDED_FILES)))
