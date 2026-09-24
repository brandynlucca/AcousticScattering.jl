# Select affected extended test files for PRs. Unknown source paths select all.
include(joinpath(@__DIR__, "..", "..", "test", "extended", "groups.jl"))

function affected_files(path::AbstractString)
    path = replace(path, '\\' => '/')
    section(prefix) = filter(file -> startswith(file, prefix), EXTENDED_FILES)
    if path in ("Project.toml", "test/Project.toml", "test/core/Project.toml",
        "test/runtests.jl", "test/extended/runtests.jl",
        "test/extended/groups.jl",
        ".github/scripts/select_extended.jl", ".github/workflows/Extended.yml") ||
       path in ("src/AcousticScattering.jl", "src/api.jl", "src/special_functions.jl")
        return EXTENDED_FILES
    elseif startswith(path, "test/extended/")
        file = path[(length("test/extended/") + 1):end]
        return file in EXTENDED_FILES ? [file] : String[]
    elseif startswith(path, "ext/AcousticScatteringMakieExt/")
        return section("plot/")
    elseif path == "src/engine/hybrid.jl"
        return ["fem/spheroid/coupled.jl"]
    elseif path == "src/engine/hybrid_general_shell.jl"
        return ["fem/sphere/coupled.jl", "fem/spheroid/coupled.jl"]
    elseif path == "src/analytical/vesm.jl"
        return ["modal/sphere/monopole.jl"]
    elseif path == "src/analytical/sphere_modal.jl"
        return section("modal/sphere/")
    elseif path == "src/analytical/spheroid_modal.jl"
        return ["modal/spheroid.jl"]
    elseif startswith(path, "src/analytical/cylinder_")
        return ["modal/cylinder.jl"]
    elseif path == "src/analytical/high_frequency.jl"
        return section("kirchhoff/")
    elseif path == "src/analytical/bent_cylinder.jl"
        return vcat(["modal/cylinder.jl"], section("kirchhoff/cylinder/"),
            section("mfs/cylinder/"))
    elseif path == "src/engine/fourier_matching.jl"
        return section("fourier/")
    elseif path in ("src/engine/mfs.jl", "src/surface_mfs.jl")
        return section("mfs/")
    elseif path == "src/engine/axisymmetric_bem.jl"
        return filter(
            file -> startswith(file, "bem/") && endswith(file, "axisymmetric.jl"),
            EXTENDED_FILES)
    elseif path in ("src/engine/full_bem.jl", "src/region_bem.jl")
        return filter(
            file -> startswith(file, "bem/") &&
                    (endswith(file, "full3d.jl") || file == "bem/arbitrary.jl"),
            EXTENDED_FILES)
    elseif path in ("src/engine/shell_bem.jl", "src/engine/edge_quadrature.jl",
        "src/engine/fluid_quadrature.jl")
        return section("bem/")
    elseif path == "src/engine/spheroid_meridian_fem.jl"
        return ["fem/spheroid/meridian.jl", "fem/spheroid/coupled.jl"]
    elseif path == "src/engine/cylinder_meridian_fem.jl"
        return ["fem/cylinder/meridian.jl"]
    elseif path == "src/engine/cylinder_elastic_radial_fem.jl"
        return ["fem/cylinder/radial.jl"]
    elseif startswith(path, "src/engine/") && occursin("fem", path)
        return section("fem/")
    elseif path == "src/engine/diagnostics.jl"
        return vcat(section("output.jl"), section("bem/"), section("fem/"),
            section("mfs/"), section("fourier/"))
    elseif path == "src/postprocessing/sweeps.jl"
        return vcat(section("sampling.jl"), section("plot/"))
    elseif startswith(path, "src/postprocessing/")
        return vcat(section("output.jl"), section("sampling.jl"), section("plot/"))
    elseif path == "src/cylinder_surface.jl"
        return vcat(section("mesh.jl"), section("bem/cylinder/"), section("mfs/cylinder/"))
    elseif path in ("src/surface_mesh.jl", "src/surface_validation.jl")
        return vcat(section("mesh.jl"), section("bem/"), section("mfs/"))
    elseif startswith(path, "src/") || startswith(path, "ext/")
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
    entries = [string("{\"task\":\"",
                   replace(replace(replace(splitext(file)[1], '/' => '-'), '_' => '-'),
                       "plot-plot" => "plot-"),
                   "\",\"file\":\"", file, "\"}") for file in selected_files]
    # A nonempty sentinel keeps fromJSON(matrix) valid for docs-only PRs.
    value = isempty(entries) ? "[{\"task\":\"none\",\"file\":\"None\"}]" :
            "[" * join(entries, ",") * "]"
    println("Extended tasks: ", value)
    open(ENV["GITHUB_OUTPUT"], "a") do output
        println(output, "selections=", value)
        println(output, "has_tests=", !isempty(selected_files))
    end
end
