module SourceLayoutChecks
using Test

include(joinpath(@__DIR__, "..", "..", ".github", "scripts", "select_extended.jl"))

@testset "Source loading and CI selection" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    seen = Set{String}()
    function visit(path)
        path = normpath(path)
        @test isfile(path)
        @test !(path in seen)
        push!(seen, path)
        source = read(path, String)
        for match in eachmatch(r"include\(\"([^\"]+)\"\)", source)
            visit(joinpath(dirname(path), match.captures[1]))
        end
    end
    visit(joinpath(root, "src", "AcousticScattering.jl"))
    files = Set(normpath(joinpath(directory, file))
    for (directory, _, names) in walkdir(joinpath(root, "src"))
    for file in names if endswith(file, ".jl"))
    @test seen == files
    @test affected_files("src/core/materials.jl") == EXTENDED_FILES
    @test affected_files("src/geometry/mesh.jl") == EXTENDED_FILES
    @test affected_files("src/unknown/new.jl") == EXTENDED_FILES
    @test "tmatrix/spheroid_farfield.jl" in
          affected_files("src/solvers/fem/volume/assembly.jl")
    @test "mfs/sphere/axisymmetric.jl" in affected_files("src/solvers/bem/axisymmetric.jl")
    @test "fem/spheroid/coupled.jl" in affected_files("src/solvers/coupled/shell_fluid.jl")
    @test isempty(affected_files("experimental/tank/signal_chain.jl"))
    @test affected_files("test/extended/tmatrix/spheroid_farfield_samples.txt") ==
          ["tmatrix/spheroid_farfield.jl"]
    # Keep every selected file exactly once while respecting the three-OS matrix.
    batches = matrix_batches(EXTENDED_FILES)
    @test length(batches)*3 <= 256
    @test reduce(vcat, batches) == EXTENDED_FILES
    @test all(!isempty, batches)
    @test isempty(matrix_entries(String[]))
    @test matrix_batches(["modal/sphere/general.jl"]) ==
          [["modal/sphere/general.jl"]]
    @test only(matrix_entries(["modal/sphere/general.jl"])) ==
          "{\"task\":\"modal-sphere-general\",\"file\":\"modal/sphere/general.jl\"}"
end
end
