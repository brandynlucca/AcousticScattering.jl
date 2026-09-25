using Test

const TEST_BASIC_GROUP = get(ENV, "TEST_BASIC_GROUP", "All")
const BASIC_GROUPS = ("Modal", "Kirchhoff", "FEM", "BEM", "MFS", "Fourier", "Plotting",
    "Utilities")
TEST_BASIC_GROUP == "All" || TEST_BASIC_GROUP in BASIC_GROUPS ||
    throw(ArgumentError("Unknown TEST_BASIC_GROUP=$TEST_BASIC_GROUP"))

@info "Running Basic group: $TEST_BASIC_GROUP"

@testset "Basic regression tests" begin
    if TEST_BASIC_GROUP in ("All", "Modal")
        @testset "Modal" begin
            include("modal/sphere.jl")
            include("modal/cylinder.jl")
            include("modal/spheroid.jl")
        end
    end
    if TEST_BASIC_GROUP in ("All", "Kirchhoff")
        @testset "Kirchhoff" begin
            include("kirchhoff/sphere.jl")
            include("kirchhoff/spheroid.jl")
            include("kirchhoff/cylinder.jl")
        end
    end
    if TEST_BASIC_GROUP in ("All", "FEM")
        @testset "FEM" begin
            include("fem/sphere.jl")
            include("fem/spheroid.jl")
            include("fem/cylinder.jl")
        end
    end
    if TEST_BASIC_GROUP in ("All", "BEM")
        @testset "BEM" begin
            include("bem/sphere.jl")
            include("bem/spheroid.jl")
            include("bem/cylinder.jl")
            include("bem/assembly.jl")
            include("bem/arbitrary.jl")
            include("bem/edge.jl")
        end
    end
    if TEST_BASIC_GROUP in ("All", "MFS")
        @testset "MFS" begin
            include("mfs/sphere.jl")
            include("mfs/spheroid.jl")
            include("mfs/cylinder.jl")
        end
    end
    if TEST_BASIC_GROUP in ("All", "Fourier")
        @testset "Fourier matching" begin
            include("fourier/sphere.jl")
            include("fourier/spheroid.jl")
            include("fourier/irregular.jl")
        end
    end
    if TEST_BASIC_GROUP in ("All", "Plotting")
        @testset "Plotting" begin
            include("plot/plot1d.jl")
            include("plot/plot2d.jl")
            include("plot/plot3d.jl")
        end
    end
    if TEST_BASIC_GROUP in ("All", "Utilities")
        @testset "Utilities" begin
            include("mesh.jl")
            include("sampling.jl")
            include("output.jl")
        end
    end
end
