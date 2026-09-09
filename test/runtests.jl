using SafeTestsets: @safetestset

const TEST_GROUPS = ("Analytical", "RadialFEM", "MeridianFEM", "Boundary", "Spheroidal",
    "Oblique", "Interfaces", "Plots1D", "Plots2D", "Plots3D")
const TEST_GROUP = get(ENV, "GROUP", isempty(ARGS) ? "All" : only(ARGS))
TEST_GROUP in ("All", "Core", TEST_GROUPS...) ||
    throw(ArgumentError("Unknown test group: $TEST_GROUP. Choose All, Core, or $(join(TEST_GROUPS, ", "))."))

if TEST_GROUP in ("All", "Core", "Analytical")
    @time @safetestset "Analytical" include("analytical.jl")
end

if TEST_GROUP in ("All", "Core", "RadialFEM")
    @time @safetestset "RadialFEM" include("radial_fem.jl")
end

if TEST_GROUP in ("All", "Core", "MeridianFEM")
    @time @safetestset "MeridianFEM" include("meridian_fem.jl")
end

if TEST_GROUP in ("All", "Core", "Boundary")
    @time @safetestset "Boundary" include("boundary.jl")
end

if TEST_GROUP in ("All", "Core", "Spheroidal")
    @time @safetestset "Spheroidal" include("spheroidal.jl")
end

if TEST_GROUP in ("All", "Core", "Oblique")
    @time @safetestset "Oblique" include("oblique.jl")
end

if TEST_GROUP in ("All", "Core", "Interfaces")
    @time @safetestset "Interfaces" include("interfaces.jl")
end

if TEST_GROUP in ("All", "Plots1D")
    @time @safetestset "Plots1D" include("visualization/plots_1d.jl")
end

if TEST_GROUP in ("All", "Plots2D")
    @time @safetestset "Plots2D" include("visualization/plots_2d.jl")
end

if TEST_GROUP in ("All", "Plots3D")
    @time @safetestset "Plots3D" include("visualization/plots_3d.jl")
end
