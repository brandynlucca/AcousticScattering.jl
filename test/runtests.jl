using SafeTestsets: @safetestset

const TEST_GROUPS = (
    "Analytical", "RadialFEM", "MeridianFEM", "Boundary", "FullBEM",
    "CylinderBEM", "RegionBEM", "NearPressure", "Spheroidal",
    "RegionShapes", "FishBEM", "LowFrequencyRegions", "GasResonance",
    "CoupledResonance", "Oblique", "Interfaces", "Plots1D",
    "Plots2D", "Plots3DModels", "Plots3DFull")
const TEST_GROUP = get(ENV, "GROUP", isempty(ARGS) ? "All" : only(ARGS))
TEST_GROUP in ("All", "Core", TEST_GROUPS...) ||
    throw(ArgumentError("Unknown test group: $TEST_GROUP. Choose All, Core, or $(join(TEST_GROUPS, ", "))."))

if TEST_GROUP in ("All", "Core", "Analytical")
    @time @safetestset "Analytical" include("analytical.jl")
end

if TEST_GROUP in ("All", "Core", "RadialFEM")
    @time @safetestset "RadialFEM" include("radial_fem.jl")
    @time @safetestset "Pressure" include("pressure.jl")
    @time @safetestset "Layered pressure" include("layered_pressure.jl")
end

if TEST_GROUP in ("All", "Core", "MeridianFEM")
    @time @safetestset "MeridianFEM" include("meridian_fem.jl")
end

if TEST_GROUP in ("All", "Core", "Boundary")
    @time @safetestset "Boundary" include("boundary.jl")
end

if TEST_GROUP in ("All", "Core", "FullBEM")
    @time @safetestset "FullBEM" include("full_bem.jl")
    @time @safetestset "Boundary pressure" include("boundary_pressure.jl")
    @time @safetestset "Axisymmetric pressure" include("axisymmetric_pressure.jl")
    @time @safetestset "BoundaryAngleSweeps" include("boundary_angle_sweeps.jl")
end

if TEST_GROUP in ("All", "Core", "CylinderBEM")
    @time @safetestset "CylinderBEM" include("cylinder_surface.jl")
    @time @safetestset "Cylinder pressure" include("cylinder_pressure.jl")
    @time @safetestset "Surface pressure" include("surface_pressure.jl")
end

if TEST_GROUP in ("All", "Core", "RegionBEM")
    @time @safetestset "RegionBEM" include("region_bem.jl")
    @time @safetestset "Region pressure" include("region_pressure.jl")
    @time @safetestset "AngleSweeps" include("angle_sweeps.jl")
end

if TEST_GROUP in ("All", "Core", "NearPressure")
    @time @safetestset "Near-interface pressure" include("near_interface_pressure.jl")
    @time @safetestset "Rim pressure" include("rim_pressure.jl")
end

if TEST_GROUP in ("All", "Core", "RegionShapes")
    @time @safetestset "RegionShapes" include("region_shapes.jl")
end

if TEST_GROUP in ("All", "Core", "FishBEM")
    @time @safetestset "FishBEM" include("fish_bem.jl")
end
if TEST_GROUP in ("All", "Core", "LowFrequencyRegions")
    @time @safetestset "LowFrequencyRegions" include("low_frequency_regions.jl")
end
if TEST_GROUP in ("All", "Core", "GasResonance")
    @time @safetestset "GasResonance" include("gas_resonance.jl")
end
if TEST_GROUP in ("All", "Core", "CoupledResonance")
    @time @safetestset "CoupledResonance" include("coupled_resonance.jl")
end

if TEST_GROUP in ("All", "Core", "Spheroidal")
    @time @safetestset "Spheroidal" include("spheroidal.jl")
end

if TEST_GROUP in ("All", "Core", "Oblique")
    @time @safetestset "Oblique" include("oblique.jl")
end

if TEST_GROUP in ("All", "Core", "Interfaces")
    @time @safetestset "Interfaces" include("interfaces.jl")
    @time @safetestset "BodyCoordinates" include("body_coordinates.jl")
end

if TEST_GROUP in ("All", "Plots1D")
    @time @safetestset "Plots1D" include("visualization/plots_1d.jl")
end

if TEST_GROUP in ("All", "Plots2D")
    @time @safetestset "Plots2D" include("visualization/plots_2d.jl")
end

if TEST_GROUP in ("All", "Plots3DModels")
    @time @safetestset "Plots3DModels" include("visualization/plots_3d.jl")
end

if TEST_GROUP in ("All", "Plots3DFull")
    @time @safetestset "Plots3DFull" include("visualization/plots_3d_full.jl")
end
