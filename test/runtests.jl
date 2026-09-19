using SafeTestsets: @safetestset

const TEST_GROUPS = (
    "Analytical", "RadialFEM", "MeridianFEM", "Boundary", "FullBEM",
    "CylinderBEM", "RegionBEM", "NearPressure", "Spheroidal",
    "RegionShapes", "FishBEM", "LowFrequencyRegions", "GasResonance",
    "CoupledResonance", "Oblique", "Interfaces", "Plots1D",
    "Plots2D", "Plots3DModels", "Plots3DFull")
const TEST_GROUP_SPEC = get(ENV, "GROUP", isempty(ARGS) ? "All" : only(ARGS))
# A comma-separated GROUP (e.g. "RadialFEM,Boundary") runs several groups in one process,
# letting a single CI job batch groups together (see the macOS-specific job in CI.yml,
# which needs fewer, larger jobs to avoid GitHub's limited macOS runner concurrency).
const REQUESTED_GROUPS = Set(split(TEST_GROUP_SPEC, ','))
all(g -> g in ("All", "Core", TEST_GROUPS...), REQUESTED_GROUPS) ||
    throw(ArgumentError("Unknown test group in: $TEST_GROUP_SPEC. Choose All, Core, or a comma-separated list from: $(join(TEST_GROUPS, ", "))."))
function _wants(name; core = true)
    any(g -> g == "All" || (core && g == "Core") || g == name, REQUESTED_GROUPS)
end

if _wants("Analytical")
    @time @safetestset "Analytical" include("analytical.jl")
end

if _wants("RadialFEM")
    @time @safetestset "RadialFEM" include("radial_fem.jl")
    @time @safetestset "Pressure" include("pressure.jl")
    @time @safetestset "Layered pressure" include("layered_pressure.jl")
end

if _wants("MeridianFEM")
    @time @safetestset "MeridianFEM" include("meridian_fem.jl")
end

if _wants("Boundary")
    @time @safetestset "Boundary" include("boundary.jl")
end

if _wants("FullBEM")
    @time @safetestset "FullBEM" include("full_bem.jl")
    @time @safetestset "Boundary pressure" include("boundary_pressure.jl")
    @time @safetestset "Axisymmetric pressure" include("axisymmetric_pressure.jl")
    @time @safetestset "BoundaryAngleSweeps" include("boundary_angle_sweeps.jl")
end

if _wants("CylinderBEM")
    @time @safetestset "CylinderBEM" include("cylinder_surface.jl")
    @time @safetestset "Cylinder pressure" include("cylinder_pressure.jl")
    @time @safetestset "Surface pressure" include("surface_pressure.jl")
end

if _wants("RegionBEM")
    @time @safetestset "RegionBEM" include("region_bem.jl")
    @time @safetestset "Region pressure" include("region_pressure.jl")
    @time @safetestset "AngleSweeps" include("angle_sweeps.jl")
end

if _wants("NearPressure")
    @time @safetestset "Near-interface pressure" include("near_interface_pressure.jl")
    @time @safetestset "Rim pressure" include("rim_pressure.jl")
end

if _wants("RegionShapes")
    @time @safetestset "RegionShapes" include("region_shapes.jl")
end

if _wants("FishBEM")
    @time @safetestset "FishBEM" include("fish_bem.jl")
end
if _wants("LowFrequencyRegions")
    @time @safetestset "LowFrequencyRegions" include("low_frequency_regions.jl")
end
if _wants("GasResonance")
    @time @safetestset "GasResonance" include("gas_resonance.jl")
end
if _wants("CoupledResonance")
    @time @safetestset "CoupledResonance" include("coupled_resonance.jl")
end

if _wants("Spheroidal")
    @time @safetestset "Spheroidal" include("spheroidal.jl")
end

if _wants("Oblique")
    @time @safetestset "Oblique" include("oblique.jl")
end

if _wants("Interfaces")
    @time @safetestset "Interfaces" include("interfaces.jl")
    @time @safetestset "BodyCoordinates" include("body_coordinates.jl")
end

if _wants("Plots1D"; core = false)
    @time @safetestset "Plots1D" include("visualization/plots_1d.jl")
end

if _wants("Plots2D"; core = false)
    @time @safetestset "Plots2D" include("visualization/plots_2d.jl")
end

if _wants("Plots3DModels"; core = false)
    @time @safetestset "Plots3DModels" include("visualization/plots_3d.jl")
end

if _wants("Plots3DFull"; core = false)
    @time @safetestset "Plots3DFull" include("visualization/plots_3d_full.jl")
end
