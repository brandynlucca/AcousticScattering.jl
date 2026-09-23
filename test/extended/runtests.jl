using SafeTestsets: @safetestset
include("groups.jl")

const EXTENDED_GROUP = get(ENV, "TEST_EXTENDED_GROUP", "All")
EXTENDED_GROUP == "All" || EXTENDED_GROUP in EXTENDED_GROUPS ||
    throw(ArgumentError("Unknown TEST_EXTENDED_GROUP=$EXTENDED_GROUP"))
const EXTENDED_SELECTED_FILES = let value = get(ENV, "TEST_EXTENDED_FILES", "All")
    value == "All" ? nothing : Set(split(value, ','))
end
selected_file(path) = (EXTENDED_GROUP == "All" || extended_group(path) == EXTENDED_GROUP) &&
    (EXTENDED_SELECTED_FILES === nothing || path in EXTENDED_SELECTED_FILES)
@info "Running extended group: $EXTENDED_GROUP"

if EXTENDED_GROUP == "All" || startswith(EXTENDED_GROUP, "modal-")
    selected_file("modal/sphere/general.jl") && @time @safetestset "Modal sphere general" include("modal/sphere/general.jl")
    selected_file("modal/sphere/concentric.jl") && @time @safetestset "Modal sphere concentric" include("modal/sphere/concentric.jl")
    selected_file("modal/sphere/monopole.jl") && @time @safetestset "Modal sphere monopole" include("modal/sphere/monopole.jl")
    selected_file("modal/spheroid.jl") && @time @safetestset "Modal spheroid" include("modal/spheroid.jl")
    selected_file("modal/cylinder.jl") && @time @safetestset "Modal cylinder" include("modal/cylinder.jl")
end

if EXTENDED_GROUP == "All" || startswith(EXTENDED_GROUP, "kirchhoff-")
    selected_file("kirchhoff/sphere/general.jl") && @time @safetestset "Kirchhoff sphere general" include("kirchhoff/sphere/general.jl")
    selected_file("kirchhoff/sphere/layer_reflection.jl") && @time @safetestset "Kirchhoff sphere layer reflection" include("kirchhoff/sphere/layer_reflection.jl")
    selected_file("kirchhoff/spheroid.jl") && @time @safetestset "Kirchhoff spheroid" include("kirchhoff/spheroid.jl")
    selected_file("kirchhoff/cylinder/surface.jl") && @time @safetestset "Kirchhoff cylinder surface" include("kirchhoff/cylinder/surface.jl")
    selected_file("kirchhoff/cylinder/integral.jl") && @time @safetestset "Kirchhoff cylinder integral" include("kirchhoff/cylinder/integral.jl")
end

if EXTENDED_GROUP == "All" || startswith(EXTENDED_GROUP, "fem-")
    selected_file("fem/sphere/radial.jl") && @time @safetestset "FEM sphere radial" include("fem/sphere/radial.jl")
    selected_file("fem/sphere/meridian.jl") && @time @safetestset "FEM sphere meridian" include("fem/sphere/meridian.jl")
    selected_file("fem/sphere/coupled.jl") && @time @safetestset "FEM sphere coupled" include("fem/sphere/coupled.jl")
    selected_file("fem/spheroid/meridian.jl") && @time @safetestset "FEM spheroid meridian" include("fem/spheroid/meridian.jl")
    selected_file("fem/spheroid/coupled.jl") && @time @safetestset "FEM spheroid coupled" include("fem/spheroid/coupled.jl")
    selected_file("fem/cylinder/radial.jl") && @time @safetestset "FEM cylinder radial" include("fem/cylinder/radial.jl")
    selected_file("fem/cylinder/meridian.jl") && @time @safetestset "FEM cylinder meridian" include("fem/cylinder/meridian.jl")
end

if EXTENDED_GROUP == "All" || startswith(EXTENDED_GROUP, "bem-")
    selected_file("bem/sphere/axisymmetric.jl") && @time @safetestset "BEM sphere axisymmetric" include("bem/sphere/axisymmetric.jl")
    selected_file("bem/sphere/full3d.jl") && @time @safetestset "BEM sphere full 3D" include("bem/sphere/full3d.jl")
    selected_file("bem/spheroid/axisymmetric.jl") && @time @safetestset "BEM spheroid axisymmetric" include("bem/spheroid/axisymmetric.jl")
    selected_file("bem/spheroid/full3d.jl") && @time @safetestset "BEM spheroid full 3D" include("bem/spheroid/full3d.jl")
    selected_file("bem/cylinder/axisymmetric.jl") && @time @safetestset "BEM cylinder axisymmetric" include("bem/cylinder/axisymmetric.jl")
    selected_file("bem/cylinder/full3d.jl") && @time @safetestset "BEM cylinder full 3D" include("bem/cylinder/full3d.jl")
    selected_file("bem/arbitrary.jl") && @time @safetestset "BEM arbitrary" include("bem/arbitrary.jl")
end

if EXTENDED_GROUP == "All" || startswith(EXTENDED_GROUP, "mfs-")
    selected_file("mfs/sphere/axisymmetric.jl") && @time @safetestset "MFS sphere axisymmetric" include("mfs/sphere/axisymmetric.jl")
    selected_file("mfs/sphere/surface.jl") && @time @safetestset "MFS sphere surface" include("mfs/sphere/surface.jl")
    selected_file("mfs/spheroid.jl") && @time @safetestset "MFS spheroid" include("mfs/spheroid.jl")
    selected_file("mfs/cylinder/axisymmetric.jl") && @time @safetestset "MFS cylinder axisymmetric" include("mfs/cylinder/axisymmetric.jl")
    selected_file("mfs/cylinder/surface.jl") && @time @safetestset "MFS cylinder surface" include("mfs/cylinder/surface.jl")
end

if EXTENDED_GROUP == "All" || startswith(EXTENDED_GROUP, "fourier-")
    selected_file("fourier/sphere.jl") && @time @safetestset "Fourier sphere" include("fourier/sphere.jl")
    selected_file("fourier/spheroid.jl") && @time @safetestset "Fourier spheroid" include("fourier/spheroid.jl")
    selected_file("fourier/irregular.jl") && @time @safetestset "Fourier irregular" include("fourier/irregular.jl")
end

if EXTENDED_GROUP in ("All", "mesh")
    selected_file("mesh.jl") && @time @safetestset "Mesh" include("mesh.jl")
end
if EXTENDED_GROUP in ("All", "sampling")
    selected_file("sampling.jl") && @time @safetestset "Sampling" include("sampling.jl")
end
if EXTENDED_GROUP in ("All", "output")
    selected_file("output.jl") && @time @safetestset "Output" include("output.jl")
end

if EXTENDED_GROUP == "All" || startswith(EXTENDED_GROUP, "plot-")
    selected_file("plot/plot1d.jl") && @time @safetestset "Plot 1D" include("plot/plot1d.jl")
    selected_file("plot/plot2d.jl") && @time @safetestset "Plot 2D" include("plot/plot2d.jl")
    selected_file("plot/plot3d.jl") && @time @safetestset "Plot 3D" include("plot/plot3d.jl")
end
