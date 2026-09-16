module AcousticScatteringMakieExt

using Makie
using AcousticScattering
using Inti
using StaticArrays: SVector

include("theme.jl")
include("sampling_plots.jl")
include("sweep_results.jl")
include("mesh_plots.jl")
include("region_plots.jl")

end
