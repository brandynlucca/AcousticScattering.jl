module AcousticScattering

using LinearAlgebra
using SpecialFunctions

include("analytical/sphere_modal.jl")
include("analytical/spheroid_modal.jl")
include("analytical/high_frequency.jl")

include("engine/bem.jl")
include("engine/fem.jl")
include("engine/hybrid.jl")

include("postprocessing/farfield.jl")
include("postprocessing/target_strength.jl")
include("postprocessing/fields.jl")
include("postprocessing/validator_io.jl")

include("ecosystem/mesh_io.jl")
include("ecosystem/solvers.jl")

end # module AcousticScattering
