module AcousticScattering

using LinearAlgebra: I, dot, mul!, Diagonal, norm, cond, diag, diagind, pinv, det,
                     SingularException
using SparseArrays: sparse
using SpecialFunctions: besselj, bessely, besselh
using SpheroidalWaves: SpheroidalWaves
using GenericLinearAlgebra: GenericLinearAlgebra
using Ferrite: Ferrite
using Inti: Inti
using Gmsh: gmsh
using HMatrices: HMatrices
using LinearMaps: LinearMap
using IterativeSolvers: IterativeSolvers
using StaticArrays: SVector
using PrecompileTools: @compile_workload

# Must precede every include below: Spheroid subtypes this directly.
abstract type AbstractBody end

include("special_functions.jl")

include("postprocessing/target_strength.jl")

include("analytical/sphere_modal.jl")
include("analytical/vesm.jl")
include("analytical/cylinder_modal.jl")
include("analytical/cylinder_elastic_modal.jl")
include("analytical/spheroid_modal.jl")
include("analytical/high_frequency.jl")
include("analytical/bent_cylinder.jl")

include("engine/axisymmetric_bem.jl")
include("engine/shell_bem.jl")
include("engine/radial_fem.jl")
include("engine/meridian_fem.jl")
include("engine/cylinder_meridian_fem.jl")
include("engine/elastic_radial_fem.jl")
include("engine/fluid_shell_radial_fem.jl")
include("engine/cylinder_elastic_radial_fem.jl")
include("engine/spheroid_meridian_fem.jl")
include("engine/mfs.jl")
include("engine/shell_fem.jl")
include("engine/shell_fem_general.jl")
include("engine/full_bem.jl")
include("engine/hybrid.jl")
include("engine/hybrid_general_shell.jl")

include("postprocessing/farfield.jl")

include("api.jl")

include("postprocessing/sweeps.jl")
include("postprocessing/revolution.jl")

export Rigid, PressureRelease, FluidFilled, GasFilled, SolidElastic
export Shelled, FluidLayer, ElasticLayer, ViscousLayer, LayeredMaterial, VacuumInterior,
       FluidInterior
export AbstractBody, Sphere, Cylinder, Spheroid, Shell
export AbstractSolution, ModalSolution, KirchhoffSolution, FEMSolution, BEMSolution,
       MFSSolution
export modal, kirchhoff, fem, bem, mfs
export target_strength, scattering_amplitude
export Mesh, mesh

@compile_workload begin
    let radius = 0.01, len = 0.07, k = 2π * 38000.0 / 1477.3
        kirchhoff(Cylinder(radius, len; radius_curvature = 1e8len), Rigid(), k; incidence_angle = π /
                                                                                                  2)
    end
    let body = Spheroid(0.05, 0.02), k = 2π * 38000.0 / 1477.4
        kirchhoff(body, Rigid(), k; incidence_angle = π / 2)
    end
end

end # module AcousticScattering
