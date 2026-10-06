module AcousticScattering

using LinearAlgebra: I, dot, mul!, Diagonal, norm, cond, diag, diagind, pinv, det,
                     SingularException, svdvals, cross, lu, svd
using LinearAlgebra: LinearAlgebra
using SparseArrays: sparse, spzeros, SparseMatrixCSC
using SpecialFunctions: besselj, bessely, besselh
using SpheroidalWaves: SpheroidalWaves
using GenericLinearAlgebra: GenericLinearAlgebra
using Ferrite: Ferrite
using Inti: Inti
using Gmsh: gmsh
using HMatrices: HMatrices
using LinearMaps: LinearMap
import LinearMaps
using IterativeSolvers: IterativeSolvers
using IncompleteLU: IncompleteLU
using NLsolve: NLsolve
using ForwardDiff: ForwardDiff
using StaticArrays: SVector, SMatrix, MVector
using PrecompileTools: @compile_workload

export Rigid, PressureRelease, Impedance, FluidFilled, GasFilled, SolidElastic,
       ViscoelasticSolid, SpatialFluid
export Shelled, FluidLayer, ElasticLayer, ViscousLayer, LayeredMaterial, VacuumInterior,
       FluidInterior
export AbstractBody, Sphere, Cylinder, Spheroid, Shell, Irregular
export IncidentField, PlaneWave, SphericalWave, BesselBeam
export incident_pressure, incident_gradient, incident_coefficient
export AbstractSolution, ModalSolution, TMatrixSolution,
       KirchhoffSolution, FEMSolution,
       BEMSolution,
       MFSSolution, FMSolution, FreeSurfaceSolution
export modal, tmatrix, kirchhoff, fem, bem, mfs, fourier,
       free_surface
export target_strength, scattering_amplitude, pressure, diagnostics
export Mesh, mesh
export components, frequency_sweep, incidence_angle_sweep, bistatic_sweep, bistatic_map

include("core/core.jl")
include("numerics/numerics.jl")
include("geometry/geometry.jl")
include("solvers/solvers.jl")
include("fields/fields.jl")
include("postprocessing/postprocessing.jl")
include("precompile.jl")

end # module AcousticScattering
