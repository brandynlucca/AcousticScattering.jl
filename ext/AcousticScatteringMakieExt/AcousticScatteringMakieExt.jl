module AcousticScatteringMakieExt

using Makie
using AcousticScattering
using Inti
using StaticArrays: SVector
using PrecompileTools: @compile_workload

include("theme.jl")
include("sampling_plots.jl")
include("sweep_results.jl")
include("mesh_plots.jl")
include("region_plots.jl")

# Backend-agnostic: constructs recipes/scene graphs (this package's own dispatch and helper
# functions, plus Makie's generic Figure/Axis/attribute machinery) without rasterizing, so it
# needs no active Makie backend (e.g. CairoMakie) during this extension's own precompilation.
@compile_workload begin
    body, boundary, k = AcousticScattering.Sphere(0.01), AcousticScattering.Rigid(),
    2π * 38000.0 / 1477.4

    freq_sweep = AcousticScattering.frequency_sweep(
        kk -> AcousticScattering.modal(body, boundary, kk), [30e3, 40e3], 1477.4)
    # Sphere/modal is rotationally symmetric (no incidence_angle); use axisymmetric bem here
    # instead, and avoid Spheroid/modal, which needs the optional SpheroidalWaves backend.
    angle_sweep = AcousticScattering.incidence_angle_sweep(
        beta -> AcousticScattering.bem(body, boundary, k; incidence_angle = beta, n = 8),
        [0.0, 0.5])
    frequencysweepplot(freq_sweep)
    incidenceanglesweepplot(angle_sweep)
    Makie.plot(freq_sweep, angle_sweep)

    # One example per genuinely distinct code path (recipe type / Axis kind / backend
    # primitive); :bistatic_cartesian and direct low-level recipe calls are omitted as
    # redundant with :bistatic_polar and the Makie.plot(sol; kind=...) calls below.
    axi_sol = AcousticScattering.bem(body, boundary, k; n = 8)
    Makie.plot(axi_sol; kind = :bistatic_polar, angles = [0.0, pi / 2, pi])
    Makie.plot(axi_sol; kind = :bistatic_map, thetas = [0.0, pi / 2], phis = [0.0, pi])
    Makie.plot(axi_sol; kind = :mesh)
    Makie.plot(axi_sol; kind = :surface_field)

    full_sol = AcousticScattering.bem(body, boundary, 100.0;
        method = :full, meshsize = 0.012, qorder = 2)
    Makie.plot(full_sol; kind = :mesh)
    Makie.plot(full_sol; kind = :surface_field)
end

end
