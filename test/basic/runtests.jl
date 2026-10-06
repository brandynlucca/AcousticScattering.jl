using Test

macro timed_include(path)
    quote
        let test_file = $(esc(path)), started = time_ns()
            try
                include(test_file)
            finally
                @info "Basic test file completed" file=test_file elapsed_seconds=(time_ns()-started)/1e9
            end
        end
    end
end

const TEST_BASIC_GROUP = get(ENV, "TEST_BASIC_GROUP", "All")
const BASIC_GROUPS = (
    "Modal", "TMatrix", "Kirchhoff", "FEM", "BEM", "MFS", "Fourier", "Plotting",
    "Utilities")
TEST_BASIC_GROUP == "All" || TEST_BASIC_GROUP in BASIC_GROUPS ||
    throw(ArgumentError("Unknown TEST_BASIC_GROUP=$TEST_BASIC_GROUP"))

@info "Running Basic group: $TEST_BASIC_GROUP"

@testset "Basic regression tests" begin
    if TEST_BASIC_GROUP in ("All", "Modal")
        elapsed = @elapsed @testset "Modal" begin
            @timed_include "modal/sphere.jl"
            @timed_include "modal/cylinder.jl"
            @timed_include "modal/cylinder_envelope.jl"
            @timed_include "modal/spheroid.jl"
            @timed_include "modal/spheroid_incident.jl"
            @timed_include "modal/vesm.jl"
        end
        @info "Basic group completed" group="Modal" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "TMatrix")
        elapsed = @elapsed @testset "T-matrix" begin
            @timed_include "tmatrix/spheroid.jl"
        end
        @info "Basic group completed" group="T-matrix" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "Kirchhoff")
        elapsed = @elapsed @testset "Kirchhoff" begin
            @timed_include "kirchhoff/sphere.jl"
            @timed_include "kirchhoff/spheroid.jl"
            @timed_include "kirchhoff/cylinder.jl"
            @timed_include "kirchhoff/high_frequency.jl"
            @timed_include "kirchhoff/envelope.jl"
            @timed_include "kirchhoff/envelope_slender.jl"
            @timed_include "kirchhoff/mesh.jl"
        end
        @info "Basic group completed" group="Kirchhoff" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "FEM")
        elapsed = @elapsed @testset "FEM" begin
            @timed_include "fem/sphere.jl"
            @timed_include "fem/spheroid.jl"
            @timed_include "fem/cylinder.jl"
            @timed_include "fem/pressure_meridian.jl"
            @timed_include "fem/meridian_reference.jl"
            @timed_include "fem/material_fields.jl"
            @timed_include "fem/mesh.jl"
            @timed_include "fem/volume.jl"
            @timed_include "fem/volume_regions.jl"
            @timed_include "fem/spatial_fluid.jl"
            @timed_include "fem/volume_rotation.jl"
            @timed_include "fem/volume_solvers.jl"
            @timed_include "fem/free_surface.jl"
        end
        @info "Basic group completed" group="FEM" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "BEM")
        elapsed = @elapsed @testset "BEM" begin
            @timed_include "bem/sphere.jl"
            @timed_include "bem/spheroid.jl"
            @timed_include "bem/cylinder.jl"
            @timed_include "bem/assembly.jl"
            @timed_include "bem/quadrature_workspace.jl"
            @timed_include "bem/quadrature_cycles.jl"
            @timed_include "bem/kernel_reuse.jl"
            @timed_include "bem/modal_kernel.jl"
            @timed_include "bem/mode_storage.jl"
            @timed_include "bem/arbitrary.jl"
            @timed_include "bem/edge.jl"
            @timed_include "bem/formulations.jl"
            @timed_include "bem/preconditioner.jl"
            @timed_include "bem/setup.jl"
            @timed_include "bem/regions_compressed.jl"
            @timed_include "bem/adaptive.jl"
        end
        @info "Basic group completed" group="BEM" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "MFS")
        elapsed = @elapsed @testset "MFS" begin
            @timed_include "mfs/sphere.jl"
            @timed_include "mfs/spheroid.jl"
            @timed_include "mfs/cylinder.jl"
            @timed_include "mfs/sweeps.jl"
            @timed_include "mfs/assembly.jl"
        end
        @info "Basic group completed" group="MFS" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "Fourier")
        elapsed = @elapsed @testset "Fourier matching" begin
            @timed_include "fourier/sphere.jl"
            @timed_include "fourier/spheroid.jl"
            @timed_include "fourier/irregular.jl"
        end
        @info "Basic group completed" group="Fourier matching" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "Plotting")
        elapsed = @elapsed @testset "Plotting" begin
            @timed_include "plot/plot1d.jl"
            @timed_include "plot/plot2d.jl"
            @timed_include "plot/plot3d.jl"
            @timed_include "plot/volume.jl"
        end
        @info "Basic group completed" group="Plotting" elapsed_seconds=elapsed
    end
    if TEST_BASIC_GROUP in ("All", "Utilities")
        elapsed = @elapsed @testset "Utilities" begin
            @timed_include "mesh.jl"
            @timed_include "sampling.jl"
            @timed_include "farfield.jl"
            @timed_include "output.jl"
            @timed_include "internals.jl"
            @timed_include "source_layout.jl"
            @timed_include "pressure.jl"
            @timed_include "incident.jl"
            @timed_include "incident_solvers.jl"
        end
        @info "Basic group completed" group="Utilities" elapsed_seconds=elapsed
    end
end
