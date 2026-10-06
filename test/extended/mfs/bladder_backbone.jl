using AcousticScattering
using Test

@testset "Pressure-release bladder and fluid backbone MFS" begin
    # Canonical two-body geometry, not the paper's unavailable X-ray mesh.
    bladder_args = (; semiaxes = (0.25, 0.25, 0.25), center = (-0.9, 0.0, 0.0))
    bone_args = (; semiaxes = (0.2, 0.2, 0.2), center = (0.9, 0.0, 0.0))
    bladder = mesh(; bladder_args..., resolution = 0.6, mesh_order = 3, qorder = 2)
    bone = mesh(; bone_args..., resolution = 0.6, mesh_order = 3, qorder = 2)
    bladder_sources = mesh(; bladder_args..., resolution = 0.7, mesh_order = 3, qorder = 2)
    bone_sources = mesh(; bone_args..., resolution = 0.7, mesh_order = 3, qorder = 2)
    bladder_check = mesh(; bladder_args..., resolution = 0.5, mesh_order = 3, qorder = 2)
    bone_check = mesh(; bone_args..., resolution = 0.5, mesh_order = 3, qorder = 2)
    material = FluidFilled(1100 / 1026, 2273 / 1490)
    kwargs = (; bladder_offset = 0.15, backbone_offset_ext = 0.12,
        backbone_offset_int = 0.12, bladder_source_mesh = bladder_sources,
        backbone_source_mesh = bone_sources, check_bladder = bladder_check,
        check_backbone = bone_check)
    combined = mfs(bladder, bone, material, 2.0; kwargs...)
    report = combined.data.diagnostics
    @test report.method === :least_squares
    @test report.bladder_oversampling > 1.3
    @test report.backbone_oversampling > 1.3
    @test report.component_residuals.backbone_velocity < 5e-4
    @test report.boundary_residual.relative_residual < 0.005
    @test report.boundary_residual.backbone_velocity < 0.01

    # When backbone material matches water, the coupled result must recover
    # the single pressure-release bladder solution.
    bladder_only = mfs(bladder, PressureRelease(), 2.0;
        offset = 0.15, source_mesh = bladder_sources, check_mesh = bladder_check)
    transparent = mfs(bladder, bone, FluidFilled(1.0, 1.0), 2.0; kwargs...)
    @test abs(scattering_amplitude(transparent) - scattering_amplitude(bladder_only)) /
          abs(scattering_amplitude(bladder_only)) < 5e-4
    @test abs(scattering_amplitude(combined) - scattering_amplitude(bladder_only)) /
          abs(scattering_amplitude(bladder_only)) > 0.02

    # A separate coupled BEM method approaches pressure release as the bladder
    # density contrast tends to zero. Its finer geometry differs from the MFS
    # collocation and source meshes.
    reference = bem([bladder, bone], [FluidFilled(1e-8, 1.0), material], 2.0;
        parents = [0, 0], incidence_angle = pi / 2)
    expected = scattering_amplitude(reference)
    actual = scattering_amplitude(combined)
    @test abs(actual - expected) / abs(expected) < 0.003
    @test abs(20log10(abs(actual / expected))) < 0.02
    @test target_strength(combined) ≈ 20log10(abs(actual)) atol = 1e-12
end
