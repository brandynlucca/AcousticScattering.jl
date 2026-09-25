using AcousticScattering
using Test

const AS = AcousticScattering

@testset "Small numerical helpers" begin
    @test AS.backscattering_cross_section(2.0 + 1.0im) ≈ 5.0
    @test AS.bem3d_elements_per_wavelength(2pi) ≈ 0.1
    @test AS.bem3d_elements_per_wavelength(2pi; elements_per_wavelength = 5) ≈ 0.2

    A = [3.0 0.0; 0.0 1.0; 0.0 0.0]
    report = AS._mfs_matrix_diagnostics(A)
    @test report.conditioning === :svd
    @test report.condition_number ≈ 3.0
    @test report.numerical_rank == 2
    skipped = AS._mfs_matrix_diagnostics(A; condition_limit = 1)
    @test skipped.conditioning === :not_computed && skipped.condition_number === nothing
    @test_throws ArgumentError AS._mfs_matrix_diagnostics(A; condition_limit = -1)
    @test AS._mfs_matrix_diagnostics([1.0 1.0; 1.0 1.0]).condition_number == Inf ||
          AS._mfs_matrix_diagnostics([1.0 1.0; 1.0 1.0]).numerical_rank == 1

    @test AS._record_refinement!(nothing, true, 0.1, 0.5, 10) === nothing
    reports = AS._SolveReports()
    AS._record_refinement!(reports, true, 0.1, 0.5, 10)
    @test reports.refinement.n_elements == 10 && reports.refinement.converged
end
