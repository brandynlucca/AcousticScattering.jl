using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

@testset "Oblique gas-spheroid resonance" begin
    @test_skip "requires SpheroidalWaves quad-precision backend, not available locally"
end
