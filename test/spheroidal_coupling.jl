using AcousticScattering
using Test

@time @testset "Spheroidal coupling at angular nodes" begin
    @test_skip "requires SpheroidalWaves backend, not available locally"
end
