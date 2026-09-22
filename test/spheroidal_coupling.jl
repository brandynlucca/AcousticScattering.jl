using AcousticScattering
using Test

@time "Spheroidal coupling at angular nodes" @testset "Spheroidal coupling at angular nodes" begin
    @test_skip "requires SpheroidalWaves backend, not available locally"
end
