using AcousticScattering
using PrecompileTools: PrecompileTools
using Test: @test, @testset

enabled = parse(Bool, only(ARGS))
@testset "Installed package (workload enabled: $enabled)" begin
    @test PrecompileTools.workload_enabled(AcousticScattering) == enabled
    wavenumber = 2pi * 38000.0 / 1477.4
    @test isapprox(target_strength(modal(Sphere(0.01), Rigid(), wavenumber)),
        -49.088291; atol = 1e-4)
    fluid = FluidFilled(1028.9 / 1026.8, 1480.3 / 1477.4)
    @test isapprox(target_strength(modal(Sphere(0.01), fluid, wavenumber)),
        -94.278687; atol = 1e-3)
    spheroid = modal(Spheroid(0.02, 0.01), Rigid(), 2pi * 12000.0 / 1477.4;
        incidence_angle = pi / 4, m_max = 6, n_max = 8)
    @test isfinite(target_strength(spheroid))
end
