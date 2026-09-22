using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(2)

@time "Reusable fluid incidence sweeps" @testset "Reusable fluid incidence sweeps" begin
    outer = mesh(; semiaxes = (1.0, 0.7, 0.8), center = (0.2, -0.1, 0.15),
        resolution = 0.9, qorder = 4)
    inner = mesh(; semiaxes = (0.25, 0.16, 0.18), center = (0.3, -0.06, 0.1),
        resolution = 0.9, qorder = 4)
    surfaces = [outer, inner]
    materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
    angles = [0.6, pi / 2, 0.6]
    k, azimuth = 0.8, 0.4
    options = (;
        formulation = :muller, equilibrate = true, condition_limit = 0, incidence_azimuth = azimuth)
    single = incidence_angle_sweep(outer, materials[1], k, angles; options...)
    fresh_single = incidence_angle_sweep(angles) do incidence_angle
        bem(outer, materials[1], k; incidence_angle, options...)
    end
    @test single.amplitudes ≈ fresh_single.amplitudes rtol = 1e-11
    @test single.target_strength ≈ fresh_single.target_strength atol = 1e-10
    @test single.labels == ["Scattered field"]
    @test single.angles == angles
    @test single.amplitudes[1] == single.amplitudes[end]

    reused = incidence_angle_sweep(surfaces, materials, k, angles;
        components = true, labels = ["body", "inclusion"], options...)
    fresh = incidence_angle_sweep(angles) do incidence_angle
        components(bem(surfaces, materials, k; incidence_angle, options...);
            labels = ["body", "inclusion"])
    end
    @test reused.amplitudes ≈ fresh.amplitudes rtol = 1e-11
    @test reused.target_strength ≈ fresh.target_strength atol = 1e-10
    @test reused.labels == fresh.labels
    @test size(reused.amplitudes) == (3, 4)
    @test reused.amplitudes[:, 4] ≈ reused.amplitudes[:, 2] + reused.amplitudes[:, 3]
    @test reused.amplitudes[1, :] == reused.amplitudes[end, :]

    let (surface, material, wavenumber) = (inner, materials[2], k)
        sweep = incidence_angle_sweep(surface, material, wavenumber, [0.7])
        direct = bem(surface, material, wavenumber; incidence_angle = 0.7)
        @test sweep.amplitudes[1] ≈ scattering_amplitude(direct) rtol = 1e-11
    end
    siblings = [outer,
        mesh(; semiaxes = (0.25, 0.16, 0.18), center = (2.0, 0.3, 0.0),
            resolution = 0.9, qorder = 4)]
    separate = incidence_angle_sweep(siblings, materials, k, angles; parents = [0, 0])
    direct = incidence_angle_sweep(angles) do incidence_angle
        bem(siblings, materials, k; parents = [0, 0], incidence_angle)
    end
    @test separate.amplitudes ≈ direct.amplitudes rtol = 1e-11
    @test separate.labels == ["Scattered field"]

    @test_throws ArgumentError incidence_angle_sweep(outer, materials[1], k, Float64[])
    @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, [NaN])
    @test_throws ArgumentError incidence_angle_sweep(outer, materials[1], k, angles;
        incidence_azimuth = Inf)
    @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
        condition_limit = -1)
    @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
        components = true, labels = ["one"])
    @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
        labels = ["body", "inclusion"])
    @test_throws ArgumentError incidence_angle_sweep(surfaces, materials[1:1], k, angles)
    @test_throws ArgumentError incidence_angle_sweep(surfaces, materials, k, angles;
        parents = [0, 2])
    @test_throws ArgumentError incidence_angle_sweep(outer, materials[1], 0.0, angles)
end
