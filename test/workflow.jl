using AcousticScattering
using Test
using LinearAlgebra: BLAS, norm

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

@time "Ellipsoid construction and complex sweeps" @testset "Ellipsoid construction and complex sweeps" begin
    surface = mesh(; semiaxes = (0.3, 0.4, 0.6), center = (0.1, -0.2, 0.3),
        rotation = (axis = (0, 1, 0), angle = pi / 2), resolution = 0.8)
    nodes = surface.body.nodes
    radii = ((nodes[1, :] .- 0.1) ./ 0.6) .^ 2 .+
            ((nodes[2, :] .+ 0.2) ./ 0.4) .^ 2 .+
            ((nodes[3, :] .- 0.3) ./ 0.3) .^ 2
    @test maximum(abs.(radii .- 1)) < 1e-10
    @test_throws ArgumentError mesh(; semiaxes = (1, 0, 1))
    @test_throws ArgumentError mesh(; semiaxes = (1, 1, 1), tip_ratio = 0)
    @test_throws ArgumentError mesh(; semiaxes = (1, 1, 1),
        rotation = (axis = (0, 0, 0), angle = 1))

    solve = k -> modal(Sphere(0.1), Rigid(), k)
    sweep = frequency_sweep(solve, [100.0, 200.0], 1500.0)
    @test sweep.amplitudes == scattering_amplitude.(solve.(sweep.k))
    @test sweep.target_strength == target_strength.(sweep.amplitudes)
    radial = frequency_sweep(k -> fem(Sphere(0.1), Rigid(), k), [100.0], 1500.0)
    @test radial.amplitudes[1] ==
          scattering_amplitude(fem(Sphere(0.1), Rigid(), only(radial.k)))
    scalar_solve = k -> fem(Sphere(0.1), Rigid(), k;
        method = :meridian, n_r = 3, n_theta = 8, l_max = 3)
    scalar = frequency_sweep(scalar_solve, [100.0], 1500.0)
    @test scalar.amplitudes === nothing
    @test scalar.target_strength[1] == target_strength(scalar_solve(only(scalar.k)))
    @test_throws ArgumentError frequency_sweep(solve, Float64[], 1500.0)
    @test_throws ArgumentError frequency_sweep(solve, [100.0], 0.0)

    surfaces = [mesh(; semiaxes = (r, r, r), resolution = 0.9, qorder = 4)
                for r in (1.0, 0.3)]
    materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
    sol = bem(surfaces, materials, 0.8; incidence_angle = 0.7, incidence_azimuth = 0.4)
    comparison = components(sol; labels = ["flesh", "bladder"])
    @test comparison.coupled === sol
    @test comparison.labels ==
          ["Coupled", "Isolated flesh", "Isolated bladder", "Coherent sum"]
    @test_throws ArgumentError components(sol; labels = ["one"])
    pattern = bistatic_sweep(comparison, [0.3, 1.2, 4.1]; azimuth = 0.6)
    @test pattern.incidence_azimuth == 0.4
    @test size(pattern.amplitudes) == (3, 4)
    @test pattern.amplitudes[:, 4] ≈ pattern.amplitudes[:, 2] + pattern.amplitudes[:, 3]
    direct = bem(
        surfaces[2], materials[2], 0.8; incidence_angle = 0.7, incidence_azimuth = 0.4)
    for (i, theta) in enumerate(pattern.angles)
        direction = [cos(theta), sin(theta) * cos(0.6), sin(theta) * sin(0.6)]
        @test pattern.amplitudes[i, 1] ≈ scattering_amplitude(sol; direction)
        @test pattern.amplitudes[i, 3] ≈ scattering_amplitude(direct; direction)
    end
    spectrum = frequency_sweep(_ -> comparison, [0.8 * 1500 / (2pi)], 1500.0)
    tilt = incidence_angle_sweep(_ -> comparison, [0.7])
    @test size(spectrum.amplitudes) == (1, 4)
    @test spectrum.amplitudes == tilt.amplitudes
    @test spectrum.amplitudes[1, 1] ≈ scattering_amplitude(sol)
    single_pattern = bistatic_sweep(sol, pattern.angles; azimuth = 0.6)
    @test single_pattern.amplitudes ≈ pattern.amplitudes[:, 1]
end
