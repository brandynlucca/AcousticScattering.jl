using AcousticScattering
using Test

@testset "Sampling" begin
    sphere = Sphere(0.01)
    frequencies = [20_000.0, 30_000.0]
    sweep = frequency_sweep(k -> modal(sphere, Rigid(), k; m_max = 3),
        frequencies, 1500.0)
    @test sweep isa AcousticScattering.FrequencySweep
    @test sweep.frequencies == frequencies
    @test sweep.k ≈ 2pi .* frequencies ./ 1500.0
    @test size(sweep.amplitudes) == (2,)
    @test sweep.target_strength ≈ target_strength.(sweep.amplitudes)

    body = Cylinder(0.1, 0.5)
    angles = [pi / 4, pi / 2]
    aspect = incidence_angle_sweep(
        angle -> modal(body, Rigid(), 1.0; incidence_angle = angle, m_max = 2),
        angles)
    @test aspect isa AcousticScattering.IncidenceAngleSweep
    @test aspect.angles == angles
    @test length(aspect.amplitudes) == length(angles)
    @test aspect.target_strength ≈ target_strength.(aspect.amplitudes)

    irregular = Irregular(1.0, Float64[], Float64[])
    fourier_angles = [pi / 3, pi / 2]
    fourier_aspect = incidence_angle_sweep(irregular, PressureRelease(), 0.5,
        fourier_angles; mapping_order = 2, continuation_steps = 1,
        m_max = 2, n_max = 2)
    @test fourier_aspect.angles == fourier_angles
    @test length(fourier_aspect.amplitudes) == 2
    @test fourier_aspect.target_strength ≈
          target_strength.(fourier_aspect.amplitudes)

    solution = mfs(Sphere(1.0), Rigid(), 0.5; incidence_angle = 0.0,
        n = 12, offset = 0.2, condition_limit = 0)
    cut = bistatic_sweep(solution, [0.0, pi / 2, pi])
    @test cut isa AcousticScattering.BistaticSweep
    @test length(cut.amplitudes) == 3
    @test cut.target_strength ≈ target_strength.(cut.amplitudes)
    grid = bistatic_map(solution, [0.0, pi / 2], [0.0, pi])
    @test grid isa AcousticScattering.BistaticMap
    @test size(grid.target_strength) == (2, 2)
    @test grid.thetas == [0.0, pi / 2]

    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    surface = mesh(nodes, triangles; qorder = 2)
    coupled = bem([surface], [FluidFilled(1.2, 1.1)], 0.3;
        condition_limit = 0)
    comparison = components(coupled; labels = ["inclusion"])
    @test comparison isa AcousticScattering.ComponentComparison
    @test length(comparison.isolated) == 1
    @test comparison.labels == ["Coupled", "Isolated inclusion", "Coherent sum"]
    @test_throws ArgumentError components(coupled; labels = ["too", "many"])
    component_cut = bistatic_sweep(comparison, [0.0, pi])
    @test size(component_cut.amplitudes) == (2, 3)
    @test component_cut.labels == comparison.labels

    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        Float64[], 1500.0)
    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        [20_000.0], 0.0)
    @test_throws ArgumentError incidence_angle_sweep(angle -> modal(body, Rigid(), 1.0),
        Float64[])
    @test_throws ArgumentError bistatic_sweep(solution, Float64[])
    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        [-1.0], 1500.0)
    @test_throws ArgumentError frequency_sweep(k -> modal(sphere, Rigid(), k),
        [NaN], 1500.0)
    @test_throws ArgumentError incidence_angle_sweep(angle -> modal(body, Rigid(), 1.0),
        [NaN])
    @test_throws ArgumentError incidence_angle_sweep(irregular, PressureRelease(), 0.5,
        Float64[])
    @test_throws ArgumentError bistatic_sweep(solution, [NaN])
    @test_throws ArgumentError bistatic_sweep(solution, [0.0]; azimuth = Inf)
    @test_throws ArgumentError incidence_angle_sweep([surface], [FluidFilled(1.2, 1.1)],
        0.3, [0.0]; labels = ["inclusion"])
end

@testset "Sweep constructors and mesh sweeps" begin
    @test AcousticScattering.FrequencySweep([1.0, 2.0], [1.0, 2.0], [-10.0, -11.0]).labels ==
          ["Scattered field"]
    @test AcousticScattering.IncidenceAngleSweep([0.0, 1.0], [-10.0, -11.0]).amplitudes ===
          nothing
    @test AcousticScattering.BistaticSweep([0.0, 1.0], 0.0, [-10.0, -11.0], 0.3).incidence_azimuth ==
          0.0

    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    tetra = mesh(nodes, triangles; qorder = 2)
    angles = [0.0, 0.5]
    rigid = incidence_angle_sweep(tetra, Rigid(), 0.3, angles;
        compression = (method = :none,))
    @test length(rigid.target_strength) == 2 && all(isfinite, rigid.target_strength)
    fluid = incidence_angle_sweep(tetra, FluidFilled(1.2, 1.1), 0.3, angles)
    @test length(fluid.amplitudes) == 2 && all(isfinite, fluid.target_strength)
    coupled = incidence_angle_sweep([tetra], [FluidFilled(1.2, 1.1)], 0.3, angles;
        components = true, labels = ["body"])
    @test size(coupled.amplitudes) == (2, 3)
    @test coupled.labels == ["Coupled", "Isolated body", "Coherent sum"]
    @test_throws ArgumentError incidence_angle_sweep(
        [tetra], [FluidFilled(1.2, 1.1)], 0.3, angles; labels = ["body"])

    scalar = frequency_sweep(
        k -> fem(Cylinder(0.05, 0.1), Rigid(), k; incidence_angle = 0.0,
            n_r = 6, n_theta = 12, m_max = 0),
        [1000.0, 2000.0], 1477.4)
    @test scalar.amplitudes === nothing && all(isfinite, scalar.target_strength)
end
