using AcousticScattering
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "Solution interface contract (every concrete AbstractSolution type)" begin
    a = 0.01
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = AS.Sphere(a)

    modal_sol = AS.modal(sphere, AS.Rigid(), k)
    kirch_sol = AS.kirchhoff(sphere, AS.Rigid(), k)
    fem_sol = AS.fem(sphere, AS.Rigid(), k)
    bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)
    mfs_sol = AS.mfs(sphere, AS.Rigid(), k; n = 16)

    @testset "target_strength works with no keywords on every concrete type" begin
        for sol in (modal_sol, kirch_sol, fem_sol, bem_sol, mfs_sol)
            @test AS.target_strength(sol) isa Float64
        end
    end

    @testset "scattering_amplitude: available for 4 of 5, explicit error for the scalar-FEM case" begin
        for sol in (modal_sol, kirch_sol, bem_sol, mfs_sol)
            @test AS.scattering_amplitude(sol) isa Complex
        end
        @test_throws ArgumentError AS.scattering_amplitude(fem_sol)
    end

    @testset "angle/azimuth keywords: supported where a real bistatic query exists, explicit error otherwise" begin
        # modal/kirchhoff/scalar-fem: no reusable surface state, angle already baked in at solve
        # time — must error, not silently ignore the keyword (a real bug caught this session:
        # target_strength(fem_sol; angle=0.3) used to silently return the unchanged backscatter
        # value instead of erroring or actually honoring the angle).
        @test_throws ArgumentError AS.target_strength(modal_sol; angle = 0.3)
        @test_throws ArgumentError AS.scattering_amplitude(modal_sol; angle = 0.3)
        @test_throws ArgumentError AS.target_strength(kirch_sol; angle = 0.3)
        @test_throws ArgumentError AS.scattering_amplitude(kirch_sol; angle = 0.3)
        @test_throws ArgumentError AS.target_strength(fem_sol; angle = 0.3)

        # bem/mfs axisymmetric: genuine reusable surface state, angle/azimuth are real queries.
        @test AS.target_strength(bem_sol; angle = pi / 2) isa Float64
        @test AS.target_strength(mfs_sol; angle = pi / 2) isa Float64
    end
end

@testset "Visualization sampling (frequency/incidence-angle sweeps)" begin
    a = 0.01
    c_water = 1477.4
    sphere = AS.Sphere(a)

    @testset "frequency_sweep: shape, k conversion, endpoint agreement" begin
        freqs = 20e3:10e3:60e3
        sweep = AS.frequency_sweep(k -> AS.modal(sphere, AS.Rigid(), k), freqs, c_water)
        @test sweep.frequencies == collect(freqs)
        @test length(sweep.k) == length(freqs)
        @test length(sweep.target_strength) == length(freqs)
        @test sweep.k ≈ 2pi .* collect(freqs) ./ c_water

        single = AS.frequency_sweep(k -> AS.modal(sphere, AS.Rigid(), k), [38000.0], c_water)
        k_direct = 2pi * 38000.0 / c_water
        ts_direct = AS.target_strength(AS.modal(sphere, AS.Rigid(), k_direct))
        @test single.target_strength[1] == ts_direct
    end

    @testset "incidence_angle_sweep: shape, endpoint agreement" begin
        k = 2pi * 38000.0 / c_water
        spheroid = AS.Spheroid(0.05, 0.02)
        angles = 0:(pi / 8):(pi / 2)
        sweep = AS.incidence_angle_sweep(
            angle -> AS.modal(spheroid, AS.Rigid(), k; incidence_angle = angle), angles)
        @test sweep.angles == collect(angles)
        @test length(sweep.target_strength) == length(angles)

        single = AS.incidence_angle_sweep(
            angle -> AS.modal(spheroid, AS.Rigid(), k; incidence_angle = angle), [pi /
                                                                                  4])
        ts_direct = AS.target_strength(AS.modal(spheroid, AS.Rigid(), k; incidence_angle = pi /
                                                                                           4))
        @test single.target_strength[1] == ts_direct
    end
end

@testset "Visualization sampling (bistatic sweep/map)" begin
    a = 0.01
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = AS.Sphere(a)
    bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)

    @testset "bistatic_sweep: shape, endpoint agreement, azimuthal periodicity" begin
        angles = 0:(pi / 4):pi
        sweep = AS.bistatic_sweep(bem_sol, angles)
        @test sweep.angles == collect(angles)
        @test length(sweep.target_strength) == length(angles)

        single = AS.bistatic_sweep(bem_sol, [pi]; azimuth = 0.3)
        ts_direct = AS.target_strength(bem_sol; angle = pi, azimuth = 0.3)
        @test single.target_strength[1] == ts_direct

        sweep_0 = AS.bistatic_sweep(bem_sol, [pi / 2]; azimuth = 0.0)
        sweep_2pi = AS.bistatic_sweep(bem_sol, [pi / 2]; azimuth = 2pi)
        @test sweep_0.target_strength[1] ≈ sweep_2pi.target_strength[1] atol = 1e-8
    end

    @testset "bistatic_sweep/BistaticSweep records the solution's actual incidence angle" begin
        β = deg2rad(30.0)
        spheroid = AS.Spheroid(0.02, 0.01)
        sol = AS.bem(spheroid, AS.Rigid(), k; incidence_angle = β, m_max = 8, n = 16)
        sweep = AS.bistatic_sweep(sol, 0:(pi / 4):pi; azimuth = pi)
        @test sweep.incidence_angle == β
    end

    @testset "bistatic_map: shape, endpoint agreement" begin
        thetas = 0:(pi / 4):pi
        phis = 0:(pi / 2):(2pi)
        map_result = AS.bistatic_map(bem_sol, thetas, phis)
        @test size(map_result.target_strength) == (length(thetas), length(phis))
        @test map_result.target_strength[2, 3] ==
              AS.target_strength(bem_sol; angle = thetas[2], azimuth = phis[3])
    end

    @testset "axisymmetric vs full-BEM cross-validation (coordinate convention correctness)" begin
        full_sol = AS.bem(sphere, AS.Rigid(), k; method = :full,
            meshsize = AS.bem3d_elements_per_wavelength(k))
        thetas = [0.0, pi / 2, pi]
        phis = [0.0, pi / 2]
        axi_map = AS.bistatic_map(bem_sol, thetas, phis)
        full_map = AS.bistatic_map(full_sol, thetas, phis)
        @test all(abs.(axi_map.target_strength .- full_map.target_strength) .< 0.5)
    end
end

@testset "Visualization sampling (revolve_panels)" begin
    a = 0.01
    c_water = 1477.4
    k = 2pi * 38000.0 / c_water
    sphere = AS.Sphere(a)

    @testset "shape, radius, and z agree with panel midpoints" begin
        bem_sol = AS.bem(sphere, AS.Rigid(), k; n = 16)
        d = bem_sol.data
        ps = AS.panels(d.mesh)
        surf = AS.revolve_panels(ps, d.p_scat_modes; n_phi = 36)
        @test size(surf.x) == size(surf.y) == size(surf.z) == size(surf.field) ==
              (36, length(ps))
        @test all(
            hypot(surf.x[i, j], surf.y[i, j]) ≈ ps[j].rhom
        for i in 1:36, j in eachindex(ps))
        @test all(surf.z[i, j] == ps[j].zm for i in 1:36, j in eachindex(ps))
    end

    @testset "axisymmetric (m=0-only) field is constant across azimuth" begin
        bem_axial = AS.bem(sphere, AS.Rigid(), k; n = 16, incidence_angle = 0.0)
        d = bem_axial.data
        @test length(d.p_scat_modes) == 1
        ps = AS.panels(d.mesh)
        surf = AS.revolve_panels(ps, d.p_scat_modes; n_phi = 36)
        @test all(surf.field[i, j] == surf.field[1, j]
        for i in 1:36, j in eachindex(ps))
    end
end

@testset "Mesh interface: geometry preservation, orientation, element counts" begin
    a = 0.01
    n = 40
    sphere = AS.Sphere(a)
    m = AS.mesh(sphere; resolution = n)

    @testset "element counts match requested resolution" begin
        @test AS.element_count(m) == n
        @test length(AS.elements(m)) == n
        @test length(AS.coordinates(m)) == n
        @test length(AS.normals(m)) == n
    end

    @testset "geometry preservation: every element sits on the sphere's own surface" begin
        for (rho, z) in AS.coordinates(m)
            @test hypot(rho, z) ≈ a atol = 1e-3 * a
        end
    end

    @testset "orientation: outward normal has positive radial component (convex body about the origin)" begin
        for ((rho, z), (nrho, nz)) in zip(AS.coordinates(m), AS.normals(m))
            @test nrho * rho + nz * z > 0
        end
    end

    @testset "spheroid geometry preservation" begin
        a2, b2 = 0.05, 0.02
        spheroid = AS.Spheroid(a2, b2)
        m2 = AS.mesh(spheroid; resolution = 30)
        for (rho, z) in AS.coordinates(m2)
            @test (rho / b2)^2 + (z / a2)^2 ≈ 1.0 atol = 1e-2
        end
    end

    @testset "full 3D mesh element counts and geometry" begin
        k = 2pi * 38000.0 / 1477.4
        m3 = AS.mesh(sphere; k = k, method = :full)
        @test AS.element_count(m3) == length(AS.coordinates(m3)) ==
              length(AS.normals(m3))
        # Gmsh's triangulated quadrature nodes approximate the sphere, they don't sit exactly on
        # it — a coarse mesh at this resolution deviates from `a` by ~1-2%, not the 0.1% the
        # axisymmetric meridian mesh above achieves, so this tolerance is deliberately looser.
        for c in AS.coordinates(m3)
            @test hypot(c...) ≈ a atol = 0.03 * a
        end
    end

    # Round-trip mesh I/O is genuinely untestable, not merely unwritten: `src/ecosystem/mesh_io.jl`
    # is a stub ("Mesh import (.stl/.msh/.vtk). Not yet implemented."), so there is no I/O
    # capability to round-trip yet. Flagged here rather than silently skipped from the suite.

    @testset "bent cylinder geometry is rejected, not silently straightened" begin
        bent = AS.Cylinder(0.01, 0.07; radius_curvature = 0.20)
        @test_throws ArgumentError AS.mesh(bent; resolution = 20)
        @test_throws ArgumentError AS.mesh(bent; k = 2pi * 38000.0 / 1477.4)
        @test_throws ArgumentError AS.fem(bent, AS.Rigid(), 100.0)
        @test_throws ArgumentError AS.fem(bent, AS.SolidElastic(7.8, 3.7, 1.9), 100.0)
    end

    @testset "resolution/thickness sanity checks" begin
        @test_throws ArgumentError AS.sphere_mesh(0.01, 2)
        @test_throws ArgumentError AS.spheroid_mesh(0.05, 0.02, 2)
        @test_throws ArgumentError AS.Shell(AS.Sphere(0.01), 5.0)
    end
end
