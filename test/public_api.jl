# Included by the Interfaces group; calls below deliberately use ordinary user imports.
@testset "Public exports and source docstrings" begin
    expected = Set((:Rigid, :PressureRelease, :FluidFilled, :GasFilled, :SolidElastic,
        :Shelled, :FluidLayer, :ElasticLayer, :ViscousLayer, :LayeredMaterial,
        :VacuumInterior, :FluidInterior, :AbstractBody, :Sphere, :Cylinder, :Spheroid,
        :Shell, :AbstractSolution, :ModalSolution, :KirchhoffSolution, :FEMSolution,
        :BEMSolution, :MFSSolution, :modal, :kirchhoff, :fem, :bem, :mfs,
        :target_strength, :scattering_amplitude, :pressure, :diagnostics, :Mesh, :mesh,
        :components, :frequency_sweep, :incidence_angle_sweep, :bistatic_sweep, :bistatic_map))
    @test Set(names(AcousticScattering)) == union(expected, Set((:AcousticScattering,)))
    for name in expected
        @test isdefined(@__MODULE__, name)
        @test getfield(@__MODULE__, name) === getfield(AcousticScattering, name)
        @test haskey(Base.Docs.meta(AcousticScattering), Base.Docs.Binding(AcousticScattering, name))
    end

    body = Sphere(0.01)
    solution = modal(body, Rigid(), 100.0)
    @test solution isa ModalSolution
    @test target_strength(solution) ≈ 20 * log10(abs(scattering_amplitude(solution)))
    @test mesh(body; resolution = 12) isa Mesh
    @test diagnostics(solution) === nothing
    @test_throws ArgumentError target_strength(solution; angle = 0.3)
    @test_throws ArgumentError mesh(body)
    @test_throws ArgumentError mesh(body; resolution = 12, k = 100.0)
end

@testset "Bent MFS ASCII grid controls" begin
    body = Cylinder(0.01, 0.07; radius_curvature = 0.20)
    ascii = mfs(body, PressureRelease(), 100.0; n_s = 6, n_phi = 8, offset = 0.003)
    legacy = mfs(body, PressureRelease(), 100.0; n_s = 6, n_φ = 8, offset = 0.003)
    @test scattering_amplitude(ascii) ≈ scattering_amplitude(legacy)
    @test length(ascii.data.points) == 6 * 8
    @test_throws ArgumentError mfs(body, Rigid(), 100.0; n_phi = 8, n_φ = 8)
    @test_throws ArgumentError mfs(body, Rigid(), 100.0; n_phi = 0)
    @test_throws ArgumentError mfs(body, Rigid(), 100.0; n_s = 0)
end

@testset "Full BEM diagnostics survive public dispatch" begin
    body = Sphere(0.01)
    k = 100.0
    options = (reltol = 1e-8, restart = 150, maxiter = 1200)
    solution = bem(body, PressureRelease(), k; method = :full, meshsize = 0.004,
        qorder = 4, compression = (method = :none,), gmres_kwargs = options)
    d = diagnostics(solution)
    @test d.converged
    @test d.method == :gmres
    @test 0 < d.iterations <= options.maxiter
    @test length(d.residual_history) == d.iterations
    @test d.relative_residual <= 2 * options.reltol
    @test d.absolute_residual >= 0
    @test d.unknown_count == d.quadrature_nodes == length(solution.data.quad)
    @test d.meshsize == 0.004
    @test d.quadrature_order == 4
    @test d.mesh_order == 2
    @test d.formulation == :burton_miller
    @test d.coupling == im / k
    @test d.compression == (method = :none,)
    @test d.correction == (method = :dim,)
    @test d.solver_options.reltol == options.reltol
    @test d.solver_options.abstol == 0
    @test scattering_amplitude(solution) ≈
          scattering_amplitude(modal(body, PressureRelease(), k)) rtol = 0.07

    failed = @test_logs (:warn, r"GMRES did not converge") bem(body, PressureRelease(), k;
        method = :full, meshsize = 0.012, qorder = 2, compression = (method = :none,),
        gmres_kwargs = (reltol = 1e-14, restart = 1, maxiter = 1))
    failed_d = diagnostics(failed)
    @test !failed_d.converged
    @test failed_d.iterations == 1
    @test failed_d.relative_residual > failed_d.solver_options.reltol

    transmission = bem(body, FluidFilled(1.05, 1.02), k;
        method = :full, meshsize = 0.012, qorder = 2)
    direct = diagnostics(transmission)
    @test direct.method == :direct
    @test direct.converged === nothing
    @test direct.iterations === nothing
    @test isempty(direct.residual_history)
    @test direct.relative_residual < 1e-10
    @test direct.unknown_count == 2 * direct.quadrature_nodes
    @test direct.formulation == :muller
    @test direct.equilibrate
    @test direct.scaled_relative_residual < 1e-10
    @test direct.compression == (method = :none,)

    # Preserve the established low-level tuple when diagnostics are not requested.
    low_level = AS.solve_full_bem(PressureRelease(), k, solution.data.quad;
        incidence_angle = pi / 2, compression = (method = :none,), gmres_kwargs = options)
    @test length(low_level) == 3
    @test low_level[1] ≈ solution.data.p_scat
    @test low_level[2] ≈ solution.data.dpdn_scat

    @test AS._linear_residual([1.0;;], [0.0], [0.0]).relative_residual == 0
    @test AS._linear_residual([1.0;;], [1.0], [0.0]).relative_residual == Inf
end
