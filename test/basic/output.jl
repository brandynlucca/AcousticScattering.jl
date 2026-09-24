using AcousticScattering
using Test

@testset "Solution outputs" begin
    body = Sphere(0.01)
    k = 0.5
    modal_solution = modal(body, Rigid(), k; m_max = 3)
    amplitude = scattering_amplitude(modal_solution)
    @test amplitude isa ComplexF64
    @test target_strength(modal_solution) isa Float64
    @test target_strength(modal_solution) ≈ 20log10(abs(amplitude))
    @test target_strength(amplitude) ≈ target_strength(modal_solution)
    @test diagnostics(modal_solution) === nothing

    point = (0.02, 0.0, 0.0)
    incident = pressure(modal_solution, point; field = :incident)
    scattered = pressure(modal_solution, point; field = :scattered)
    total = pressure(modal_solution, point)
    @test incident isa ComplexF64
    @test scattered isa ComplexF64
    @test total ≈ incident + scattered
    @test pressure(modal_solution, [point, (0.03, 0.0, 0.0)]) isa Vector{ComplexF64}
    @test length(pressure(modal_solution, [0.02 0.03; 0.0 0.0; 0.0 0.0])) == 2

    fluid_solution = modal(body, FluidFilled(1.2, 1.1), k; m_max = 3)
    @test pressure(fluid_solution, (0.0, 0.0, 0.0); field = :interior) isa ComplexF64

    boundary_solution = mfs(Sphere(1.0), Rigid(), k; incidence_angle = 0.0,
        n = 12, offset = 0.2, condition_limit = 0)
    report = diagnostics(boundary_solution)
    @test report isa NamedTuple
    @test report.relative_residual < 1e-4
    @test scattering_amplitude(boundary_solution) isa ComplexF64
    @test target_strength(boundary_solution) isa Float64
    @test pressure(boundary_solution, (1.5, 0.0, 0.0); field = :total) isa ComplexF64

    @test_throws ArgumentError pressure(modal_solution, point; field = :unknown)
    @test_throws ArgumentError pressure(modal_solution, point; region = 0)
    @test_throws ArgumentError pressure(modal_solution, [0.02, 0.0])
    @test_throws ArgumentError pressure(modal_solution, [0.02 0.03; 0.0 0.0])
    @test_throws ArgumentError pressure(modal_solution, [(0.02, 0.0)])
    @test_throws ArgumentError pressure(modal_solution, (NaN, 0.0, 0.0))
    @test_throws ArgumentError pressure(modal_solution, (0.0, 0.0, 0.0);
        field = :scattered)
    @test_throws ArgumentError pressure(modal_solution, point; field = :interior)
    @test_throws ArgumentError pressure(boundary_solution, (0.0, 0.0, 0.0);
        field = :interior)
    @test_throws ArgumentError target_strength(modal_solution; angle = 0.0)
    @test_throws ArgumentError scattering_amplitude(modal_solution; angle = 0.0)
end

let
    @time "Public exports and source docstrings" @testset "Public exports and source docstrings" begin
        expected = Set((:Rigid, :PressureRelease, :FluidFilled, :GasFilled, :SolidElastic,
            :Shelled, :FluidLayer, :ElasticLayer, :ViscousLayer, :LayeredMaterial,
            :VacuumInterior, :FluidInterior, :AbstractBody, :Sphere, :Cylinder, :Spheroid,
            :Shell, :Irregular, :AbstractSolution, :ModalSolution,
            :KirchhoffSolution, :FEMSolution,
            :BEMSolution, :MFSSolution, :FMSolution, :modal, :kirchhoff, :fem, :bem, :mfs, :fourier,
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
end
