using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "FEM reports across discretizations" @testset "FEM reports across discretizations" begin
        sphere, cylinder, spheroid = Sphere(1.0), Cylinder(1.0, 3.0), Spheroid(1.2, 1.0)
        solid = SolidElastic(2.7, 4.0, 2.0)
        elastic_shell = Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8)
        fluid_shell = Shelled(FluidLayer(1.2, 1.1), FluidInterior(1.0, 1.0), 0.8)
        for body in (sphere, cylinder), boundary in (solid, elastic_shell)

            solution = fem(body, boundary, 1.0; n_elements = 8, m_max = 2)
            d = diagnostics(solution)
            @test d.converged === nothing
            @test d.relative_residual < 1e-9
            @test Set(r.component for r in d.systems) == Set((:radial_basis, :interface))
            @test Set(r.mode for r in d.systems) == Set(0:2)
        end
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1), fluid_shell,
            Shelled(FluidLayer(1.2, 1.1), VacuumInterior(), 0.8))
            solution = boundary isa FluidFilled ?
                       fem(
                sphere, boundary, 1.0; n_elements_int = 8, n_elements_ext = 8, m_max = 2) :
                       fem(sphere, boundary, 1.0; n_elements = 8, m_max = 2)
            @test diagnostics(solution).relative_residual < 1e-10
            @test Set(r.mode for r in diagnostics(solution).systems) == Set(0:2)
        end
        for body in (sphere, cylinder, spheroid), boundary in (Rigid(), PressureRelease())

            solution = body isa Sphere ?
                       fem(
                body, boundary, 1.0; method = :meridian, n_r = 3, n_theta = 8, l_max = 3) :
                       fem(body, boundary, 1.0; n_r = 3, n_theta = 8, l_max = 3, m_max = 2)
            d = diagnostics(solution)
            @test d.relative_residual < 1e-10
            @test Set(r.mode for r in d.systems) == (body isa Sphere ? Set([0]) : Set(0:2))
            @test all(r.n_r == 3 && r.n_theta == 8 for r in d.systems)
        end
        for method in (:thin, :general)
            solution = fem(Shell(Spheroid(1.2, 1.0), 0.02), Shelled(0.3, 2700.0, 70e9),
                1000.0, 1500.0, method == :thin ? 0.0 : 1000.0, 1500.0, 1.0;
                method, incidence_angle = 0.0, n_eta = 9, n_t = 3, m_max = 0)
            d = diagnostics(solution)
            @test d.discretization == method
            @test d.relative_residual < 1e-8
            @test only(d.systems).mode == 0
            @test d.solver_options.n_eta == 9
        end
    end
end
