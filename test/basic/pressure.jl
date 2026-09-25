using AcousticScattering
using Test

@testset "Pressure evaluation across solvers" begin
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    tetra = mesh(nodes, triangles; qorder = 2)
    inside, outside = (0.1, 0.1, 0.1), (2.0, 0.0, 0.0)

    @testset "Full BEM on an explicit mesh" begin
        rigid = bem(tetra, Rigid(), 0.3; compression = (method = :none,))
        @test pressure(rigid, collect(outside)) ≈ pressure(rigid, outside)
        @test isfinite(pressure(rigid, (1.0, 0.0, 0.0)))
        @test_throws ArgumentError pressure(rigid, inside)

        fluid = bem(tetra, FluidFilled(1.2, 1.1), 0.3)
        @test pressure(fluid, inside) ≈ pressure(fluid, inside; field = :interior)
        @test pressure(fluid, [inside, outside]) ≈
              [pressure(fluid, inside), pressure(fluid, outside)]
        @test_throws ArgumentError pressure(fluid, inside; field = :scattered)
    end

    @testset "Axisymmetric BEM interior fields" begin
        fluid = FluidFilled(1.2, 1.1)
        sphere = bem(Sphere(0.01), fluid, 100.0; n = 16)
        @test isfinite(pressure(sphere, (0.001, 0.0, 0.0); field = :interior))
        cylinder = bem(Cylinder(0.01, 0.03), fluid, 100.0; n = 16)
        @test isfinite(pressure(cylinder, (0.001, 0.0, 0.0); field = :interior))
        @test isfinite(pressure(cylinder, (0.05, 0.0, 0.0)))
    end

    @testset "Sphere shells agree between modal and radial FEM" begin
        k, center = 50.0, (0.001, 0.0, 0.0)
        shell = Shelled(FluidLayer(1.1, 1.02), FluidInterior(1.05, 1.02), 0.8)
        modal_shell = modal(Sphere(0.01), shell, k; m_max = 3)
        fem_shell = fem(Sphere(0.01), shell, k; n_elements = 20, m_max = 3)
        @test pressure(fem_shell, center; field = :interior)≈pressure(
            modal_shell, center; field = :interior) rtol=1e-3
        @test isfinite(pressure(fem_shell, (0.009, 0.0, 0.0); field = :shell))

        filled = FluidFilled(1.05, 1.02)
        fem_filled = fem(Sphere(0.01), filled, k;
            n_elements_int = 20, n_elements_ext = 20, m_max = 3)
        @test pressure(fem_filled, center; field = :interior)≈pressure(
            modal(Sphere(0.01), filled, k; m_max = 3), center; field = :interior) rtol=0.02
        @test isfinite(pressure(fem_filled, (0.012, 0.0, 0.0)))

        elastic = Shelled(ElasticLayer(2.0, 2.0, 1.0), FluidInterior(1.0, 1.0), 0.8)
        @test isfinite(pressure(modal(Sphere(0.01), elastic, k; m_max = 3), center;
            field = :interior))
    end
end
