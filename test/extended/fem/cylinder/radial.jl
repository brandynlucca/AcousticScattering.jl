using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "elastic sphere/cylinder radial FEM (vs modal series)" @testset "elastic sphere/cylinder radial FEM (vs modal series)" begin
        freq = 38000.0
        c_ext = 1477.4
        rho_ext = 1026.8
        k = 2pi * freq / c_ext
        rho_shell, G, lambda = 2700.0, 2.6e10, 5.3e10
        cL_shell = sqrt((lambda + 2G) / rho_shell)
        cT_shell = sqrt(G / rho_shell)
        radius_shell, radius_fluid = 0.05, 0.048

        solid = AS.SolidElastic(14900.0 / rho_ext, 6853.0 / c_ext, 4171.0 / c_ext)
        sphere1 = AS.Sphere(0.019)
        @test AS.target_strength(AS.fem(sphere1, solid, k; n_elements = 200)) ≈
              AS.target_strength(AS.modal(sphere1, solid, k; m_max = 13)) atol = 0.05

        shell_water = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
            AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), radius_fluid / radius_shell)
        sphere_shell = AS.Sphere(radius_shell)
        @test AS.target_strength(AS.fem(
            sphere_shell, shell_water, k; n_elements = 200, m_max = 20)) ≈
              AS.target_strength(AS.modal(sphere_shell, shell_water, k; m_max = 20)) atol = 0.001

        radius, length = 0.05, 0.5
        rho0, c1, c2 = 7810.0, 5973.0, 3193.0
        solid_cyl = AS.SolidElastic(rho0 / rho_ext, c1 / c_ext, c2 / c_ext)
        cyl = AS.Cylinder(radius, length)
        @test AS.target_strength(AS.fem(cyl, solid_cyl, k; n_elements = 200)) ≈
              AS.target_strength(AS.modal(cyl, solid_cyl, k)) atol = 0.01

        shell_cyl = AS.Shelled(
            AS.ElasticLayer(rho_shell / rho_ext, cL_shell / c_ext, cT_shell / c_ext),
            AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), radius_fluid / radius_shell)
        @test AS.target_strength(AS.fem(cyl, shell_cyl, k; n_elements = 200)) ≈
              AS.target_strength(AS.modal(cyl, shell_cyl, k)) atol = 0.001

        for angle_deg in (90.0, 30.0)
            ang = deg2rad(angle_deg)
            @test AS.target_strength(AS.fem(
                cyl, solid_cyl, k; incidence_angle = ang, n_elements = 200)) ≈
                  AS.target_strength(AS.modal(cyl, solid_cyl, k; incidence_angle = ang)) atol = 0.01
            @test AS.target_strength(AS.fem(
                cyl, shell_cyl, k; incidence_angle = ang, n_elements = 200)) ≈
                  AS.target_strength(AS.modal(cyl, shell_cyl, k; incidence_angle = ang)) atol = 0.001
        end

        ts_solid_modal = AS.target_strength(AS.modal(sphere_shell, solid, k))
        shell_limit_diffs = Float64[]
        for rr in (0.1, 0.005)
            bc = AS.Shelled(
                AS.ElasticLayer(solid.density_contrast, solid.speed_longitudinal_contrast,
                    solid.speed_transversal_contrast),
                AS.FluidInterior(1026.8 / rho_ext, 1477.4 / c_ext), rr)
            ts_fem = AS.target_strength(AS.fem(
                sphere_shell, bc, k; n_elements = 200, m_max = 20))
            push!(shell_limit_diffs, abs(ts_fem - ts_solid_modal))
        end
        @test shell_limit_diffs[1] > 10 * maximum(shell_limit_diffs[2:end])
        @test maximum(shell_limit_diffs[2:end]) < 0.05
    end
end
