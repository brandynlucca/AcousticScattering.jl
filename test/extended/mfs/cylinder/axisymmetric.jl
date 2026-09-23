using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Axisymmetric MFS (cylinder with spheroidal endcaps, axial incidence)" @testset "Axisymmetric MFS (cylinder with spheroidal endcaps, axial incidence)" begin
        radius, cyl_length, endcap_depth = 0.01, 0.05, 0.01
        c_water = 1477.3
        freq = 38000.0
        k = 2pi * freq / c_water
        mesh = AS.cylinder_spheroidal_endcap_mesh(radius, cyl_length, endcap_depth, 112)
        capped_cyl = AS.Cylinder(radius, cyl_length; endcap_depth = endcap_depth)

        for boundary in (AS.Rigid(), AS.PressureRelease())
            p_bem, dpdn_bem, ps_bem = AS.solve_axial(boundary, k, mesh; rtol = 1e-5)
            ts_bem = AS.target_strength(ps_bem, p_bem, dpdn_bem, k, pi)
            for offset_frac in (0.1, 0.5)
                ts_mfs = AS.target_strength(AS.mfs(
                    capped_cyl, boundary, k; incidence_angle = 0.0,
                    offset = offset_frac * radius, n = 112))
                @test ts_mfs ≈ ts_bem atol = 0.05
            end
        end
    end
end
