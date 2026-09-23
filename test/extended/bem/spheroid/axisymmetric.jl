using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Oblique spheroid complex references at strong contrast" @testset "Oblique spheroid complex references at strong contrast" begin
        body = AS.Spheroid(0.03, 0.01)
        k = 20.0
        boundary = AS.FluidFilled(5.0, 2.0)
        numerical = AS.bem(body, boundary, k; method = :axisymmetric,
            n = 32, m_max = 4, incidence_angle = pi / 3)
        reference = AS.modal(body, boundary, k; incidence_angle = pi / 3,
            m_max = 4, n_max = 6)
        @test isfinite(AS.scattering_amplitude(numerical))
        @test abs(AS.target_strength(numerical) - AS.target_strength(reference)) < 3
    end
end

let
    @time "Oblique gas-spheroid resonance" @testset "Oblique gas-spheroid resonance" begin
        body = AS.Spheroid(0.025, 0.0075)
        boundary = AS.GasFilled(0.00129, 0.23)
        k = 2pi * 500 / 1477.4
        numerical = AS.bem(body, boundary, k; method = :axisymmetric,
            n = 32, m_max = 4, incidence_angle = pi / 3)
        reference = AS.modal(body, boundary, k; incidence_angle = pi / 3,
            m_max = 4, n_max = 6, precision = :quad)
        @test isfinite(AS.scattering_amplitude(numerical))
        @test abs(AS.target_strength(numerical) - AS.target_strength(reference)) < 3
    end
end

let
    @time "Oblique (multi-Fourier-mode) incidence" @testset "Oblique (multi-Fourier-mode) incidence" begin
        c_water = 1477.4

        @time @testset "$(typeof(boundary)): sphere rotational symmetry (independent of solve_axial)" for boundary in (AS.Rigid(), AS.PressureRelease())
            a = 0.01
            freq = 38000.0
            k = 2pi * freq / c_water
            body = AS.Sphere(a)
            mesh = AS.sphere_mesh(a, 12)
            ts_modal = AS.target_strength(AS.modal(body, boundary, k))

            for β_deg in (0.0, 90.0)
                β = deg2rad(β_deg)
                sol = AS._bem_oblique(body, boundary, k, mesh, β; m_max = 5, rtol = 1e-3)
                ts = AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test ts ≈ ts_modal atol = 0.3
            end
        end

        @time "no-argument target_strength/scattering_amplitude matches the antipodal backscatter query, at the actual incidence angle used" @testset "no-argument target_strength/scattering_amplitude matches the antipodal backscatter query, at the actual incidence angle used" begin
            a = 0.01
            freq = 38000.0
            k = 2pi * freq / c_water
            body = AS.Sphere(a)
            for β_deg in (0.0, 30.0, 90.0)
                β = deg2rad(β_deg)
                sol = AS.bem(body, AS.Rigid(), k; incidence_angle = β, m_max = 8, n = 16)
                @test AS.target_strength(sol) ==
                      AS.target_strength(sol; angle = pi - β, azimuth = pi)
                @test AS.scattering_amplitude(sol) ==
                      AS.scattering_amplitude(sol; angle = pi - β, azimuth = pi)
            end

            spheroid = AS.Spheroid(0.05, 0.02)
            β = deg2rad(30.0)
            sol_mfs = AS.mfs(
                spheroid, AS.Rigid(), k; incidence_angle = β, m_max = 8, offset = 0.015)
            @test AS.target_strength(sol_mfs) ==
                  AS.target_strength(sol_mfs; angle = pi - β, azimuth = pi)
        end

        @time @testset "$(typeof(boundary)): prolate spheroid cross-check against the analytical modal series" for boundary in (AS.Rigid(), AS.PressureRelease())
            spheroid = AS.Spheroid(0.05, 0.02)
            k = 20.0
            for beta in (0.0, pi / 2)
                reference = AS.modal(spheroid, boundary, k;
                    incidence_angle = beta, m_max = 4, n_max = 6)
                numerical = AS.bem(spheroid, boundary, k; method = :axisymmetric,
                    n = 32, incidence_angle = beta, m_max = 4)
                @test abs(AS.target_strength(numerical) -
                          AS.target_strength(reference)) < 2
            end
        end

        @time @testset "$(typeof(boundary)): finite cylinder cross-check against FCMS" for boundary in (AS.Rigid(), AS.PressureRelease())
            radius, length = 0.01, 1.0
            freq = 20000.0
            k = 2pi * freq / c_water
            body = AS.Cylinder(radius, length)
            β = π / 2

            ts_modal = AS.target_strength(AS.modal(
                body, boundary, k; incidence_angle = β, m_max = 30))
            sol = AS.bem(
                body, boundary, k; n = 140, incidence_angle = β, m_max = 4, rtol = 1e-4)
            ts_bem = AS.target_strength(sol; angle = π - β, azimuth = π)
            @test ts_bem ≈ ts_modal atol = 0.01
        end

        @time "solve_oblique(::FluidFilled,...): reduced 2n×2n system correctness" @testset "solve_oblique(::FluidFilled,...): reduced 2n×2n system correctness" begin
            a_pol, b_eq = 0.01, 0.07
            spheroid = AS.Spheroid(a_pol, b_eq)
            c_med = 1477.3
            k = 2pi * 38000.0 / c_med
            mesh = AS.spheroid_mesh(a_pol, b_eq, 24)
            β = deg2rad(8.0)

            p_rigid, d_rigid, ps_rigid = AS.solve_oblique(
                AS.Rigid(), k, mesh, β; m_max = 15)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, d_rigid, k, π - β, π)
            stiff = AS.FluidFilled(1e8, 1e8)
            p_stiff, d_stiff, ps_stiff = AS.solve_oblique(stiff, k, mesh, β; m_max = 15)
            ts_stiff = AS.target_strength(ps_stiff, p_stiff, d_stiff, k, π - β, π)
            @test ts_stiff ≈ ts_rigid atol = 1e-4

            trivial = AS.FluidFilled(1.0, 1.0)
            p_triv, d_triv, _ = AS.solve_oblique(trivial, k, mesh, β; m_max = 10)
            @test maximum(abs, p_triv[1]) < 0.05
            @test maximum(abs, d_triv[1]) < 5.0

            bc = AS.FluidFilled(1028.9 / 1026.8, 1480.3 / c_med)
            k_small = 20.0
            reference = AS.modal(spheroid, bc, k_small; incidence_angle = β,
                m_max = 4, n_max = 6, precision = :quad)
            numerical = AS.bem(spheroid, bc, k_small; method = :axisymmetric,
                n = 32, m_max = 4, incidence_angle = β)
            @test abs(AS.target_strength(numerical) -
                      AS.target_strength(reference)) < 3
        end

        @time "_azimuthal_fixed_order: far-pair quadrature converges under mesh refinement" @testset "_azimuthal_fixed_order: far-pair quadrature converges under mesh refinement" begin
            a_pol, b_eq = 0.01, 0.07
            k = 2pi * 38000.0 / 1477.3
            β_r = deg2rad(8.0)
            maxR(panels) = begin
                mesh_r = AS.spheroid_mesh(a_pol, b_eq, panels)
                ps_r = AS.panels(mesh_r)
                n_r = length(ps_r)
                I_r = Matrix{ComplexF64}(I, n_r, n_r)
                K_int, V_int, _ = AS.assemble_cbie_operators(mesh_r, k; m = 0, rtol = 1e-6)
                p_inc = ComplexF64[AS._p_inc_mode(0, k, β_r, p.rhom, p.zm) for p in ps_r]
                dpdn_inc = ComplexF64[AS._dpdn_inc_mode(
                                          0, k, β_r, p.rhom, p.zm, p.nrho, p.nz)
                                      for p in ps_r]
                R = (0.5 * I_r + K_int) * p_inc - V_int * dpdn_inc
                maximum(abs, R)
            end
            R48, R96 = maxR(48), maxR(96)
            @test R96 < 0.6 * R48
            @test R96 < 0.01
        end
    end
end
