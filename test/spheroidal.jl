using AcousticScattering
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "Spheroid modal series (rigid/pressure-release/fluid-filled)" begin
    k = 2pi * 38000.0 / 1477.4

    @testset "converges to the sphere modal series as eccentricity -> 0" begin
        a_sphere = 0.05
        sphere = AS.Sphere(a_sphere)
        for (boundary, sphere_ts) in (
            (AS.Rigid(), AS.target_strength(AS.modal(sphere, AS.Rigid(), k))),
            (AS.PressureRelease(),
            AS.target_strength(AS.modal(sphere, AS.PressureRelease(), k)))
        )
            errs = Float64[]
            for eps in (0.01, 0.001, 0.0001)
                body = AS.Spheroid(a_sphere + eps, a_sphere - eps)
                ts = AS.target_strength(AS.modal(
                    body, boundary, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
                push!(errs, abs(ts - sphere_ts))
            end
            @test errs[1] > errs[2] > errs[3]
            @test errs[3] < 0.1
        end

        g, h = 1028.9 / 1026.8, 1480.3 / 1477.4
        fluid = AS.FluidFilled(g, h)
        sphere_ts = AS.target_strength(AS.modal(sphere, fluid, k))
        errs = Float64[]
        for eps in (0.001, 0.0001, 0.00001)
            body = AS.Spheroid(a_sphere + eps, a_sphere - eps)
            ts = AS.target_strength(AS.modal(
                body, fluid, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
            push!(errs, abs(ts - sphere_ts))
        end
        @test errs[1] > errs[2] > errs[3]
        @test errs[3] < 0.1
    end

    @testset "fluid-filled limiting cases (independent cross-checks)" begin
        body = AS.Spheroid(0.02, 0.005)
        stiff = AS.FluidFilled(1e6, 1e6)
        ts_stiff = AS.target_strength(AS.modal(
            body, stiff, k; incidence_angle = 0.0, m_max = 16, n_max = 16))
        ts_rigid = AS.target_strength(AS.modal(
            body, AS.Rigid(), k; incidence_angle = 0.0, m_max = 16, n_max = 16))
        @test ts_stiff ≈ ts_rigid atol = 1e-3

        ts_pr = AS.target_strength(AS.modal(body, AS.PressureRelease(), k;
            incidence_angle = 0.0, m_max = 16, n_max = 16))
        diffs = Float64[]
        for g_soft in (0.1, 0.01, 0.001)
            soft = AS.FluidFilled(g_soft, 0.5)
            ts_soft = AS.target_strength(AS.modal(
                body, soft, k; incidence_angle = 0.0, m_max = 16, n_max = 16))
            push!(diffs, abs(ts_soft - ts_pr))
        end
        @test diffs[1] > diffs[2] > diffs[3]
        @test diffs[3] < 0.1
    end

    @testset "fluid-filled full off-diagonal coupling (Furusawa Eq. 4)" begin
        q = 3.0
        xi0 = 1.05
        a = q * xi0
        b = sqrt(a^2 - q^2)
        body = AS.Spheroid(a, b)
        k_ref = 1.0
        bc = AS.FluidFilled(1050 / 1026, 1.02; coupling = :full)
        f = AS.scattering_amplitude(AS.modal(body, bc, k_ref;
            incidence_angle = 0.3, m_max = 2, n_max = 4))
        f_bs = f * k_ref / (-2im)
        @test real(f_bs) ≈ 9.10954776790933e-06 atol = 1e-14
        @test imag(f_bs) ≈ 0.00326545204233986 atol = 1e-12

        a_sphere = 0.05
        g, h = 1028.9 / 1026.8, 1480.3 / 1477.4
        bc_full = AS.FluidFilled(g, h; coupling = :full)
        bc_diag = AS.FluidFilled(g, h; coupling = :diagonal)
        errs = Float64[]
        for eps in (0.001, 0.0001, 0.00001)
            body_ns = AS.Spheroid(a_sphere + eps, a_sphere - eps)
            ts_full = AS.target_strength(AS.modal(
                body_ns, bc_full, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
            ts_diag = AS.target_strength(AS.modal(
                body_ns, bc_diag, k; incidence_angle = 0.0, m_max = 20, n_max = 20))
            push!(errs, abs(ts_full - ts_diag))
        end
        @test errs[1] > errs[2] > errs[3]
        @test errs[3] < 0.01

        body_fem = AS.Spheroid(0.08, 0.02)
        bc_fem_full = AS.FluidFilled(1.05, 1.05; coupling = :full)
        bc_fem_diag = AS.FluidFilled(1.05, 1.05; coupling = :diagonal)
        c_water = 1500.0
        for (freq_hz, ts_comsol) in ((38000.0, -72.301), (60000.0, -73.284))
            k_fem = 2pi * freq_hz / c_water
            ka_fem = k_fem * 0.08
            order = ceil(Int, 1.8 * ka_fem)
            ts_full = AS.target_strength(AS.modal(body_fem, bc_fem_full, k_fem;
                incidence_angle = deg2rad(30.0), m_max = order, n_max = order))
            ts_diag = AS.target_strength(AS.modal(body_fem, bc_fem_diag, k_fem;
                incidence_angle = deg2rad(30.0), m_max = order, n_max = order))
            @test ts_full ≈ ts_comsol atol = 0.5
            @test abs(ts_diag - ts_comsol) > 2.0
        end
    end

    @testset "fluid-filled full coupling: quad-precision special functions (Jech et al. 2015)" begin
        c_med, rho_med = 1477.3, 1026.8
        rho_ws, c_ws = 1028.9, 1480.3
        body_ws = AS.Spheroid(0.07, 0.01)
        bc_ws = AS.FluidFilled(rho_ws / rho_med, c_ws / c_med)
        rho_gas, c_gas = 1.24, 345.0
        bc_gas = AS.FluidFilled(rho_gas / rho_med, c_gas / c_med)
        golden_ws = [(12.0, -87.05, 14, 14), (48.0, -84.91, 18, 18),
            (88.0, -88.95, 10, 22), (128.0, -105.33, 30, 30)]
        for (freq_khz, ts_bench, m_max, n_max) in golden_ws
            k_ws = 2pi * freq_khz * 1000 / c_med
            ts = AS.target_strength(AS.modal(
                body_ws, bc_ws, k_ws; incidence_angle = pi / 2,
                m_max = m_max, n_max = n_max, precision = :quad))
            @test ts ≈ ts_bench atol = 0.3
        end

        golden_gas = [
            (12.0, -30.13435708789646, 14, 14), (48.0, -28.594046949257944, 10, 56)]
        for (freq_khz, ts_ref, m_max, n_max) in golden_gas
            k_gas = 2pi * freq_khz * 1000 / c_med
            ts = AS.target_strength(AS.modal(
                body_ws, bc_gas, k_gas; incidence_angle = pi / 2,
                m_max = m_max, n_max = n_max, precision = :quad))
            @test ts ≈ ts_ref atol = 0.01
        end
    end

    @testset "Rigid/PressureRelease: precision keyword now actually reaches the modal coefficient" begin
        body = AS.Spheroid(0.10, 0.03)
        k_safe = 2pi * 38000.0 / 1477.4
        for boundary in (AS.Rigid(), AS.PressureRelease())
            ts_q = AS.target_strength(AS.modal(
                body, boundary, k_safe; incidence_angle = 0.0,
                m_max = 10, n_max = 10, precision = :quad))
            ts_d = AS.target_strength(AS.modal(
                body, boundary, k_safe; incidence_angle = 0.0,
                m_max = 10, n_max = 10, precision = :double))
            @test isfinite(ts_q)
            @test ts_q ≈ ts_d atol = 0.01
        end
    end

    @testset "fluid-filled full coupling: no spurious warnings off-axis of m>=1" begin
        body = AS.Spheroid(0.02, 0.005)
        stiff = AS.FluidFilled(1e6, 1e6; coupling = :full)
        @test_logs AS.modal(
            body, stiff, k; incidence_angle = 0.0, m_max = 16, n_max = 16)
    end

    @testset "mode-order convergence (prolate, rigid, endon)" begin
        body = AS.Spheroid(0.10, 0.03)
        ts16 = AS.target_strength(AS.modal(body, AS.Rigid(), k; m_max = 16, n_max = 16))
        ts20 = AS.target_strength(AS.modal(body, AS.Rigid(), k; m_max = 20, n_max = 20))
        ts24 = AS.target_strength(AS.modal(body, AS.Rigid(), k; m_max = 24, n_max = 24))
        @test ts20 ≈ ts24 atol = 0.01
        @test abs(ts24 - ts20) < abs(ts20 - ts16)
    end

    @testset "oblate cross-check against the Kirchhoff baseline" begin
        body = AS.Spheroid(0.03, 0.10)
        cases = vcat(
            [(AS.Rigid(), pi / 2, f) for f in (38.0, 76.0)],
            [(AS.PressureRelease(), pi / 2, f) for f in (38.0, 76.0)],
            [(AS.Rigid(), 0.0, 38.0)],
            [(AS.PressureRelease(), 0.0, 38.0)]
        )
        for (boundary, angle, freq_khz) in cases
            let k_ka = 2pi * freq_khz * 1000 / 1477.4
                ka_eq = k_ka * 0.10
                margin = freq_khz >= 76.0 ? 3 : 8
                order = max(24, ceil(Int, ka_eq) + margin)
                precision = freq_khz < 45.0 ? :double : :quad
                ts_modal = AS.target_strength(AS.modal(
                    body, boundary, k_ka; incidence_angle = angle,
                    m_max = order, n_max = order, precision = precision))
                ts_ka = AS.target_strength(AS.kirchhoff(body, boundary, k_ka; incidence_angle = angle))
                @test ts_modal ≈ ts_ka atol = 1.5
            end
        end
    end
end
