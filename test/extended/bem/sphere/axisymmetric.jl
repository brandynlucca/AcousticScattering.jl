using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "fluid shell sphere BEM (vs modal series)" @testset "fluid shell sphere BEM (vs modal series)" begin
        rho_ext, c_ext = 1026.8, 1477.4
        a = 0.05
        rr = 0.9
        k = 2pi * 12000.0 / c_ext

        sphere = AS.Sphere(a)
        bc_g = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), rr)
        ts_modal = AS.target_strength(AS.modal(sphere, bc_g, k; m_max = 20))
        diffs_g = Float64[]
        for n in (24, 40)
            sol = AS.bem(sphere, bc_g, k; n = n, rtol = 1e-5)
            push!(diffs_g, abs(AS.target_strength(sol) - ts_modal))
        end
        @test issorted(diffs_g, rev = true)
        @test diffs_g[end] < 0.05

        bc_pr = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), rr)
        ts_modal_pr = AS.target_strength(AS.modal(sphere, bc_pr, k; m_max = 20))
        diffs_pr = Float64[]
        for n in (24, 40)
            sol = AS.bem(sphere, bc_pr, k; n = n, rtol = 1e-5)
            push!(diffs_pr, abs(AS.target_strength(sol) - ts_modal_pr))
        end
        @test issorted(diffs_pr, rev = true)
        @test diffs_pr[end] < 0.02
    end

    @time "AxisymmetricBEM (sphere, axial incidence)" @testset "AxisymmetricBEM (sphere, axial incidence)" begin
        a = 0.01
        c_water = 1477.4

        mesh = AS.sphere_mesh(a, 48)
        k_static = 1e-6
        K, _, _ = AS.assemble_cbie_operators(mesh, k_static; rtol = 1e-8)
        @test maximum(abs.(sum(K; dims = 2) .+ 0.5)) < 1e-7

        freq = 38000.0
        k = 2pi * freq / c_water

        sphere = AS.Sphere(a)
        for boundary in (AS.Rigid(), AS.PressureRelease())
            @time "$(typeof(boundary))" @testset "$(typeof(boundary))" begin
                ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))

                sol16 = AS.bem(
                    sphere, boundary, k; n = 16, incidence_angle = 0.0, rtol = 1e-5)
                ts_back_16 = AS.target_strength(sol16; angle = pi)
                ts_fwd_16 = AS.target_strength(sol16; angle = 0.0)

                sol32 = AS.bem(
                    sphere, boundary, k; n = 32, incidence_angle = 0.0, rtol = 1e-5)
                ts_back_32 = AS.target_strength(sol32; angle = pi)

                @test ts_back_16 ≈ ts_modal atol = 0.06
                @test ts_back_32 ≈ ts_modal atol = 0.02
                @test abs(ts_back_32 - ts_modal) < abs(ts_back_16 - ts_modal)

                ts_modal_fwd = AS.target_strength(AS.modal(sphere, boundary, k; angle = 0.0))
                @test ts_fwd_16 ≈ ts_modal_fwd atol = 0.06
            end
        end
    end

    @time "AxisymmetricBEM fluid-filled/transmission (sphere, axial incidence)" @testset "AxisymmetricBEM fluid-filled/transmission (sphere, axial incidence)" begin
        a = 0.05
        c_water = 1477.4
        freq = 38000.0
        k = 2pi * freq / c_water
        sphere = AS.Sphere(a)

        @time "rigid limit (gh >> 1)" @testset "rigid limit (gh >> 1)" begin
            stiff = AS.FluidFilled(1e8, 1e8)
            ts_stiff = AS.target_strength(AS.bem(
                sphere, stiff, k; incidence_angle = 0.0, n = 24))
            ts_rigid = AS.target_strength(AS.bem(
                sphere, AS.Rigid(), k; incidence_angle = 0.0, n = 24))
            @test ts_stiff ≈ ts_rigid atol = 0.25
        end

        @time "pressure-release limit (g << 1)" @testset "pressure-release limit (g << 1)" begin
            soft = AS.FluidFilled(1e-8, 1.0)
            ts_soft = AS.target_strength(AS.bem(
                sphere, soft, k; incidence_angle = 0.0, n = 24))
            ts_pr = AS.target_strength(AS.bem(
                sphere, AS.PressureRelease(), k; incidence_angle = 0.0, n = 24))
            @test ts_soft ≈ ts_pr atol = 0.1
        end

        @time "converges to the analytical FluidFilled sphere modal series" @testset "converges to the analytical FluidFilled sphere modal series" begin
            g, h = 1.05, 1.02
            bc = AS.FluidFilled(g, h)
            ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
            ts_bem = AS.target_strength(AS.bem(
                sphere, bc, k; incidence_angle = 0.0, n = 48))
            @test abs(ts_bem - ts_modal) < 0.15
        end
    end
end

let
    @time "Axisymmetric BEM diagnostic contracts" @testset "Axisymmetric BEM diagnostic contracts" begin
        for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1)),
            beta in (0.0, pi / 3)

            solution = bem(
                Sphere(1.0), boundary, 1.0; n = 12, incidence_angle = beta, m_max = 2)
            report = diagnostics(solution)
            @test report.converged === report.iterations === nothing
            @test report.relative_residual < 1e-10
            @test Set(r.mode for r in report.systems) ==
                  (iszero(beta) ? Set([0]) : Set(0:2))
            @test report.solver_options.n == 12
            low = iszero(beta) ? AS.solve_axial(boundary, 1.0, solution.data.mesh) :
                  AS.solve_oblique(boundary, 1.0, solution.data.mesh, beta; m_max = 2)
            @test solution.data.p_scat_modes == (iszero(beta) ? [low[1]] : low[1])
            @test solution.data.dpdn_scat_modes == (iszero(beta) ? [low[2]] : low[2])
        end
    end

    @time "BEM gas-sphere resonance and mesh refinement" @testset "BEM gas-sphere resonance and mesh refinement" begin
        # NOTE: the fine (n=192) and resonance-peak (n=256) refinement checks are covered at
        # full fidelity in perf/diagnostics.jl; here a single coarse off-resonance k is cheap.
        body, boundary, k = Sphere(1.0), FluidFilled(0.0012, 0.23), 0.014
        reference = modal(body, boundary, k)
        solution = bem(body, boundary, k; incidence_angle = 0.0, n = 96, rtol = 1e-8)
        @test abs(target_strength(solution) - target_strength(reference)) < 0.1
        @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.01
        scaled = bem(Sphere(0.01), boundary, 1.4;
            incidence_angle = pi / 3, n = 96, m_max = 2, rtol = 1e-8)
        scaled_reference = modal(Sphere(0.01), boundary, 1.4)
        @test abs(target_strength(scaled) - target_strength(scaled_reference)) < 0.1
        @test scattering_amplitude(scaled) ≈ scattering_amplitude(scaled_reference) rtol = 0.01
    end
end
