using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "Adaptive FEM refinement status" @testset "Adaptive FEM refinement status" begin
        for boundary in (Rigid(), FluidFilled(1.2, 1.1))
            solution = fem(Sphere(1.0), boundary, 1.0; adaptive = true, m_max = 3,
                n_elements_start = 16, max_n_elements = 128, target_tol = 0.01)
            report = diagnostics(solution)
            @test report.converged === nothing
            @test report.refinement.converged
            @test report.refinement.change_db < report.refinement.target_tol == 0.01
            @test 16 < report.refinement.n_elements <= 128
            @test abs(target_strength(solution) -
                      target_strength(modal(Sphere(1.0), boundary, 1.0; m_max = 3))) < 0.1
            failed = @test_logs (:warn, r"did not converge") fem(
                Sphere(1.0), boundary, 1.0;
                adaptive = true, m_max = 2, n_elements_start = 8, max_n_elements = 8)
            @test !diagnostics(failed).refinement.converged
            @test diagnostics(failed).refinement.change_db === nothing
        end
    end
end

let
    @time "Elastic and layered radial FEM complex sphere references" @testset "Elastic and layered radial FEM complex sphere references" begin
        body = Sphere(1.0)
        cases = (
            (SolidElastic(2.7, 4.0, 2.0), (0.4, 3.2), 400),
            (Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8),
                (0.4, 1.92), 640),
            (Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(0.0012, 0.23), 0.8),
                (0.4, 3.2), 640),
            (Shelled(FluidLayer(1.04, 1.04), VacuumInterior(), 0.8), (0.4, 4.0), 200),
            (Shelled(FluidLayer(1.04, 1.04), FluidInterior(1.2, 1.1), 0.8),
                (0.4, 4.0), 200),
            (Shelled(FluidLayer(1.04, 1.04), FluidInterior(0.0012, 0.23), 0.8),
                (0.016, 1.6), 200))
        for (boundary, ks, n_elements) in cases, k in ks

            reference = modal(body, boundary, k; m_max = 14)
            solution = fem(body, boundary, k; n_elements, m_max = 14)
            amplitude = scattering_amplitude(solution)
            @test amplitude ≈ scattering_amplitude(reference) rtol = 0.001
            @test abs(target_strength(solution) - target_strength(reference)) < 0.01
            @test target_strength(solution) == target_strength(amplitude)
            @test diagnostics(solution).relative_residual < 1e-8
            @test length(solution.data.modes) == 15
            @test_throws ArgumentError scattering_amplitude(solution; angle = 0.3)
            mode = first(solution.data.modes)
            if hasproperty(mode, :elastic)
                @test length(mode.elastic.radii) == length(mode.elastic.longitudinal) ==
                      n_elements + 1
                @test all(isfinite, mode.elastic.longitudinal)
                @test mode.elastic.shear === nothing
                dipole = solution.data.modes[2].elastic
                @test length(dipole.shear) == length(dipole.radii)
                @test all(isfinite, dipole.shear)
                @test first(mode.elastic.radii) ≈ (boundary isa SolidElastic ? 0.0 : 0.8)
                @test last(mode.elastic.radii) ≈ 1.0
            else
                @test length(mode.shell.radii) == length(mode.shell.pressure) ==
                      n_elements + 1
                @test all(isfinite, mode.shell.pressure)
                @test first(mode.shell.radii) ≈ 0.8
                @test last(mode.shell.radii) ≈ 1.0
            end
            if boundary isa SolidElastic || boundary.interior isa VacuumInterior
                @test mode.interior === nothing
            else
                @test isfinite(mode.interior.coefficient)
                @test mode.interior.radius ≈ 0.8
                @test mode.interior.wavenumber ≈ k / boundary.interior.soundspeed_contrast
            end
        end
    end

    @time "Solid sphere complex rigid and shrinking-cavity limits" @testset "Solid sphere complex rigid and shrinking-cavity limits" begin
        body = Sphere(1.0)
        stiff = SolidElastic(5000.0, 40.0, 20.0)
        solid = SolidElastic(2.7, 4.0, 2.0)
        shell = Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.005)
        for k in (0.4, 3.2)
            rigid = scattering_amplitude(modal(body, Rigid(), k))
            @test scattering_amplitude(modal(body, stiff, k)) ≈ rigid rtol = 0.001
            @test scattering_amplitude(fem(body, stiff, k; n_elements = 200)) ≈ rigid rtol = 0.001
            @test scattering_amplitude(modal(body, shell, k)) ≈
                  scattering_amplitude(modal(body, solid, k)) rtol = 1e-5
        end
    end

    @time "Elastic radial refinement and interior coupling" @testset "Elastic radial refinement and interior coupling" begin
        body = Sphere(1.0)
        boundary = Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8)
        reference = scattering_amplitude(modal(body, boundary, 1.8; m_max = 12))
        coarse = scattering_amplitude(fem(
            body, boundary, 1.8; n_elements = 160, m_max = 12))
        fine = scattering_amplitude(fem(body, boundary, 1.8; n_elements = 640, m_max = 12))
        @test abs(fine - reference) < abs(coarse - reference) / 8

        identical = Shelled(
            ElasticLayer(2.7, 4.0, 2.0; interior_coupling = :identical_fluid),
            FluidInterior(0.0012, 0.23), 0.8)
        result = fem(body, identical, 1.6; n_elements = 640, m_max = 12)
        water = fem(body, boundary, 1.6; n_elements = 640, m_max = 12)
        @test scattering_amplitude(result) == scattering_amplitude(water)
        @test result.data.modes == water.data.modes
        @test scattering_amplitude(result) ≈
              scattering_amplitude(modal(body, identical, 1.6; m_max = 12)) rtol = 0.001
    end
end

let
    @time "radial FEM (sphere)" @testset "radial FEM (sphere)" begin
        c_water = 1477.4
        a = 0.01
        freq = 38000.0
        k = 2pi * freq / c_water

        sphere = AS.Sphere(a)
        ts_modal_rigid = AS.target_strength(AS.modal(sphere, AS.Rigid(), k))
        ts_fem_rigid = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; R = 3a, n_elements = 200))
        @test ts_fem_rigid ≈ ts_modal_rigid atol = 1e-3

        ts_modal_pr = AS.target_strength(AS.modal(sphere, AS.PressureRelease(), k))
        ts_fem_pr = AS.target_strength(AS.fem(
            sphere, AS.PressureRelease(), k; R = 3a, n_elements = 200))
        @test ts_fem_pr ≈ ts_modal_pr atol = 1e-3

        ts_fem_R2 = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; R = 5a, n_elements = 200))
        @test ts_fem_rigid ≈ ts_fem_R2 atol = 1e-3

        ts_fem_quad = AS.target_strength(AS.fem(
            sphere, AS.Rigid(), k; R = 3a, n_elements = 100, order = 2))
        @test ts_fem_quad ≈ ts_modal_rigid atol = 1e-3

        reference_soundspeed = 1477.3
        range_cases_khz = (12.0, 200.0)
        for freq_khz in range_cases_khz
            kk = 2pi * freq_khz * 1000 / reference_soundspeed
            ts_modal = AS.target_strength(AS.modal(sphere, AS.Rigid(), kk))
            ts_fem_adaptive = AS.target_strength(AS.fem(
                sphere, AS.Rigid(), kk; R = 3a, adaptive = true, target_tol = 0.005))
            @test ts_fem_adaptive ≈ ts_modal atol = 0.01
        end

        range_cases_pr_khz = (12.0, 200.0)
        for freq_khz in range_cases_pr_khz
            kk = 2pi * freq_khz * 1000 / reference_soundspeed
            ts_modal_pr2 = AS.target_strength(AS.modal(sphere, AS.PressureRelease(), kk))
            ts_fem_pr_adaptive = AS.target_strength(AS.fem(
                sphere, AS.PressureRelease(), kk; R = 3a,
                adaptive = true, target_tol = 0.005))
            @test ts_fem_pr_adaptive ≈ ts_modal_pr2 atol = 0.01
        end
    end

    @time "fluid shell sphere radial FEM (vs modal series)" @testset "fluid shell sphere radial FEM (vs modal series)" begin
        freq = 38000.0
        c_ext = 1477.4
        rho_ext = 1026.8
        k = 2pi * freq / c_ext
        a = 0.05
        rr = 0.9

        sphere = AS.Sphere(a)
        bc_pr = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), rr)
        @test AS.target_strength(AS.fem(sphere, bc_pr, k; n_elements = 200, m_max = 20)) ≈
              AS.target_strength(AS.modal(sphere, bc_pr, k; m_max = 20)) atol = 1e-4

        bc_g = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), rr)
        @test AS.target_strength(AS.fem(sphere, bc_g, k; n_elements = 200, m_max = 20)) ≈
              AS.target_strength(AS.modal(sphere, bc_g, k; m_max = 20)) atol = 1e-4

        bc_plain_fluid = AS.FluidFilled(1.24 / rho_ext, 345.0 / c_ext)
        ts_plain = AS.target_strength(AS.modal(sphere, bc_plain_fluid, k; m_max = 20))
        bc_g2 = AS.Shelled(AS.FluidLayer(1.24 / rho_ext, 345.0 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), 0.999)
        @test AS.target_strength(AS.fem(sphere, bc_g2, k; n_elements = 400, m_max = 20)) ≈
              ts_plain atol = 1e-3

        bc_g3 = AS.Shelled(AS.FluidLayer(1.24 / rho_ext, 345.0 / c_ext),
            AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), 0.999999)
        @test AS.reflection_coefficient(bc_g3, k, a) ≈
              AS.reflection_coefficient(AS.FluidFilled(1.24 / rho_ext, 345.0 / c_ext)) atol = 1e-6
        bc_pr3 = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), 0.999999)
        @test AS.reflection_coefficient(bc_pr3, k, a) ≈
              AS.reflection_coefficient(AS.PressureRelease()) atol = 1e-4

        for f in (5e3, 38e3, 120e3, 250e3)
            kk = 2pi * f / c_ext
            @test abs(AS.reflection_coefficient(bc_g, kk, a)) <= 1.0 + 1e-10
        end

        ts_ka = AS.target_strength(AS.kirchhoff(sphere, bc_g, 2pi * 200e3 / c_ext))
        ts_modal_hika = AS.target_strength(AS.modal(sphere, bc_g, 2pi * 200e3 / c_ext; m_max = 40))
        @test ts_ka ≈ ts_modal_hika atol = 0.1
    end
end

let
    @time "Complex acoustic radial FEM against modal spheres" @testset "Complex acoustic radial FEM against modal spheres" begin
        body = Sphere(1.0)
        for boundary in (Rigid(), PressureRelease()), k in (0.4, 4.0), order in (1, 2)
            solution = fem(body, boundary, k; R = 3.0, n_elements = 200, order)
            reference = modal(body, boundary, k)
            amplitude = scattering_amplitude(solution)
            @test amplitude ≈ scattering_amplitude(reference) rtol = 0.001
            @test abs(target_strength(solution) - target_strength(reference)) < 0.01
            @test target_strength(solution) == target_strength(amplitude)
            mode = first(solution.data.modes)
            @test mode.order == order
            @test length(mode.exterior.pressure) == length(mode.exterior.radii) ==
                  order * 200 + 1
            @test extrema(mode.exterior.radii) == (1.0, 3.0)
            @test all(isfinite, mode.exterior.pressure)
            @test mode.interior === nothing
            @test_throws ArgumentError scattering_amplitude(solution; angle = 0.3)
        end

        for (boundary, ks) in ((FluidFilled(1.04, 1.04), (0.4, 1.6)),
            (FluidFilled(10.0, 0.5), (0.4, 1.6)),
            (GasFilled(0.0012, 0.23), (0.012, 0.0138, 0.016)))
            for k in ks
                solution = fem(
                    body, boundary, k; n_elements_int = 200, n_elements_ext = 100)
                reference = modal(body, boundary, k)
                @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 0.001
                @test abs(target_strength(solution) - target_strength(reference)) < 0.01
                mode = first(solution.data.modes)
                @test length(mode.interior.pressure) == length(mode.interior.radii) == 201
                @test length(mode.exterior.pressure) == length(mode.exterior.radii) == 101
                @test extrema(mode.interior.radii) == (0.0, 1.0)
                @test extrema(mode.exterior.radii) == (1.0, 1.2)
                @test all(isfinite, mode.interior.pressure)
                @test isfinite(mode.interior.interface_flux)
                @test isfinite(mode.exterior.interface_flux)
            end
        end
    end

    @time "Adaptive radial FEM retains the final complex solution" @testset "Adaptive radial FEM retains the final complex solution" begin
        body, k = Sphere(1.0), 1.0
        for (boundary, element_options) in ((Rigid(), (; order = 1)),
            (PressureRelease(), (; order = 1)), (FluidFilled(1.2, 1.1), (;)))
            solution = fem(body, boundary, k; adaptive = true, m_max = 3,
                n_elements_start = 16, max_n_elements = 128, target_tol = 0.001,
                element_options...)
            report = diagnostics(solution)
            @test report.refinement.converged
            n = report.refinement.n_elements
            options = boundary isa FluidFilled ?
                      (; n_elements_int = n, n_elements_ext = n) : (; n_elements = n)
            fixed = fem(body, boundary, k; m_max = 3, options..., element_options...)
            @test scattering_amplitude(solution) == scattering_amplitude(fixed)
            @test solution.data.modes == fixed.data.modes
            @test scattering_amplitude(solution) ≈
                  scattering_amplitude(modal(body, boundary, k; m_max = 3)) rtol = 0.001
            @test target_strength(solution) ==
                  AcousticScattering.radial_fem_target_strength(
                boundary, k, body.radius, 1.2; m_max = 3, options..., element_options...)

            failed = @test_logs (:warn, r"did not converge") fem(body, boundary, k;
                adaptive = true, m_max = 3, n_elements_start = 4, max_n_elements = 8,
                target_tol = 0.0, element_options...)
            @test !diagnostics(failed).refinement.converged
            options = boundary isa FluidFilled ?
                      (; n_elements_int = 8, n_elements_ext = 8) : (; n_elements = 8)
            finest = fem(body, boundary, k; m_max = 3, options..., element_options...)
            @test failed.data.modes == finest.data.modes
            @test scattering_amplitude(failed) == scattering_amplitude(finest)
        end
    end
end
