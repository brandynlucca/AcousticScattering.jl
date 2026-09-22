using AcousticScattering
using Test

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
            @test length(mode.shell.radii) == length(mode.shell.pressure) == n_elements + 1
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
    coarse = scattering_amplitude(fem(body, boundary, 1.8; n_elements = 160, m_max = 12))
    fine = scattering_amplitude(fem(body, boundary, 1.8; n_elements = 640, m_max = 12))
    @test abs(fine - reference) < abs(coarse - reference) / 8

    identical = Shelled(ElasticLayer(2.7, 4.0, 2.0; interior_coupling = :identical_fluid),
        FluidInterior(0.0012, 0.23), 0.8)
    result = fem(body, identical, 1.6; n_elements = 640, m_max = 12)
    water = fem(body, boundary, 1.6; n_elements = 640, m_max = 12)
    @test scattering_amplitude(result) == scattering_amplitude(water)
    @test result.data.modes == water.data.modes
    @test scattering_amplitude(result) ≈
          scattering_amplitude(modal(body, identical, 1.6; m_max = 12)) rtol = 0.001
end

@time "Elastic and layered radial FEM frequency sweeps" @testset "Elastic and layered radial FEM frequency sweeps" begin
    body = Sphere(0.01)
    for boundary in (SolidElastic(2.7, 4.0, 2.0),
        Shelled(ElasticLayer(2.7, 4.0, 2.0), FluidInterior(1.0, 1.0), 0.8),
        Shelled(FluidLayer(1.04, 1.04), VacuumInterior(), 0.8),
        Shelled(FluidLayer(1.04, 1.04), FluidInterior(0.0012, 0.23), 0.8))
        solve = k -> fem(body, boundary, k; n_elements = 400, m_max = 10)
        sweep = frequency_sweep(solve, [12000.0, 38000.0], 1500.0)
        @test sweep.amplitudes == scattering_amplitude.(solve.(sweep.k))
        reference = scattering_amplitude.([modal(body, boundary, k; m_max = 10)
                                           for k in sweep.k])
        @test sweep.amplitudes ≈ reference rtol = 0.001
        @test sweep.target_strength == target_strength.(sweep.amplitudes)
    end
end
