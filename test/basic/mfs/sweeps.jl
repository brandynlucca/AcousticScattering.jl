using AcousticScattering
using Test

@testset "Reusable axisymmetric MFS angle sweeps" begin
    angles = [0.0, 0.45, pi / 2, pi, 0.45]
    boundaries = (Rigid(), PressureRelease(), Impedance(1.3 - 0.6im), FluidFilled(1.2, 1.1))
    cases = Tuple{AbstractBody, AcousticScattering.AbstractBoundaryCondition, Int}[(
                                                                                       Sphere(1.0),
                                                                                       boundary,
                                                                                       oversampling)
                                                                                   for boundary in boundaries
                                                                                   for oversampling in (1, 2)]
    append!(cases,
        [(body, boundary, 2)
         for body in (Spheroid(1.2, 0.8), Spheroid(0.8, 1.2), Cylinder(0.5, 1.0),
            Cylinder(0.5, 1.0; endcap_depth = 0.3))
         for boundary in (Rigid(), FluidFilled(1.2, 1.1))])
    for (body, boundary, oversampling) in cases
        offsets = boundary isa FluidFilled ? (; offset_ext = 0.12, offset_int = 0.18) : (;)
        options = (; n = 12, m_max = 3, offset = 0.15, oversampling,
            rtol = 1e-6, condition_limit = 512, offsets...)
        result = incidence_angle_sweep(mfs, body, boundary, 0.7, angles;
            options..., return_diagnostics = true)
        @test result.sweep isa AcousticScattering.IncidenceAngleSweep
        @test result.sweep.angles == angles
        @test result.sweep.labels == ["Scattered field"]
        @test result.sweep.amplitudes[2] == result.sweep.amplitudes[5]
        for (i, beta) in enumerate(angles)
            fresh = mfs(body, boundary, 0.7; incidence_angle = beta, options...)
            @test result.sweep.amplitudes[i]≈scattering_amplitude(fresh) rtol=1e-10 atol=1e-12
            @test result.sweep.target_strength[i] ≈ target_strength(fresh) atol = 1e-8
            expected = diagnostics(fresh)
            actual = result.diagnostics[i]
            @test Dict(pairs(actual.solver_options)) == Dict(pairs(expected.solver_options))
            @test actual.unknown_count == expected.unknown_count
            @test actual.equation_count == expected.equation_count
            @test length(actual.systems) == (iszero(beta) ? 1 : 4)
            for (a, b) in zip(actual.systems, expected.systems)
                @test keys(a) == keys(b)
                for key in keys(a)
                    x, y = getproperty(a, key), getproperty(b, key)
                    if x isa NamedTuple
                        @test x.absolute_residual≈y.absolute_residual rtol=1e-7 atol=1e-12
                        @test x.relative_residual≈y.relative_residual rtol=1e-7 atol=1e-12
                    elseif x isa AbstractFloat
                        @test x≈y rtol=1e-7 atol=1e-12
                    else
                        @test x == y
                    end
                end
            end
        end
    end

    body, boundary = Spheroid(1.2, 0.8), Rigid()
    options = (; n = 12, m_max = 3, offset = 0.15, condition_limit = 0)
    sweep = incidence_angle_sweep(mfs, body, boundary, 0.7, angles; options...)
    reversed = incidence_angle_sweep(mfs, body, boundary, 0.7, reverse(angles); options...)
    @test reversed.amplitudes == reverse(sweep.amplitudes)
    axial = incidence_angle_sweep(mfs, body, boundary, 0.7, [0.0, 0.0];
        options..., return_diagnostics = true)
    @test axial.sweep.amplitudes == fill(first(sweep.amplitudes), 2)
    @test all(d -> length(d.systems) == 1, axial.diagnostics)
    @test axial.diagnostics[1].systems[1].conditioning == :not_computed
    @test axial.diagnostics[1].systems[1].condition_number === nothing
    changed = incidence_angle_sweep(mfs, body, boundary, 0.9, [0.45]; options...)
    @test only(changed.amplitudes) ≈ scattering_amplitude(
        mfs(body, boundary, 0.9; incidence_angle = 0.45, options...))
    @test only(changed.amplitudes) != sweep.amplitudes[2]
    # Default offsets and tolerances, plus a non-axial mode-zero truncation.
    for material in (Impedance(1.3 - 0.6im), FluidFilled(1.2, 1.1))
        sample = incidence_angle_sweep(mfs, body, material, 0.7, [-0.4]; n = 12, m_max = 0)
        fresh = mfs(body, material, 0.7; incidence_angle = -0.4, n = 12, m_max = 0)
        @test only(sample.amplitudes) ≈ scattering_amplitude(fresh) rtol = 1e-10
    end
    for bad in (Float64[], [NaN], [Inf])
        @test_throws ArgumentError incidence_angle_sweep(mfs, body, boundary, 0.7, bad)
    end
    for bad in ((; oversampling = 0), (; condition_limit = -1), (; m_max = -1),
        (; offset_ext = 0.2), (; offset_int = 0.2))
        @test_throws ArgumentError incidence_angle_sweep(
            mfs, body, boundary, 0.7, [0.4]; bad...)
    end
    @test_throws ArgumentError incidence_angle_sweep(mfs,
        Cylinder(0.1, 0.5; radius_curvature = 2.0), boundary, 0.7, [0.4])
end
