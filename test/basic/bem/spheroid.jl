using AcousticScattering
using Test

@testset "Spheroid" begin
    k = 0.5
    for (shape, body) in ((:prolate, Spheroid(1.2, 1.0)),
        (:oblate, Spheroid(1.0, 1.2)))
        @testset "$shape" begin
            @test body.kind == shape
            for (name, boundary) in (("rigid", Rigid()),
                ("pressure-release", PressureRelease()),
                ("fluid-filled", FluidFilled(1.2, 1.1)))
                @testset "$name" begin
                    oblique = (shape == :prolate && boundary isa PressureRelease) ||
                              (shape == :oblate && boundary isa FluidFilled)
                    incidence_angle = oblique ? pi / 3 : 0.0
                    solution = bem(body, boundary, k; incidence_angle, n = 8, m_max = 1)
                    @test solution isa BEMSolution
                    @test solution.method == :axisymmetric
                    @test scattering_amplitude(solution) isa ComplexF64
                    @test isfinite(target_strength(solution))
                end
            end
        end
    end

    full = bem(Spheroid(1.2, 1.0), Rigid(), k;
        method = :full, meshsize = 1.0, mesh_order = 1, qorder = 2,
        compression = (method = :none,))
    @test full isa BEMSolution
    @test full.method == :full
    @test isfinite(scattering_amplitude(full))
end

@testset "Curved prolate BEM" begin
    body, k = Spheroid(0.02, 0.015), 100.0
    for bc in (Rigid(), PressureRelease(), FluidFilled(1.2, 0.85), FluidFilled(0.1, 0.6))
        for beta in (0.0, 0.6, pi/2)
            curved = bem(body, bc, k; incidence_angle = beta, m_max = 6,
                n = 16, boundary_order = 5, adaptive = false, rtol = 1e-8, mode_threads = 2)
            reference = fem(body, bc, k; order = 6, incidence_angle = beta, m_max = 6,
                n_r = 8, n_theta = 32, l_max = 14, retain_fields = false)
            for (theta, phi) in ((pi-beta, pi), (0.4, 0.7), (-0.4, 0.7))
                @test isapprox(scattering_amplitude(curved; angle = theta, azimuth = phi),
                    scattering_amplitude(reference; angle = theta, azimuth = phi); rtol = 3e-6, atol = 1e-11)
            end
            @test curved.method===:axisymmetric
            @test diagnostics(curved).representation_residual<1e-7
            @test length(diagnostics(curved).systems)==(beta==pi/2 ? 7 : 14)
        end
    end
    for beta in (0.0, 0.6, pi/2)
        transparent = bem(body, FluidFilled(1.0, 1.0), k;
            incidence_angle = beta, m_max = 3, n = 12, adaptive = false)
        @test iszero(scattering_amplitude(transparent))
    end
    for bc in (Rigid(), PressureRelease(), FluidFilled(1.2, 0.85))
        curved = bem(Spheroid(1.00000001, 1.0), bc, 2.0; incidence_angle = 0.0,
            m_max = 0, n = 16, boundary_order = 5, adaptive = false, rtol = 1e-9)
        exact = modal(Sphere(1.0), bc, 2.0)
        for point in ((1.6, 0.4, 0.2), (0.3, 1.4, 0.2))
            @test isapprox(pressure(curved, point; field = :scattered),
                pressure(exact, point; field = :scattered); rtol = 1e-6)
        end
        if bc isa FluidFilled
            @test isapprox(pressure(curved, (0.2, 0.1, 0.1); field = :interior),
                pressure(exact, (0.2, 0.1, 0.1); field = :interior); rtol = 1e-6)
        end
    end
    # This axial high-frequency case exposed premature stopping at order five.
    # Independent spherical-DtN FEM gives -63.623002330122134 dB.
    axial = bem(Spheroid(0.07, 0.01), Rigid(), 2pi*400000/1477.3;
        incidence_angle = 0.0, m_max = 0, target_tol = 0.001, mode_threads = 2)
    @test diagnostics(axial).refinement.converged
    @test abs(target_strength(axial)-(-63.623002330122134))<0.002
    @test isfinite(pressure(axial, (0.1, 0.0, 0.0)))
    @test_throws ArgumentError scattering_amplitude(axial; angle = NaN)
    @test_throws ArgumentError bem(Spheroid(0.01, 0.02), Rigid(), 100.0; method = :curved)
    @test_throws ArgumentError bem(body, Rigid(), k; incidence_azimuth = 0.1)
    for controls in ((; boundary_order = 0), (; chief_points = 0), (; rtol = 0.0), (;
        n = 1),
        (; m_max = -1), (; mode_threads = 0), (; target_tol = 0.0))
        @test_throws ArgumentError bem(body, Rigid(), k; controls...)
    end
end
