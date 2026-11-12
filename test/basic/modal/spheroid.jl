using AcousticScattering
using Test

@testset "Spheroid" begin
    radius = 0.02
    k = 2pi * 12000.0 / 1477.4
    sphere = Sphere(radius)
    bodies = (
        (:prolate, Spheroid(radius + 0.0002, radius - 0.0002)),
        (:oblate, Spheroid(radius - 0.0002, radius + 0.0002))
    )
    boundaries = (
        ("rigid", Rigid()),
        ("pressure-release", PressureRelease()),
        ("fluid-filled, full coupling", FluidFilled(1.05, 1.02; coupling = :full)),
        ("fluid-filled, diagonal coupling", FluidFilled(1.05, 1.02; coupling = :diagonal))
    )

    for (shape, body) in bodies
        @testset "$shape" begin
            @test body.kind == shape
            for (name, boundary) in boundaries
                @testset "$name" begin
                    solution = if boundary isa FluidFilled && boundary.coupling === :full
                        modal(body, boundary, k;
                            incidence_angle = pi / 3, m_max = 2, n_max = 6, n_quad = 12)
                    else
                        modal(body, boundary, k;
                            incidence_angle = pi / 3, m_max = 2, n_max = 6)
                    end
                    sphere_reference = modal(sphere, boundary, k; m_max = 6)
                    @test solution isa ModalSolution
                    @test scattering_amplitude(solution) isa ComplexF64
                    @test target_strength(solution) isa Float64
                    @test isfinite(target_strength(solution))
                    @test target_strength(solution) ≈
                          target_strength(sphere_reference) atol = 1.0
                end
            end
        end
    end

    @testset "Prolate full-coupling reference" begin
        q, xi0 = 3.0, 1.05
        body = Spheroid(q * xi0, sqrt((q * xi0)^2 - q^2))
        boundary = FluidFilled(1050 / 1026, 1.02; coupling = :full)
        solution = modal(body, boundary, 1.0;
            incidence_angle = 0.3, m_max = 2, n_max = 4)
        scaled = scattering_amplitude(solution) / (-2im)
        @test real(scaled) ≈ 9.10954776790933e-6 atol = 1e-14
        @test imag(scaled) ≈ 0.00326545204233986 atol = 1e-12
    end
end

@testset "Reusable spheroid modal solutions" begin
    AS = AcousticScattering
    angles = [0.0, 0.7, pi / 2, pi]
    phis = [0.0, 0.8, pi]
    k = 90.0
    options = (; m_max = 3, n_max = 8)
    for body in (Spheroid(0.03, 0.01), Spheroid(0.01, 0.03)),
        boundary in (Rigid(), PressureRelease(), FluidFilled(1.05, 1.02),
            FluidFilled(0.0012, 0.23), FluidFilled(1.05, 1.02; coupling = :diagonal))

        precision = boundary isa FluidFilled ? :quad : :double
        kernel = boundary isa FluidFilled && boundary.coupling === :full ?
                 AS._form_function_full_fluid : AS._form_function_diagonal
        reference(theta, phi; incidence = 0.6) = kernel(boundary, k, body;
            incidence_angle = incidence, incidence_azimuth = 0.3,
            scatter_angle = theta, scatter_azimuth = phi, precision, options...)
        solution = modal(body, boundary, k; incidence_angle = 0.6,
            incidence_azimuth = 0.3, scatter_angle = 1.2, scatter_azimuth = 0.4,
            precision, options...)
        @test scattering_amplitude(solution)≈reference(1.2, 0.4) rtol=2e-10 atol=1e-13
        @test scattering_amplitude(solution; angle = 0.7, azimuth = 0.8)≈
        reference(0.7, 0.8) rtol=2e-10 atol=1e-13
        sweep = bistatic_sweep(solution, angles; azimuth = 0.8)
        expected = [reference(a, 0.8) for a in angles]
        @test sweep.amplitudes≈expected rtol=2e-10 atol=1e-13
        @test sweep.target_strength ≈ target_strength.(expected) atol = 1e-7
        map = bistatic_map(solution, angles, phis)
        @test map.target_strength ≈
              [target_strength(reference(a, p)) for a in angles, p in phis] atol = 1e-7
        incidence = incidence_angle_sweep(body, boundary, k, angles;
            method = :modal, incidence_azimuth = 0.3, precision, options...)
        @test incidence.amplitudes≈
        [reference(pi-a, pi+0.3; incidence = a) for a in angles] rtol=2e-10 atol=1e-13
    end
    solution = modal(Spheroid(0.03, 0.01), Rigid(), k; options...)
    @test_throws ArgumentError bistatic_sweep(solution, Float64[])
    @test_throws ArgumentError scattering_amplitude(solution; angle = NaN)
    @test_throws ArgumentError scattering_amplitude(solution; direction = [1, 0, 0])
    @test_throws ArgumentError modal(solution.body, Rigid(), 0.0)
    @test_throws ArgumentError modal(solution.body, Rigid(), k; n_max = -1)
    @test_throws ArgumentError modal(solution.body, Rigid(), k; overlap = :invalid)
    gas = FluidFilled(0.0012, 0.23)
    quad = modal(solution.body, gas, k; precision = :quad, options...)
    ambient = precision(BigFloat)
    setprecision(BigFloat, 64) do
        low = modal(solution.body, gas, k; precision = :quad, options...)
        @test scattering_amplitude(low) ≈ scattering_amplitude(quad) rtol = 1e-13
        @test scattering_amplitude(low; angle = 0.7) ≈
              scattering_amplitude(quad; angle = 0.7) rtol = 1e-13
        @test precision(BigFloat) == 64
    end
    @test precision(BigFloat) == ambient
    if isdefined(AS.SpheroidalWaves, :dmn)
        expected = modal(solution.body, gas, k; overlap = :coefficients, options...)
        setprecision(BigFloat, 32) do
            low = modal(solution.body, gas, k; overlap = :coefficients, options...)
            @test scattering_amplitude(low) ≈ scattering_amplitude(expected) rtol = 1e-13
            @test precision(BigFloat) == 32
        end
    end
    @test AS.form_function(Rigid(), k, solution.body; options...) ≈
          scattering_amplitude(modal(
        solution.body, Rigid(), k; incidence_angle = 0.0, options...))
end

@testset "Spheroid overlap representations" begin
    AS = AcousticScattering
    if isdefined(AS.SpheroidalWaves, :dmn)
        for kind in (:prolate, :oblate)
            overlap = AS._spheroid_angular_overlaps(Float64, 2, 2:8, 3.0, 3.0, kind;
                n_quad = 64, precision = :quad, overlap = :coefficients)
            @test overlap ≈ Matrix{Float64}(AS.I, 7, 7) atol = 1e-12
            quadrature = AS._spheroid_angular_overlaps(Float64, 2, 2:8, 3.0, 11.0, kind;
                n_quad = 65, precision = :quad, overlap = :quadrature)
            coefficients = AS._spheroid_angular_overlaps(Float64, 2, 2:8, 3.0, 11.0, kind;
                n_quad = 64, precision = :quad, overlap = :coefficients)
            @test quadrature≈coefficients rtol=1e-11 atol=1e-13
        end
    end
end

@testset "Spheroid broadside parity and reflected directions" begin
    AS = AcousticScattering
    boundary = FluidFilled(0.0012, 0.23; coupling = :full)
    k = 90.0
    options = (; m_max = 2, n_max = 6, n_quad = 48, precision = :quad)
    angles = [0.0, 0.7, pi / 2, pi - 0.7, pi]
    for body in (Spheroid(0.03, 0.01), Spheroid(0.01, 0.03))
        reference(theta; incidence = pi / 2) = AS._form_function_full_fluid(
            boundary, k, body; incidence_angle = incidence, incidence_azimuth = 0.3,
            scatter_angle = theta, scatter_azimuth = 0.4, options...)
        solution = modal(body, boundary, k; incidence_angle = pi / 2,
            incidence_azimuth = 0.3, scatter_angle = 0.7, scatter_azimuth = 0.4,
            options...)
        sweep = bistatic_sweep(solution, angles; azimuth = 0.4)
        @test sweep.amplitudes≈[reference(theta) for theta in angles] rtol=2e-10 atol=1e-13
        @test scattering_amplitude(solution)≈reference(0.7) rtol=2e-10 atol=1e-13
        @test sweep.amplitudes[2]≈sweep.amplitudes[4] rtol=1e-12 atol=1e-14
        for coefficients in solution.data.coefficients
            @test all(iszero, coefficients[2:2:end])
        end

        # A nearby incident direction excites the odd sector, even when the
        # angular offset is smaller than the usual default isapprox tolerance.
        incidence = pi / 2 + 1e-8
        nearby = modal(body, boundary, k; incidence_angle = incidence,
            incidence_azimuth = 0.3, scatter_angle = 0.7, scatter_azimuth = 0.4,
            options...)
        @test any(coefficients -> any(value -> !iszero(value), coefficients[2:2:end]),
            nearby.data.coefficients)
        reflected = bistatic_sweep(nearby, [0.7, pi - 0.7]; azimuth = 0.4)
        @test reflected.amplitudes≈
        [reference(theta; incidence) for theta in (0.7, pi-0.7)] rtol=2e-10 atol=1e-13
        @test !isapprox(reflected.amplitudes[1], reflected.amplitudes[2];
            rtol = 1e-12, atol = 1e-15)

        # A sweep containing both broadside and oblique illumination must
        # retain both parity sectors for its shared boundary solve.
        incident_angles = [pi / 2, 0.6, pi - 0.6]
        incidence_sweep = incidence_angle_sweep(body, boundary, k, incident_angles;
            method = :modal, incidence_azimuth = 0.3, options...)
        individual = [scattering_amplitude(modal(body, boundary, k;
                          incidence_angle = theta, incidence_azimuth = 0.3,
                          options...)) for theta in incident_angles]
        @test incidence_sweep.amplitudes≈individual rtol=2e-10 atol=1e-13
    end
end
