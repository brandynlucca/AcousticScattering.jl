using AcousticScattering
using Test
using LinearAlgebra
using StaticArrays

@testset "Reusable incident fields" begin
    k = 3.2
    direction = normalize(SVector(1.0, 2.0, -1.0))
    point = SVector(0.17, -0.11, 0.09)
    amplitude, phase = 0.4 + 0.7im, 0.3
    scale = amplitude * cis(phase)
    fields = (PlaneWave(; direction, amplitude, phase),
        SphericalWave(1.4; direction, amplitude, phase),
        BesselBeam(0.6; direction, amplitude, phase))

    @testset "Analytic fields and spherical reconstruction" begin
        for field in fields
            h = 1e-6
            gradient = SVector{3}(ntuple(3) do i
                step = SVector{3}(ntuple(j -> i == j ? h : 0.0, 3))
                (incident_pressure(field, k, point + step) -
                 incident_pressure(field, k, point - step)) / (2h)
            end)
            @test incident_gradient(field, k, point) ≈ gradient rtol = 2e-9
            r = norm(point)
            mu = dot(direction, point) / r
            expansion = sum(0:22) do l
                (2l + 1) * incident_coefficient(field, l, k) *
                AcousticScattering.js(l, k * r) * AcousticScattering.legendre_p(l, mu)
            end
            @test expansion ≈ incident_pressure(field, k, point) rtol = 2e-13
            @test_throws ArgumentError incident_pressure(field, 0.0, point)
            @test_throws ArgumentError incident_gradient(field, k, (NaN, 0, 0))
            @test_throws ArgumentError incident_coefficient(field, -1, k)
        end
        @test incident_pressure(fields[1], k, point) ≈
              scale * cis(k * dot(direction, point))
        r = norm(point + 1.4direction)
        @test incident_pressure(fields[2], k, point) ≈ scale * cis(k * r) / r
        # Independent angular spectrum: a uniform cone of plane waves.
        u = normalize(cross(direction, SVector(0.0, 0.0, 1.0)))
        v = cross(direction, u)
        ring = sum(0:127) do j
            azimuth = 2pi * j / 128
            d = cos(0.6) * direction + sin(0.6) * (cos(azimuth) * u + sin(azimuth) * v)
            scale * cis(k * dot(d, point)) / 128
        end
        @test incident_pressure(fields[3], k, point) ≈ ring rtol = 3e-15
        axial = BesselBeam(0.0; direction, amplitude, phase)
        @test incident_pressure(axial, k, point) ≈ incident_pressure(fields[1], k, point)
        @test incident_gradient(axial, k, point) ≈ incident_gradient(fields[1], k, point)
        @test incident_gradient(BesselBeam(0.6), k, (0, 0, 0)) ≈
              SVector(im * k * cos(0.6), 0im, 0im)
        for make in (PlaneWave, (; kwargs...) -> SphericalWave(1; kwargs...),
            (; kwargs...) -> BesselBeam(0.4; kwargs...))
            @test_throws ArgumentError make(; direction = (0, 0, 0))
            @test_throws ArgumentError make(; direction = (1, 2))
            @test_throws ArgumentError make(; amplitude = Inf)
            @test_throws ArgumentError make(; phase = NaN)
        end
        @test_throws ArgumentError incident_pressure(SphericalWave(1), k, (-1, 0, 0))
        @test_throws ArgumentError incident_gradient(SphericalWave(1), k, (-1, 0, 0))
        callbacks = IncidentField(x -> scale * cis(k * dot(direction, x)),
            x -> im * k * direction * scale * cis(k * dot(direction, x)))
        @test incident_pressure(callbacks, k, point) ≈
              incident_pressure(fields[1], k, point)
        @test incident_gradient(callbacks, k, point) ≈
              incident_gradient(fields[1], k, point)
        @test_throws ArgumentError incident_coefficient(callbacks, 1, k)
    end

    @testset "Modal beam near fields and interfaces" begin
        body = Sphere(0.2)
        for field in fields
            solution = modal(body, PressureRelease(), k; incident = field, m_max = 24)
            for normal in (direction, normalize(point), SVector(0.0, 1.0, 0.0))
                @test abs(pressure(solution, body.radius * normal)) < 2e-13
            end
            @test pressure(solution, point; field = :incident) ≈
                  incident_pressure(field, k, point)
            # Rotation changes the Cartesian samples, not the axis-relative modal amplitudes.
            unrotated = field isa PlaneWave ? PlaneWave(; amplitude, phase) :
                        field isa SphericalWave ? SphericalWave(1.4; amplitude, phase) :
                        BesselBeam(0.6; amplitude, phase)
            reference = modal(body, PressureRelease(), k; incident = unrotated, m_max = 24)
            @test scattering_amplitude(solution) == scattering_amplitude(reference)
            @test pressure(solution, 0.3direction) ≈ pressure(reference, (0.3, 0, 0)) rtol = 1e-13
            matched = modal(body, FluidFilled(1.0, 1.0), k; incident = field, m_max = 24)
            @test pressure(matched, (0, 0, 0)) ≈ incident_pressure(field, k, (0, 0, 0)) rtol = 1e-13
            @test pressure(matched, 0.1direction) ≈
                  incident_pressure(field, k, 0.1direction) rtol = 1e-13
            fluid = modal(body, FluidFilled(1.3, 0.8), k; incident = field, m_max = 24)
            p = body.radius * direction
            @test pressure(fluid, p; field = :interior) ≈ pressure(fluid, p) rtol = 1e-12
            h = 1e-6
            inner = (3pressure(fluid, p; field = :interior) -
                     4pressure(fluid, p - h * direction) +
                     pressure(fluid, p - 2h * direction)) / (2h)
            outer = (-3pressure(fluid, p) + 4pressure(fluid, p + h * direction) -
                     pressure(fluid, p + 2h * direction)) / (2h)
            @test inner / 1.3 ≈ outer rtol = 2e-8
        end
        @test_throws ArgumentError modal(body, Rigid(), k; incident = SphericalWave(0.2))
        @test_throws ArgumentError modal(body, Rigid(), k; incident = SphericalWave(0.1))
        @test_throws ArgumentError modal(body, Rigid(), k;
            incident = IncidentField(x -> 1, x -> (0, 0, 0)))
    end

    @testset "Numerical field adapters and legacy angle frames" begin
        beta, alpha = 0.7, 0.2
        for d in (AcousticScattering._bem3d_incidence_direction(beta, alpha),
            AcousticScattering._volume_direction(beta, alpha))
            @test AcousticScattering._resolve_incident(k, beta, alpha;
                incident = PlaneWave(), plane_direction = d) === nothing
            resolved = AcousticScattering._resolve_incident(k, beta, alpha;
                incident = PlaneWave(; amplitude, phase), plane_direction = d)
            @test incident_pressure(resolved, k, point) ≈ scale * cis(k * dot(d, point))
            @test incident_gradient(resolved, k, point) ≈
                  im * k * d * scale * cis(k * dot(d, point))
        end
        for field in fields
            resolved = AcousticScattering._resolve_incident(k, beta, alpha; incident = field)
            @test incident_pressure(resolved, k, point) ==
                  incident_pressure(field, k, point)
            @test incident_gradient(resolved, k, point) ==
                  incident_gradient(field, k, point)
        end
    end
end
