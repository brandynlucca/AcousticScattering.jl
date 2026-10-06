using AcousticScattering
using Test
using LinearAlgebra
using StaticArrays

@testset "Spheroidal incident projection" begin
    beta, alpha = 0.7, 0.4
    d = SVector(cos(beta), sin(beta) * cos(alpha), sin(beta) * sin(alpha))
    scale = 0.4 + 0.7im
    observation = (; scatter_angle = 1.4, scatter_azimuth = 2.1)
    orders = (; m_max = 5, n_max = 7)
    for body in (Spheroid(1.5, 1.0), Spheroid(0.7, 1.0)),
        boundary in (Rigid(), PressureRelease(), FluidFilled(1.3, 0.8),
            FluidFilled(1.3, 0.8; coupling = :diagonal))

        k = 1.2
        reference = modal(
            body, boundary, k; incidence_angle = beta, incidence_azimuth = alpha,
            observation..., orders...)
        field = PlaneWave(; direction = d, amplitude = scale)
        projected = modal(body, boundary, k; incident = field, observation..., orders...)
        @test scattering_amplitude(projected) ≈ scale * scattering_amplitude(reference) rtol = 2e-10
        forced_projection = modal(body, boundary, k; incident = PlaneWave(),
            incidence_angle = beta, incidence_azimuth = alpha, incident_n_eta = 40,
            observation..., orders...)
        @test scattering_amplitude(forced_projection) ≈ scattering_amplitude(reference) rtol = 2e-10
        # Zero-angle beam has an independent pointwise path but identical forcing.
        beam = modal(
            body, boundary, k; incident = BesselBeam(0; direction = d, amplitude = scale),
            observation..., orders...)
        @test scattering_amplitude(beam) ≈ scattering_amplitude(projected) rtol = 2e-12
        # Two nonsymmetric wave directions require both cosine and sine components.
        p(x) = scale * cis(k * dot(d, x)) + 0.3cis(k * x[3])
        g(x) = im * k *
               (scale * cis(k * dot(d, x)) * d + 0.3cis(k * x[3]) * SVector(0, 0, 1))
        combined = modal(
            body, boundary, k; incident = IncidentField(p, g), observation..., orders...)
        other = modal(
            body, boundary, k; incidence_angle = pi / 2, incidence_azimuth = pi / 2,
            observation..., orders...)
        @test scattering_amplitude(combined) ≈
              scale * scattering_amplitude(reference) + 0.3scattering_amplitude(other) rtol = 2e-10
    end
    body = Spheroid(1.4, 1.0)
    elastic = SolidElastic(2.7, 4.2, 2.1)
    reference = tmatrix(
        body, elastic, 0.8; incidence_angle = beta, incidence_azimuth = alpha,
        observation..., orders...)
    projected = tmatrix(
        body, elastic, 0.8; incident = PlaneWave(; direction = d, amplitude = scale),
        observation..., orders...)
    @test scattering_amplitude(projected) ≈ scale * scattering_amplitude(reference) rtol = 2e-9
    forced_projection = tmatrix(body, elastic, 0.8; incidence_angle = beta,
        incidence_azimuth = alpha, incident_n_eta = 40, observation..., orders...)
    @test scattering_amplitude(forced_projection) ≈ scattering_amplitude(reference) rtol = 2e-9
    for body in (Spheroid(1.5, 1.0), Spheroid(0.7, 1.0)),
        field in (BesselBeam(0.6; direction = d), SphericalWave(3.0; direction = d))

        kwargs = (; incident = field, observation..., m_max = 6, n_max = 10)
        baseline = modal(body, Rigid(), 1.2; kwargs...)
        quadrature = modal(
            body, Rigid(), 1.2; kwargs..., incident_n_eta = 64, incident_n_phi = 64)
        refined = modal(
            body, Rigid(), 1.2; incident = field, observation..., m_max = 8, n_max = 12)
        @test scattering_amplitude(baseline) ≈ scattering_amplitude(quadrature) rtol = 1e-10
        @test scattering_amplitude(baseline) ≈ scattering_amplitude(refined) rtol = 2e-7
    end
    # Direct surface MFS enforces the same beam boundary data without spheroidal projection.
    small = Spheroid(0.2, 0.12)
    surface = mesh(small; method = :full, resolution = 0.08, mesh_order = 2, qorder = 4)
    sources = mesh(small; method = :full, resolution = 0.08, mesh_order = 2, qorder = 1)
    field = BesselBeam(0.6; direction = d)
    observation_direction = SVector(cos(1.4), sin(1.4) * cos(2.1), sin(1.4) * sin(2.1))
    for boundary in (Rigid(), FluidFilled(1.3, 0.8))
        projected = modal(small, boundary, 5.0; incident = field, observation..., orders...)
        if boundary isa FluidFilled
            surface = mesh(
                small; method = :full, resolution = 0.04, mesh_order = 2, qorder = 4)
            sources = mesh(
                small; method = :full, resolution = 0.04, mesh_order = 2, qorder = 1)
        end
        offsets = boundary isa FluidFilled ? (; offset_ext = 0.04, offset_int = 0.04) :
                  (; offset = 0.04)
        numerical = mfs(surface, boundary, 5.0; incident = field, source_mesh = sources,
            offsets..., condition_limit = 0)
        @test scattering_amplitude(projected) ≈
              scattering_amplitude(numerical; direction = observation_direction) rtol = 0.01
    end
    @test_throws ArgumentError modal(body, Rigid(), 1.0; incident = SphericalWave(0.5))
    @test_throws ArgumentError modal(body, Rigid(), 1.0; incident = BesselBeam(0.4),
        m_max = 4, n_max = 5, incident_n_phi = 8)
    @test_throws ArgumentError modal(body, Rigid(), 1.0; incident = BesselBeam(0.4),
        m_max = 4, n_max = 5, incident_n_eta = 5)
    # On-axis scattering of a cone equals any one constituent's axial far field.
    shell = Shelled(ElasticLayer(2.7, 4.2, 2.1), FluidInterior(1.1, 0.9), 0.85)
    for boundary in (elastic, shell)
        beam = tmatrix(body, boundary, 0.8; incident = BesselBeam(0.6),
            incidence_angle = 0, scatter_angle = pi, m_max = 0, n_max = 8, check = false)
        plane = tmatrix(body, boundary, 0.8; incidence_angle = 0.6, scatter_angle = pi,
            m_max = 0, n_max = 8, check = false)
        @test scattering_amplitude(beam) ≈ scattering_amplitude(plane) rtol = 2e-9
    end
end
