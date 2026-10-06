using AcousticScattering
using Test
using LinearAlgebra
using StaticArrays

@testset "Incident beams across numerical solvers" begin
    body, k = Sphere(0.2), 3.2
    d = normalize(SVector(1.0, 2.0, -1.0))
    scale = 0.4 + 0.7im
    point = SVector(0.3, -0.1, 0.1)
    surface = mesh(body; method = :full, resolution = 0.1, mesh_order = 2, qorder = 4)
    sources = mesh(body; method = :full, resolution = 0.1, mesh_order = 2, qorder = 1)
    for field in (PlaneWave(; direction = d, amplitude = scale),
        SphericalWave(1.4; direction = d, amplitude = scale),
        BesselBeam(0.6; direction = d, amplitude = scale))
        reference = modal(body, PressureRelease(), k; incident = field, m_max = 18)
        solution = mfs(surface, PressureRelease(), k; incident = field,
            source_mesh = sources, offset = 0.07, condition_limit = 0)
        @test scattering_amplitude(solution; direction = -d) ≈
              scattering_amplitude(reference) rtol = 0.003
        @test pressure(solution, point; field = :scattered) ≈
              pressure(reference, point; field = :scattered) rtol = 0.003
        @test pressure(solution, point; field = :incident) ≈
              incident_pressure(field, k, point)
    end

    # Reuse the same mesh to isolate direction and phase from discretization.
    beta, alpha = 0.7, 0.2
    opts = (;
        incidence_angle = beta, incidence_azimuth = alpha, compression = (method = :none,))
    baseline = bem(surface, PressureRelease(), k; opts...)
    scaled = bem(
        surface, PressureRelease(), k; opts..., incident = PlaneWave(; amplitude = scale))
    @test scattering_amplitude(scaled) ≈ scale * scattering_amplitude(baseline) rtol = 1e-7
    field = BesselBeam(0.6; direction = d, amplitude = scale)
    solution = bem(
        surface, PressureRelease(), k; incident = field, compression = (method = :none,))
    reference = modal(body, PressureRelease(), k; incident = field, m_max = 18)
    @test scattering_amplitude(solution; direction = -d) ≈ scattering_amplitude(reference) rtol = 0.01
    @test pressure(solution, point; field = :incident) ≈ incident_pressure(field, k, point)

    # Volume FEM receives Cartesian fields without applying its legacy angle-frame transform.
    volume = fem(body, PressureRelease(), k; method = :volume, incident = field,
        closure = :dtn, points_per_wavelength = 5, solver = :direct)
    @test scattering_amplitude(volume; angle = acos(-d[3]), azimuth = atan(-d[2], -d[1])) ≈
          scattering_amplitude(reference) rtol = 0.03
    @test pressure(volume, point; field = :incident) ≈ incident_pressure(field, k, point)
end
