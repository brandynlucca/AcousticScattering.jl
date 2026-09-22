using AcousticScattering
using LinearAlgebra: BLAS, norm
using Test

const AS = AcousticScattering
BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

@time "Body coordinates and directions" @testset "Body coordinates and directions" begin
    @test AS._bem3d_incidence_direction(0.0, 0.0) ≈ [1, 0, 0]
    @test AS._bem3d_incidence_direction(pi/2, 0.0) ≈ [0, 1, 0] atol=1e-15
    @test AS._bem3d_incidence_direction(pi/2, pi/2) ≈ [0, 0, 1] atol=1e-15
    for body in (Spheroid(0.06, 0.02), Cylinder(0.02, 0.12; endcap_depth = 0.02))
        for method in (:axisymmetric, :full)
            surface = mesh(body; method, resolution = method === :full ? 0.05 : 40)
            points = if method === :full
                AS.coordinates(surface)
            else
                panels = AS.panels(surface.data)
                revolved = AS.revolve_panels(panels, [zeros(length(panels))]; n_phi = 16)
                collect(zip(vec(revolved.x), vec(revolved.y), vec(revolved.z)))
            end
            extents = [maximum(p[j] for p in points)-minimum(p[j] for p in points)
                       for j in 1:3]
            @test extents[1] > 2extents[2]
            @test extents[1] > 2extents[3]
        end
    end
end

@time "Rotated geometry preserves complex amplitudes" @testset "Rotated geometry preserves complex amplitudes" begin
    original = mesh(; semiaxes = (0.06, 0.018, 0.025), resolution = 0.8, qorder = 5, tip_ratio = 0.4)
    angle = 0.37
    rotation = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1]
    nodes = rotation*original.body.nodes
    rotated = mesh(nodes, hcat(original.body.connectivity...); qorder = 5)
    @test rotated.body.nodes ≈ nodes
    @test maximum(norm.(AS.normals(rotated) .-
                        [rotation*n for n in AS.normals(original)])) < 1e-10
    beta, alpha = pi/3, 0.4
    incident = [cos(beta), sin(beta)*cos(alpha), sin(beta)*sin(alpha)]
    rotated_incident = rotation*incident
    for material in (FluidFilled(1.04, 1.04),)
        a = bem(original, material, 1.0; incidence_angle = beta, incidence_azimuth = alpha)
        b = bem(rotated, material, 1.0; incidence_angle = acos(rotated_incident[1]),
            incidence_azimuth = atan(rotated_incident[3], rotated_incident[2]))
        for direction in (-incident, incident, [0.0, 0.0, 1.0])
            reference = scattering_amplitude(a; direction)
            actual = scattering_amplitude(b; direction = rotation*direction)
            @test actual ≈ reference rtol=1e-5
        end
    end
end
