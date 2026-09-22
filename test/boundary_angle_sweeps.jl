using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(2)

@time "Rigid and pressure-release incidence sweeps" @testset "Rigid and pressure-release incidence sweeps" begin
    surface = mesh(
        Sphere(1.0); method = :full, resolution = 0.4, mesh_order = 3, qorder = 4)
    angles = [pi / 6, pi / 2, pi / 6]
    gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200)
    for boundary in (Rigid(),),
        compression in ((method = :hmatrix, tol = 1e-7),),
        (formulation, k) in ((:cbie, 0.8),)
        options = (; formulation, compression, gmres_kwargs, incidence_azimuth = 0.4)
        reused = incidence_angle_sweep(surface, boundary, k, angles; options...)
        fresh = incidence_angle_sweep(angles) do incidence_angle
            solution = bem(surface, boundary, k; incidence_angle, options...)
            @test diagnostics(solution).converged
            @test diagnostics(solution).relative_residual < 2e-9
            solution
        end
        @test reused.amplitudes ≈ fresh.amplitudes rtol = 1e-10
        @test reused.target_strength ≈ fresh.target_strength atol = 1e-9
        @test reused.amplitudes[1] ≈ reused.amplitudes[end] rtol = 1e-12
        @test reused.angles == angles
        @test reused.labels == ["Scattered field"]
        reference = scattering_amplitude(modal(Sphere(1.0), boundary, k))
        @test maximum(abs.(reused.amplitudes .- reference)) / abs(reference) < 0.01
        @test maximum(abs.(reused.target_strength .- target_strength(reference))) < 0.1
    end
    for boundary in (Rigid(), PressureRelease())
        @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 1.0, Float64[])
        @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 1.0, [NaN])
        @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 1.0, angles;
            incidence_azimuth = Inf)
        @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 0.0, angles)
        @test_throws ArgumentError incidence_angle_sweep(surface, boundary, 1.0, angles;
            formulation = :unknown)
        @test_throws ArgumentError bem(surface, boundary, 1.0; incidence_angle = NaN)
    end
end

@time "Bent-surface incidence sweeps" @testset "Bent-surface incidence sweeps" begin
    body = Cylinder(0.5, 2.0; radius_curvature = 2.0, endcap_depth = 0.5)
    surface = mesh(body; method = :full, resolution = 0.32, mesh_order = 3, qorder = 4)
    collocation = mesh(body; method = :full, resolution = 0.3, mesh_order = 3)
    sources = mesh(body; method = :full, resolution = 0.32, mesh_order = 3, qorder = 1)
    options = (; incidence_azimuth = 0.4, compression = (method = :hmatrix, tol = 1e-7),
        gmres_kwargs = (reltol = 1e-9, restart = 150, maxiter = 1200))
    angles = [pi / 6, pi / 3]
    for boundary in (Rigid(),)
        reused = incidence_angle_sweep(surface, boundary, 1.0, angles; options...)
        fresh = incidence_angle_sweep(
            a -> bem(surface, boundary, 1.0;
                incidence_angle = a, options...), angles)
        independent = incidence_angle_sweep(angles) do incidence_angle
            mfs(collocation, boundary, 1.0; source_mesh = sources, offset = 0.3,
                incidence_angle, incidence_azimuth = 0.4, condition_limit = 0)
        end
        @test reused.amplitudes ≈ fresh.amplitudes rtol = 1e-10
        @test maximum(abs.(reused.target_strength - independent.target_strength)) < 0.1
        @test maximum(abs.((reused.amplitudes - independent.amplitudes) ./
                           independent.amplitudes)) < 0.01
        if boundary isa Rigid
            reference = [-0.115044319980511 + 0.02223869525583432im,
                -0.2798784736915524 - 0.03732557787576306im]
            @test maximum(abs.((reused.amplitudes - reference) ./ reference)) < 0.01
            @test maximum(abs.(reused.target_strength - target_strength.(reference))) < 0.1
        end
    end
end
