using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(4)

@testset "Near-source normal derivatives" begin
    for distance in (1e-3, 1e-5, 1e-7, 1e-9), m in (0, 3)

        actual = AcousticScattering._azimuthal_dGdn_field(
            0.5, 0.5, 0.5, 0.6, 0.8, 0.5-distance, 0.5-distance; m)
        expected = AcousticScattering._azimuthal_dGdn(
            0.5, 0.5-distance, 0.5-distance, 0.5, 0.5, 0.6, 0.8; m)
        @test isapprox(actual, expected; rtol = 1e-6, atol = 1e-12)
    end
end

function compare_rim_pressure(actual, expected)
    for (got, wanted) in zip(actual, expected)
        @test isapprox(got, wanted; rtol = 1e-3, atol = 1e-12)
        @test abs(20log10(abs(got/wanted))) < 0.01
    end
end

@testset "Edge quadrature on smooth surfaces" begin
    body, k = Sphere(0.5), 0.5
    surface = mesh(body; method = :full, resolution = 0.2, mesh_order = 3, qorder = 5)
    for boundary in (Rigid(), PressureRelease())
        options = (; formulation = :cbie, compression = (method = :none,),
            correction = (method = :edge,),
            gmres_kwargs = (reltol = 1e-10, restart = 150, maxiter = 600))
        solution = bem(body, boundary, k; method = :full, meshsize = 0.2, mesh_order = 3,
            qorder = 5, incidence_angle = 0.0, options...)
        reference = modal(body, boundary, k)
        points = [(0.5, 0.0, 0.0), (0.500001, 0.0, 0.0), (0.8, 0.1, 0.2)]
        compare_rim_pressure(pressure(solution, points; field = :scattered),
            pressure(reference, points; field = :scattered))
        amplitude = scattering_amplitude(reference)
        compare_rim_pressure([scattering_amplitude(solution)], [amplitude])
        sweep = incidence_angle_sweep(surface, boundary, k, [0.0, pi/3]; options...)
        compare_rim_pressure(sweep.amplitudes, fill(amplitude, length(sweep.amplitudes)))
        @test_throws ArgumentError bem(surface, boundary, k; formulation = :cbie,
            compression = (method = :none,), correction = (method = :edge, rtol = 0.0))
    end
end

@testset "Fluid edge quadrature on a sphere" begin
    body, boundary, k = Sphere(0.5), FluidFilled(1.2, 1.1), 0.5
    solution = bem(body, boundary, k; method = :full, meshsize = 0.2, mesh_order = 3,
        qorder = 7, incidence_angle = 0.0, formulation = :cbie,
        correction = (method = :edge,), condition_limit = 0)
    reference = modal(body, boundary, k)
    for (field, points) in (
        (:scattered, [(0.5+delta, 0.0, 0.0) for delta in (0.0, 1e-8, 0.001, 0.05, 0.5)]),
        (:interior, [(0.0, 0.0, 0.0), (0.49, 0.0, 0.0), (0.5, 0.0, 0.0)]))
        compare_rim_pressure(pressure(solution, points; field), pressure(reference, points; field))
    end
    compare_rim_pressure([scattering_amplitude(solution)], [scattering_amplitude(reference)])
end

@testset "Full surface layer evaluation at flat-cylinder rims" begin
    body, k = Cylinder(0.5, 1.0), 0.5
    source = (0.1, 0.05, -0.02)
    points = [(side*(0.5+delta), 0.3+0.6delta, 0.4+0.8delta)
              for side in (-1, 1) for delta in (0.0, 1e-8, 1e-4, 0.001, 0.01, 0.05)]
    expected = [AcousticScattering._green3d(k, point, source) for point in points]
    for resolution in (0.2, 0.15)
        quadrature = mesh(body; method = :full, resolution, mesh_order = 3, qorder = 5).data
        p = [AcousticScattering._green3d(k, Tuple(q.coords), source) for q in quadrature]
        dp = [AcousticScattering._dgreen3d_dn(k, Tuple(q.coords), Tuple(q.normal), source)
              for q in quadrature]
        actual = AcousticScattering._pressure_layer_values(
            quadrature, k, points, false, p, dp)
        compare_rim_pressure(actual, expected)
    end
end

@testset "Rigid flat-cylinder rim continuity and refinement" begin
    body, boundary, k = Cylinder(0.5, 1.0), Rigid(), 0.5
    points = [(side*(0.5+delta), 0.3+0.6delta, 0.4+0.8delta)
              for side in (-1, 1) for delta in (0.0, 1e-8, 1e-4, 0.001, 0.01, 0.05)]
    values = Vector{ComplexF64}[]
    amplitudes = ComplexF64[]
    for n in (256, 512)
        solution = bem(body, boundary, k; n, incidence_angle = pi/3, m_max = 6)
        push!(values, pressure(solution, points; field = :scattered))
        push!(amplitudes, scattering_amplitude(solution))
        compare_rim_pressure(last(values)[[1, 7]], last(values)[[2, 8]])
    end
    compare_rim_pressure(last(values), first(values))
    full_values = Vector{ComplexF64}[]
    for meshsize in (0.12, 0.1)
        solution = bem(
            body, boundary, k; method = :full, meshsize, mesh_order = 3, qorder = 5,
            incidence_angle = pi/3, formulation = :cbie, compression = (method = :none,),
            correction = (method = :edge,), gmres_kwargs = (
                reltol = 1e-11, restart = 150, maxiter = 1200))
        @test diagnostics(solution).converged
        @test diagnostics(solution).relative_residual < 1e-10
        actual = pressure(solution, points; field = :scattered)
        push!(full_values, actual)
        compare_rim_pressure(actual[[1, 7]], actual[[2, 8]])
        amplitude = scattering_amplitude(solution)
        compare_rim_pressure([amplitude], [last(amplitudes)])
        direction = [-cos(pi/3), -sin(pi/3), 0.0]
        far = pressure(solution, Tuple(1e5*direction); field = :scattered)*1e5*cis(-k*1e5)
        compare_rim_pressure([far], [amplitude])
    end
    compare_rim_pressure(last(full_values), first(full_values))
    compare_rim_pressure(last(full_values), last(values))
end

@testset "Fluid-filled flat-cylinder rims" begin
    body, boundary, k = Cylinder(0.5, 1.0), FluidFilled(1.2, 1.1), 0.5
    points = [(side*(0.5+delta), 0.3+0.6delta, 0.4+0.8delta)
              for side in (-1, 1) for delta in (0.0, 1e-8, 1e-4, 0.001, 0.01, 0.05)]
    bem_values, mfs_values = Vector{ComplexF64}[], Vector{ComplexF64}[]
    amplitudes = Tuple{ComplexF64, ComplexF64}[]
    for n in (384, 512)
        boundary_solution = bem(body, boundary, k; n, incidence_angle = pi/3, m_max = 6)
        source_solution = mfs(body, boundary, k; n, oversampling = 2, offset = 0.015,
            incidence_angle = pi/3, m_max = 6, condition_limit = 0)
        push!(bem_values, pressure(boundary_solution, points; field = :scattered))
        push!(mfs_values, pressure(source_solution, points; field = :scattered))
        push!(amplitudes, (scattering_amplitude(boundary_solution),
            scattering_amplitude(source_solution)))
        compare_rim_pressure(last(bem_values), last(mfs_values))
        for values in (last(bem_values), last(mfs_values))
            compare_rim_pressure(values[[1, 7]], values[[2, 8]])
        end
        for solution in (boundary_solution, source_solution)
            rim = points[[1, 7]]
            compare_rim_pressure(pressure(solution, rim), pressure(solution, rim; field = :interior))
        end
    end
    compare_rim_pressure(last(bem_values), first(bem_values))
    compare_rim_pressure(last(mfs_values), first(mfs_values))
    full_values = Vector{ComplexF64}[]
    for meshsize in (0.25, 0.2)
        solution = bem(body, boundary, k; method = :full, meshsize, mesh_order = 3,
            qorder = 7, incidence_angle = pi/3, formulation = :cbie,
            correction = (method = :edge,), condition_limit = 0)
        @test diagnostics(solution).relative_residual < 1e-10
        actual = pressure(solution, points; field = :scattered)
        push!(full_values, actual)
        compare_rim_pressure(actual, last(bem_values))
        compare_rim_pressure(actual, last(mfs_values))
        compare_rim_pressure(actual[[1, 7]], actual[[2, 8]])
        rim = points[[1, 7]]
        compare_rim_pressure(pressure(solution, rim), pressure(solution, rim; field = :interior))
        amplitude = scattering_amplitude(solution)
        for expected in last(amplitudes)
            compare_rim_pressure([amplitude], [expected])
        end
        direction = [-cos(pi/3), -sin(pi/3), 0.0]
        far = pressure(solution, Tuple(1e5*direction); field = :scattered)*1e5*cis(-k*1e5)
        compare_rim_pressure([far], [amplitude])
    end
    compare_rim_pressure(last(full_values), first(full_values))
end

@testset "Pressure-release flat-cylinder rims" begin
    body, boundary, k = Cylinder(0.5, 1.0), PressureRelease(), 0.5
    points = [(side*(0.5+delta), 0.3+0.6delta, 0.4+0.8delta)
              for side in (-1, 1) for delta in (0.0, 1e-8, 1e-4, 0.001, 0.01, 0.05)]
    bem_values, mfs_values = Vector{ComplexF64}[], Vector{ComplexF64}[]
    amplitudes = ComplexF64[]
    for n in (256, 512)
        boundary_solution = bem(body, boundary, k; n, incidence_angle = pi/3, m_max = 6)
        source_solution = mfs(body, boundary, k; n, oversampling = 2, offset = 0.015,
            incidence_angle = pi/3, m_max = 6, condition_limit = 0)
        push!(bem_values, pressure(boundary_solution, points; field = :scattered))
        push!(mfs_values, pressure(source_solution, points; field = :scattered))
        push!(amplitudes, scattering_amplitude(source_solution))
        compare_rim_pressure(last(bem_values), last(mfs_values))
        for solution in (boundary_solution, source_solution)
            rim = points[[1, 7]]
            compare_rim_pressure(pressure(solution, rim; field = :scattered),
                -pressure(solution, rim; field = :incident))
        end
    end
    compare_rim_pressure(last(bem_values), first(bem_values))
    compare_rim_pressure(last(mfs_values), first(mfs_values))
    full_values = Vector{ComplexF64}[]
    for resolution in (0.2, 0.15)
        surface = mesh(body; method = :full, resolution, mesh_order = 3, qorder = 5)
        solution = bem(surface, boundary, k; incidence_angle = pi/3,
            formulation = :cbie, compression = (method = :none,), correction = (method = :edge,),
            gmres_kwargs = (reltol = 1e-10, restart = 300, maxiter = 1600))
        @test diagnostics(solution).converged
        @test diagnostics(solution).relative_residual < 1e-9
        values = pressure(solution, points; field = :scattered)
        push!(full_values, values)
        compare_rim_pressure(values, last(bem_values))
        compare_rim_pressure(values, last(mfs_values))
        compare_rim_pressure(values[[1, 7]], -pressure(solution, points[[1, 7]]; field = :incident))
        compare_rim_pressure(values[[1, 7]], values[[2, 8]])
        amplitude = scattering_amplitude(solution)
        compare_rim_pressure([amplitude], [last(amplitudes)])
        direction = [-cos(pi/3), -sin(pi/3), 0.0]
        far = pressure(solution, Tuple(1e5*direction); field = :scattered)*1e5*cis(-k*1e5)
        compare_rim_pressure([far], [amplitude])
        @test_throws ArgumentError bem(surface, FluidFilled(1.2, 1.1), k; correction = (method = :edge,))
        @test_throws ArgumentError bem(surface, boundary, k; correction = (method = :edge,))
    end
    compare_rim_pressure(last(full_values), first(full_values))
end
