using AcousticScattering
using Test
using LinearAlgebra

module GradedSphereReference

# Independent radial ODE and spherical-wave matching; no AcousticScattering import.
using LinearAlgebra, SpecialFunctions

function graded_state(ell, radius; a, k, g0, c0, alpha, beta, steps)
    stop = log(radius / a)
    start = -16.0
    step = (stop - start) / steps
    state = [1.0, 0.0]
    function rhs(s, y)
        r = a * exp(s)
        t2 = exp(2s)
        rg = 2alpha * t2 / (1 + alpha * t2)
        c = c0 * (1 + beta * t2)
        return [y[2], -(2ell + 1 - rg) * y[2] - (k^2 * r^2 / c^2 - ell * rg) * y[1]]
    end
    for j in 0:(steps - 1)
        s = start + j * step
        k1 = rhs(s, state)
        k2 = rhs(s + step / 2, state + step / 2 * k1)
        k3 = rhs(s + step / 2, state + step / 2 * k2)
        k4 = rhs(s + step, state + step * k3)
        state += step / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return state
end

sj(ell, x) = sqrt(pi / (2x)) * besselj(ell + 0.5, x)
sh(ell, x) = sqrt(pi / (2x)) * (besselj(ell + 0.5, x) + im * bessely(ell + 0.5, x))
function legendre_values(n, x)
    values = [1.0, x]
    for ell in 1:(n - 1)
        push!(values, ((2ell + 1) * x * values[end] - ell * values[end - 1]) / (ell + 1))
    end
    return values[1:(n + 1)]
end

function graded_reference(; a = 0.5, k = 2.0, g0 = 1.2, c0 = 0.9,
        alpha = 0.5, beta = 0.2, steps = 2000, nmax = 10)
    settings = (; a, k, g0, c0, alpha, beta, steps)
    edge = [graded_state(ell, a; settings...) for ell in 0:nmax]
    coefficients = ComplexF64[]
    traces = ComplexF64[]
    x = k * a
    for ell in 0:nmax
        v, w = edge[ell + 1]
        Y = (ell + w / v) / (a * g0 * (1 + alpha))
        j, h = sj(ell, x), sh(ell, x)
        dj, dh = ell / x * j - sj(ell + 1, x), ell / x * h - sh(ell + 1, x)
        A = (Y * j - k * dj) / (k * dh - Y * h)
        push!(coefficients, A)
        push!(traces, j + A * h)
    end
    farfield = angle -> -im / k * sum((2ell + 1) * coefficients[ell + 1] *
                            legendre_values(nmax, cos(angle))[ell + 1] for ell in 0:nmax)
    function interior(point)
        r = norm(point)
        P = legendre_values(nmax, point[3] / r)
        return sum(im^ell * (2ell + 1) * traces[ell + 1] * (r / a)^ell *
                   graded_state(ell, r; settings...)[1] / edge[ell + 1][1] * P[ell + 1]
        for ell in 0:nmax)
    end
    return (; farfield, interior)
end

end
using .GradedSphereReference: graded_reference

@testset "Spatial fluid volume FEM" begin
    AS = AcousticScattering
    a, k = 0.5, 2.0
    angles = (0.0, pi / 2, pi)
    points = [(0.0, 0.0, 0.2), (0.15, 0.1, -0.2), (0.0, 0.0, 0.4)]
    rho = x -> 1.2 * (1 + 0.5 * sum(abs2, x) / a^2)
    speed = x -> 0.9 * (1 + 0.2 * sum(abs2, x) / a^2)
    material = SpatialFluid(rho, speed; min_soundspeed_contrast = 0.9)
    reference = graded_reference()
    exact = reference.farfield.(angles)
    exact_pressure = reference.interior.(points)
    scale, pressure_scale = maximum(abs, exact), maximum(abs, exact_pressure)

    @testset "Reference convergence and homogeneous limit" begin
        refined = graded_reference(steps = 4000, nmax = 12)
        constant = graded_reference(alpha = 0.0, beta = 0.0)
        for angle in angles
            @test abs(reference.farfield(angle) - refined.farfield(angle)) < 1e-9
            @test constant.farfield(angle) ≈ scattering_amplitude(
                modal(Sphere(a), FluidFilled(1.2, 0.9), k; angle)) atol = 1e-9
        end
        for point in points
            @test abs(reference.interior(point) - refined.interior(point)) < 1e-9
        end
    end

    @testset "Material validation" begin
        @test_throws ArgumentError SpatialFluid(1.0, 1.0; min_soundspeed_contrast = 0.0)
        @test_throws ArgumentError SpatialFluid(-1.0, 1.0; min_soundspeed_contrast = 1.0)
        @test_throws ArgumentError SpatialFluid(1.0, 1.0im; min_soundspeed_contrast = 1.0)
        @test_throws ArgumentError SpatialFluid(1.0, 0.8; min_soundspeed_contrast = 1.0)
        for bad in (SpatialFluid(x -> NaN, 1.0; min_soundspeed_contrast = 1.0),
            SpatialFluid(1.0, x -> 0.5; min_soundspeed_contrast = 1.0))
            @test_throws ArgumentError AS._volume_fluid_parameters(
                AS._SpatialFluidCoefficients(bad, k), zeros(3))
        end
        @test_throws ArgumentError fem(Sphere(a), material, k; method = :radial)
        @test_throws ArgumentError free_surface(Sphere(a), material, k, 2.0)
    end

    @testset "Quadrature profiles and mesh refinement" begin
        errors = Tuple{Float64, Float64}[]
        for h in (0.25, 0.1)
            solution = fem(Sphere(a), material, k; incidence_angle = 0.0,
                h, h_body = h, solver_tolerance = 1e-10)
            values = [scattering_amplitude(solution; angle) for angle in angles]
            fields = pressure(solution, points; field = :total)
            push!(errors,
                (maximum(abs.(values .- exact)) / scale,
                    maximum(abs.(fields .- exact_pressure)) / pressure_scale))
            @test diagnostics(solution).residual < 1e-8
            @test errors[end][1] < 1e-3
            @test errors[end][2] < 1e-3
        end
        @test errors[end][1] < errors[1][1] / 2
        @test errors[end][2] < errors[1][2] / 2
        @info "Spatial fluid refinement" errors
    end

    @testset "Constant callbacks recover homogeneous FEM" begin
        constant = SpatialFluid(x -> 1.2, 1.1; min_soundspeed_contrast = 1.1)
        options = (; method = :volume, incidence_angle = 0.0, h = 0.2, h_body = 0.2)
        spatial = fem(Sphere(a), constant, k; options...)
        homogeneous = fem(Sphere(a), FluidFilled(1.2, 1.1), k; options...)
        for angle in angles
            @test scattering_amplitude(spatial; angle) ≈
                  scattering_amplitude(homogeneous; angle) rtol = 1e-10
        end
        @test pressure(spatial, points; field = :total) ≈
              pressure(homogeneous, points; field = :total) rtol = 1e-10
    end

    @testset "Global coordinates of translated region" begin
        center = [0.1, 0.05, -0.1]
        shifted = SpatialFluid(x -> rho(x - center), x -> speed(x - center);
            min_soundspeed_contrast = 0.9)
        solution = fem([Sphere(a)], [shifted], k; centers = [center],
            incidence_angle = 0.0, h = 0.12, h_body = 0.12, solver_tolerance = 1e-10)
        for angle in angles
            direction = [sin(angle), 0.0, cos(angle)]
            phase = cis(k * dot([0.0, 0.0, 1.0] - direction, center))
            @test abs(scattering_amplitude(solution; angle, azimuth = 0.0) -
                      phase * reference.farfield(angle)) / scale < 2e-3
        end
        values = pressure(solution, [collect(p) + center for p in points]; field = :total)
        @test maximum(abs.(values .- cis(k * center[3]) .* exact_pressure)) /
              pressure_scale < 2e-3
    end

    @testset "Incidence reuse with a graded region" begin
        sweep = incidence_angle_sweep([Sphere(a)], [material], k, [0.0, pi / 2];
            h = 0.16, h_body = 0.16, solver_tolerance = 1e-10)
        @test maximum(abs.(sweep.amplitudes .- reference.farfield(pi))) / scale < 1e-3
    end

    @testset "Spatial fluid core in an elastic shell" begin
        solid = SolidElastic(2.7, 4.2, 2.1)
        core = SpatialFluid(1.2, x -> 1.1; min_soundspeed_contrast = 1.1)
        solution = fem([Sphere(a), Sphere(0.3)], [solid, core], k;
            incidence_angle = 0.0, h = 0.2, h_body = 0.1)
        shell = Shelled(ElasticLayer(2.7, 4.2, 2.1), FluidInterior(1.2, 1.1), 0.6)
        exact_shell = [scattering_amplitude(modal(Sphere(a), shell, k; angle))
                       for angle in angles]
        actual = [scattering_amplitude(solution; angle) for angle in angles]
        @test maximum(abs.(actual .- exact_shell)) / maximum(abs, exact_shell) < 0.01
    end

    @testset "Damped solid coupling through a matched spatial fluid" begin
        # Independent e3Dss case 1; the added unit-contrast outer fluid region
        # has no physical interface with the exterior.
        E, nu, density = 70e9, 0.33, 2700.0
        cL = sqrt(E * (1 - nu) / (density * (1 + nu) * (1 - 2nu))) / 1500
        cT = sqrt(E / (2density * (1 + nu))) / 1500
        solid = ViscoelasticSolid(2.7, cL, cT;
            loss_longitudinal = 0.05, loss_transversal = 0.05)
        fluid = SpatialFluid(x -> 1.0, 1.0; min_soundspeed_contrast = 1.0)
        solution = fem([Sphere(0.15), Sphere(0.1)], [fluid, solid], 20.0;
            incidence_angle = 0.0, h = 0.025, h_body = 0.02)
        expected = 0.0281125345366979 + 0.0389760655908978im
        error = abs(scattering_amplitude(solution; angle = pi) - expected) / abs(expected)
        @test error < 5e-3
        @info "e3Dss damped solid with matched SpatialFluid shell" error
    end
end
