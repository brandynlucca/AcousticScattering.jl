using AcousticScattering
using LinearAlgebra: BLAS
using Test

BLAS.set_num_threads(min(4, Sys.CPU_THREADS))

function fish_surfaces(; resolution = 0.5, qorder = 5)
    options = (; resolution, qorder, tip_ratio = 0.4)
    return [mesh(; semiaxes = (0.10, 0.018, 0.025), options...),
        mesh(; semiaxes = (0.025, 0.006, 0.009), center = (0.01, 0.003, 0),
            rotation = (axis = (0, 0, 1), angle = deg2rad(10)), options...)]
end

@testset "Fluid quadrature transitions" begin
    materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
    directions = ([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    samples = Matrix{ComplexF64}[]
    for (resolution, qorder) in ((0.5, 5), (0.45, 5), (0.5, 7))
        surfaces = fish_surfaces(; resolution, qorder)
        @info "Quadrature transition discretization" resolution qorder
        solutions = (bem(surfaces, materials, 2pi * 1850 / 1477.4; incidence_angle = pi/2),
            bem(surfaces, materials, 2pi * 2000 / 1477.4; incidence_angle = pi/2),
            bem(surfaces[1], materials[1], 2pi * 2300 / 1477.4; incidence_angle = pi/2))
        push!(samples,
            [scattering_amplitude(solution; direction)
             for direction in directions, solution in solutions])
    end
    for refined in samples[2:3]
        db = maximum(abs.(target_strength.(refined) - target_strength.(first(samples))))
        errors = abs.(refined - first(samples)) ./ abs.(refined)
        relative = maximum(errors)
        @test db < 0.1
        @test relative < 0.01
        frequency = (1850, 2000, 2300)[argmax(errors)[2]]
        @info "Quadrature transition refinement" maximum_db=db maximum_complex=relative frequency
    end
end

@testset "Synthetic fish and displaced bladder" begin
    materials = [FluidFilled(1.04, 1.04), GasFilled(0.00129, 0.23)]
    cases = ((1500.0, 90.0), (2250.0, 60.0), (2250.0, 90.0),
        (2250.0, 120.0), (3000.0, 90.0))
    samples = Matrix{ComplexF64}[]
    for (resolution, qorder) in ((0.5, 5), (0.45, 5), (0.5, 7))
        surfaces = fish_surfaces(; resolution, qorder)
        @info "Fish discretization" resolution qorder
        amplitudes = zeros(ComplexF64, length(cases) * 3, 3)
        for (j, (frequency, degrees)) in enumerate(cases)
            beta = deg2rad(degrees)
            incident = [cos(beta), sin(beta), 0.0]
            directions = (-incident, incident, [0.0, 0.0, 1.0])
            k = 2pi * frequency / 1477.4
            solutions = (bem(surfaces, materials, k; incidence_angle = beta),
                bem(surfaces[1], materials[1], k; incidence_angle = beta),
                bem(surfaces[2], materials[2], k; incidence_angle = beta))
            for (m, solution) in enumerate(solutions),
                (n, direction) in enumerate(directions)

                amplitudes[3(j - 1) + n, m] = scattering_amplitude(solution; direction)
            end
            if (resolution, qorder, frequency, degrees) == (0.5, 5, 2250.0, 90.0)
                for (m, geometry, material) in ((1, surfaces, materials),
                    (2, surfaces[1], materials[1]))
                    pressure = bem(geometry, material, k;
                        incidence_angle = beta, formulation = :cbie)
                    reference = [scattering_amplitude(pressure; direction)
                                 for direction in directions]
                    response = amplitudes[(3j - 2):3j, m]
                    db = maximum(abs.(target_strength.(response) -
                                      target_strength.(reference)))
                    relative = maximum(abs.(response - reference) ./ abs.(reference))
                    @test db < 0.1
                    @test relative < 0.01
                    @info "Fish formulation comparison" model=(:coupled, :flesh)[m] maximum_db=db maximum_complex=relative
                end
            end
            @test diagnostics(first(solutions)).scaled_relative_residual < 1e-10
        end
        @test all(isfinite, amplitudes)
        push!(samples, amplitudes)
    end
    for refined in samples[2:3]
        errors = abs.(refined - first(samples)) ./ abs.(refined)
        relative = maximum(errors)
        db = maximum(abs.(target_strength.(refined) - target_strength.(first(samples))))
        @test db < 0.1
        @test relative < 0.01
        worst = argmax(errors)
        model = (:coupled, :flesh, :bladder)[worst[2]]
        @info "Fish component refinement" maximum_db=db maximum_complex=relative model case=cases[cld(
            worst[1], 3)]
    end
    coupled, flesh, bladder = eachcol(last(samples))
    coherent = flesh + bladder
    difference_db = maximum(abs.(target_strength.(coupled) - target_strength.(coherent)))
    difference_complex = maximum(abs.(coupled - coherent) ./ abs.(coupled))
    @test difference_complex > 0.01
    @info "Coupled versus coherent components" maximum_db=difference_db maximum_complex=difference_complex
end
