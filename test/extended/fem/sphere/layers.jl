using AcousticScattering
using Test
using LinearAlgebra

BLAS.set_num_threads(1)
include("../../../helpers/layered_fem.jl")

# Locate the first isolated resonance and its power half-maximum crossings
# using only public solver amplitudes. Brackets are dimensionless k*gas_radius.
function layered_resonance(solve; tolerance = 1e-9)
    power(q) = abs2(scattering_amplitude(solve(q)))
    lower, upper = 0.008, 0.05
    ratio = (sqrt(5) - 1) / 2
    left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    pleft, pright = power(left), power(right)
    while upper - lower > tolerance
        if pleft < pright
            lower, left, pleft = left, right, pright
            right = lower + ratio * (upper - lower)
            pright = power(right)
        else
            upper, right, pright = right, left, pleft
            left = upper - ratio * (upper - lower)
            pleft = power(left)
        end
    end
    peak = (lower + upper) / 2
    half_power = power(peak) / 2
    function crossing(lower, upper)
        sign_lower = sign(power(lower) - half_power)
        @test sign_lower != sign(power(upper) - half_power)
        while upper - lower > tolerance
            middle = (lower + upper) / 2
            if sign(power(middle) - half_power) == sign_lower
                lower = middle
            else
                upper = middle
            end
        end
        return (lower + upper) / 2
    end
    left = crossing(0.008, peak)
    right = crossing(peak, 0.05)
    return (; peak, width = right - left)
end

@testset "Lossy layered sphere: independent displacement FEM" begin
    # Physical radii [m], moduli [Pa], viscosities [Pa s], and exterior
    # rho=1000 kg/m^3, c=1500 m/s. These are specified numerical benchmarks.
    # The peak amplitudes [m] also have an independently integrated radial
    # force-balance reference, with <1e-9 relative refinement uncertainty.
    cases = (
        (shear_modulus = 0.3e6, bulk = 50 / 3, shear = 25.0,
            samples = (0.01626841315035625, 0.017485150695167997, 0.01902372390771196),
            peak = -0.007567729953482754 + 0.12460170464872054im),
        (shear_modulus = 1.06e6, bulk = 50 / 3, shear = 25.0,
            samples = (0.023672017496512644, 0.025052784506221705, 0.02670792530218686),
            peak = -0.005801332441953473 + 0.16039196551913582im),
        (shear_modulus = 2.22e6, bulk = 50 / 3, shear = 25.0,
            samples = (0.031822365977184096, 0.033434073359267784, 0.03531952660808097),
            peak = -0.0037380960115807097 + 0.18501992438134737im),
        (shear_modulus = 1.06e6, bulk = 25 / 3, shear = 12.5,
            samples = (0.024156677746378686, 0.02500958462787464, 0.025959930510248024),
            peak = -0.0032247845786163073 + 0.26721408743016417im),
        (shear_modulus = 1.06e6, bulk = 100 / 3, shear = 50.0,
            samples = (0.02285479311625998, 0.025227774722815885, 0.02853907076954408),
            peak = -0.00756157921193882 + 0.08912643334539794im),
        (shear_modulus = 1.06e6, bulk = 40.0, shear = 7.5,
            samples = (0.02436676514477617, 0.025000389544852154, 0.025686270656262562),
            peak = -0.0008904556953923901 + 0.3642497163169916im))
    body = Sphere(0.12)
    core = FluidInterior(1.26 / 1000, 330 / 1500)
    for c in cases
        wall = ElasticLayer(1.05, sqrt((1149.3e6 + 2c.shear_modulus) / 1050) / 1500,
            sqrt(c.shear_modulus / 1050) / 1500)
        flesh = ViscousLayer(1500.0, 1.05, 1.0,
            (c.bulk + 4c.shear / 3) / 1050, c.shear / 1050)
        boundary = Shelled(LayeredMaterial(flesh, wall, 0.022 / 0.12),
            core, 0.02 / 0.12)
        for (index, q) in enumerate(c.samples)
            k = q / 0.02
            reference = scattering_amplitude(modal(body, boundary, k))
            solutions = [fem(body, boundary, k; method = :radial, n_elements = n)
                         for n in (64, 128, 256)]
            amplitudes = scattering_amplitude.(solutions)
            errors = abs.(amplitudes .- reference) ./ abs(reference)
            @test errors[3] < errors[2] < errors[1]
            @test errors[3] < 5e-4
            @test abs(20log10(abs(amplitudes[3] / reference))) < 0.005
            @test abs(amplitudes[3] - amplitudes[2]) / abs(reference) < 0.005
            @test diagnostics(solutions[3]).relative_residual < 1e-7
            coefficient = im * k * amplitudes[3]
            @test -real(coefficient) - abs2(coefficient) >= -1e-10
            index == 2 && @test isapprox(amplitudes[3], c.peak; rtol = 5e-4)
        end
        # The fixed samples alone would not measure a shifted FEM resonance.
        solve(q) = fem(body, boundary, q / 0.02; method = :radial, n_elements = 256)
        coarse_search = layered_resonance(solve; tolerance = 1e-7)
        refined_search = layered_resonance(solve; tolerance = 1e-9)
        reference_width = c.samples[3] - c.samples[1]
        @test refined_search.peak ≈ c.samples[2] rtol = 0.001
        @test refined_search.width ≈ reference_width rtol = 0.001
        @test refined_search.peak ≈ coarse_search.peak rtol = 1e-5
        @test refined_search.width ≈ coarse_search.width rtol = 1e-4
    end
end

@testset "Layered displacement FEM limiting cases and arbitrary stacks" begin
    a, k = 1.0, 0.8
    body = Sphere(a)
    core = FluidInterior(0.2, 0.7)
    wall = ElasticLayer(1.3, 1.7, 0.6)
    transparent = ViscousLayer(1500.0, 1.0, 1.0, 0.0, 0.0)
    reduced = modal(Sphere(0.7), Shelled(wall, core, 0.4 / 0.7), k; m_max = 0)
    layered = Shelled(LayeredMaterial(transparent, wall, 0.7), core, 0.4)
    @test scattering_amplitude(fem(body, layered, k; method = :radial, n_elements = 128)) ≈
          scattering_amplitude(reduced) rtol = 1e-6

    # More than two layers, including a fluid between two elastic layers.
    stack = Shelled(
        LayeredMaterial(wall,
            LayeredMaterial(FluidLayer(1.2, 0.9), wall, 0.65), 0.8), core, 0.3)
    @test scattering_amplitude(fem(body, stack, k; method = :radial, n_elements = 128)) ≈
          scattering_amplitude(modal(body, stack, k; m_max = 0)) rtol = 1e-6

    for (compressional, shear) in ((1000.0, 0.0), (400.0, 300.0), (1000.0, 300.0))
        viscous = ViscousLayer(1500.0, 1.1, 1.05, compressional, shear)
        simple = Shelled(viscous, core, 0.4)
        subdivided = Shelled(
            LayeredMaterial(viscous,
                LayeredMaterial(viscous, viscous, 0.7), 0.8), core, 0.4)
        f = scattering_amplitude(fem(body, simple, k; method = :radial, n_elements = 128))
        @test scattering_amplitude(fem(
            body, subdivided, k; method = :radial, n_elements = 128)) ≈ f rtol = 1e-6
        boundary = Shelled(LayeredMaterial(viscous, wall, 0.7), core, 0.4)
        @test scattering_amplitude(fem(
            body, boundary, k; method = :radial, n_elements = 128)) ≈
              scattering_amplitude(modal(body, boundary, k)) rtol = 1e-6
    end
    inviscid = Shelled(
        LayeredMaterial(
            ViscousLayer(1500.0, 1.1, 1.05, 0.0, 0.0), wall, 0.7), core, 0.4)
    vanishing = Shelled(
        LayeredMaterial(
            ViscousLayer(1500.0, 1.1, 1.05, 1e-7, 3e-8), wall, 0.7), core, 0.4)
    @test scattering_amplitude(fem(body, inviscid, k; method = :radial)) ≈
          scattering_amplitude(fem(body, vanishing, k; method = :radial)) rtol = 1e-8
end

@testset "Complete coupled volume FEM for viscous and elastic layers" begin
    body, k = Sphere(1.0), 1.0
    boundary = Shelled(
        LayeredMaterial(
            ViscousLayer(1500.0, 1.1, 1.05, 1000.0, 300.0),
            ElasticLayer(1.3, 1.7, 0.6), 0.7),
        FluidInterior(0.2, 0.7),
        0.4)
    # A regular spherical incident wave isolates the monopole without a
    # truncation of the volume solver's displacement/pressure space.
    incident = IncidentField(x -> sinc(k * norm(x) / pi),
        x -> iszero(norm(x)) ? zeros(ComplexF64, 3) :
             (cos(k * norm(x)) - sinc(k * norm(x) / pi)) / norm(x)^2 .* collect(x))
    reference = scattering_amplitude(fem(
        body, boundary, k; method = :radial, n_elements = 128))
    amplitudes = Vector{ComplexF64}[]
    for h in (0.20, 0.15, 0.12)
        solution = fem(body, boundary, k; method = :volume, incident,
            closure = :dtn, dtn_order = 6, h, h_body = h)
        values = [scattering_amplitude(solution; angle) for angle in (0.0, pi / 2, pi)]
        push!(amplitudes, values)
        @test diagnostics(solution).residual < 1e-7
        @test maximum(abs.(values .- sum(values) / 3)) / abs(reference) < 0.002
    end
    errors = [maximum(abs.(values .- reference)) / abs(reference) for values in amplitudes]
    @test errors[3] < errors[1] / 2
    @test errors[3] < 0.005
    @test maximum(abs.(amplitudes[3] .- amplitudes[2])) / abs(reference) < 0.002

    moved_closure = fem(body, boundary, k; incident,
        closure = :dtn, domain_radius = 1.6, dtn_order = 8, h = 0.15, h_body = 0.15)
    @test scattering_amplitude(moved_closure) ≈ amplitudes[2][2] rtol = 0.005
    @test scattering_amplitude(moved_closure) ≈ reference rtol = 0.005

    for k_plane in (0.5, 1.0)
        full = fem(body, boundary, k_plane; method = :volume, incidence_angle = 0.0,
            closure = :dtn, h = 0.15, h_body = 0.15)
        projected = layered_angular_average(full; n = 32)
        @test projected ≈ layered_angular_average(full; n = 16) rtol = 1e-5
        monopole = scattering_amplitude(fem(
            body, boundary, k_plane; method = :radial, n_elements = 128))
        @test projected ≈ monopole rtol = 0.005
        full_backscatter = scattering_amplitude(full)
        omitted = abs(full_backscatter - monopole) / abs(full_backscatter)
        if k_plane == 0.5
            @test omitted < 0.05
        else
            @test omitted > 0.5
        end
    end
end

@testset "Volume FEM zero-shear fluid limit" begin
    body, k = Sphere(1.0), 1.0
    wall, core = ElasticLayer(1.3, 1.7, 0.6), FluidInterior(0.2, 0.7)
    for compressional in (0.0, 1000.0)
        boundary = Shelled(
            LayeredMaterial(
                ViscousLayer(1500.0, 1.1, 1.05, compressional, 0.0), wall, 0.7),
            core, 0.4)
        # The lossless forward amplitude nearly cancels. Resolve that small
        # complex field instead of relaxing its relative-error tolerance.
        h = iszero(compressional) ? 0.06 : 0.15
        solution = fem(body, boundary, k; method = :volume,
            incidence_angle = 0.0, closure = :dtn, h, h_body = h)
        reference = scattering_amplitude(fem(
            body, boundary, k; method = :radial, n_elements = 128))
        @test layered_angular_average(solution; n = 32) ≈ reference rtol = 0.005
        # Zero shear uses acoustic pressure and does not impose wall no-slip.
        @test isfinite(pressure(solution, (0.85, 0.0, 0.0)))
        if iszero(compressional)
            coarse = fem(body, boundary, k; method = :volume,
                incidence_angle = 0.0, closure = :dtn, h = 0.15, h_body = 0.15)
            acoustic = Shelled(LayeredMaterial(FluidLayer(1.1, 1.05), wall, 0.7), core, 0.4)
            for angle in (0.0, pi / 2, pi)
                expected = scattering_amplitude(modal(body, acoustic, k; angle, m_max = 10))
                @test scattering_amplitude(solution; angle)≈expected rtol=0.01 atol=1e-6
                @test abs(scattering_amplitude(solution; angle) - expected) <
                      abs(scattering_amplitude(coarse; angle) - expected)
            end
        end
    end
end
