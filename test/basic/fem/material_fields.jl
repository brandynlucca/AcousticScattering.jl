using AcousticScattering
using Test
using LinearAlgebra

const AS = AcousticScattering

function radial_axis_displacement(solution, r)
    total = zero(ComplexF64)
    for (i, mode) in enumerate(solution.data.modes)
        ell = i - 1
        radii = mode.elastic.radii
        j = clamp(searchsortedlast(radii, r), 1, length(radii) - 1)
        h = radii[j + 1] - radii[j]
        dphi = (mode.elastic.longitudinal[j + 1] -
                mode.elastic.longitudinal[j]) / h
        psi = mode.elastic.shear === nothing ? 0im :
              mode.elastic.shear[j] +
              (r - radii[j]) / h *
              (mode.elastic.shear[j + 1] - mode.elastic.shear[j])
        total += dphi + ell * (ell + 1) * psi / r
    end
    return total / solution.k^2
end

function set_affine_displacement!(solution, gradient)
    system = solution.data.system
    values = solution.data.solution
    for (node_id, node) in enumerate(system.grid.nodes)
        exact = gradient * collect(node.x)
        for component in 1:3
            values[AS._displacement_dof(system.n_nodes, node_id, component)] = exact[component]
        end
    end
end

@testset "Volume FEM material fields" begin
    k = 2.0
    rho_ext, c_ext = 1000.0, 1500.0
    point = (0.1, 0.08, 0.06)
    gradient = [0.2 0.1 0.0; -0.05 0.3 0.02; 0.0 0.04 -0.1]
    pressure_scale = 2.0 + 0.3im
    bulk_scale = rho_ext * c_ext^2

    @time "Elastic volume fields" @testset "Elastic volume fields" begin
        material = SolidElastic(2.7, 4.2, 2.1)
        volume = fem(Sphere(0.5), material, k; method = :volume,
            incidence_angle = pi / 2, points_per_wavelength = 6, closure = :dtn)
        radial = fem(Sphere(0.5), material, k; method = :radial,
            n_elements = 160, m_max = 6)
        actual = volume((0.2, 0.0, 0.0); quantity = :displacement,
            density_exterior = rho_ext, soundspeed_exterior = c_ext)[1]
        reference = radial_axis_displacement(radial, 0.2) / bulk_scale
        @test abs(actual - reference) / abs(reference) < 0.02
        @test isapprox(volume(point; quantity = :stress),
            transpose(volume(point; quantity = :stress)); rtol = 1e-12)

        saved = copy(volume.data.solution)
        try
            set_affine_displacement!(volume, gradient)
            expected_u = pressure_scale / bulk_scale .* (gradient * collect(point))
            rho, cL, cT = material.density_contrast,
            material.speed_longitudinal_contrast,
            material.speed_transversal_contrast
            mu = rho * cT^2
            lame = rho * (cL^2 - 2cT^2)
            expected_stress = pressure_scale .* (
                lame * tr(gradient) * Matrix{Float64}(I, 3, 3) +
                mu * (gradient + transpose(gradient)))
            @test volume(point; quantity = :displacement,
                density_exterior = rho_ext, soundspeed_exterior = c_ext,
                pressure_amplitude = pressure_scale) ≈ expected_u rtol = 1e-8
            @test volume(point; quantity = :velocity,
                density_exterior = rho_ext, soundspeed_exterior = c_ext,
                pressure_amplitude = pressure_scale) ≈ -im * k * c_ext * expected_u rtol = 1e-8
            @test volume(point; quantity = :stress,
                pressure_amplitude = pressure_scale) ≈ expected_stress rtol = 1e-8
            cell_id = findfirst(==(AS._REGION_SOLID), volume.data.system.labels)
            @test AS._volume_cell_mean_dilatation(volume.data, cell_id) ≈
                  tr(gradient) rtol = 1e-8
            @test length(volume([point, point]; quantity = :displacement,
                density_exterior = rho_ext, soundspeed_exterior = c_ext)) == 2
            @test volume(collect(point); quantity = :stress,
                pressure_amplitude = pressure_scale) ≈ expected_stress rtol = 1e-8
            @test all(s -> isapprox(s, expected_stress; rtol = 1e-8),
                volume(hcat(collect(point), collect(point)); quantity = :stress,
                    pressure_amplitude = pressure_scale))
            @test size(volume(fill(point, 2, 3); quantity = :stress)) == (2, 3)
        finally
            copyto!(volume.data.solution, saved)
        end
        @test_throws ArgumentError volume(point; quantity = :unknown)
        exterior_point = ((volume.body.radius + volume.data.system.R) / 2, 0.0, 0.0)
        @test_throws ArgumentError volume(exterior_point; quantity = :stress)
        @test_throws ArgumentError volume(point; quantity = :displacement,
            density_exterior = 0.0, soundspeed_exterior = c_ext)
        @test_throws ArgumentError volume(point; quantity = :stress,
            pressure_amplitude = NaN)
        @test_throws ArgumentError volume([0.1, 0.2]; quantity = :stress)
        @test_throws ArgumentError volume((NaN, 0.0, 0.0); quantity = :stress)
        @test_throws ArgumentError volume("point"; quantity = :stress)
        @test volume(exterior_point; field = :scattered) ==
              pressure(volume, exterior_point; field = :scattered)
    end

    @time "Viscous volume fields" @testset "Viscous volume fields" begin
        material = ViscousLayer(c_ext, 1.05, 1.02, 50.0, 100.0)
        volume = fem([Sphere(0.5)], [material], k; method = :volume,
            incidence_angle = 0.0, points_per_wavelength = 6, closure = :dtn)
        @test all(isfinite,
            volume(point; quantity = :velocity,
                density_exterior = rho_ext, soundspeed_exterior = c_ext))
        @test all(isfinite, volume(point; quantity = :stress))

        saved = copy(volume.data.solution)
        try
            set_affine_displacement!(volume, gradient)
            rho = material.density_contrast
            mu = -im * rho * (k / c_ext) * material.kinematic_viscosity_shear
            longitudinal = rho * (material.soundspeed_contrast^2 -
                            im * (k / c_ext) *
                            material.kinematic_viscosity_compressional)
            lame = longitudinal - 2mu
            expected_stress = pressure_scale .* (
                lame * tr(gradient) * Matrix{ComplexF64}(I, 3, 3) +
                mu * (gradient + transpose(gradient)))
            expected_velocity = -im * k * c_ext * pressure_scale / bulk_scale .* (
                gradient * collect(point))
            @test volume(point; quantity = :stress,
                pressure_amplitude = pressure_scale) ≈ expected_stress rtol = 1e-8
            @test volume(point; quantity = :velocity,
                density_exterior = rho_ext, soundspeed_exterior = c_ext,
                pressure_amplitude = pressure_scale) ≈ expected_velocity rtol = 1e-8
            @test volume(point; quantity = :displacement,
                density_exterior = rho_ext, soundspeed_exterior = c_ext,
                pressure_amplitude = pressure_scale) ≈
                  pressure_scale / bulk_scale .* (gradient * collect(point)) rtol = 1e-8
        finally
            copyto!(volume.data.solution, saved)
        end
        @test_throws ArgumentError volume(point; quantity = :displacement)
        @test_throws ArgumentError volume(point; quantity = :velocity,
            density_exterior = rho_ext, soundspeed_exterior = 1490.0)
        @test_throws ArgumentError volume(point; quantity = :displacement,
            density_exterior = rho_ext, soundspeed_exterior = 1490.0)
        @test_throws ArgumentError volume(point; quantity = :stress,
            soundspeed_exterior = 1490.0)
    end
end
