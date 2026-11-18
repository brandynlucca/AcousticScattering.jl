using AcousticScattering
using Test
using LinearAlgebra
using SpecialFunctions

const AS = AcousticScattering
BLAS.set_num_threads(1)

let
    @time "ShellFEM (axisymmetric shell operator)" @testset "ShellFEM (axisymmetric shell operator)" begin
        geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)

        @time "geometry" @testset "geometry" begin
            @test geometry.focal_radius ≈ 0.0342928563989645
            @test geometry.semimajor_inner ≈ 0.03490343822605447
            @test geometry.semiminor_inner ≈ 0.006500000000000001
            @test geometry.xi_outer ≈ 1.0206207261596576
            @test geometry.xi_inner ≈ 1.0178049276498415
            @test geometry.xi_mid ≈ 1.0192128269047496
            @test geometry.semimajor_mid ≈ 0.03495171911302723
            @test geometry.semiminor_mid ≈ 0.006754455489227317
            @test geometry.a_shape ≈ 1.0192128269047496
            @test geometry.aspect_ratio ≈ 5.174616839028955
            @test geometry.bending_epsilon ≈ 1.705382018748745e-5

            @test_throws ArgumentError AS.ProlateShellGeometry(0.007, 0.035, 0.0005)
            @test_throws ArgumentError AS.ProlateShellGeometry(0.035, 0.007, 0.007)
            @test_throws ArgumentError AS.Shelled(0.6, 2565.0, 70e9)
        end

        material = AS.Shelled(0.32, 2565.0, 70e9)

        @time @testset "assembled system, freq=$freq" for freq in (12000.0, 38000.0)
            sys = AS.assemble_shell_system(geometry, material.material, freq; n_eta = 9)

            omega_hat_golden = freq == 12000.0 ? 0.4779315959914645 : 1.5134500539729712
            @test sys.nondimensional_frequency ≈ omega_hat_golden

            @test size(sys.dynamic_matrix) == (27, 27)
            @test all(isfinite, sys.dynamic_matrix)
            @test norm(sys.load_scale_q) ≈ 1.4057207588996422e-11 rtol = 1e-9
        end
    end

    @time "Independent prolate shell frequencies" @testset "Independent prolate shell frequencies" begin
        geometry = AS.ProlateShellGeometry(1.0097040331600713, 0.7207583814259635, 0.02745)
        material = Shelled(0.3, 2700.0, 70e9).material
        frequency = AS.extensional_plate_speed(material) / (2pi * geometry.semimajor_mid)
        static = AS.assemble_shell_system(geometry, material, 0.0; n_eta = 257)
        dynamic = AS.assemble_shell_system(geometry, material, frequency; n_eta = 257)
        squared_frequencies = eigvals(-static.dynamic_matrix,
            dynamic.dynamic_matrix - static.dynamic_matrix)
        frequencies = sort(sqrt.(real.(filter(
            value -> abs(imag(value)) < 1e-6 && real(value) > 0.1, squared_frequencies))))
        # First three flexural modes; reference frequencies are rounded to 0.001.
        for (computed, expected) in zip(frequencies[1:3], (1.043, 1.285, 1.372))
            @test computed≈expected atol=0.0005 rtol=0
        end
    end

    @time "Coupled shell/fluid axisymmetric scattering" @testset "Coupled shell/fluid axisymmetric scattering" begin
        geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
        shell_body = AS.Shell(AS.Spheroid(0.035, 0.007), 0.0005)
        rho_ext, c_ext = 1026.8, 1477.3
        rho_int, c_int = 1077.3, 1575.0
        freq = 12000.0
        k = 2pi * freq / c_ext

        @time "geometry/interpolation building blocks" @testset "geometry/interpolation building blocks" begin
            eta = AS.uniform_eta_grid(9; pole_offset = 0.001)
            mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            ps = AS.panels(mesh)
            @test length(ps) == 8

            a, b = geometry.semimajor_length, geometry.semiminor_length
            for p in ps
                g = (p.rhom / b^2, p.zm / a^2)
                gn = hypot(g...)
                @test (p.nrho * g[1] + p.nz * g[2]) / gn ≈ 1.0 atol = 1e-9
            end

            L_s2b = AS.shell_to_bem_interpolation(9)
            L_b2s = AS.bem_to_shell_interpolation(9)
            f = geometry.focal_radius
            interp_eta = L_s2b * eta
            actual_eta = [p.zm / (f * geometry.xi_outer) for p in ps]
            @test interp_eta ≈ actual_eta atol = 1e-12

            roundtrip = L_b2s * interp_eta
            @test roundtrip[2:8] ≈ eta[2:8] atol = 1e-12
        end

        @time "rigid limit (independent cross-check against solve_axial)" @testset "rigid limit (independent cross-check against solve_axial)" begin
            n_eta = 33
            eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
            outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

            # Scale density too, so rigid-body motion also vanishes.
            stiff_material = AS.Shelled(0.32, 2565.0 * 1e16 / 70e9, 1e16)
            sol = AS.fem(shell_body, stiff_material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            @test abs(AS.target_strength(sol) - ts_rigid) < 0.1
        end

        @time "thin-shell backscatter: mesh and pole refinement below 0.1 dB" @testset "thin-shell backscatter: mesh and pole refinement below 0.1 dB" begin
            # NOTE: the pole-offset refinement comparison (n_eta=257 at two offsets) is covered
            # at full fidelity in perf/radial_fem.jl; here a single configuration checks sanity.
            material = AS.Shelled(0.32, 2565.0, 70e9)
            sol = AS.fem(shell_body, material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = 257,
                pole_offset = 0.001, rtol = 1e-4)
            @test all(isfinite, sol.data.p_ext_modes[1])
            @test all(isfinite, sol.data.dpdn_ext_modes[1])
            @test all(isfinite, sol.data.shell_state)
            ts = AS.target_strength(sol)
            @test -150.0 < ts < 0.0
        end

        @time "fluid-filled: rigid limit (independent cross-check)" @testset "fluid-filled: rigid limit (independent cross-check)" begin
            n_eta = 33
            eta = AS.uniform_eta_grid(n_eta; pole_offset = 0.001)
            outer_mesh = AS.prolate_confocal_mesh(geometry, eta; surface = :outer)
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_axial(AS.Rigid(), k, outer_mesh; rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi)

            stiff_material = AS.Shelled(0.32, 2565.0 * 1e16 / 70e9, 1e16)
            sol = AS.fem(shell_body, stiff_material, rho_ext, c_ext, rho_int, c_int, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            @test maximum(abs, sol.data.p_int_modes[1]) < 1.0
            @test abs(AS.target_strength(sol) - ts_rigid) < 0.1
        end

        @time "fluid-filled: zero-interior-density limit recovers the vacuum-backed solver" @testset "fluid-filled: zero-interior-density limit recovers the vacuum-backed solver" begin
            n_eta = 9
            material = AS.Shelled(0.32, 2565.0, 70e9)
            sol_vac = AS.fem(shell_body, material, rho_ext, c_ext, 0.0, 1.0, k;
                method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
            ts_vac = AS.target_strength(sol_vac)

            diffs = Float64[]
            for rho_int_test in (1.0, 1e-3, 1e-6)
                sol = AS.fem(
                    shell_body, material, rho_ext, c_ext, rho_int_test, c_int, k;
                    method = :thin, incidence_angle = 0.0, n_eta = n_eta, pole_offset = 0.001, rtol = 1e-4)
                ts = AS.target_strength(sol)
                push!(diffs, abs(ts - ts_vac))
            end
            @test diffs[1] > diffs[2] > diffs[3]
            @test diffs[3] < 1e-4
        end
    end

    @time "General (Fourier-mode) solid shell FEM and oblique coupling" @testset "General (Fourier-mode) solid shell FEM and oblique coupling" begin
        reference_geometry = AS.ProlateShellGeometry(0.035, 0.007, 0.0005)
        geometry = AS.ProlateShellGeometry(0.02, 0.005, 0.0005)
        c_water = 1477.4
        freq_hz = 12000.0
        k = 2pi * freq_hz / c_water
        n_eta = 13
        n_t = 3

        @time "outer dimensions and confocal cavity" @testset "outer dimensions and confocal cavity" begin
            mesh = AS.build_structured_shell_strip(reference_geometry, 9, 3; pole_offset = 0.001)
            @test AS.Ferrite.getnnodes(mesh.grid) == 27
            @test AS.Ferrite.getncells(mesh.grid) == 32
            @test mesh.outer_ρ[1] ≈ 0.000312971244685512 atol = 1e-12
            @test mesh.outer_z[end] ≈ 0.034965 atol = 1e-12
            @test mesh.inner_ρ[5] ≈ 0.0065 atol = 1e-12
        end

        @time "elastic reciprocity and unit-normal projections (m=0,1,2)" @testset "elastic reciprocity and unit-normal projections (m=0,1,2)" begin
            mesh = AS.build_structured_shell_strip(reference_geometry, 9, 3; pole_offset = 0.001)
            omega = 2pi * 12000.0
            for m in (0, 1, 2)
                ops = AS.assemble_shell_modal_operators(mesh, m, omega, 2565.0, 70e9, 0.32)
                @test ops.dynamic_matrix ≈ ops.dynamic_matrix' rtol = 1e-12
                @test tr(ops.B_out * ops.B_out') ≈ 9.0 atol = 1e-10
                @test tr(ops.B_in * ops.B_in') ≈ 9.0 atol = 1e-10
            end
        end

        @time "coupled oblique rigid limit (independent cross-check against solve_oblique)" @testset "coupled oblique rigid limit (independent cross-check against solve_oblique)" begin
            beta = deg2rad(45.0)
            m_max = 3
            shell_mesh = AS.build_structured_shell_strip(geometry, n_eta, n_t; pole_offset = 0.001)
            outer_mesh = AS.MeridianMesh(reverse(shell_mesh.outer_ρ), reverse(shell_mesh.outer_z))
            p_rigid, dpdn_rigid, ps_rigid = AS.solve_oblique(
                AS.Rigid(), k, outer_mesh, beta; m_max = m_max, rtol = 1e-4)
            ts_rigid = AS.target_strength(ps_rigid, p_rigid, dpdn_rigid, k, pi - beta, pi)

            shell_body2 = AS.Shell(AS.Spheroid(0.02, 0.005), 0.0005)
            material_stiff = AS.Shelled(0.3, 1e6, 1e14)
            sol = AS.fem(
                shell_body2, material_stiff, 1026.8, c_water, 1026.8, c_water, k;
                method = :general, incidence_angle = beta, m_max = m_max, n_eta = n_eta,
                n_t = n_t, pole_offset = 0.001, rtol = 1e-4)
            ts_stiff = AS.target_strength(sol; angle = pi - beta, azimuth = pi)

            @test ts_stiff ≈ ts_rigid atol = 0.1
            @test AS.target_strength(sol) == ts_stiff
        end
    end

    @time "Confocal elastic shell against independent outputs" @testset "Confocal elastic shell against independent outputs" begin
        solution = fem(Shell(Spheroid(1.5, 1.0), 0.2), Shelled(0.33, 2700.0, 70e9),
            1000.0, 1500.0, 1000.0, 1500.0, 1.0;
            method = :general, incidence_angle = pi / 3, n_eta = 161, n_t = 17,
            m_max = 6, pole_offset = 1e-4, rtol = 1e-5)
        for (angle, azimuth, reference) in (
            (2pi / 3, pi, -0.106487925705214 + 0.0747097439019188im),
            (pi / 3, 0.0, 0.0814480252840919 + 0.0804910782181885im),
            (pi / 2, pi / 2, -0.467543104429619 + 0.0525831637378128im))
            @test abs(target_strength(solution; angle, azimuth) - 20log10(abs(reference))) <
                  0.1
            @test scattering_amplitude(solution; angle, azimuth) ≈ reference rtol = 0.01
        end
    end
end

@testset "Total interior pressure preserves the coupled weak equations" begin
    body = Spheroid(0.02, 0.02/1.01)
    boundary = GasFilled(1.24/1026.8, 345/1477.3)
    k = 60.0

    function incident_traces(surface, m, beta, parity)
        function trace(rho, z, normal_rho, normal_z)
            argument = k*rho*sin(beta)
            value = besselj(m, argument)
            derivative = m==0 ? -besselj(1, argument) :
                         (besselj(m-1, argument)-besselj(m+1, argument))/2
            phase = (m==0 ? 1.0 : 2.0)*im^m*cis(k*z*cos(beta))
            return phase*value,
                phase*k*(sin(beta)*normal_rho*derivative+im*cos(beta)*normal_z*value)
        end
        values = map(eachindex(surface.chi)) do j
            rho, z = surface.rho[j], surface.z[j]
            nr, nz = surface.nrho[j], surface.nz[j]
            pressure, derivative = trace(rho, z, nr, nz)
            if parity != 0
                reflected_pressure, reflected_derivative = trace(rho, -z, nr, -nz)
                pressure = (pressure+parity*reflected_pressure)/2
                derivative = (derivative+parity*reflected_derivative)/2
            end
            (; pressure, derivative)
        end
        return getproperty.(values, :pressure), getproperty.(values, :derivative)
    end

    # Assemble every volume degree of freedom before imposing constraints.
    # This reference uses neither a Robin inverse nor angular diagonalization.
    for (beta, half, parity) in ((deg2rad(37.0), false, 0),
            (deg2rad(37.0), true, 1), (deg2rad(37.0), true, -1), (pi/2, true, 0)),
        m in (0, 2)

        p, nr, nt, nb, boundary_degree = 3, 2, half ? 4 : 8, half ? 6 : 12, 4
        mesh = AS._spheroid_fembem_confocal(boundary, body.a, body.b, nr, nt, p;
            half, assemble_bulk = true, quadrature_extra = 2)
        surface = AS._spheroid_curved_boundary(body.a, body.b, nb, boundary_degree+1;
            half)
        trace, flux = AS._spheroid_fembem_trace(mesh, surface)
        operators = AS._spheroid_curved_operators(surface, k, m:m, Val(1),
            Val(boundary_degree+1); rtol = 1e-9, chief_points = half ? 4 : 8,
            split = parity != 0, quadrature_extra = 2)
        offset = parity==-1 ? 2 : 0
        K, V = operators[1+offset][1], operators[2+offset][1]
        nbe = length(surface.chi)
        D = -K
        for j in 1:nbe
            D[j, j] += 0.5
        end
        projection = qr(V)
        reduced_D = (projection.Q' * D)[1:nbe, :]
        reduced_V = Matrix(projection.R)

        radial_nodes, angular_nodes = nr*p+1, nt*p+1
        volume_nodes = radial_nodes*angular_nodes
        volume = zeros(ComplexF64, volume_nodes, volume_nodes)
        for cell in mesh.cells
            volume[cell.ids, cell.ids] += cell.S+m^2*cell.A-k^2*cell.M
        end
        interface = (volume_nodes-angular_nodes+1):volume_nodes
        volume_trace = zeros(Float64, nbe, volume_nodes)
        volume_trace[:, interface] = trace
        volume_flux = zeros(Float64, volume_nodes, nbe)
        volume_flux[interface, :] = flux

        fixed = Int[]
        if m>0
            append!(fixed, 1:angular_nodes)
            append!(fixed, 1:angular_nodes:volume_nodes)
            half || append!(fixed, angular_nodes:angular_nodes:volume_nodes)
        end
        parity==-1 && append!(fixed, angular_nodes:angular_nodes:volume_nodes)
        free = setdiff(1:volume_nodes, fixed)
        incident, derivative = incident_traces(surface, m, beta, parity)
        matrix = [volume[free, free] -volume_flux[free, :];
                  reduced_D*volume_trace[:, free] reduced_V]
        rhs = vcat(volume_flux[free, :]*derivative, reduced_D*incident)
        reference = matrix\rhs
        total = reference[1:length(free)]
        expected_derivative = reference[(length(free)+1):end]
        expected_pressure = volume_trace[:, free]*total-incident
        @test norm(matrix*reference-rhs) <= 2e-11*norm(rhs)
        fem_rhs = volume_flux[free, :]*(expected_derivative+derivative)
        @test norm(volume[free, free]*total-fem_rhs) <= 2e-10*norm(rhs)

        # Test both elimination orders against the independently assembled
        # equations. The outgoing derivative is scattered, not density scaled.
        for interior_reduction in (:angular, :radial)
            reports = AS._SolveReports()
            numerical = AS._spheroid_fembem_tensor_mode(mesh, surface, trace, flux,
                K, V, k, m, beta, reports, 1;
                parity, interior_reduction, interior_field = :total)
            @test only(reports.systems).interior_reduction ===
                  (interior_reduction===:angular ? :tensor : :radial)
            @test numerical.pressure ≈ expected_pressure rtol=2e-9 atol=1e-11
            @test numerical.derivative ≈ expected_derivative rtol=2e-9 atol=1e-9
            for (angle, azimuth) in ((pi-beta, pi), (0.4, 0.7), (2.1, 1.2))
                expected = AS._SpheroidSurfaceMode(surface, expected_pressure,
                    expected_derivative)
                @test AS._spheroid_curved_amplitude(numerical, k, m, angle, azimuth) ≈
                      AS._spheroid_curved_amplitude(expected, k, m, angle, azimuth) rtol=2e-9 atol=1e-12
            end
        end
    end
end

@testset "Total interior pressure preserves equatorial reconstruction" begin
    body, boundary, k = Spheroid(0.02, 0.015), GasFilled(0.0012, 0.23), 60.0
    options = (; m_max = 2, mode_threads = 2, rtol = 1e-9, chief_points = 8,
        interior_field = :total, quadrature_extra = 2)
    for beta in (deg2rad(37.0), pi/2)
        full = AS._spheroid_fembem_level(body, boundary, k, 3, 16, 12, 4, 4;
            options..., half = false, incidence_angle = beta, split_symmetry = false,
            reports = AS._SolveReports())
        reduced = AS._spheroid_fembem_level(body, boundary, k, 3, 16, 12, 4, 4;
            options..., half = beta==pi/2, incidence_angle = beta,
            reports = AS._SolveReports())
        for (m, (expected, actual)) in enumerate(zip(full, reduced))
            pressure = beta==pi/2 ? vcat(actual.pressure, reverse(actual.pressure)) :
                       actual.pressure
            derivative = beta==pi/2 ? vcat(actual.derivative, reverse(actual.derivative)) :
                         actual.derivative
            @test pressure ≈ expected.pressure rtol=2e-8 atol=1e-10
            @test derivative ≈ expected.derivative rtol=2e-8 atol=1e-8
            for (angle, azimuth) in ((pi-beta, pi), (0.4, 0.7), (2.1, 1.2))
                @test AS._spheroid_curved_amplitude(actual, k, m-1, angle, azimuth) ≈
                      AS._spheroid_curved_amplitude(expected, k, m-1, angle, azimuth) rtol=2e-8 atol=1e-11
            end
        end
    end
end

@testset "Independent refinement of total and scattered interior gas fields" begin
    body, boundary, k = Spheroid(0.02, 0.02/1.01), GasFilled(1.24/1026.8, 345/1477.3), 60.0
    pairing = Coupling(interior = :fem, exterior = :bem)
    options = (; coupling = pairing, m_max = 8, mode_threads = 2, adaptive = false,
        order = 6, boundary_order = 5, rtol = 1e-9, quadrature_extra = 4)
    @test_throws ArgumentError coupled(body, boundary, k; options..., interior_field = :invalid)
    for beta in (deg2rad(37.0), pi/2)
        amplitudes = Dict{Symbol, Vector{Vector{ComplexF64}}}()
        for interior_field in (:total, :scattered)
            amplitudes[interior_field] = Vector{ComplexF64}[]
            for (nr, nt, nb) in ((4, 24, 20), (6, 32, 28))
                solution = coupled(body, boundary, k; options..., interior_field,
                    incidence_angle = beta, n_r = nr, n_theta = nt, n_boundary = nb)
                @test diagnostics(solution).solver_options.interior_field === interior_field
                @test diagnostics(solution).relative_residual < 1e-10
                push!(amplitudes[interior_field],
                    [scattering_amplitude(solution; angle, azimuth)
                     for (angle, azimuth) in ((pi-beta, pi), (0.4, 0.7), (2.1, 1.2))])
            end
            @test amplitudes[interior_field][1] ≈ amplitudes[interior_field][2] rtol=1e-5 atol=1e-10
        end
        @test amplitudes[:total][2] ≈ amplitudes[:scattered][2] rtol=2e-5 atol=1e-10
    end
end

@testset "Batched coupled incidences retain independent adaptive results" begin
    body, boundary, k = Spheroid(0.02, 0.015), GasFilled(0.0012, 0.23), 100.0
    angles = [deg2rad(37.0), pi/2, 0.0]
    pairing = Coupling(interior = :fem, exterior = :bem)
    options = (; coupling = pairing, n_r = 1, n_theta = 4, n_boundary = 4,
        order = 3, boundary_order = 3, m_max = 4, mode_threads = 2,
        rtol = 1e-8, adaptive = true, target_tol = 0.005, max_refinements = 4)
    lengths = Int[]
    for interior_field in (:total, :scattered)
        fresh = [coupled(body, boundary, k; options..., interior_field,
                     incidence_angle = angle) for angle in angles]
        batched = incidence_angle_sweep(coupled, body, boundary, k, angles;
            options..., interior_field, return_diagnostics = true)
        @test batched.sweep.angles == angles
        @test length(batched.diagnostics) == length(angles)
        for i in eachindex(angles)
            expected, actual = diagnostics(fresh[i]), batched.diagnostics[i]
            @test expected.refinement.converged
            @test batched.sweep.amplitudes[i] ≈ scattering_amplitude(fresh[i]) rtol=1e-10 atol=1e-12
            @test batched.sweep.target_strength[i] ≈ target_strength(fresh[i]) atol=1e-9
            @test actual.refinement.converged == expected.refinement.converged
            @test actual.refinement.mode_refinements == expected.refinement.mode_refinements
            @test actual.refinement.mode_n_r == expected.refinement.mode_n_r
            @test actual.refinement.mode_n_theta == expected.refinement.mode_n_theta
            @test actual.refinement.mode_n_boundary == expected.refinement.mode_n_boundary
            @test actual.refinement.change_db ≈ expected.refinement.change_db rtol=1e-7 atol=1e-10
            @test actual.solver_options.incidence_angle == angles[i]
            @test actual.solver_options.interior_field === interior_field
            @test getproperty.(actual.refinement.history, :modes) ==
                  getproperty.(expected.refinement.history, :modes)
            push!(lengths, length(actual.refinement.history))
        end
    end
    # End-on and off-axis loads must be allowed to finish at different stages.
    @test length(unique(lengths)) > 1

    seen = Int[]
    on_solution = (index, solution)->begin
        @test diagnostics(solution).solver_options.incidence_angle == angles[index]
        push!(seen, index)
    end
    completed = AS._coupled_angle_solutions(body, boundary, k, angles;
        options..., interior_field = :total, on_solution)
    @test sort(seen) == collect(eachindex(angles))
    @test length(completed) == length(angles)
    @test get(task_local_storage(), :acousticscattering_fembem_batch, nothing) === nothing

    callback_error = ErrorException("test callback failure")
    callbacks = Int[]
    failure = try
        AS._coupled_angle_solutions(body, boundary, k, angles;
            options..., interior_field = :total,
            on_solution = (index, _)->begin
                push!(callbacks, index)
                throw(callback_error)
            end)
        nothing
    catch error
        error
    end
    @test failure === callback_error
    @test length(callbacks) == 1
    @test get(task_local_storage(), :acousticscattering_fembem_batch, nothing) === nothing
    @test get(task_local_storage(), :acousticscattering_numerical_reuse, nothing) === nothing

    # A failed angle or callback must release the other suspended solves and
    # leave a fresh sweep usable. Duplicate angles retain their own output slots.
    @test_throws ArgumentError AS._coupled_angle_solutions(body, boundary, k, angles;
        options..., interior_field = :invalid)
    resumed = incidence_angle_sweep(coupled, body, boundary, k,
        [angles[3], angles[1], angles[1]];
        options..., interior_field = :total, return_diagnostics = true)
    @test resumed.sweep.amplitudes[1] ≈ scattering_amplitude(completed[3]) rtol=1e-10 atol=1e-12
    @test resumed.sweep.amplitudes[2] ≈ scattering_amplitude(completed[1]) rtol=1e-10 atol=1e-12
    @test resumed.sweep.amplitudes[3] == resumed.sweep.amplitudes[2]
    @test resumed.diagnostics[2].refinement.mode_refinements ==
          resumed.diagnostics[3].refinement.mode_refinements
    @test_throws ArgumentError incidence_angle_sweep(coupled, body, boundary, k,
        Float64[]; options...)
    @test_throws ArgumentError incidence_angle_sweep(coupled, body, boundary, k,
        [0.0, NaN]; options...)
end

@testset "Coupled batches share geometry and retain separate field buffers" begin
    body, k = Spheroid(0.02, 0.015), 100.0
    angles = [deg2rad(37.0), pi/2, 0.0]
    pairing = Coupling(interior = :fem, exterior = :bem)
    for (boundary, interior_field) in
        ((FluidFilled(1028.9/1026.8, 1480.3/1477.3), :scattered),
         (GasFilled(1.24/1026.8, 345/1477.3), :total))

        cache = Dict{Any, Any}()
        options = (; m_max = 3, mode_threads = 1, rtol = 1e-9, chief_points = 8,
            interior_field, geometry_cache = cache)
        oblique = AS._spheroid_fembem_level_setup(body, boundary, k, 3, 16, 12, 4, 4;
            options..., half = false, incidence_angle = angles[1], reports = AS._SolveReports())
        broadside = AS._spheroid_fembem_level_setup(body, boundary, k, 3, 16, 12, 4, 4;
            options..., half = true, incidence_angle = angles[2], reports = AS._SolveReports())
        @test length(cache) == 1
        @test oblique.mesh === broadside.mesh
        @test oblique.be === broadside.be
        @test oblique.T === broadside.T
        @test oblique.B === broadside.B
        @test oblique.fields !== broadside.fields
        @test oblique.reports !== broadside.reports
        @test isempty(oblique.mesh.load_points) == (interior_field===:total)
        refined = AS._spheroid_fembem_level_setup(body, boundary, k, 3, 16, 12, 4, 4;
            options..., half = true, incidence_angle = angles[2], quadrature_extra = 2,
            reports = AS._SolveReports())
        @test length(cache) == 2
        @test refined.mesh !== broadside.mesh
        @test refined.B !== broadside.B

        solve_options = (; coupling = pairing, n_r = 3, n_theta = 16, n_boundary = 12,
            order = 4, boundary_order = 4, m_max = 3, mode_threads = 1,
            rtol = 1e-9, interior_field, adaptive = false)
        fresh = [coupled(body, boundary, k; solve_options..., incidence_angle = angle)
                 for angle in angles]
        batched = incidence_angle_sweep(coupled, body, boundary, k, angles;
            solve_options..., return_diagnostics = true)
        for i in eachindex(angles)
            @test batched.sweep.amplitudes[i] ≈ scattering_amplitude(fresh[i]) rtol=1e-10 atol=1e-13
            @test batched.diagnostics[i].relative_residual < 1e-10
        end
    end
end
