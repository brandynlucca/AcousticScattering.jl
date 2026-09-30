using AcousticScattering
using Test
using LinearAlgebra

const AS = AcousticScattering

@testset "Fluid sweep solution recycling" begin
    A = ComplexF64[2 1im 0; 0 3 1; 0.2 0 5]
    rows, cols = [2.0, 0.5, 3.0], [0.5, 4.0, 2.0]
    factor = (; scaled_A = A ./ rows ./ transpose(cols), row_norms = rows,
        col_norms = cols, preconditioner = AS.IterativeSolvers.Identity(),
        options = (; reltol = 1e-10, abstol = 0.0, restart = 3, maxiter = 3))
    recycled = AS._fluid_sweep_factor(factor, 2, 20)
    b = ComplexF64[1, 0.3im, 0.2]
    first_solve = AS._solve_compressed_fluid_system(recycled, b)
    repeated = AS._solve_compressed_fluid_system(recycled, b)
    @test repeated.history.iters == 0
    @test repeated.history.isconverged
    @test repeated.x ≈ first_solve.x rtol = 1e-9
    @test recycled.recycling.count == 1
    # Tiny changes must use the original RHS tolerance, not a tighter tolerance
    # proportional to the already-small starting residual.
    perturbed = b + 1e-12*ComplexF64[0, 1, 1im]
    close = AS._solve_compressed_fluid_system(recycled, perturbed)
    @test close.history.iters == 0
    @test norm((A*close.x - perturbed) ./ rows) <= 1e-10*norm(perturbed ./ rows)
    for rhs in (ComplexF64[1im, 2, -3], reverse(b), b, zeros(ComplexF64, 3))
        solved = AS._solve_compressed_fluid_system(recycled, rhs)
        @test solved.history.isconverged
        @test solved.x≈A\rhs rtol=1e-8 atol=1e-12
        @test norm((A*solved.x-rhs) ./ rows) <= 1e-10*norm(rhs ./ rows)
        @test recycled.recycling.count <= 2
    end
    @test size(recycled.recycling.basis) == (3, 2)
    @test AS._fluid_sweep_factor(factor, 0, 20) === factor
    @test AS._fluid_sweep_factor(factor, 8, 1) === factor
    @test AS._fluid_sweep_factor((; dense = true), 8, 20) == (; dense = true)
    @test AS._fluid_sweep_factor(factor, 8, 20).recycling.count == 0
    @test_throws ArgumentError AS._fluid_sweep_factor(factor, -1, 20)
    # A poor but residual-reducing starting guess exhausts the one-step budget;
    # a zero start solves this eigenvector RHS in one step. Exercise the fallback.
    easy = merge(factor,
        (; scaled_A = Diagonal(ComplexF64[1, 2, 4]),
            row_norms = ones(3), col_norms = ones(3),
            options = merge(factor.options, (; maxiter = 1))))
    fallback = AS._fluid_sweep_factor(easy, 2, 20)
    fallback.recycling.count = 1
    fallback.recycling.basis[:, 1] = [0.5, 0.2, 0]
    fallback.recycling.images[:, 1] = [1, 0, 0]
    solved = AS._solve_compressed_fluid_system(fallback, ComplexF64[1, 0, 0])
    @test solved.history.isconverged
    @test solved.history.iters == 2
    @test length(solved.history[:resnorm]) == 2
    @test solved.x ≈ [1, 0, 0]
    @test fallback.recycling.count == 1
    fallback.recycling.basis[:, 1] .= NaN
    guarded = AS._solve_compressed_fluid_system(fallback, ComplexF64[1, 0, 0])
    @test guarded.history.isconverged
    @test guarded.x ≈ [1, 0, 0]
    failed = AS._fluid_sweep_factor(easy, 2, 20)
    result = AS._solve_compressed_fluid_system(failed, ComplexF64[1, 1, 1])
    @test !result.history.isconverged
    @test failed.recycling.count == 0
    # Absolute tolerances remain effective even for a small RHS.
    absolute = AS._fluid_sweep_factor(
        merge(easy,
            (; options = merge(easy.options, (; abstol = 1e-8)))), 2, 20)
    small = AS._solve_compressed_fluid_system(absolute, fill(1e-10 + 0im, 3))
    @test small.history.isconverged
    @test small.history.iters == 0
end

@testset "Fluid hierarchical vector workspaces preserve operators" begin
    source = mesh(Sphere(1.0); method = :full, resolution = 0.5, qorder = 4).data
    target = mesh(Sphere(0.5); method = :full, resolution = 0.4, qorder = 4).data
    op = AS.Inti.Helmholtz(; k = 0.3, dim = 3)
    compression = (method = :hmatrix, tol = 1e-9)
    S, D = AS._fluid_layer_operators(op, source, source, (method = :dim,); compression)
    cross = AS._fluid_layer_operators(op, target, source,
        (method = :dim, target_location = :inside); compression)
    @test AS.HMatrices.rowperm(S) != collect(1:length(source))
    @test any(leaf -> AS.HMatrices.data(leaf) isa AS.HMatrices.RkMatrix, AS.HMatrices.leaves(S))
    @test length(target) != length(source)
    x = ComplexF64[cis(i/7) for i in 1:length(source)]
    saved = copy(x)
    for A in (S, D, cross...)
        wrapped = AS._fluid_operator_workspace(A)
        for (alpha, beta) in ((1, 0), (0.3+0.7im, -0.4+0.2im), (0, 1))
            initial = ComplexF64[cis(i/11) for i in 1:size(A, 1)]
            expected = alpha*(A*x) + beta*initial
            actual = copy(initial)
            mul!(actual, wrapped, x, alpha, beta)
            @test actual≈expected rtol=2e-12 atol=1e-13
            @test x == saved
        end
        output = fill(ComplexF64(NaN), size(A, 1))
        mul!(output, wrapped, x)
        @test output≈A*x rtol=2e-12 atol=1e-13
        fill!(output, 1)
        mul!(output, wrapped, fill(ComplexF64(NaN), length(x)), 0, 1)
        @test all(isone, output)
    end
    F = lu(S; rtol = 1e-11)
    wrapped = AS._fluid_inverse_workspace(F)
    for rhs in (x, conj.(x))
        expected, actual = copy(rhs), copy(rhs)
        ldiv!(F, expected)
        ldiv!(wrapped, actual)
        @test actual ≈ expected rtol = 2e-11
    end
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 1), AS._fluid_operator_workspace(S), x)
    @test_throws DimensionMismatch ldiv!(wrapped, zeros(ComplexF64, 1))

    # LU compression samples both columns and conjugate rows of products in
    # hierarchical order. Exercise nonzero subtree offsets and low-rank additions.
    hm = AS.HMatrices
    children = hm.children(S)
    pairs = [(children[2, j], children[j, 1]) for j in axes(children, 2)]
    m, n = size(first(pairs)[1], 1), size(first(pairs)[2], 2)
    @test first(hm.rowrange(first(pairs)[1])) > 1
    R = hm.RkMatrix(ComplexF64[cis(i*j/13) for i in 1:m, j in 1:2],
        ComplexF64[cis(i*j/17) for i in 1:n, j in 1:2])
    P = hm.RkMatrix(ComplexF64[cis(i/19) for i in 1:m, j in 1:1],
        ComplexF64[cis(i/23) for i in 1:n, j in 1:1])
    for products in (pairs, empty(pairs)), lowrank in ((R, P), (nothing, P))

        multiplier = -0.7 + 0.4im
        operator = hm.MulLinearOp{ComplexF64}(lowrank..., products, multiplier)
        expected = zeros(ComplexF64, m, n)
        for (A, B) in products
            expected .+= multiplier .*
                         (Matrix(A; global_index = false)*Matrix(B; global_index = false))
        end
        for term in lowrank
            term === nothing || (expected .+= Matrix(term))
        end
        capacity = max(size(S)...)
        workspace = AS._FluidACAProduct(operator, zeros(ComplexF64, capacity),
            zeros(ComplexF64, capacity), zeros(ComplexF64, capacity))
        for (actual, reference) in ((workspace, expected), (workspace', expected'))
            output = fill(ComplexF64(NaN), size(actual, 1))
            for j in unique([1, cld(size(actual, 2), 2), size(actual, 2), 1])
                hm.getblock!(output, actual, axes(actual, 1), j)
                @test output≈reference[:, j] rtol=2e-12 atol=1e-13
            end
        end
    end
    # A separate rectangular product checks conjugate rows with different lengths.
    rectangular_H = only(filter(part -> part.lmap isa hm.HMatrix, first(cross).maps)).lmap
    rectangular = hm.MulLinearOp{ComplexF64}(nothing, nothing, [(rectangular_H, S)], 0.3im)
    capacity = max(size(S)..., size(rectangular_H)...)
    workspace = AS._FluidACAProduct(rectangular, zeros(ComplexF64, capacity),
        zeros(ComplexF64, capacity), zeros(ComplexF64, capacity))
    expected = 0.3im .*
               (Matrix(rectangular_H; global_index = false)*Matrix(S; global_index = false))
    for (actual, reference) in ((workspace, expected), (workspace', expected'))
        output = zeros(ComplexF64, size(actual, 1))
        for j in (1, size(actual, 2), 1)
            hm.getblock!(output, actual, axes(actual, 1), j)
            @test output≈reference[:, j] rtol=2e-12 atol=1e-13
        end
    end
    # Factorization must preserve the original operator and its physical ordering.
    before = Matrix(S; global_index = false)
    factor = AS._fluid_inverse_workspace(AS._fluid_assembly_lu(S; rtol = 1e-11))
    for rhs in (x, conj.(x))
        expected, actual = copy(rhs), copy(rhs)
        ldiv!(wrapped, expected)
        ldiv!(factor, actual)
        @test actual ≈ expected rtol = 1e-8
        @test norm(S*actual-rhs)/norm(rhs) < 1e-8
    end
    @test Matrix(S; global_index = false) == before
end

@testset "Fluid factorization preserves equations and diagnostics" begin
    coefficients = ComplexF64[4 1+im 0 1; 1-im 5 1 0; 0 1 6 im; 1 0 -im 7]
    rows, cols = [1e-5, 1e3, 2.0, 4e-2], [1e4, 1e-2, 1.0, 100.0]
    A = rows .* coefficients .* transpose(cols)
    original = copy(A)
    for equilibrate in (false, true), condition_limit in (0, 4), norm_floor in (0.0, 0.5)
        factor = AS._factor_fluid_system(A; equilibrate, condition_limit, norm_floor)
        @test A == original
        scaled_A = equilibrate ? A ./ factor.row_norms ./ transpose(factor.col_norms) : A
        if condition_limit > 0
            @test factor.diagnostics.condition_number ≈ cond(A)
            @test factor.diagnostics.scaled_condition_number ≈ cond(scaled_A)
        else
            @test factor.diagnostics.condition_number === nothing
            @test factor.diagnostics.scaled_condition_number === nothing
        end
        for phase in (1.0, im)
            expected = phase .* ComplexF64[1, -2im, 3, 0.5 + im] ./ cols
            b = A * expected
            solved = AS._solve_fluid_system(factor, b)
            @test solved.x ≈ expected rtol = 1e-12
            @test A == original
            # A deliberately perturbed solution keeps the residual above rounding noise.
            x = solved.x .+ 1e-4 ./ cols
            residual = A * x - b
            report = AS._fluid_residual_report(residual, b, factor.row_norms)
            @test report.absolute_residual ≈ norm(residual)
            @test report.relative_residual ≈ norm(residual) / norm(b)
            y = equilibrate ? x .* factor.col_norms : x
            scaled_rhs = equilibrate ? b ./ factor.row_norms : b
            explicit = scaled_A * y - scaled_rhs
            @test report.scaled_absolute_residual ≈ norm(explicit) rtol = 1e-8
            @test report.scaled_relative_residual ≈ norm(explicit) / norm(scaled_rhs) rtol = 1e-8
        end
        b = zeros(ComplexF64, 4)
        @test iszero(norm(AS._solve_fluid_system(factor, b).x))
        zero_report = AS._fluid_residual_report(b, b, factor.row_norms)
        @test zero_report.relative_residual == zero_report.scaled_relative_residual == 0
        nonzero_report = AS._fluid_residual_report(ones(4), b, factor.row_norms)
        @test nonzero_report.relative_residual == nonzero_report.scaled_relative_residual ==
              Inf
    end
    # The factorization must allocate only one matrix-sized workspace after warmup.
    A = Matrix{ComplexF64}(I, 256, 256)
    for equilibrate in (false, true)
        AS._factor_fluid_system(A; equilibrate, condition_limit = 0)
        bytes = @allocated AS._factor_fluid_system(A; equilibrate, condition_limit = 0)
        @test bytes < 1.5 * sizeof(A)
        @test A == I
    end
end

@testset "Full BEM formulation variants" begin
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    tetra = mesh(nodes, triangles; qorder = 2)
    backscatter = AS.SVector(-1.0, 0.0, 0.0)
    for boundary in (Rigid(), PressureRelease())
        solution = bem(tetra, boundary, 0.3;
            formulation = :cbie, compression = (method = :none,))
        @test diagnostics(solution).formulation === :cbie
        @test isfinite(target_strength(solution))
        p, dp, quad = AS.solve_full_bem(boundary, 0.3, tetra.data;
            compression = (method = :none,), return_diagnostics = false)
        @test length(p) == length(dp) == length(quad)
        @test isfinite(AS.target_strength(quad, backscatter, 0.3, p, dp))
    end

    fluid = FluidFilled(1.2, 1.1)
    for (k, formulation) in ((0.3, :muller), (0.3, :cbie), (4.0, :muller))
        @test isfinite(target_strength(bem(tetra, fluid, k; formulation)))
    end
    @test diagnostics(bem(tetra, fluid, 4.0)).derivative_evaluation.exterior === :direct
    p, dp, quad = AS.solve_full_bem(fluid, 0.3, tetra.data; return_diagnostics = false)
    @test length(p) == length(dp) == length(quad)
end

@testset "CBIE elimination preserves the four-trace equations" begin
    function four_trace_matrix(quad, material, k)
        g = material.density_contrast
        exterior = AS.Inti.Helmholtz(; k, dim = 3)
        interior = AS.Inti.Helmholtz(; k = k / material.soundspeed_contrast, dim = 3)
        S, D = AS._fluid_layer_operators(
            exterior, quad, quad, (method = :dim,); regular = false)
        Si, Di = AS._fluid_layer_operators(
            interior, quad, quad, (method = :dim,); regular = false)
        n = length(quad)
        identity = Matrix{ComplexF64}(I, n, n)
        zero_block = zeros(ComplexF64, n, n)
        return [0.5I-D S zero_block zero_block;
                zero_block zero_block 0.5I+Di -Si;
                identity zero_block -identity zero_block;
                zero_block identity zero_block -identity/g]
    end
    sphere = mesh(Sphere(1.0); method = :full, resolution = 0.8, qorder = 4)
    spheroid = mesh(Spheroid(1.5, 1.0); method = :full, resolution = 0.9, qorder = 4)
    for (surface, material, k) in ((sphere, GasFilled(0.0012, 0.23), 0.0138),
        (sphere, GasFilled(0.0012, 0.23), 1.0),
        (sphere, FluidFilled(1000.0, 2.0), 1.0),
        (sphere, FluidFilled(1.0, 1.0), 1.0),
        (spheroid, FluidFilled(1.05, 1.02), 1.0))
        n = length(surface.data)
        original = four_trace_matrix(surface.data, material, k)
        for equilibrate in (false, true)
            factor = AS._factor_fluid_system(original; equilibrate, condition_limit = 0)
            for incidence_angle in (0.0, pi / 3)
                direction = AS._bem3d_incidence_direction(incidence_angle, 0.4)
                p_inc = [cis(k * dot(direction, q.coords)) for q in surface.data]
                d_inc = [im * k * dot(direction, q.normal) * p_inc[i]
                         for (i, q) in enumerate(surface.data)]
                rhs = [zeros(ComplexF64, 2n); -p_inc; -d_inc]
                reference = AS._solve_fluid_system(factor, rhs).x
                solution = bem(surface, material, k; formulation = :cbie,
                    equilibrate, condition_limit = 0, incidence_angle, incidence_azimuth = 0.4)
                p, d = solution.data.p_scat, solution.data.dpdn_scat
                @test p≈reference[1:n] rtol=1e-8 atol=1e-10
                @test d≈reference[(n + 1):(2n)] rtol=1e-8 atol=1e-10
                reconstructed = [p; d; p + p_inc; material.density_contrast .* (d + d_inc)]
                @test AS._linear_residual(original, reconstructed, rhs).relative_residual <
                      1e-9
                report = diagnostics(solution)
                @test report.unknown_count == 2n
                @test report.relative_residual < 1e-9
                @test report.scaled_relative_residual < 1e-9
                for observation in (-direction, direction, AS.SVector(0.0, 0.0, 1.0))
                    expected = AS.far_field(surface.data, observation, k,
                        reference[1:n], reference[(n + 1):(2n)])
                    @test scattering_amplitude(solution; direction = observation)≈expected rtol=1e-8 atol=1e-10
                end
            end
        end
    end
    # The reduced incident forcing must also work when the factorization is reused.
    angles = [0.0, pi / 3, 0.0]
    material = GasFilled(0.0012, 0.23)
    reused = incidence_angle_sweep(sphere, material, 1.0, angles;
        formulation = :cbie, condition_limit = 0)
    fresh = [scattering_amplitude(bem(sphere, material, 1.0;
                 formulation = :cbie, condition_limit = 0, incidence_angle))
             for incidence_angle in angles]
    @test vec(reused.amplitudes) ≈ fresh rtol = 1e-11
    @test reused.amplitudes[1] == reused.amplitudes[end]
end

@testset "Compressed fluid Müller equations and reuse" begin
    surface = mesh(Sphere(1.0); method = :full, resolution = 0.8, qorder = 4)
    compression = (method = :hmatrix, tol = 1e-9)
    for (material, k) in ((FluidFilled(1.05, 1.02), 0.3),
        (FluidFilled(1.05, 1.02), 1.0), (FluidFilled(1.05, 1.02), 2.0),
        (GasFilled(0.0012, 0.23), 0.0138), (GasFilled(0.0012, 0.23), 1.0),
        (FluidFilled(1000.0, 2.0), 1.0), (FluidFilled(1.0, 1.0), 0.3),
        (FluidFilled(1.0001, 1.0001), 0.3),
        (FluidFilled(1.2, 1.0), 0.3), (FluidFilled(1.2, 1.0), 2.0))
        dense = AS._assemble_full_fluid(material, k, surface.data)
        dense_factor = AS._factor_fluid_system(dense.A; condition_limit = 0)
        reference = bem(surface, material, k; incidence_angle = pi/3,
            incidence_azimuth = 0.4, condition_limit = 0)
        for equilibrate in (true, false)
            solution = bem(surface, material, k; compression, equilibrate,
                incidence_angle = pi/3, incidence_azimuth = 0.4, condition_limit = 0)
            report = diagnostics(solution)
            @test report.converged
            @test report.method === :gmres
            @test report.preconditioner === :twolevel
            @test report.condition_number === nothing
            @test report.iterations > 0
            @test report.derivative_evaluation ==
                  diagnostics(reference).derivative_evaluation
            @test solution.data.p_scat≈reference.data.p_scat rtol=2e-5 atol=2e-8
            @test solution.data.dpdn_scat≈reference.data.dpdn_scat rtol=2e-5 atol=2e-8
            direction = AS._bem3d_incidence_direction(pi/3, 0.4)
            pinc = [cis(k*dot(direction, q.coords)) for q in surface.data]
            dinc = [im*k*dot(direction, q.normal)*pinc[j]
                    for (j, q) in enumerate(surface.data)]
            x = [solution.data.p_scat + pinc; solution.data.dpdn_scat + dinc]
            rhs = [pinc; dinc]
            # Independent residual in the original dense, corrected equations.
            residual = AS._fluid_residual_report(dense.A*x-rhs, rhs, dense_factor.row_norms)
            @test residual.scaled_relative_residual < 1e-7
            for observation in (-direction, direction, AS.SVector(0.0, 0.0, 1.0))
                @test scattering_amplitude(solution; direction = observation)≈
                scattering_amplitude(reference; direction = observation) rtol=2e-5 atol=2e-9
            end
            if !equilibrate
                @test report.relative_residual ≈ report.scaled_relative_residual
            end
        end
    end
    angles = [0.0, pi/3, 0.0]
    material = FluidFilled(1.05, 1.02)
    default_compressed = bem(surface, material, 0.3; compression = (method = :hmatrix,))
    default_reference = bem(surface, material, 0.3; condition_limit = 0)
    @test diagnostics(default_compressed).compression.tol == 1e-8
    @test scattering_amplitude(default_compressed) ≈ scattering_amplitude(default_reference) rtol = 2e-5
    sweep = incidence_angle_sweep(
        surface, material, 0.3, angles; compression, condition_limit = 0)
    fresh = [scattering_amplitude(bem(surface, material, 0.3; compression,
                 incidence_angle, condition_limit = 0)) for incidence_angle in angles]
    @test vec(sweep.amplitudes) ≈ fresh rtol = 1e-8
    @test sweep.amplitudes[1] ≈ sweep.amplitudes[end] rtol = 1e-12
    for body in (Sphere(0.01), Spheroid(0.015, 0.01))
        small = mesh(body; method = :full, resolution = 0.008, qorder = 4)
        reference = bem(small, material, 30.0; condition_limit = 0,
            incidence_angle = pi/3, incidence_azimuth = 0.4)
        solution = bem(small, material, 30.0; compression, condition_limit = 0,
            incidence_angle = pi/3, incidence_azimuth = 0.4)
        @test diagnostics(solution).converged
        @test solution.data.p_scat ≈ reference.data.p_scat rtol = 2e-5
        @test solution.data.dpdn_scat ≈ reference.data.dpdn_scat rtol = 2e-5
        @test scattering_amplitude(solution) ≈ scattering_amplitude(reference) rtol = 2e-5
    end
    for options in ((; formulation = :cbie, compression),
        (; correction = (method = :edge,), compression),
        (; compression = (method = :fmm,)),
        (; compression = (method = :hmatrix, tol = 0.0)),
        (; compression, gmres_kwargs = (; Pl = I)),
        (; gmres_kwargs = (; reltol = 1e-8)))
        @test_throws ArgumentError bem(surface, material, 0.3; options...)
    end
    @test_logs (:warn, r"GMRES did not converge") begin
        failed = bem(surface, material, 0.3; compression,
            gmres_kwargs = (; maxiter = 1, reltol = 1e-14))
        @test !diagnostics(failed).converged
    end
end

@testset "Coupled-region target strength" begin
    nodes = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4]
    tetra = mesh(nodes, triangles; qorder = 2)
    region = bem([tetra], [FluidFilled(1.2, 1.1)], 0.3)
    @test isfinite(target_strength(region))
    @test target_strength(region) ≈ target_strength(scattering_amplitude(region))
end
