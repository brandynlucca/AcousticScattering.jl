using AcousticScattering
using LinearAlgebra
using SparseArrays
using Test

@testset "Fluid setup equilibration without local factors" begin
    AS = AcousticScattering
    blocks = (ComplexF64[4 1+im 0; 2im 3 0.1; 0 1 2],
        ComplexF64[0 0; 0 0], ComplexF64[1e-12 1e3; 1e-8 1e4])
    for block in blocks, floor in (0.0, eps(Float64), 0.5)

        original = copy(block)
        rows, cols = AS._fluid_block_norms(block, floor)
        expected_rows = max.(vec(maximum(abs, block; dims = 2)), floor)
        expected_rows[iszero.(expected_rows)] .= 1
        expected_cols = max.(vec(maximum(abs, block ./ expected_rows; dims = 1)), floor)
        expected_cols[iszero.(expected_cols)] .= 1
        @test rows == expected_rows
        @test cols == expected_cols
        @test block == original
    end
end

@testset "Fluid hierarchical setup workspaces" begin
    AS, hm = AcousticScattering, AcousticScattering.HMatrices
    points = [AS.SVector(mod(37i, 128)/128, 0.0, 0.0) for i in 1:128]
    kernel(x, y) = ComplexF64((x == y ? 8 : 0) + cis(0.5norm(x-y))/(1+norm(x-y)))
    tree = hm.ClusterTree(points, hm.CardinalitySplitter(; nmax = 8))
    H = hm.assemble_hmatrix(hm.KernelMatrix(kernel, points, points), tree, tree;
        adm = hm.StrongAdmissibilityStd(3), comp = hm.PartialACA(; rtol = 1e-12),
        threads = false, distributed = false)
    dense = Matrix(H)
    @test hm.rowperm(H) != collect(1:128)
    @test any(leaf -> hm.data(leaf) isa hm.RkMatrix, hm.leaves(H))
    target = [p + AS.SVector(2.0, 0, 0) for p in points[1:83]]
    targettree = hm.ClusterTree(target, hm.CardinalitySplitter(; nmax = 8))
    rectangular = hm.assemble_hmatrix(
        hm.KernelMatrix(kernel, target, points), targettree, tree;
        adm = hm.StrongAdmissibilityStd(3), comp = hm.PartialACA(; rtol = 1e-12),
        threads = false, distributed = false)
    correction = sparse([1, 30, 83], [4, 2, 127], ComplexF64[0.3im, 0.2, -0.1], 83, 128)
    corrected = AS.LinearMap(AS._fluid_operator_workspace(rectangular)) +
                AS.LinearMap(correction)
    cases = ((AS._fluid_operator_workspace(H), dense),
        (AS._fluid_operator_workspace(rectangular), Matrix(rectangular)),
        (corrected, Matrix(rectangular)+Matrix(correction)))
    for (operator, reference) in cases
        batch = AS._fluid_batch_workspace(operator, 7)
        for width in (1, 7, 3, 7), (alpha, beta) in ((1, 0), (0.3+0.2im, -0.4im))

            X = ComplexF64[cis(i*j/17) for i in 1:128, j in 1:width]
            initial = ComplexF64[cis(i*j/11) for i in axes(reference, 1), j in 1:width]
            saved = copy(X)
            expected = alpha*(reference*X)+beta*initial
            actual = copy(initial)
            mul!(actual, batch, X, alpha, beta)
            @test actual≈expected rtol=3e-12 atol=2e-13
            @test X == saved
        end
    end
    batch = AS._fluid_batch_workspace(first(first(cases)), 7)
    X = fill(ComplexF64(NaN), 128, 3)
    Y = ones(ComplexF64, 128, 3)
    mul!(Y, batch, X, 0, 1)
    @test all(isone, Y)
    mul!(Y, batch, X, 0, 0)
    @test all(iszero, Y)
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 128, 8), batch, zeros(ComplexF64, 128, 8))
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 127, 3), batch, zeros(ComplexF64, 128, 3))

    old = AS._FluidAssemblyCompressor(hm.PartialACA(; rtol = 1e-12))
    newer = AS._FluidLUCompressor(old)
    oldfactor, newfactor = lu(H, old), AS._fluid_assembly_lu(H; rtol = 1e-12)
    @test Matrix(H) == dense
    @test Matrix(newfactor.factors; global_index = false) ≈
          Matrix(oldfactor.factors; global_index = false) rtol = 3e-11
    inverse = AS._fluid_batch_workspace(AS._fluid_inverse_workspace(newfactor), 7)
    for width in (7, 1, 3, 7)
        rhs = ComplexF64[cis(i*j/13) for i in 1:128, j in 1:width]
        actual = copy(rhs)
        ldiv!(inverse, actual)
        @test actual ≈ dense\rhs rtol = 5e-10
        @test norm(dense*actual-rhs)/norm(rhs) < 1e-10
    end
    @test_throws DimensionMismatch ldiv!(inverse, zeros(ComplexF64, 128, 8))
    @test_throws DimensionMismatch ldiv!(inverse, zeros(ComplexF64, 127, 3))
    # Compare update traversal for full and triangular block selections, complex
    # multipliers, original-matrix preservation and retained independent factors.
    for flag in ('N', 'U', 'L')
        original = deepcopy(H)
        expected, actual = deepcopy(H), deepcopy(H)
        hm.hmul!(expected, H, H, 0.3+0.2im, 0.7, old, nothing, flag)
        hm.hmul!(actual, H, H, 0.3+0.2im, 0.7, newer, nothing, flag)
        @test Matrix(actual) ≈ Matrix(expected) rtol = 2e-10
        @test Matrix(H) == Matrix(original)
    end
    jobs = [Threads.@spawn AS._fluid_assembly_lu(H; rtol = 1e-12) for _ in 1:2]
    for job in jobs
        factor = fetch(job)
        rhs = ComplexF64[cis(i/13) for i in 1:128]
        actual = copy(rhs)
        ldiv!(factor, actual)
        @test actual ≈ dense\rhs rtol = 5e-10
    end
    @test Matrix(H) == dense
end

@testset "Coupled coarse images preserve vector equations" begin
    AS = AcousticScattering
    surfaces = [mesh(Sphere(1.0); method = :full, resolution = 1.0, qorder = 4),
        mesh(Sphere(0.4); method = :full, resolution = 0.4, qorder = 4)]
    materials = [FluidFilled(1.05, 1.02), GasFilled(0.0012, 0.23)]
    for k in (0.03, 1.0)
        system = AS._assemble_region_bem(surfaces, materials, k;
            compression = (method = :hmatrix, tol = 1e-9))
        factor = AS._factor_full_fluid(system; equilibrate = true, condition_limit = 0)
        basis = factor.preconditioner.basis
        original = copy(basis)
        expected = hcat([factor.scaled_A*view(basis, :, j) for j in axes(basis, 2)]...)
        for width in (1, 7, 18, 40)
            actual = similar(basis)
            AS._region_coarse_images!(actual, system.coarse_interactions, basis,
                factor.row_norms, factor.col_norms; batchsize = width)
            @test actual ≈ expected rtol = 5e-12
            @test basis == original
        end
        @test_throws ArgumentError AS._region_coarse_images!(similar(basis),
            system.coarse_interactions, basis, factor.row_norms, factor.col_norms; batchsize = 0)
    end
end
