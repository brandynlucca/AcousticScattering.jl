# Matrix workspaces are temporary, owned by coarse-space setup. They reuse the
# stored hierarchy and permit BLAS matrix products without per-leaf temporaries.
struct _FluidHBatchWorkspace{H} <: AbstractMatrix{ComplexF64}
    matrix::H
    input::Matrix{ComplexF64}
    output::Matrix{ComplexF64}
    rank_work::Matrix{ComplexF64}
end

struct _FluidHLUBatchWorkspace{H}
    factors::H
    work::Matrix{ComplexF64}
    rank_work::Matrix{ComplexF64}
end

Base.size(A::_FluidHBatchWorkspace) = size(A.matrix)

_fluid_batch_workspace(A, width) = A
function _fluid_batch_workspace(A::_FluidHMatrixWorkspace, width)
    return _FluidHBatchWorkspace(A.matrix,
        zeros(ComplexF64, size(A, 2), width), zeros(ComplexF64, size(A, 1), width),
        zeros(ComplexF64, length(A.rank_work), width))
end
function _fluid_batch_workspace(F::_FluidHLUWorkspace, width)
    return _FluidHLUBatchWorkspace(F.factors,
        zeros(ComplexF64, length(F.work), width),
        zeros(ComplexF64, length(F.rank_work), width))
end
function _fluid_batch_workspace(A::LinearMaps.WrappedMap, width)
    LinearMap(_fluid_batch_workspace(A.lmap, width))
end
function _fluid_batch_workspace(A::LinearMaps.LinearCombination, width)
    reduce(+, map(part -> _fluid_batch_workspace(part, width), A.maps))
end

function _fluid_hmul_batch!(Y, H, X, work, alpha)
    if HMatrices.isleaf(H)
        target = view(Y, HMatrices.rowrange(H), :)
        source = view(X, HMatrices.colrange(H), :)
        data = HMatrices.data(H)
        if data isa HMatrices.RkMatrix
            tmp = view(work, 1:size(data.A, 2), 1:size(X, 2))
            mul!(tmp, data.B', source)
            mul!(target, data.A, tmp, alpha, 1)
        else
            mul!(target, data, source, alpha, 1)
        end
    else
        for child in HMatrices.children(H)
            _fluid_hmul_batch!(Y, child, X, work, alpha)
        end
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, A::_FluidHBatchWorkspace,
        X::AbstractMatrix, alpha::Number = 1, beta::Number = 0)
    width = size(X, 2)
    size(X, 1) == size(A, 2) && size(Y) == (size(A, 1), width) &&
    width <= size(A.input, 2) ||
        throw(DimensionMismatch("fluid hierarchical batch product"))
    if iszero(alpha)
        return iszero(beta) ? fill!(Y, 0) : LinearAlgebra.rmul!(Y, beta)
    end
    rows, columns = HMatrices.rowperm(A.matrix), HMatrices.colperm(A.matrix)
    input, output = view(A.input, :, 1:width), view(A.output, :, 1:width)
    for j in 1:width, i in eachindex(columns)

        input[i, j] = alpha*X[columns[i], j]
    end
    if iszero(beta)
        fill!(output, 0)
    else
        for j in 1:width, i in eachindex(rows)

            output[i, j] = beta*Y[rows[i], j]
        end
    end
    _fluid_hmul_batch!(output, A.matrix, input, A.rank_work, 1)
    for j in 1:width, i in eachindex(rows)

        Y[rows[i], j] = output[i, j]
    end
    return Y
end

function _fluid_htriangular_batch!(H, Y, work, lower)
    if HMatrices.isleaf(H)
        block = HMatrices.data(H)::Matrix{ComplexF64}
        triangular = lower ? LinearAlgebra.UnitLowerTriangular(block) :
                     LinearAlgebra.UpperTriangular(block)
        LinearAlgebra.ldiv!(triangular, view(Y, HMatrices.colrange(H), :))
    else
        children = HMatrices.children(H)
        count = size(children, 1)
        for i in (lower ? (1:count) : (count:-1:1))
            for j in (lower ? (1:(i - 1)) : ((i + 1):count))
                _fluid_hmul_batch!(Y, children[i, j], Y, work, -1)
            end
            _fluid_htriangular_batch!(children[i, i], Y, work, lower)
        end
    end
    return Y
end

function LinearAlgebra.ldiv!(F::_FluidHLUBatchWorkspace, Y::AbstractMatrix)
    width = size(Y, 2)
    size(Y, 1) == size(F.work, 1) && width <= size(F.work, 2) ||
        throw(DimensionMismatch("fluid hierarchical batch inverse"))
    rows, columns = HMatrices.rowperm(F.factors), HMatrices.colperm(F.factors)
    work = view(F.work, :, 1:width)
    for j in 1:width, i in eachindex(columns)

        work[i, j] = Y[columns[i], j]
    end
    _fluid_htriangular_batch!(F.factors, work, F.rank_work, true)
    _fluid_htriangular_batch!(F.factors, work, F.rank_work, false)
    for j in 1:width, i in eachindex(rows)

        Y[rows[i], j] = work[i, j]
    end
    return Y
end

function _fluid_coarse_images!(output, system, basis, rows, cols, scaled_A)
    if hasproperty(system, :coarse_interactions)
        return _region_coarse_images!(output, system.coarse_interactions, basis, rows, cols)
    end
    for j in axes(basis, 2)
        mul!(view(output, :, j), scaled_A, view(basis, :, j))
    end
    return output
end

function _region_coarse_images!(output, interactions, basis, rows, cols; batchsize = 18)
    # Eighteen columns cover the pressure/flux harmonics of one interface. Keep
    # each input buffer below 8 MiB on larger meshes, with a minimum of one column.
    batchsize > 0 || throw(ArgumentError("coarse batch size must be positive"))
    n = length(rows) ÷ 2
    width = min(batchsize, size(basis, 2), max(1, (8*1024^2) ÷ (16n)))
    output .= basis ./ cols
    for part in interactions
        S, D, K, H = map(A -> _fluid_batch_workspace(A, width), (
            part.S, part.D, part.K, part.H))
        factor = _fluid_batch_workspace(part.factor, width)
        prepared = merge(part, (; S, D, K, H, factor))
        input_p = zeros(ComplexF64, length(part.columns), width)
        input_v = similar(input_p)
        pressure_work = zeros(ComplexF64, length(part.rows), width)
        flux_work = similar(pressure_work)
        for firstcolumn in 1:width:size(basis, 2)
            columns = firstcolumn:min(firstcolumn + width - 1, size(basis, 2))
            active = 1:length(columns)
            p, v = view(input_p, :, active), view(input_v, :, active)
            p .= view(basis, part.columns, columns) ./ view(cols, part.columns)
            v .= view(basis, part.columns .+ n, columns) ./ view(cols, part.columns .+ n)
            # Skip exactly zero source blocks, including their sparse corrections.
            # Never discard merely small entries introduced by orthogonalization.
            all(iszero, p) && all(iszero, v) && continue
            chunk = merge(prepared,
                (; pressure_work = view(pressure_work, :, active),
                    flux_work = view(flux_work, :, active)))
            _apply_region_interaction!(view(output, part.rows, columns),
                view(output, part.rows .+ n, columns), chunk, p, v)
        end
    end
    output ./= rows
    return output
end

# Guarded mixed-precision factorization for opt-in dense fluid BEM solves.
module MixedRefinement
using LinearAlgebra

mutable struct RefinedLU
    A::Matrix{ComplexF64} # Borrowed, read-only original equations.
    rows::Vector{Float64}
    columns::Vector{Float64}
    matrix_norm::Float64
    factor::Any
    precision::Symbol
    reason::Symbol
    reciprocal_condition::Float64
end

function promote!(F::RefinedLU, reason)
    # Release the low-precision factors before allocating the double workspace.
    F.factor = nothing
    scaled = F.A ./ F.rows ./ transpose(F.columns)
    F.factor = lu!(scaled)
    F.precision = :double
    F.reason = reason
    return F
end

function factorize_refined(A::Matrix{ComplexF64}; condition_guard = 10eps(Float32))
    n, m = size(A)
    n == m && n > 0 || throw(DimensionMismatch("expected a nonempty square matrix"))
    all(isfinite, A) || throw(ArgumentError("matrix must be finite"))
    isfinite(condition_guard) && condition_guard >= 0 ||
        throw(ArgumentError("invalid condition guard"))
    rows = vec(maximum(abs, A; dims = 2))
    rows[iszero.(rows)] .= 1
    columns, sums = zeros(n), zeros(n)
    @inbounds for j in 1:n, i in 1:n

        columns[j] = max(columns[j], abs(A[i, j] / rows[i]))
        sums[i] += abs(A[i, j])
    end
    all(isfinite, sums) || throw(ArgumentError("matrix norm overflow"))
    columns[iszero.(columns)] .= 1
    low = Matrix{ComplexF32}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n

        low[i, j] = A[i, j] / rows[i] / columns[j]
    end
    F = RefinedLU(A, rows, columns, maximum(sums), nothing, :single, :none, NaN)
    all(isfinite, low) || return promote!(F, :conversion)
    anorm = opnorm(low, Inf)
    F.factor = lu!(low; check = false)
    issuccess(F.factor) || return promote!(F, :single_singular)
    F.reciprocal_condition = Float64(LAPACK.gecon!('I', F.factor.factors, anorm))
    if !isfinite(F.reciprocal_condition) || F.reciprocal_condition <= condition_guard
        promote!(F, :condition)
    end
    return F
end

function correction(F, rhs)
    scaled = rhs ./ F.rows
    magnitude = norm(scaled, Inf)
    isfinite(magnitude) || return fill(ComplexF64(NaN), length(rhs))
    iszero(magnitude) && return zeros(ComplexF64, length(rhs))
    # Normalize each RHS/residual independently to avoid narrowing overflow or
    # losing a uniformly tiny residual to Float32 underflow.
    work = F.precision === :single ? ComplexF32.(scaled ./ magnitude) : scaled ./ magnitude
    value = ComplexF64.(F.factor \ work) .* magnitude ./ F.columns
    return value
end

function backward_errors(F, x, b)
    residual = b - F.A * x
    denominator = abs.(b)
    absx = abs.(x)
    @inbounds for j in axes(F.A, 2), i in axes(F.A, 1)

        denominator[i] += abs(F.A[i, j]) * absx[j]
    end
    all(isfinite, residual) && all(isfinite, denominator) ||
        return (; residual, normwise = Inf, componentwise = Inf)
    componentwise = 0.0
    for i in eachindex(residual)
        ratio = iszero(denominator[i]) ?
                (iszero(residual[i]) ? 0.0 : Inf) : abs(residual[i]) / denominator[i]
        componentwise = max(componentwise, ratio)
    end
    scale = F.matrix_norm * norm(x, Inf) + norm(b, Inf)
    normwise = !isfinite(scale) ? Inf :
               iszero(scale) ? (iszero(norm(residual, Inf)) ? 0.0 : Inf) :
               norm(residual, Inf) / scale
    return (; residual, normwise, componentwise)
end

function solve_refined(F::RefinedLU, b::Vector{ComplexF64}; tolerance = 1e-13,
        maxiter = 8)
    length(b) == size(F.A, 1) || throw(DimensionMismatch("right-hand side length"))
    all(isfinite, b) || throw(ArgumentError("right-hand side must be finite"))
    isfinite(tolerance) && 0 < tolerance < 1 || throw(ArgumentError("invalid tolerance"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    x = correction(F, b)
    history = NamedTuple[]
    corrections, attempts, stalls = 0, 0, 0
    previous = Inf
    while true
        errors = backward_errors(F, x, b)
        metric = max(errors.normwise, errors.componentwise)
        push!(history, (; precision = F.precision, errors.normwise, errors.componentwise))
        if isfinite(metric) && metric <= tolerance
            return (; x, accepted = true, precision = F.precision, reason = F.reason,
                corrections, history, errors.normwise, errors.componentwise)
        end
        stalls = metric >= previous * 0.9 ? stalls + 1 : 0
        reason = !isfinite(metric) ? :nonfinite :
                 attempts >= maxiter ? :iteration_limit : stalls >= 2 ? :stagnation : :none
        if reason !== :none
            F.precision === :double &&
                error("double-precision fallback failed backward-error acceptance: $reason")
            promote!(F, reason)
            x = correction(F, b)
            previous, attempts, stalls = Inf, 0, 0
            continue
        end
        delta = correction(F, errors.residual)
        updated = x + delta
        if updated == x
            F.precision === :double && error("double-precision correction stagnated")
            promote!(F, :stagnation)
            x = correction(F, b)
            previous, attempts, stalls = Inf, 0, 0
            continue
        end
        x = updated
        previous = metric
        corrections += 1
        attempts += 1
    end
end

end
