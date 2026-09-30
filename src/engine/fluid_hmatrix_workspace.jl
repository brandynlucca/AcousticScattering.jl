# Reuse vector workspaces when applying fixed fluid operators. Hierarchical storage
# and factorization remain owned by HMatrices; no dense global matrix is formed.
struct _FluidHMatrixWorkspace{H} <: AbstractMatrix{ComplexF64}
    matrix::H
    input::Vector{ComplexF64}
    output::Vector{ComplexF64}
    rank_work::Vector{ComplexF64}
    groups::Vector{Vector{H}}
    outputs::Vector{Vector{ComplexF64}}
    rank_works::Vector{Vector{ComplexF64}}
end

struct _FluidHLUWorkspace{H}
    factors::H
    work::Vector{ComplexF64}
    rank_work::Vector{ComplexF64}
end

function _fluid_rank_workspace(H)
    rank = maximum(HMatrices.leaves(H); init = 0) do leaf
        data = HMatrices.data(leaf)
        data isa HMatrices.RkMatrix ? size(data.A, 2) : 0
    end
    return zeros(ComplexF64, rank)
end

_fluid_operator_workspace(A) = A
function _fluid_operator_workspace(A::HMatrices.HMatrix)
    leaves = HMatrices.leaves(A)
    workers = min(Threads.nthreads(), length(leaves))
    groups = [typeof(A)[] for _ in 1:workers]
    costs = zeros(Int, workers)
    cost(leaf) = begin
        data = HMatrices.data(leaf)
        data isa HMatrices.RkMatrix ? size(data.A, 2)*sum(size(data)) : prod(size(data))
    end
    for leaf in sort(leaves; by = cost, rev = true)
        j = argmin(costs)
        push!(groups[j], leaf)
        costs[j] += cost(leaf)
    end
    rank_work = _fluid_rank_workspace(A)
    return _FluidHMatrixWorkspace(A, zeros(ComplexF64, size(A, 2)),
        zeros(ComplexF64, size(A, 1)), rank_work, groups,
        [zeros(ComplexF64, size(A, 1)) for _ in groups],
        [similar(rank_work) for _ in groups])
end
function _fluid_operator_workspace(A::LinearMaps.WrappedMap)
    return LinearMap(_fluid_operator_workspace(A.lmap))
end
function _fluid_operator_workspace(A::LinearMaps.LinearCombination)
    return reduce(+, map(_fluid_operator_workspace, A.maps))
end
function _fluid_inverse_workspace(F)
    return _FluidHLUWorkspace(F.factors, zeros(ComplexF64, size(F.factors, 1)),
        _fluid_rank_workspace(F.factors))
end
_fluid_inverse_workspace(::Nothing) = nothing

Base.size(A::_FluidHMatrixWorkspace) = size(A.matrix)

# Work in the hierarchical (permuted) ordering. In particular, retain vector BLAS
# operations in triangular updates, instead of representing a vector as an n×1 matrix.
function _fluid_hmul_local!(y, H, x, work, alpha)
    if HMatrices.isleaf(H)
        rows, columns = HMatrices.rowrange(H), HMatrices.colrange(H)
        data = HMatrices.data(H)
        target, source = view(y, rows), view(x, columns)
        if data isa HMatrices.RkMatrix
            tmp = view(work, 1:size(data.A, 2))
            mul!(tmp, data.B', source)
            mul!(target, data.A, tmp, alpha, 1)
        else
            mul!(target, data, source, alpha, 1)
        end
    else
        for child in HMatrices.children(H)
            _fluid_hmul_local!(y, child, x, work, alpha)
        end
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::_FluidHMatrixWorkspace,
        x::AbstractVector, alpha::Number = 1, beta::Number = 0)
    length(x) == size(A, 2) && length(y) == size(A, 1) ||
        throw(DimensionMismatch("fluid hierarchical product"))
    if iszero(alpha)
        return iszero(beta) ? fill!(y, 0) : LinearAlgebra.rmul!(y, beta)
    end
    rows, columns = HMatrices.rowperm(A.matrix), HMatrices.colperm(A.matrix)
    for i in eachindex(columns)
        A.input[i] = alpha*x[columns[i]]
    end
    if iszero(beta)
        fill!(A.output, 0)
    else
        for i in eachindex(rows)
            A.output[i] = beta*y[rows[i]]
        end
    end
    if length(A.groups) == 1
        _fluid_hmul_local!(A.output, A.matrix, A.input, A.rank_work, 1)
    else
        # Assign one persistent buffer to each task, independently of thread IDs.
        # Reduce in a fixed order after every task finishes.
        Threads.@threads for j in eachindex(A.groups)
            output = A.outputs[j]
            fill!(output, 0)
            for leaf in A.groups[j]
                _fluid_hmul_local!(output, leaf, A.input, A.rank_works[j], 1)
            end
        end
        for output in A.outputs
            A.output .+= output
        end
    end
    for i in eachindex(rows)
        y[rows[i]] = A.output[i]
    end
    return y
end

function _fluid_htriangular!(H, y, work, lower)
    if HMatrices.isleaf(H)
        block = HMatrices.data(H)
        triangular = lower ? LinearAlgebra.UnitLowerTriangular(block) :
                     LinearAlgebra.UpperTriangular(block)
        LinearAlgebra.ldiv!(triangular, view(y, HMatrices.colrange(H)))
    else
        children = HMatrices.children(H)
        count = size(children, 1)
        for i in (lower ? (1:count) : (count:-1:1))
            for j in (lower ? (1:(i - 1)) : ((i + 1):count))
                _fluid_hmul_local!(y, children[i, j], y, work, -1)
            end
            _fluid_htriangular!(children[i, i], y, work, lower)
        end
    end
    return y
end

function LinearAlgebra.ldiv!(F::_FluidHLUWorkspace, y::AbstractVector)
    length(y) == length(F.work) || throw(DimensionMismatch("fluid hierarchical inverse"))
    rows, columns = HMatrices.rowperm(F.factors), HMatrices.colperm(F.factors)
    for i in eachindex(columns)
        F.work[i] = y[columns[i]]
    end
    _fluid_htriangular!(F.factors, F.work, F.rank_work, true)
    _fluid_htriangular!(F.factors, F.work, F.rank_work, false)
    for i in eachindex(rows)
        y[rows[i]] = F.work[i]
    end
    return y
end
