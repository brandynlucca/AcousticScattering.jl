# Local workspaces for columns sampled by ACA during hierarchical LU updates.
# The dependency owns the LU/ACA algorithms, pivoting and compression criteria.
struct _FluidACAProduct{L} <: AbstractMatrix{ComplexF64}
    operator::L
    product::Vector{ComplexF64}
    rank_work::Vector{ComplexF64}
    column_work::Vector{ComplexF64}
end

Base.size(A::_FluidACAProduct) = size(A.operator)

function _fluid_aca_mul!(y, H, x, work, offset)
    if HMatrices.isleaf(H)
        rows = HMatrices.rowrange(H) .- offset[1]
        columns = HMatrices.colrange(H) .- offset[2]
        data = HMatrices.data(H)
        target, source = view(y, rows), view(x, columns)
        if data isa HMatrices.RkMatrix
            tmp = view(work, 1:size(data.A, 2))
            mul!(tmp, data.B', source)
            mul!(target, data.A, tmp, 1, 1)
        elseif data isa LinearAlgebra.Adjoint{<:Any, <:HMatrices.RkMatrix}
            R = parent(data)
            tmp = view(work, 1:size(R.A, 2))
            mul!(tmp, R.A', source)
            mul!(target, R.B, tmp, 1, 1)
        else
            mul!(target, data, source, 1, 1)
        end
    else
        for child in HMatrices.children(H)
            _fluid_aca_mul!(y, child, x, work, offset)
        end
    end
    return y
end

function _fluid_aca_column!(out, R::HMatrices.RkMatrix, j, work, add)
    column = view(work, 1:size(R.B, 2))
    column .= conj.(view(R.B, j, :))
    return mul!(out, R.A, column, 1, add)
end

function _fluid_aca_column!(
        out, R::LinearAlgebra.Adjoint{<:Any, <:HMatrices.RkMatrix}, j, work, add)
    return _fluid_aca_adjoint_column!(out, parent(R), j, work, add)
end

function _fluid_aca_adjoint_column!(out, R, j, work, add)
    column = view(work, 1:size(R.A, 2))
    column .= conj.(view(R.A, j, :))
    return mul!(out, R.B, column, 1, add)
end

function _fluid_aca_column!(
        out, H::Union{HMatrices.HMatrix, LinearAlgebra.Adjoint{<:Any, <:HMatrices.HMatrix}},
        j, work, add = false)
    add && error("hierarchical column addition is unsupported")
    _fluid_aca_hcolumn!(out, H, j, work, first(HMatrices.rowrange(H))-1)
    return out
end

function _fluid_aca_hcolumn!(out, H, j, work, row_offset)
    if HMatrices.hasdata(H)
        data = HMatrices.data(H)
        rows = HMatrices.rowrange(H) .- row_offset
        local_j = j - first(HMatrices.colrange(H)) + 1
        target = view(out, rows)
        if data isa
           Union{HMatrices.RkMatrix, LinearAlgebra.Adjoint{<:Any, <:HMatrices.RkMatrix}}
            _fluid_aca_column!(target, data, local_j, work, false)
        else
            target .= view(data, :, local_j)
        end
    end
    for child in HMatrices.children(H)
        j in HMatrices.colrange(child) &&
            _fluid_aca_hcolumn!(out, child, j, work, row_offset)
    end
    return out
end

function _fluid_aca_product_column!(out, W, j, ::Val{transposed}) where {transposed}
    L = W.operator
    fill!(out, 0)
    for (left, right) in L.pairs
        A, B = transposed ? (right', left') : (left, right)
        tmp = view(W.product, 1:size(B, 1))
        fill!(tmp, 0)
        _fluid_aca_column!(tmp, B, j + first(HMatrices.colrange(B))-1, W.column_work)
        _fluid_aca_mul!(out, A, tmp, W.rank_work, HMatrices.offset(A))
    end
    LinearAlgebra.rmul!(out, transposed ? conj(L.multiplier) : L.multiplier)
    for R in (L.R, L.P)
        R === nothing && continue
        _fluid_aca_column!(out, transposed ? R' : R, j, W.column_work, true)
    end
    return out
end

function HMatrices.getblock!(out, W::_FluidACAProduct, rows, j::Int)
    rows == axes(W, 1) || throw(DimensionMismatch("ACA product rows"))
    return _fluid_aca_product_column!(out, W, j, Val(false))
end

function HMatrices.getblock!(out, W::LinearAlgebra.Adjoint{<:Any, <: _FluidACAProduct}, rows, j::Int)
    rows == axes(W, 1) || throw(DimensionMismatch("ACA adjoint product rows"))
    return _fluid_aca_product_column!(out, parent(W), j, Val(true))
end

struct _FluidAssemblyCompressor{C}
    compressor::C
end

function (compress::_FluidAssemblyCompressor)(operator, rows, columns, buffer)
    if operator isa Union{HMatrices.MulLinearOp, _FluidLUUpdate}
        n = maximum((size(A, 2) for (A, _) in operator.pairs); init = 0)
        capacity = max(n, size(operator)...)
        workspace = _FluidACAProduct(operator, zeros(ComplexF64, n),
            zeros(ComplexF64, capacity), zeros(ComplexF64, capacity))
        return compress.compressor(workspace, rows, columns, buffer)
    end
    return compress.compressor(operator, rows, columns, buffer)
end

function _fluid_assembly_lu(A; rtol)
    lu(A,
        _FluidLUCompressor(_FluidAssemblyCompressor(HMatrices.PartialACA(; rtol))))
end
