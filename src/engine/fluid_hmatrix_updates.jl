# Inherited low-rank updates only need restrictions of their factors. Keep those
# restrictions as views until the child is recompressed; retained factors remain
# ordinary HMatrices.RkMatrix objects with independent storage.
struct _FluidLowRankView{A, B}
    A::A
    B::B
end

Base.adjoint(R::_FluidLowRankView) = _FluidLowRankView(R.B, R.A)

function _fluid_aca_column!(out, R::_FluidLowRankView, j, work, add)
    column = view(work, 1:size(R.B, 2))
    column .= conj.(view(R.B, j, :))
    return mul!(out, R.A, column, 1, add)
end

struct _FluidLUUpdate{R, P, V, S} <: AbstractMatrix{ComplexF64}
    R::R
    P::P
    pairs::V
    multiplier::S
    dims::Tuple{Int, Int}
end

Base.size(L::_FluidLUUpdate) = L.dims

struct _FluidLUCompressor{C}
    compressor::C
end

function (c::_FluidLUCompressor)(operator, rows, columns, buffer)
    c.compressor(operator, rows, columns, buffer)
end

#=
The update traversal below is adapted from HMatrices.jl's _hmul!/execute_node!.
It is selected only by our compressor type. LU, planning, dense updates and ACA
remain owned by HMatrices; no methods for dependency-only argument types change.

MIT License
Copyright (c) 2021 Luiz M. Faria <maltezfaria@gmail.com> and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#
function HMatrices._hmul!(C::HMatrices.HMatrix, compressor::_FluidLUCompressor,
        plan, alpha, inherited, buffers, flag)
    products = get(plan, C, Tuple{typeof(C), typeof(C)}[])
    if inherited !== nothing || !isempty(products)
        if HMatrices.isleaf(C) && !HMatrices.isadmissible(C)
            target = HMatrices.data(C)
            for (A, B) in products
                HMatrices._mul_dense!(target, A, B, alpha)
            end
            inherited === nothing ||
                mul!(target, inherited.A, inherited.B', true, true)
        else
            operator = _FluidLUUpdate(
                HMatrices.data(C), inherited, products, alpha, size(C))
            buffer = take!(buffers)
            updated = try
                compressor(operator, axes(operator, 1), axes(operator, 2), buffer)
            finally
                put!(buffers, buffer)
            end
            HMatrices.setdata!(C, updated)
        end
    end
    update = HMatrices.data(C)
    pivot = HMatrices.pivot(C)
    children = HMatrices.children(C)
    for i in axes(children, 1), j in axes(children, 2)

        (flag == 'U' && i > j || flag == 'L' && j > i) && continue
        child = children[i, j]
        restriction = if update === nothing
            nothing
        else
            rows = HMatrices.rowrange(child) .- pivot[1] .+ 1
            columns = HMatrices.colrange(child) .- pivot[2] .+ 1
            _FluidLowRankView(view(update.A, rows, :), view(update.B, columns, :))
        end
        HMatrices._hmul!(child, compressor, plan, alpha, restriction, buffers,
            i == j ? flag : 'N')
    end
    HMatrices.isleaf(C) || HMatrices.setdata!(C, nothing)
    return C
end
