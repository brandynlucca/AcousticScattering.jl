const _VOLUME_ITERATIVE_DOFS = 60_000

# An incomplete LU factorization kept in single precision, applied to double precision vectors.
struct _SinglePrecisionILU{F}
    factor::F
    input::Vector{ComplexF32}
    output::Vector{ComplexF32}
end

function LinearAlgebra.ldiv!(y, P::_SinglePrecisionILU, x)
    P.input .= x
    LinearAlgebra.ldiv!(P.output, P.factor, P.input)
    y .= P.output
    return y
end

LinearAlgebra.ldiv!(P::_SinglePrecisionILU, x) = LinearAlgebra.ldiv!(x, P, x)

# Linear solver of the active dofs whose factorizations are built on first use and reused for every right-hand side.
mutable struct _VolumeLinearSolver
    matrix::SparseMatrixCSC{ComplexF64, <:Integer}
    shifted::Union{Nothing, SparseMatrixCSC{ComplexF64, <:Integer}}
    ilu_tolerance::Float64
    tolerance::Float64
    direct::Any
    incomplete::Any
    mode::Symbol
    dtn::Any
    operator::Any
end

# Product of the system operator with `x`, which adds the Dirichlet-to-Neumann term when it is applied without its dense matrix.
function _apply(solver::_VolumeLinearSolver, x)
    y = solver.matrix * x
    if solver.dtn !== nothing
        positions = solver.dtn.positions
        y[positions] .-= _dtn_apply(solver.dtn, x[positions])
    end
    return y
end

function _volume_operator(matrix, dtn, solver_ref)
    dtn === nothing && return matrix
    n = size(matrix, 1)
    return LinearMap{ComplexF64}(n; ismutating = true) do y, x
        y .= _apply(solver_ref[], x)
    end
end

function _direct_factor!(solver::_VolumeLinearSolver)
    if solver.direct === nothing
        A = SparseMatrixCSC{ComplexF64, Int}(solver.matrix)
        if solver.dtn !== nothing
            f, n = solver.dtn, size(A, 1)
            T = f.Cr' * (f.weights .* f.Cr) + f.Ci' * (f.weights .* f.Ci)
            A = A - sparse(repeat(f.positions, outer = length(f.positions)),
                repeat(f.positions, inner = length(f.positions)), vec(T), n, n)
        end
        solver.direct = lu(A)
    end
    return solver.direct
end

function _incomplete_factor!(solver::_VolumeLinearSolver, scale)
    factor = IncompleteLU.ilu(
        SparseMatrixCSC{ComplexF32, eltype(solver.shifted.colptr)}(solver.shifted);
        τ = scale * solver.ilu_tolerance)
    n = size(solver.matrix, 1)
    solver.incomplete = _SinglePrecisionILU(factor, zeros(ComplexF32, n), zeros(ComplexF32, n))
    return solver.incomplete
end

# Solution of `matrix * x = rhs` and the solver used and its iteration count.
function _solve!(solver::_VolumeLinearSolver, rhs)
    if solver.mode === :iterative
        for scale in (1.0, 0.2)
            preconditioner = scale == 1.0 && solver.incomplete !== nothing ?
                             solver.incomplete : _incomplete_factor!(solver, scale)
            x, history = IterativeSolvers.gmres(solver.operator, rhs; Pr = preconditioner,
                restart = 50, maxiter = 500, reltol = solver.tolerance, log = true)
            iterations = history.iters
            converged = history.isconverged
            # The single-precision preconditioner limits the accuracy of the recurrence, so the true residual is refined.
            for _ in 1:2
                converged || break
                residual = rhs - _apply(solver, x)
                norm(residual) <= solver.tolerance * norm(rhs) && break
                correction, refinement = IterativeSolvers.gmres(solver.operator, residual;
                    Pr = preconditioner, restart = 50, maxiter = 200,
                    reltol = min(0.5, 0.5solver.tolerance * norm(rhs) / norm(residual)),
                    log = true)
                x += correction
                iterations += refinement.iters
                converged = refinement.isconverged
            end
            converged && return x, (; solver = :iterative, iterations)
            solver.incomplete = nothing
        end
        @warn "the iterative volume solve did not converge, using the direct solver"
        solver.mode = :direct
        solver.shifted = nothing
    end
    return _direct_factor!(solver) \ rhs, (; solver = :direct, iterations = 0)
end

# Everything of a volume problem that does not depend on the incident direction.
