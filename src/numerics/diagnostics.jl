mutable struct _SolveReports
    systems::Vector{NamedTuple}
    lock::ReentrantLock
    refinement::Union{Nothing, NamedTuple}
end

_SolveReports() = _SolveReports(NamedTuple[], ReentrantLock(), nothing)

function _linear_residual(A, x, b)
    return _residual_report(A * x - b, b)
end

function _residual_report(residual, b)
    absolute = norm(residual)
    rhs_norm = norm(b)
    relative = iszero(rhs_norm) ? (iszero(absolute) ? 0.0 : Inf) : absolute / rhs_norm
    return (; absolute_residual = absolute, relative_residual = relative)
end

function _record_solve!(reports, A, x, b; mode = nothing, component = :field, kwargs...)
    reports === nothing && return nothing
    report = merge(_linear_residual(A, x, b),
        (; mode, component, equation_count = size(A, 1), unknown_count = size(A, 2),
            method = size(A, 1) == size(A, 2) ? :direct : :least_squares, kwargs...))
    lock(reports.lock) do
        push!(reports.systems, report)
    end
    return nothing
end

function _solve_reported(A, b, reports; kwargs...)
    x = A \ b
    _record_solve!(reports, A, x, b; kwargs...)
    return x
end

function _summarize_solves(reports; method, solver_options, kwargs...)
    systems = reports.systems
    isempty(systems) && return nothing
    return (; method = :direct, discretization = method,
        converged = nothing, iterations = nothing, residual_history = nothing,
        absolute_residual = maximum(r.absolute_residual for r in systems),
        relative_residual = maximum(r.relative_residual for r in systems),
        unknown_count = maximum(r.unknown_count for r in systems),
        equation_count = maximum(r.equation_count for r in systems),
        systems, solver_options, refinement = reports.refinement, kwargs...)
end

function _record_refinement!(reports, converged, change_db, target_tol, n_elements)
    reports === nothing && return nothing
    reports.refinement = (; converged, change_db, target_tol, n_elements)
    return nothing
end

function _mfs_matrix_diagnostics(A; condition_limit::Integer = 512)
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    size(A, 2) > condition_limit && return (;
        condition_number = nothing, numerical_rank = nothing, rank_tolerance = nothing,
        conditioning = :not_computed, condition_limit)
    values = svdvals(A)
    tolerance = max(size(A)...) * eps(eltype(values)) * first(values)
    return (; condition_number = iszero(last(values)) ? Inf : first(values) / last(values),
        numerical_rank = count(>(tolerance), values), rank_tolerance = tolerance,
        conditioning = :svd, condition_limit)
end
