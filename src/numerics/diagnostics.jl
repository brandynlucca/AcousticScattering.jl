mutable struct _SolveReports
    systems::Vector{NamedTuple}
    lock::ReentrantLock
    refinement::Union{Nothing, NamedTuple}
end

_SolveReports() = _SolveReports(NamedTuple[], ReentrantLock(), nothing)

# A sweep owns its cache. No operator is reused at a different frequency.
# Bound retained storage so refinement at one angle cannot accumulate all of
# its fine-grid operators while the remaining angles wait.
mutable struct _NumericalReuse
    entries::Dict{Any, Tuple{Any, Int}}
    order::Vector{Any}
    bytes::Int
    limit::Int
    lock::ReentrantLock
    retain_prefix::Bool
end
function _NumericalReuse(; retain_prefix = true)
    _NumericalReuse(
        Dict{Any, Tuple{Any, Int}}(), Any[], 0, 512*1024^2, ReentrantLock(), retain_prefix)
end
function _numerical_reuse_context()
    get(task_local_storage(), :acousticscattering_numerical_reuse, nothing)
end

function _with_numerical_reuse(f; retain_prefix = true)
    _numerical_reuse_context()===nothing || return f()
    return task_local_storage(f, :acousticscattering_numerical_reuse, _NumericalReuse(;
        retain_prefix))
end

_reuse_numerical(f, ::Nothing, key) = f()
function _numerical_factor_key(key)
    key isa Tuple && !isempty(key) &&
        first(key) in
        (:fembem_operator, :fembem_radial_operator, :fembem_exterior_factor, :curved_factor)
end
function _numerical_shared_key(key)
    key isa Tuple && !isempty(key) &&
        first(key) in (:curved_operators, :chief, :fembem_angular, :fembem_radial,
            :fembem_radial_inverse, :fembem_exterior)
end
function _evict_numerical!(cache, key)
    deleteat!(cache.order, findfirst(isequal(key), cache.order))
    cache.bytes -= pop!(cache.entries, key)[2]
end
function _reuse_numerical(f, cache::_NumericalReuse, key)
    found = lock(cache.lock) do
        if haskey(cache.entries, key)
            deleteat!(cache.order, findfirst(isequal(key), cache.order))
            push!(cache.order, key)
            return cache.entries[key]
        end
        return nothing
    end
    found===nothing || return found[1]
    value = f()
    bytes = Base.summarysize(value)
    bytes>cache.limit && return value
    lock(cache.lock) do
        haskey(cache.entries, key) && return
        # Complete solve factors contain large derived maps and apply to fewer
        # configurations than the shared BEM rows and FEM eigensystems. Reserve
        # most storage for those shared components to avoid cyclic eviction.
        if _numerical_factor_key(key)
            budget = cache.limit÷4
            bytes>budget && return
            used = sum(entry[2] for (k, entry) in cache.entries if _numerical_factor_key(k); init = 0)
            while used+bytes>budget
                oldest = first(k for k in cache.order if _numerical_factor_key(k))
                used -= cache.entries[oldest][2]
                _evict_numerical!(cache, oldest)
            end
        end
        while cache.bytes+bytes>cache.limit
            factor = findfirst(_numerical_factor_key, cache.order)
            # Angles visit modes in the same order. Evicting shared components
            # during that scan would discard every entry just before its reuse
            # when the full working set exceeds the cap. Keep the admitted
            # prefix and compute later components without retaining them.
            if cache.retain_prefix && factor===nothing &&
               (_numerical_shared_key(key) || _numerical_factor_key(key))
                return
            end
            oldest = factor===nothing ? first(cache.order) : cache.order[factor]
            _evict_numerical!(cache, oldest)
        end
        cache.entries[key] = (value, bytes)
        push!(cache.order, key)
        cache.bytes += bytes
    end
    return value
end

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
