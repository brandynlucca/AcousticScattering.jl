# Compressed single-interface Müller operators. No global dense matrix is assembled.
function _fluid_diagonal_blocks(H)
    blocks = Tuple{Vector{Int}, Matrix{ComplexF64}}[]
    permutation = HMatrices.rowperm(H)
    function visit(node)
        if HMatrices.isleaf(node)
            indices = permutation[HMatrices.rowrange(node)]
            push!(blocks, (indices, HMatrices.data(node)))
        else
            children = HMatrices.children(node)
            for j in axes(children, 1)
                visit(children[j, j])
            end
        end
    end
    visit(H)
    return blocks
end

function _assemble_compressed_fluid(boundary, k, quad; formulation, correction, compression)
    compression.method === :hmatrix ||
        throw(ArgumentError("fluid compression supports only :none or :hmatrix"))
    formulation === :muller && correction.method === :dim ||
        throw(ArgumentError("compressed fluid BEM requires formulation=:muller and correction=:dim"))
    tol = get(compression, :tol, 1e-8)
    isfinite(tol) && 0 < tol < 1 ||
        throw(ArgumentError("compression tolerance must lie in (0, 1)"))
    compression = (; method = :hmatrix, tol)
    n, g, ki = length(quad), boundary.density_contrast, k/boundary.soundspeed_contrast
    bounds = _fluid_quadrature_size(quad)
    regular = _fluid_regular_range(max(k, ki), bounds)
    reconstruct = max(k, ki)*bounds.radius <= pi/2
    oe, oi = Inti.Helmholtz(; k, dim = 3), Inti.Helmholtz(; k = ki, dim = 3)
    Se, De = _fluid_layer_operators(oe, quad, quad, correction; regular, compression)
    Si, Di = ki == k ? (Se, De) :
             _fluid_layer_operators(oi, quad, quad, correction; regular, compression)
    # Preserve K = S⁻¹DS and H = S⁻¹(D²-I/4) without dense products.
    if reconstruct
        Fe = _fluid_assembly_lu(Se; rtol = tol/100)
        Fi = ki == k ? Fe : _fluid_assembly_lu(Si; rtol = tol/100)
        Ke = He = Ki = Hi = nothing
    else
        Ke, He = _fluid_layer_operators(oe, quad, quad, correction;
            derivative = true, regular, compression)
        Ki, Hi = ki == k ? (Ke, He) :
                 _fluid_layer_operators(oi, quad, quad, correction;
            derivative = true, regular, compression)
        Fe = Fi = nothing
    end
    local_Se, local_De = _fluid_diagonal_blocks(Se), _fluid_diagonal_blocks(De)
    local_Si, local_Di = _fluid_diagonal_blocks(Si), _fluid_diagonal_blocks(Di)
    if !reconstruct
        local_Ke, local_He = _fluid_diagonal_blocks(Ke), _fluid_diagonal_blocks(He)
        local_Ki, local_Hi = _fluid_diagonal_blocks(Ki), _fluid_diagonal_blocks(Hi)
    end
    blocks = Tuple{Vector{Int}, Matrix{ComplexF64}}[]
    for j in eachindex(local_Se)
        indices, se = local_Se[j]
        de, si, di = last(local_De[j]), last(local_Si[j]), last(local_Di[j])
        if reconstruct
            fe, fi = lu(se), lu(si)
            ke, he = fe \ (de*se), fe \ (de*de - 0.25I)
            kii, hi = fi \ (di*si), fi \ (di*di - 0.25I)
        else
            ke, he = last(local_Ke[j]), last(local_He[j])
            kii, hi = last(local_Ki[j]), last(local_Hi[j])
        end
        block = [I-de+di se-g*si; -he+hi/g I+ke-kii]
        push!(blocks, ([indices; indices .+ n], block))
    end
    Se, De, Si, Di = map(_fluid_operator_workspace, (Se, De, Si, Di))
    Ke, He, Ki, Hi = map(_fluid_operator_workspace, (Ke, He, Ki, Hi))
    Fe, Fi = map(_fluid_inverse_workspace, (Fe, Fi))
    temp_e, temp_i = zeros(ComplexF64, n), zeros(ComplexF64, n)
    derivative_e, derivative_i = similar(temp_e), similar(temp_i)
    A = LinearMap{ComplexF64}(2n, 2n) do y, x
        p, d = view(x, 1:n), view(x, (n + 1):2n)
        yp, yd = view(y, 1:n), view(y, (n + 1):2n)
        mul!(temp_e, Se, d)
        mul!(temp_e, De, p, -1, 1)
        mul!(temp_i, Si, d, g, 0)
        mul!(temp_i, Di, p, -1, 1)
        @. yp = p + temp_e - temp_i
        if reconstruct
            mul!(derivative_e, De, temp_e)
            mul!(derivative_i, Di, temp_i)
            @. derivative_e += 0.25p
            @. derivative_i += 0.25p
            LinearAlgebra.ldiv!(Fe, derivative_e)
            LinearAlgebra.ldiv!(Fi, derivative_i)
            @. yd = d + derivative_e - derivative_i/g
        else
            mul!(yd, He, p, -1, 0)
            mul!(yd, Hi, p, inv(g), 1)
            mul!(yd, Ke, d, 1, 1)
            mul!(yd, Ki, d, -1, 1)
            yd .+= d
        end
        return y
    end
    derivative_evaluation = reconstruct ?
                            (exterior = :calderon, interior = :calderon) :
                            (exterior = :direct, interior = :direct)
    return (;
        A, blocks, quad, k, formulation, correction, compression, derivative_evaluation)
end

struct _FluidBlockPreconditioner{F}
    indices::Vector{Vector{Int}}
    factors::Vector{F}
    workspace::Vector{ComplexF64}
end

struct _FluidInterfacePreconditioner{B}
    interfaces::B
    row_norms::Vector{Float64}
    col_norms::Vector{Float64}
    input::Vector{ComplexF64}
    output::Vector{ComplexF64}
end

function _fluid_interface_preconditioner(interfaces, rows, cols)
    blocks = map(interfaces) do interface
        (; density_exterior, density_interior) = interface
        scale = max(density_exterior, density_interior)
        a, b = density_exterior/scale, density_interior/scale
        weight = 4*(a/(a+b))*(b/(a+b))
        pressure_weight = weight*(density_exterior-density_interior)
        flux_weight = ((a-b)/(a+b)^2)/scale
        merge(interface, (; weight, pressure_weight, flux_weight))
    end
    return _FluidInterfacePreconditioner(blocks, rows, cols,
        zeros(ComplexF64, length(rows)), zeros(ComplexF64, length(cols)))
end

function LinearAlgebra.ldiv!(P::_FluidInterfacePreconditioner, x::AbstractVector)
    @. P.input = x*P.row_norms
    n = length(x) ÷ 2
    for block in P.interfaces
        (; indices, S, factor, weight, pressure_weight, flux_weight) = block
        p, v = view(P.input, indices), view(P.input, indices .+ n)
        yp, yv = view(P.output, indices), view(P.output, indices .+ n)
        if iszero(pressure_weight) && iszero(flux_weight)
            copyto!(yp, p)
            copyto!(yv, v)
        else
            mul!(yp, S, v)
            copyto!(yv, p)
            LinearAlgebra.ldiv!(factor, yv)
            @. yp = weight*p-pressure_weight*yp
            @. yv = weight*v+flux_weight*yv
        end
    end
    @. x = P.output*P.col_norms
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::_FluidInterfacePreconditioner, x::AbstractVector)
    copyto!(y, x)
    return LinearAlgebra.ldiv!(P, y)
end

struct _FluidCoarsePreconditioner{B, F}
    local_blocks::B
    basis::Matrix{ComplexF64}
    correction::Matrix{ComplexF64}
    factor::F
    workspace::Vector{ComplexF64}
end

function LinearAlgebra.ldiv!(P::_FluidCoarsePreconditioner, x::AbstractVector)
    mul!(P.workspace, P.basis', x)
    LinearAlgebra.ldiv!(P.factor, P.workspace)
    LinearAlgebra.ldiv!(P.local_blocks, x)
    mul!(x, P.correction, P.workspace, 1, 1)
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::_FluidCoarsePreconditioner, x::AbstractVector)
    copyto!(y, x)
    return LinearAlgebra.ldiv!(P, y)
end

function LinearAlgebra.ldiv!(P::_FluidBlockPreconditioner, x::AbstractVector)
    for (indices, factor) in zip(P.indices, P.factors)
        workspace = view(P.workspace, 1:length(indices))
        workspace .= view(x, indices)
        LinearAlgebra.ldiv!(factor, workspace)
        x[indices] .= workspace
    end
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::_FluidBlockPreconditioner, x::AbstractVector)
    copyto!(y, x)
    return LinearAlgebra.ldiv!(P, y)
end

function _fluid_coarse_basis(quads, ranges, n)
    basis = zeros(ComplexF64, 2n, 18length(quads))
    for (i, quad) in enumerate(quads)
        bounds = _fluid_quadrature_size(quad)
        traces = [first(_regular_helmholtz_traces(q, bounds.center, bounds.radius, 0.0, 2))
                  for q in quad]
        pressure = reduce(vcat, transpose.(traces))
        columns = (18(i - 1) + 1):(18(i - 1) + 9)
        basis[ranges[i], columns] = pressure
        basis[ranges[i] .+ n, columns .+ 9] = pressure
    end
    return basis
end

function _fluid_block_norms(block, norm_floor)
    rows = max.(vec(maximum(abs, block; dims = 2)), norm_floor)
    rows = ifelse.(iszero.(rows), 1.0, rows)
    cols = ones(size(block, 2))
    for j in axes(block, 2)
        value = 0.0
        for i in axes(block, 1)
            value = max(value, abs(block[i, j]/rows[i]))
        end
        value = max(value, norm_floor)
        cols[j] = iszero(value) ? 1.0 : value
    end
    return rows, cols
end

function _factor_full_fluid(
        system; equilibrate, condition_limit, gmres_kwargs = (;), norm_floor = 0.0,
        precision::Symbol = :double, refinement::NamedTuple = (;))
    if !hasproperty(system, :compression)
        isempty(gmres_kwargs) ||
            throw(ArgumentError("gmres_kwargs requires fluid compression"))
        return _factor_fluid_system(system.A; equilibrate, condition_limit, norm_floor,
            precision, refinement)
    end
    precision === :double && isempty(refinement) ||
        throw(ArgumentError("mixed precision requires dense fluid BEM"))
    any(key -> haskey(gmres_kwargs, key), (:Pl, :Pr, :log)) &&
        throw(ArgumentError("fluid gmres_kwargs cannot override Pl, Pr or log"))
    rows, cols = ones(size(system.A, 1)), ones(size(system.A, 2))
    interface_preconditioning = hasproperty(system, :preconditioner_interfaces) &&
                                system.preconditioner_interfaces !== nothing
    if interface_preconditioning
        if equilibrate
            for (indices, block) in system.blocks
                row_norms, col_norms = _fluid_block_norms(block, norm_floor)
                rows[indices], cols[indices] = row_norms, col_norms
            end
        end
    else
        factors = [_factor_fluid_system(block; equilibrate, condition_limit = 0, norm_floor)
                   for (_, block) in system.blocks]
        for ((indices, _), factor) in zip(system.blocks, factors)
            if equilibrate
                rows[indices] = factor.row_norms
                cols[indices] = factor.col_norms
            end
        end
    end
    # In the Calderon regime, retain the full single-layer principal part on each
    # interface: [I (rho_e-rho_i)S; (1/rho_e-1/rho_i)S^-1/4 I]
    local_blocks = interface_preconditioning ?
                   _fluid_interface_preconditioner(system.preconditioner_interfaces, rows, cols) :
                   _FluidBlockPreconditioner(
        first.(system.blocks), getproperty.(factors, :factorization),
        zeros(ComplexF64, maximum(length(first(block)) for block in system.blocks)))
    work = zeros(ComplexF64, length(cols))
    scaled_A = LinearMap{ComplexF64}(size(system.A)...) do y, x
        @. work = x/cols
        mul!(y, system.A, work)
        y ./= rows
        return y
    end
    # A small global space captures smooth interface modes, including gas resonances
    n = size(system.A, 1) ÷ 2
    basis = hasproperty(system, :quads) ?
            _fluid_coarse_basis(system.quads, system.ranges, n) :
            _fluid_coarse_basis((system.quad,), (1:n,), n)
    basis .*= cols
    basis = Matrix(LinearAlgebra.qr(basis).Q)
    applied = similar(basis)
    _fluid_coarse_images!(applied, system, basis, rows, cols, scaled_A)
    coarse_factor = lu(basis' * applied)
    for j in axes(applied, 2)
        LinearAlgebra.ldiv!(local_blocks, view(applied, :, j))
    end
    preconditioner = _FluidCoarsePreconditioner(
        local_blocks, basis, basis-applied, coarse_factor,
        zeros(ComplexF64, size(basis, 2)))
    options = merge(
        (; reltol = 1e-10, abstol = 0.0,
            restart = hasproperty(system, :quads) ? 600 : 150, maxiter = 500,
            orth_meth = IterativeSolvers.DGKS()),
        gmres_kwargs)
    return (; scaled_A, preconditioner, row_norms = rows, col_norms = cols, options,
        diagnostics = (; equilibrate, condition_limit, condition_number = nothing,
            scaled_condition_number = nothing, conditioning = :not_computed,
            equilibration = equilibrate ? :local_blocks : :none, preconditioner = :twolevel,
            local_preconditioner = interface_preconditioning ? :calderon : :block_jacobi))
end

mutable struct _FluidSolutionSubspace
    basis::Matrix{ComplexF64}
    images::Matrix{ComplexF64}
    count::Int
end

function _fluid_sweep_factor(factor, dimension, samples)
    dimension >= 0 || throw(ArgumentError("recycle_dimension must be nonnegative"))
    hasproperty(factor, :scaled_A) || return factor
    n = length(factor.col_norms)
    # Bound retained solution/image pairs by 64 MiB as well as the requested count.
    capacity = min(dimension, max(samples - 1, 0), n, (64 * 1024^2) ÷ (32n))
    capacity == 0 && return factor
    basis = zeros(ComplexF64, n, capacity)
    return merge(factor, (; recycling = _FluidSolutionSubspace(basis, similar(basis), 0)))
end

function _fluid_recycle_update!(space, x, image)
    space.count == size(space.basis, 2) && return
    u, c = copy(x), copy(image)
    original_norm = norm(c)
    # Reorthogonalize images and apply identical transformations to their solutions.
    # Thus A*U = C, C'*C = I, and U*(C'*b) minimizes the residual in this space.
    for _ in 1:2, j in 1:space.count

        coefficient = dot(view(space.images, :, j), c)
        c .-= coefficient .* view(space.images, :, j)
        u .-= coefficient .* view(space.basis, :, j)
    end
    image_norm = norm(c)
    if isfinite(image_norm) && image_norm > sqrt(eps(Float64))*original_norm
        space.count += 1
        space.basis[:, space.count] .= u ./ image_norm
        space.images[:, space.count] .= c ./ image_norm
    end
    return nothing
end

function _fluid_recycle_refine!(x, history, image, factor, rhs, target)
    residual = rhs - image
    residual_norm = norm(residual)
    # Reconcile a recursive-residual convergence estimate with the true residual.
    # Keep corrections inside this attempt's original total iteration budget.
    for _ in 1:2
        remaining = factor.options.maxiter - history.iters
        (residual_norm <= target || !history.isconverged || remaining <= 0) && break
        options = merge(factor.options,
            (; reltol = 0.0, abstol = target/10, maxiter = remaining))
        correction, extra = IterativeSolvers.gmres(factor.scaled_A, residual;
            Pr = factor.preconditioner, log = true, options...)
        history.iters += extra.iters
        history.mvps += extra.mvps
        append!(history[:resnorm], extra[:resnorm])
        candidate = x + correction
        candidate_image = factor.scaled_A*candidate
        candidate_residual = rhs - candidate_image
        candidate_norm = norm(candidate_residual)
        candidate_norm < residual_norm || break
        copyto!(x, candidate)
        image, residual, residual_norm = candidate_image, candidate_residual, candidate_norm
    end
    return image, residual_norm <= target
end

function _solve_recycled_fluid_system(factor, b)
    space = factor.recycling
    rhs = b ./ factor.row_norms
    rhs_norm = norm(rhs)
    target = max(factor.options.abstol, factor.options.reltol*rhs_norm)
    x = zeros(ComplexF64, length(rhs))
    reused = space.count > 0 && rhs_norm > 0
    if reused
        basis, images = view(space.basis, :, 1:space.count),
        view(space.images, :, 1:space.count)
        mul!(x, basis, images' * rhs)
        initial_residual = norm(factor.scaled_A*x - rhs)
        if initial_residual <= target
            # Avoid allocating a full Arnoldi basis when the retained space suffices.
            history = IterativeSolvers.ConvergenceHistory(;
                restart = factor.options.restart)
            history[:resnorm] = Float64[]
            history[:reltol], history[:abstol] = factor.options.reltol,
            factor.options.abstol
            history.isconverged = true
            history.mvps = 1
            return (; x = x ./ factor.col_norms, history)
        end
        # Guard against loss of the image/solution relation in ill-conditioned spaces.
        if !(initial_residual < rhs_norm)
            fill!(x, 0)
            reused = false
            space.count = 0
        end
    end
    # gmres! otherwise scales reltol by the initial residual of the warm start.
    options = merge(factor.options, (;
        reltol = 0.0, abstol = target, initially_zero = !reused))
    x, history = IterativeSolvers.gmres!(x, factor.scaled_A, rhs;
        Pr = factor.preconditioner, log = true, options...)
    image = factor.scaled_A*x
    image, accepted = _fluid_recycle_refine!(x, history, image, factor, rhs, target)
    if reused && !accepted
        attempted_history = history
        x, history = IterativeSolvers.gmres(factor.scaled_A, rhs;
            Pr = factor.preconditioner, log = true, factor.options...)
        image = factor.scaled_A*x
        image, accepted = _fluid_recycle_refine!(x, history, image, factor, rhs, target)
        history.iters += attempted_history.iters
        history.mvps += attempted_history.mvps
        prepend!(history[:resnorm], attempted_history[:resnorm])
        # An unsuccessful recycled solve must not seed later angles.
        space.count = 0
    end
    history.isconverged = accepted
    accepted && _fluid_recycle_update!(space, x, image)
    return (; x = x ./ factor.col_norms, history)
end

function _solve_compressed_fluid_system(factor, b)
    hasproperty(factor, :recycling) && return _solve_recycled_fluid_system(factor, b)
    scaled_x, history = IterativeSolvers.gmres(factor.scaled_A, b ./ factor.row_norms;
        Pr = factor.preconditioner, log = true, factor.options...)
    return (; x = scaled_x ./ factor.col_norms, history)
end

function _solve_compressed_fluid(system, factor;
        incidence_angle, incidence_azimuth, return_diagnostics, incident = nothing)
    (; A, quad, k, formulation, correction, compression, derivative_evaluation) = system
    n = length(quad)
    pinc, dinc = _incident_traces(quad, k, incidence_angle, incidence_azimuth, incident)
    b = [pinc; dinc]
    (; x, history) = _solve_compressed_fluid_system(factor, b)
    residual = _fluid_residual_report(A*x-b, b, factor.row_norms)
    history.isconverged ||
        @warn "Compressed fluid BEM GMRES did not converge" iterations=history.iters residual=residual.relative_residual
    p, d = x[1:n] - pinc, x[(n + 1):2n] - dinc
    if return_diagnostics
        report = merge(residual, factor.diagnostics,
            (; method = :gmres, converged = history.isconverged,
                iterations = history.iters,
                formulation, derivative_evaluation, residual_history = history[:resnorm],
                unknown_count = 2n, quadrature_nodes = n, compression, correction,
                solver_options = factor.options))
        return p, d, quad, report
    end
    return p, d, quad
end
