"""
    bem3d_elements_per_wavelength(k; elements_per_wavelength=10)

Recommended Gmsh `meshsize` [m] for [`gmsh_sphere_mesh`](@ref)/
[`gmsh_spheroid_mesh`](@ref) at wavenumber `k` [1/m], targeting
`elements_per_wavelength` triangle edges per acoustic wavelength
(`λ = 2π/k`). This wavelength rule is a starting resolution: refine mesh size
and quadrature order independently to assess geometric and integration errors.
"""
function bem3d_elements_per_wavelength(k::Real; elements_per_wavelength::Real = 10)
    (2π / k) / elements_per_wavelength
end

# Polar angle is from +x; azimuth is from +y toward +z.
function _bem3d_incidence_direction(incidence_angle::Real, incidence_azimuth::Real)
    β, α = incidence_angle, incidence_azimuth
    return SVector(cos(β), sin(β) * cos(α), sin(β) * sin(α))
end

"""
    solve_full_bem(boundary::Union{Rigid,PressureRelease}, k, quad;
                   incidence_angle=0.0, incidence_azimuth=0.0,
                   formulation=:burton_miller,
                   compression=(method=:hmatrix, tol=1e-5),
                   correction=(method=:dim,),
                   gmres_kwargs=(reltol=1e-4, restart=150, maxiter=1200),
                   return_diagnostics=false)

Solve a boundary integral equation for a rigid or pressure-release scatterer over the
full 3D surface `quad` (from [`gmsh_sphere_mesh`](@ref)/
[`gmsh_spheroid_mesh`](@ref)) at a unit-amplitude plane wave arriving from
`incidence_angle`/`incidence_azimuth` in rad (see `_bem3d_incidence_direction`),
using Inti operators and GMRES, with optional H-matrix compression.
`formulation=:burton_miller` combines the pressure and normal-derivative equations
with coupling `im/k` for the outgoing `exp(im*k*r)` kernel and outward body normals.
It removes fictitious interior resonances. `formulation=:cbie` selects an uncombined
equation, which can be singular at interior resonances. Both require `k > 0`.

For rigid or pressure-release rims, `formulation=:cbie`, `compression=(method=:none,)` and
`correction=(method=:edge,)` select adaptive edge-weighted single-layer quadrature.
Pressure-release boundaries use the total normal derivative as the unknown. Rigid boundaries
use an indirect single-layer density and its target-normal derivative equation; pressure
evaluation through [`bem`](@ref) retains that density. Returned traces remain scattered traces.
Triangles may meet at most one sharp edge each. Quadrature tolerances `rtol`, `atol`
and `maxsubdiv` can be supplied in `correction` independently of `gmres_kwargs`.

Returns `(p_scat, dpdn_scat, quad)`, the surface scattered pressure and
its normal derivative at every quadrature node, and `quad` itself (for
`far_field`). With `return_diagnostics=true`, append a fourth element containing
convergence history, recomputed linear residuals and solver settings.
"""
function solve_full_bem(boundary::Union{Rigid, PressureRelease}, k::Real, quad;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        incident = nothing,
        formulation::Symbol = :burton_miller,
        compression = (method = :hmatrix, tol = 1e-5),
        correction = (method = :dim,),
        gmres_kwargs = (reltol = 1e-4, restart = 150, maxiter = 1200),
        return_diagnostics::Bool = false, _density = nothing)
    all(isfinite, (incidence_angle, incidence_azimuth)) ||
        throw(ArgumentError("incidence angles must be finite"))
    system = _assemble_full_boundary(
        boundary, k, quad; formulation, compression, correction)
    return _solve_full_boundary(system; incidence_angle, incidence_azimuth,
        incident, gmres_kwargs, return_diagnostics, _density)
end

function _assemble_full_boundary(boundary::Union{Rigid, PressureRelease}, k::Real, quad;
        formulation::Symbol = :burton_miller,
        compression = (method = :hmatrix, tol = 1e-5), correction = (method = :dim,))
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    formulation in (:cbie, :burton_miller) ||
        throw(ArgumentError("formulation must be :cbie or :burton_miller"))
    op = Inti.Helmholtz(; k = k, dim = 3)
    if correction.method === :edge
        formulation === :cbie && compression.method === :none ||
            throw(ArgumentError("edge quadrature requires formulation=:cbie and compression=(method=:none,)"))
        enrich = boundary isa Rigid
        S = _edge_single_layer(op, quad, quad, correction; enrich)
        K = enrich ?
            _edge_single_layer(op, quad, quad, correction; derivative = true, enrich) :
            nothing
        A = enrich ?
            LinearMap{ComplexF64}((y, x) -> (mul!(y, K, x); y .= 0.5 .* x .- y), length(quad)) :
            S
        return (; A, S, D = nothing, K, H = nothing, boundary, k, quad,
            coupling = 0.0im, formulation, compression, correction, Pl = nothing)
    end
    S, D = Inti.single_double_layer(;
        op, target = quad, source = quad, compression, correction)
    n = length(quad)
    coupling = formulation == :burton_miller ? im / k : 0.0im
    K = H = nothing
    if formulation == :burton_miller
        K, H = Inti.adj_double_layer_hypersingular(;
            op, target = quad, source = quad, compression, correction)
    end

    # Calderon preconditioning (Pl=lu(H)) cuts PressureRelease/burton_miller far-field error on ill-conditioned rims. The dual (Pl=lu(S)) worsens Rigid, so it stays one-sided (docs/DEVELOPMENT_PRIORITIES.md #4c).
    Pl = boundary isa PressureRelease && formulation == :burton_miller &&
         compression.method === :none ? lu(H) : nothing

    if boundary isa Rigid
        if formulation == :burton_miller
            work = Vector{ComplexF64}(undef, n)
            A = LinearMap{ComplexF64}(n, n) do y, x
                mul!(y, D, x)
                mul!(work, H, x)
                y .= 0.5 .* x .- y .- coupling .* work
            end
        else
            A = LinearMap{ComplexF64}((y, x) -> (mul!(y, D, x); y .= 0.5 .* x .- y), n, n)
        end
    else
        if formulation == :burton_miller
            work = Vector{ComplexF64}(undef, n)
            A = LinearMap{ComplexF64}(n, n) do y, x
                mul!(y, S, x)
                mul!(work, K, x)
                y .+= coupling .* (0.5 .* x .+ work)
            end
        else
            A = LinearMap{ComplexF64}((y, x) -> mul!(y, S, x), n, n)
        end
    end
    return (; A, S, D, K, H, boundary, k, quad, coupling,
        formulation, compression, correction, Pl)
end

function _solve_full_boundary(system;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        incident = nothing,
        gmres_kwargs = (reltol = 1e-4, restart = 150, maxiter = 1200),
        return_diagnostics::Bool = true, _density = nothing)
    (; A, S, D, K, H, boundary, k, quad, coupling,
        formulation, compression, correction, Pl) = system
    incident = _resolve_incident(
        k, incidence_angle, incidence_azimuth; incident)
    n = length(quad)
    p_inc, dpdn_inc = _incident_traces(
        quad, k, incidence_angle, incidence_azimuth, incident)
    if correction.method === :edge
        rhs = boundary isa Rigid ? dpdn_inc : p_inc
    elseif boundary isa Rigid
        rhs = S * dpdn_inc
        if formulation == :burton_miller
            rhs .+= coupling .* (0.5 .* dpdn_inc .+ K * dpdn_inc)
        end
    else
        rhs = (D * (-p_inc)) .+ (0.5 .* p_inc)
        if formulation == :burton_miller
            rhs .+= coupling .* (H * (-p_inc))
        end
    end
    preconditioner_kwargs = Pl === nothing ? (;) : (; Pl)
    x, hist = IterativeSolvers.gmres(
        A, rhs; log = true, merge(preconditioner_kwargs, gmres_kwargs)...)
    hist.isconverged ||
        @warn "solve_full_bem: GMRES did not converge to the requested tolerance, result may be inaccurate" boundary formulation iters = hist.iters
    p_scat = boundary isa Rigid ? x : -p_inc
    dpdn_scat = boundary isa Rigid ? -dpdn_inc : x
    if correction.method === :edge
        if boundary isa Rigid
            p_scat = S*x
            _density === nothing || (_density[] = x)
        else
            dpdn_scat = x-dpdn_inc
        end
    end
    if return_diagnostics
        diagnostics = merge(_linear_residual(A, x, rhs),
            (
                method = :gmres, converged = hist.isconverged, iterations = hist.iters,
                formulation = formulation, coupling = coupling,
                residual_history = copy(hist[:resnorm]), unknown_count = n,
                quadrature_nodes = n, compression = compression, correction = correction,
                solver_options = merge(
                    (abstol = 0.0, reltol = sqrt(eps(Float64)),
                        restart = min(20, n), maxiter = n),
                    (; gmres_kwargs...))))
        return p_scat, dpdn_scat, quad, diagnostics
    end
    return p_scat, dpdn_scat, quad
end

"""
    _fluid_derivative_operators(op, target, source, S, D, correction; regular=true, reconstruct=true)

Evaluate the normal traces of the fluid layer operators. For coincident surfaces with
density interpolation, `reconstruct=true` and `k*radius <= π/2`,
recover them from the Calderón identities
`S*K = D*S` and `S*H = D*D - I/4`. Radii measure distances from the mean node
position and are rigid-motion invariant.
This uses the pressure operators consistently at low frequency; it requires a dense
single-layer factorization and is not suitable near its interior Dirichlet resonances.
Other interactions retain direct derivative quadrature.

See van 't Wout et al. (2022), doi:10.1016/j.camwa.2021.11.021, for the operator
identities; their hypersingular operator has the opposite sign to `H` here.
"""
function _fluid_derivative_operators(
        op, target, source, S, D, correction; regular = true, reconstruct = true)
    if reconstruct && target === source && correction.method === :dim
        if op.k * _fluid_quadrature_size(source).radius <= pi/2
            factor = lu(S)
            return factor \ (D * S), factor \ (D * D - 0.25I), :calderon
        end
    end
    K, H = _fluid_layer_operators(
        op, target, source, correction; derivative = true, regular)
    return K, H, :direct
end

"""
    solve_full_bem(boundary::FluidFilled, k, quad;
                   incidence_angle=0.0, incidence_azimuth=0.0,
                   formulation=:muller, equilibrate=true, condition_limit=512,
                   correction=(method=:dim,), compression=(method=:none,),
                   gmres_kwargs=(;), return_diagnostics=false)

Solve fluid transmission over the full 3D surface `quad`. The default is a dense direct solve.
`compression=(method=:hmatrix, tol=1e-8)` selects compressed, block-preconditioned GMRES
for single-interface Müller with density interpolation. Near-field corrections are retained;
low-frequency Calderón products use hierarchical LU inverses at `tol/100`.
`gmres_kwargs` overrides the defaults `(reltol=1e-10, abstol=0, restart=150, maxiter=500)`
and DGKS reorthogonalization. The two-level preconditioner acts on the right.
Compression and iteration tolerances do not bound the physical discretization error.
`formulation=:muller` uses the pressure and exterior normal derivative as two
unknown traces. `formulation=:cbie` eliminates the two interior traces through pressure
and normal-velocity continuity, solving for the two exterior scattered traces. Its
right-hand side retains the discrete interior operators applied to the incident traces.
With `correction=(method=:edge,)`, CBIE instead uses total traces and material-dependent
corner powers for pressure and normal flux.
For edge quadrature, the constant-density Laplace double-layer identity regularizes both
pressure equations; this removes the constant background before integrating the singular kernel.
Edge quadrature requires a closed conforming triangular surface and does not use
hypersingular operators.
Use `bem` with `pressure` and `scattering_amplitude` to retain the weighted corner
interpolation when evaluating fields.
For dense solves, `equilibrate=true` scales rows and columns by their maximum absolute entry before
factorization, then recovers the physical traces. This scaling does not change the equations.
Both formulations require positive finite wavenumber and material contrasts.
For dense single-interface fluid/gas solves, `precision=:mixed` opts into
ComplexF32 LU with original-equation ComplexF64 residual refinement.
It requires `equilibrate=true`. The default `precision=:double` is unchanged.
`refinement=(tolerance=1e-13, maxiter=8, condition_guard=10eps(Float32))`
controls normwise and componentwise backward-error acceptance and the
single-precision reciprocal-condition guard. Conditioning, stagnation,
nonfinite corrections or iteration limits trigger ComplexF64 factorization.
Failed full-precision acceptance raises an error. Diagnostics report the actual
factor precision, promotion reason, correction history and both backward errors.
This option does not apply to compressed or multiple-interface solves.
With density interpolation, regular-wave pressure quadrature requires
`k*radius <= 1` and `2k*rms_radius <= 1` in both
media. The enclosing radius measures distances from the mean node position; the RMS
radius uses surface quadrature area weights. The quadrature choice is shared by both
media to preserve cancellation at weak-contrast interfaces. Otherwise, both media
use direct density interpolation. Separately, Calderón reconstruction of self normal
derivatives requires `k*radius <= π/2` in both media and uses those pressure operators.

Returns `(p_scat, dpdn_scat, quad)`, the *exterior scattered* pressure and
its normal derivative, in the same form `far_field`/
[`target_strength`](@ref) expect (the interior trace is discarded, matching
`axisymmetric_bem.jl`'s own `solve_axial(::FluidFilled, ...)` convention).
With `return_diagnostics=true`, append original and scaled system residuals and settings.
For at most `condition_limit` unknowns, also compute both matrix 2-norm condition numbers.
Zero disables this SVD calculation. A standard direct solve has no iterative convergence flag or history.
Compressed solves instead use maxima within the local preconditioner blocks for scaling,
skip SVD condition estimates, and report GMRES convergence and scaled residual history.
Their recomputed residuals describe the compressed system, including approximate Calderón inverses.
`derivative_evaluation` records the exterior and interior operator evaluation methods.
"""
function solve_full_bem(boundary::FluidFilled, k::Real, quad;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        incident = nothing,
        formulation::Symbol = :muller, equilibrate::Bool = true,
        condition_limit::Integer = 512,
        correction = (method = :dim,), return_diagnostics::Bool = false,
        compression = (method = :none,), gmres_kwargs = (;),
        precision::Symbol = :double, refinement::NamedTuple = (;))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    system = _assemble_full_fluid(boundary, k, quad; formulation, correction, compression)
    factor = _factor_full_fluid(system; equilibrate, gmres_kwargs, precision, refinement,
        condition_limit = return_diagnostics ? condition_limit : 0)
    return _solve_full_fluid(system, factor; incidence_angle, incidence_azimuth,
        incident, return_diagnostics)
end

function _assemble_full_fluid(boundary::FluidFilled, k::Real, quad;
        formulation::Symbol = :muller, correction = (method = :dim,),
        compression = (method = :none,))
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    formulation in (:muller, :cbie) ||
        throw(ArgumentError("fluid formulation must be :muller or :cbie"))
    g = boundary.density_contrast
    h = boundary.soundspeed_contrast
    isfinite(g) && g > 0 && isfinite(h) && h > 0 ||
        throw(ArgumentError("fluid density and soundspeed contrasts must be finite and positive"))
    k_int = k / boundary.soundspeed_contrast
    if compression.method !== :none
        return _assemble_compressed_fluid(
            boundary, k, quad; formulation, correction, compression)
    end
    if correction.method === :edge
        formulation === :cbie ||
            throw(ArgumentError("fluid edge quadrature requires formulation=:cbie"))
        pressure, flux = _edge_fluid_patches(quad, g)
        op_ext, op_int = Inti.Helmholtz(; k, dim = 3), Inti.Helmholtz(; k = k_int, dim = 3)
        S, D = _edge_fluid_operators(op_ext, quad, correction, pressure, flux)
        Si, Di = _edge_fluid_operators(op_int, quad, correction, pressure, flux)
        # Enforce the constant-density Laplace identity in both media.
        laplace = _edge_single_layer(Inti.Helmholtz(; k = 0.0, dim = 3),
            quad, quad, correction; patches = pressure, double_layer = true)
        defect = vec(sum(laplace; dims = 2)) .+ 0.5
        D[diagind(D)] .-= defect
        Di[diagind(Di)] .-= defect
        A = [0.5I-D S; 0.5I+Di -g*Si]
        return (; A, quad, k, formulation, correction,
            derivative_evaluation = (exterior = :not_used, interior = :not_used))
    end
    bounds = _fluid_quadrature_size(quad)
    regular = formulation === :muller && _fluid_regular_range(max(k, k_int), bounds)
    reconstruct = max(k, k_int) * bounds.radius <= pi/2

    op_ext = Inti.Helmholtz(; k = k, dim = 3)
    op_int = Inti.Helmholtz(; k = k_int, dim = 3)
    S_ext, D_ext = _fluid_layer_operators(op_ext, quad, quad, correction;
        regular)
    S_int, D_int = _fluid_layer_operators(op_int, quad, quad, correction;
        regular)
    n = length(quad)

    derivative_evaluation = nothing
    if formulation == :muller
        Id = Matrix{ComplexF64}(I, n, n)
        K_ext, H_ext, exterior = _fluid_derivative_operators(
            op_ext, quad, quad, S_ext, D_ext, correction; regular, reconstruct)
        K_int, H_int, interior = _fluid_derivative_operators(
            op_int, quad, quad, S_int, D_int, correction; regular, reconstruct)
        derivative_evaluation = (; exterior, interior)
        A = [Id-D_ext+D_int S_ext-g*S_int; -H_ext+H_int/g Id+K_ext-K_int]
    else
        # p_int = p_scat + p_inc, d_int = g * (d_scat + d_inc).
        A = [0.5I-D_ext S_ext; 0.5I+D_int -g*S_int]
    end
    return (; A, quad, k, formulation, derivative_evaluation, correction)
end

function _mixed_refinement_options(options)
    all(key -> key in (:tolerance, :maxiter, :condition_guard), keys(options)) ||
        throw(ArgumentError("unknown refinement option"))
    tolerance = get(options, :tolerance, 1e-13)
    maxiter = get(options, :maxiter, 8)
    condition_guard = get(options, :condition_guard, 10eps(Float32))
    isfinite(tolerance) && 0 < tolerance < 1 ||
        throw(ArgumentError("invalid refinement tolerance"))
    maxiter isa Integer && maxiter >= 0 ||
        throw(ArgumentError("maxiter must be a nonnegative integer"))
    isfinite(condition_guard) && condition_guard >= 0 ||
        throw(ArgumentError("invalid condition guard"))
    return (; tolerance, maxiter, condition_guard)
end

function _factor_fluid_system(A; equilibrate::Bool = true,
        condition_limit::Integer = 512, norm_floor = 0.0,
        precision::Symbol = :double, refinement::NamedTuple = (;))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    precision in (:double, :mixed) ||
        throw(ArgumentError("precision must be :double or :mixed"))
    if precision === :mixed
        equilibrate || throw(ArgumentError("mixed precision requires equilibrate=true"))
        iszero(norm_floor) ||
            throw(ArgumentError("mixed precision does not support norm_floor"))
        options = _mixed_refinement_options(refinement)
        factorization = MixedRefinement.factorize_refined(A; options.condition_guard)
        row_norms, col_norms = factorization.rows, factorization.columns
        compute_condition = size(A, 1) <= condition_limit
        diagnostics = (; equilibrate, condition_limit,
            condition_number = compute_condition ? cond(A) : nothing,
            scaled_condition_number = compute_condition ?
                                      cond(A ./ row_norms ./ transpose(col_norms)) :
                                      nothing,
            conditioning = compute_condition ? :svd : :not_computed)
        return (; factorization, row_norms, col_norms, diagnostics, refinement = options)
    end
    isempty(refinement) ||
        throw(ArgumentError("refinement options require precision=:mixed"))
    if equilibrate
        row_norms = max.(vec(maximum(abs, A; dims = 2)), norm_floor)
        row_norms = ifelse.(iszero.(row_norms), 1.0, row_norms)
        scaled_A = A ./ row_norms
        col_norms = max.(vec(maximum(abs, scaled_A; dims = 1)), norm_floor)
        col_norms = ifelse.(iszero.(col_norms), 1.0, col_norms)
        scaled_A ./= transpose(col_norms)
    else
        scaled_A = copy(A)
        row_norms = col_norms = nothing
    end
    compute_condition = size(A, 1) <= condition_limit
    condition_number = compute_condition ? cond(A) : nothing
    scaled_condition_number = compute_condition ?
                              (equilibrate ? cond(scaled_A) : condition_number) : nothing
    # Keep A for true residuals; the scaled workspace becomes the LU storage.
    factorization = LinearAlgebra.lu!(scaled_A)
    diagnostics = (;
        equilibrate, condition_limit, condition_number, scaled_condition_number,
        conditioning = compute_condition ? :svd : :not_computed)
    return (; factorization, row_norms, col_norms, diagnostics)
end

function _fluid_residual_report(residual, b, row_norms)
    report = _residual_report(residual, b)
    # The residual in equilibrated equations is D_r^-1 * (A*x - b).
    # Evaluate it from the original system, independently of the overwritten LU workspace.
    scaled = row_norms === nothing ? report :
             _residual_report(residual ./ row_norms, b ./ row_norms)
    return merge(report,
        (; scaled_relative_residual = scaled.relative_residual,
            scaled_absolute_residual = scaled.absolute_residual))
end

function _solve_fluid_system(factor, b)
    if factor.factorization isa MixedRefinement.RefinedLU
        solved = MixedRefinement.solve_refined(factor.factorization, b;
            tolerance = factor.refinement.tolerance, maxiter = factor.refinement.maxiter)
        refinement_report = (; precision = :mixed, factor_precision = solved.precision,
            refinement_reason = solved.reason, refinement_iterations = solved.corrections,
            normwise_backward_error = solved.normwise,
            componentwise_backward_error = solved.componentwise,
            refinement_history = solved.history,
            refinement_options = factor.refinement,
            single_reciprocal_condition = factor.factorization.reciprocal_condition)
        return (; solved.x, scaled_x = solved.x .* factor.col_norms,
            scaled_b = b ./ factor.row_norms, refinement_report)
    end
    scaled_b = factor.row_norms === nothing ? b : b ./ factor.row_norms
    scaled_x = factor.factorization \ scaled_b
    x = factor.col_norms === nothing ? scaled_x : scaled_x ./ factor.col_norms
    return (; x, scaled_x, scaled_b)
end

function _fluid_forcing(system, p_inc, dpdn_inc)
    n = length(system.quad)
    if system.correction.method === :edge
        return [p_inc; zeros(ComplexF64, n)]
    elseif system.formulation === :muller
        return [p_inc; dpdn_inc]
    end
    rhs = zeros(ComplexF64, 2n)
    # Preserve the eliminated interior equation's discrete forcing.
    mul!(view(rhs, (n + 1):(2n)), view(system.A, (n + 1):(2n), :),
        [p_inc; dpdn_inc], -1, 0)
    return rhs
end

function _solve_full_fluid(system, factor;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        incident = nothing,
        return_diagnostics::Bool = true)
    incident = _resolve_incident(
        system.k, incidence_angle, incidence_azimuth; incident)
    hasproperty(system, :compression) &&
        return _solve_compressed_fluid(system, factor; incidence_angle, incidence_azimuth,
            incident, return_diagnostics)
    (; A, quad, k, formulation, derivative_evaluation, correction) = system
    n = length(quad)
    p_inc, dpdn_inc = _incident_traces(
        quad, k, incidence_angle, incidence_azimuth, incident)
    b = _fluid_forcing(system, p_inc, dpdn_inc)
    solved = _solve_fluid_system(factor, b)
    x = solved.x
    p_scat = x[1:n]
    dpdn_scat = x[(n + 1):(2n)]
    if formulation == :muller || correction.method === :edge
        p_scat = p_scat - p_inc
        dpdn_scat = dpdn_scat - dpdn_inc
    end

    if return_diagnostics
        diagnostics = merge(
            _fluid_residual_report(A * x - b, b, factor.row_norms), factor.diagnostics,
            (
                method = :direct, converged = nothing, iterations = nothing,
                formulation, derivative_evaluation,
                residual_history = Float64[], unknown_count = size(A, 1), quadrature_nodes = n,
                compression = (method = :none,), correction = correction, solver_options = (;)))
        if hasproperty(solved, :refinement_report)
            diagnostics = merge(diagnostics, solved.refinement_report,
                (; method = :mixed_refinement, converged = true,
                    iterations = solved.refinement_report.refinement_iterations))
        end
        return p_scat, dpdn_scat, quad, diagnostics
    end
    return p_scat, dpdn_scat, quad
end

"""
    far_field(quad, xhat, k, p_scat, dpdn_scat)

Far-field scattering amplitude f(x̂) [m] extrapolated from a full 3D BEM
surface solution (as returned by [`solve_full_bem`](@ref)) at observation
direction `xhat` (a unit `SVector{3}`, the direction the receiver looks
*from*, so `xhat = -d̂` is monostatic backscatter for an incident direction
`d̂`), via the same Kirchhoff-Helmholtz far-field reduction
`axisymmetric_bem.jl`'s own `far_field` uses, now a genuine 3D surface
integral (no azimuthal closed form needed, since there's no assumed
symmetry to exploit).
"""
function far_field(quad, xhat::AbstractVector, k::Real, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number})
    total = zero(ComplexF64)
    for (i, q) in enumerate(quad)
        y = q.coords
        xdotn = dot(xhat, q.normal)
        xdoty = dot(xhat, y)
        total += (im * k * xdotn * p_scat[i] + dpdn_scat[i]) * cis(-k * xdoty) * q.weight
    end
    return -total / (4π)
end

"""
    target_strength(quad, xhat, k, p_scat, dpdn_scat)

Target strength [dB re 1 m²] of a full 3D BEM solution at observation
direction `xhat` (see `far_field`).
"""
function target_strength(
        quad, xhat::AbstractVector, k::Real, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number})
    return target_strength(far_field(quad, xhat, k, p_scat, dpdn_scat))
end
