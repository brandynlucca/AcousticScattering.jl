# Full-3D Nyström BEM with Inti operators and Gmsh surface meshes.

"""
    gmsh_sphere_mesh(radius; meshsize, qorder=4, mesh_order=2)

Full 3D triangulated surface quadrature (an `Inti.Quadrature`) for a sphere
of the given `radius` [m], via Gmsh's OpenCASCADE kernel. `meshsize` [m] is
the target element edge length, see [`bem3d_elements_per_wavelength`](@ref)
for a wavelength-based default. `mesh_order=2` uses curved quadratic triangles;
`mesh_order=1` uses flat triangles and `mesh_order=3` uses cubic triangles.
`qorder` controls integration separately.
"""
function gmsh_sphere_mesh(radius::Real; meshsize::Real, qorder::Integer = 4,
        mesh_order::Integer = 2)
    meshsize > 0 || throw(ArgumentError("meshsize must be positive, got $meshsize"))
    mesh_order in (1, 2, 3) || throw(ArgumentError("mesh_order must be 1, 2 or 3"))
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("sphere")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(mesh_order)
        gmsh.model.mesh.affineTransform([0.0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
    return Inti.Quadrature(view(msh, Γ); qorder = qorder)
end

"""
    gmsh_spheroid_mesh(a, b; meshsize, qorder=4, mesh_order=2)

Full 3D triangulated surface quadrature for a prolate/oblate spheroid with
semi-axis `a` [m] along the x axis of symmetry and equatorial semi-axis `b`
[m] (matching [`Spheroid`](@ref)'s convention: prolate if `a > b`, oblate
if `a < b`), via Gmsh (a unit sphere, non-uniformly scaled with OpenCASCADE's
`dilate`). `mesh_order=2` uses curved quadratic triangles; `mesh_order=1` uses
flat triangles and `mesh_order=3` uses cubic triangles. `qorder` controls integration separately.
"""
function gmsh_spheroid_mesh(a::Real, b::Real; meshsize::Real, qorder::Integer = 4,
        mesh_order::Integer = 2)
    meshsize > 0 || throw(ArgumentError("meshsize must be positive, got $meshsize"))
    mesh_order in (1, 2, 3) || throw(ArgumentError("mesh_order must be 1, 2 or 3"))
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("spheroid")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        tag = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, tag)], 0.0, 0.0, 0.0, b, b, a)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(mesh_order)
        gmsh.model.mesh.affineTransform([0.0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
    return Inti.Quadrature(view(msh, Γ); qorder = qorder)
end

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
`incidence_angle`/`incidence_azimuth` [rad] (see [`_bem3d_incidence_direction`](@ref)),
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
[`far_field`](@ref)). With `return_diagnostics=true`, append a fourth element containing
convergence history, recomputed linear residuals and solver settings.
"""
function solve_full_bem(boundary::Union{Rigid, PressureRelease}, k::Real, quad;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
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
        gmres_kwargs, return_diagnostics, _density)
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
            coupling = 0.0im, formulation, compression, correction)
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
        formulation, compression, correction)
end

function _solve_full_boundary(system;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        gmres_kwargs = (reltol = 1e-4, restart = 150, maxiter = 1200),
        return_diagnostics::Bool = true, _density = nothing)
    (; A, S, D, K, H, boundary, k, quad, coupling,
        formulation, compression, correction) = system
    direction = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    n = length(quad)
    p_inc = ComplexF64[cis(k * dot(direction, q.coords)) for q in quad]
    dpdn_inc = ComplexF64[im * k * dot(direction, q.normal) * p_inc[i]
                          for (i, q) in enumerate(quad)]
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
    x, hist = IterativeSolvers.gmres(A, rhs; log = true, gmres_kwargs...)
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
                   correction=(method=:dim,), return_diagnostics=false)

Solve fluid transmission over the full 3D surface `quad` using a dense direct solve.
`formulation=:muller` uses the pressure and exterior normal derivative as two
unknown traces. `formulation=:cbie` uses four traces and explicit interface conditions.
With `correction=(method=:edge,)`, CBIE eliminates the interior traces through those
conditions and uses material-dependent corner powers for pressure and normal flux.
The constant-density Laplace double-layer identity regularizes both pressure equations;
this removes the constant background before integrating the singular part of the kernel.
Edge quadrature requires a closed conforming triangular surface and does not use
hypersingular operators.
Use `bem` with `pressure` and `scattering_amplitude` to retain the weighted corner
interpolation when evaluating fields.
`equilibrate=true` scales rows and columns by their maximum absolute entry before
factorization, then recovers the physical traces. This scaling does not change the equations.
Both formulations require positive finite wavenumber and material contrasts.
With density interpolation, regular-wave pressure quadrature requires
`k*radius <= 1` and `2k*rms_radius <= 1` in both
media. The enclosing radius measures distances from the mean node position; the RMS
radius uses surface quadrature area weights. The quadrature choice is shared by both
media to preserve cancellation at weak-contrast interfaces. Otherwise, both media
use direct density interpolation. Separately, Calderón reconstruction of self normal
derivatives requires `k*radius <= π/2` in both media and uses those pressure operators.

Returns `(p_scat, dpdn_scat, quad)`, the *exterior scattered* pressure and
its normal derivative, in the same form [`far_field`](@ref)/
[`target_strength`](@ref) expect (the interior trace is discarded, matching
`axisymmetric_bem.jl`'s own `solve_axial(::FluidFilled, ...)` convention).
With `return_diagnostics=true`, append original and scaled system residuals and settings.
For at most `condition_limit` unknowns, also compute both matrix 2-norm condition numbers.
Zero disables this SVD calculation. A direct solve has no iterative convergence flag or history.
`derivative_evaluation` records the exterior and interior operator evaluation methods.
"""
function solve_full_bem(boundary::FluidFilled, k::Real, quad;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        formulation::Symbol = :muller, equilibrate::Bool = true,
        condition_limit::Integer = 512,
        correction = (method = :dim,), return_diagnostics::Bool = false)
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    system = _assemble_full_fluid(boundary, k, quad; formulation, correction)
    factor = _factor_fluid_system(system.A; equilibrate,
        condition_limit = return_diagnostics ? condition_limit : 0)
    return _solve_full_fluid(system, factor; incidence_angle, incidence_azimuth,
        return_diagnostics)
end

function _assemble_full_fluid(boundary::FluidFilled, k::Real, quad;
        formulation::Symbol = :muller, correction = (method = :dim,))
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    formulation in (:muller, :cbie) ||
        throw(ArgumentError("fluid formulation must be :muller or :cbie"))
    g = boundary.density_contrast
    h = boundary.soundspeed_contrast
    isfinite(g) && g > 0 && isfinite(h) && h > 0 ||
        throw(ArgumentError("fluid density and soundspeed contrasts must be finite and positive"))
    k_int = k / boundary.soundspeed_contrast
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

    Id = Matrix{ComplexF64}(I, n, n)
    derivative_evaluation = nothing
    if formulation == :muller
        K_ext, H_ext, exterior = _fluid_derivative_operators(
            op_ext, quad, quad, S_ext, D_ext, correction; regular, reconstruct)
        K_int, H_int, interior = _fluid_derivative_operators(
            op_int, quad, quad, S_int, D_int, correction; regular, reconstruct)
        derivative_evaluation = (; exterior, interior)
        A = [Id-D_ext+D_int S_ext-g*S_int; -H_ext+H_int/g Id+K_ext-K_int]
    else
        off_p, off_d, off_pi, off_di = 0, n, 2n, 3n
        A = zeros(ComplexF64, 4n, 4n)
        rows = 1:n
        A[rows, (off_p + 1):(off_p + n)] = 0.5 .* Id .- D_ext
        A[rows, (off_d + 1):(off_d + n)] = S_ext
        rows = (n + 1):(2n)
        A[rows, (off_pi + 1):(off_pi + n)] = 0.5 .* Id .+ D_int
        A[rows, (off_di + 1):(off_di + n)] = -S_int
        rows = (2n + 1):(3n)
        A[rows, (off_p + 1):(off_p + n)] = Id
        A[rows, (off_pi + 1):(off_pi + n)] = -Id
        rows = (3n + 1):(4n)
        A[rows, (off_d + 1):(off_d + n)] = Id
        A[rows, (off_di + 1):(off_di + n)] = -Id ./ g
    end
    return (; A, quad, k, formulation, derivative_evaluation, correction)
end

function _factor_fluid_system(A; equilibrate::Bool = true,
        condition_limit::Integer = 512, norm_floor = 0.0)
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    if equilibrate
        row_norms = max.(vec(maximum(abs, A; dims = 2)), norm_floor)
        row_norms = ifelse.(iszero.(row_norms), 1.0, row_norms)
        scaled_A = A ./ row_norms
        col_norms = max.(vec(maximum(abs, scaled_A; dims = 1)), norm_floor)
        col_norms = ifelse.(iszero.(col_norms), 1.0, col_norms)
        scaled_A ./= transpose(col_norms)
    else
        scaled_A = A
        row_norms = col_norms = nothing
    end
    factorization = lu(scaled_A)
    compute_condition = size(A, 1) <= condition_limit
    condition_number = compute_condition ? cond(A) : nothing
    scaled_condition_number = compute_condition ?
                              (equilibrate ? cond(scaled_A) : condition_number) : nothing
    diagnostics = (;
        equilibrate, condition_limit, condition_number, scaled_condition_number,
        conditioning = compute_condition ? :svd : :not_computed)
    return (; factorization, scaled_A, row_norms, col_norms, diagnostics)
end

function _solve_fluid_system(factor, b)
    scaled_b = factor.row_norms === nothing ? b : b ./ factor.row_norms
    scaled_x = factor.factorization \ scaled_b
    x = factor.col_norms === nothing ? scaled_x : scaled_x ./ factor.col_norms
    return (; x, scaled_x, scaled_b)
end

function _solve_full_fluid(system, factor;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        return_diagnostics::Bool = true)
    (; A, quad, k, formulation, derivative_evaluation, correction) = system
    direction = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    n = length(quad)
    p_inc = ComplexF64[cis(k * dot(direction, q.coords)) for q in quad]
    dpdn_inc = ComplexF64[im * k * dot(direction, q.normal) * p_inc[i]
                          for (i, q) in enumerate(quad)]
    b = correction.method === :edge ? [p_inc; zeros(ComplexF64, n)] :
        formulation === :muller ? [p_inc; dpdn_inc] :
        [zeros(ComplexF64, 2n); -p_inc; -dpdn_inc]
    (; x, scaled_x, scaled_b) = _solve_fluid_system(factor, b)
    p_scat = x[1:n]
    dpdn_scat = x[(n + 1):(2n)]
    if formulation == :muller || correction.method === :edge
        p_scat = p_scat - p_inc
        dpdn_scat = dpdn_scat - dpdn_inc
    end

    if return_diagnostics
        scaled_report = _linear_residual(factor.scaled_A, scaled_x, scaled_b)
        diagnostics = merge(_linear_residual(A, x, b), factor.diagnostics,
            (
                method = :direct, converged = nothing, iterations = nothing,
                formulation, derivative_evaluation,
                scaled_relative_residual = scaled_report.relative_residual,
                scaled_absolute_residual = scaled_report.absolute_residual,
                residual_history = Float64[], unknown_count = size(A, 1), quadrature_nodes = n,
                compression = (method = :none,), correction = correction, solver_options = (;)))
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
direction `xhat` (see [`far_field`](@ref)).
"""
function target_strength(
        quad, xhat::AbstractVector, k::Real, p_scat::AbstractVector{<:Number},
        dpdn_scat::AbstractVector{<:Number})
    return target_strength(far_field(quad, xhat, k, p_scat, dpdn_scat))
end
