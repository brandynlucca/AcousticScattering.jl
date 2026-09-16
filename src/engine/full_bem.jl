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

Solve the direct boundary integral equation for a rigid or pressure-release scatterer over the
full 3D surface `quad` (from [`gmsh_sphere_mesh`](@ref)/
[`gmsh_spheroid_mesh`](@ref)) at a unit-amplitude plane wave arriving from
`incidence_angle`/`incidence_azimuth` [rad] (see [`_bem3d_incidence_direction`](@ref)),
using Inti operators and GMRES, with optional H-matrix compression.
`formulation=:burton_miller` combines the pressure and normal-derivative equations
with coupling `im/k` for the outgoing `exp(im*k*r)` kernel and outward body normals.
It removes fictitious interior resonances. `formulation=:cbie` selects the conventional
pressure equation, which can be singular at these frequencies. Both require `k > 0`.

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
        return_diagnostics::Bool = false)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    formulation in (:cbie, :burton_miller) ||
        throw(ArgumentError("formulation must be :cbie or :burton_miller"))
    d̂ = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    op = Inti.Helmholtz(; k = k, dim = 3)
    S, D = Inti.single_double_layer(;
        op, target = quad, source = quad, compression, correction)
    n = length(quad)
    coupling = formulation == :burton_miller ? im / k : 0.0im
    reciprocal = boundary isa PressureRelease && compression.method != :fmm
    if formulation == :burton_miller && (boundary isa Rigid || !reciprocal)
        K, H = Inti.adj_double_layer_hypersingular(;
            op, target = quad, source = quad, compression, correction)
    end

    p_inc = ComplexF64[cis(k * dot(d̂, q.coords)) for q in quad]
    dpdn_inc = ComplexF64[im * k * dot(d̂, q.normal) * p_inc[i]
                          for (i, q) in enumerate(quad)]

    if boundary isa Rigid
        dpdn_scat = -dpdn_inc
        rhs = S * dpdn_inc
        if formulation == :burton_miller
            work = Vector{ComplexF64}(undef, n)
            A = LinearMap{ComplexF64}(n, n) do y, x
                mul!(y, D, x)
                mul!(work, H, x)
                y .= 0.5 .* x .- y .- coupling .* work
            end
            rhs .+= coupling .* (0.5 .* dpdn_inc .+ K * dpdn_inc)
        else
            A = LinearMap{ComplexF64}((y, x) -> (mul!(y, D, x); y .= 0.5 .* x .- y), n, n)
        end
        p_scat, hist = IterativeSolvers.gmres(A, rhs; log = true, gmres_kwargs...)
        hist.isconverged ||
            @warn "solve_full_bem: GMRES did not converge to the requested tolerance, result may be inaccurate" boundary formulation iters = hist.iters
    else
        p_scat = -p_inc
        rhs = (D * p_scat) .- (0.5 .* p_scat)
        if formulation == :burton_miller
            work = Vector{ComplexF64}(undef, n)
            weighted = similar(work)
            weights = [q.weight for q in quad]
            A = LinearMap{ComplexF64}(n, n) do y, x
                mul!(y, S, x)
                # Reciprocity gives the adjoint operator as a quadrature-weighted transpose.
                if reciprocal
                    weighted .= weights .* conj.(x)
                    mul!(work, adjoint(D), weighted)
                    y .+= coupling .* (0.5 .* x .+ conj.(work) ./ weights)
                else
                    mul!(work, K, x)
                    y .+= coupling .* (0.5 .* x .+ work)
                end
            end
            # Use the analytic incident field as forcing for the total normal trace.
            rhs = reciprocal ? p_inc .+ coupling .* dpdn_inc :
                  rhs .+ coupling .* (H * p_scat)
        else
            A = LinearMap{ComplexF64}((y, x) -> mul!(y, S, x), n, n)
        end
        dpdn_scat, hist = IterativeSolvers.gmres(A, rhs; log = true, gmres_kwargs...)
        formulation == :burton_miller && reciprocal && (dpdn_scat .-= dpdn_inc)
        hist.isconverged ||
            @warn "solve_full_bem: GMRES did not converge to the requested tolerance, result may be inaccurate" boundary formulation iters = hist.iters
    end

    if return_diagnostics
        x = boundary isa Rigid ? p_scat : dpdn_scat
        reciprocal && formulation == :burton_miller && (x = x .+ dpdn_inc)
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
    _fluid_derivative_operators(op, target, source, S, D, correction; regular=true)

Evaluate the normal traces of the fluid layer operators. For coincident surfaces with
density interpolation, `regular=true`, `k*radius <= 1` and `2k*rms_radius <= 1`,
recover them from the Calderón identities
`S*K = D*S` and `S*H = D*D - I/4`. Radii measure distances from the mean node
position; the RMS radius uses surface quadrature area weights. Both are rigid-motion invariant.
This uses the pressure operators consistently at low frequency; it requires a dense
single-layer factorization and is not suitable near its interior Dirichlet resonances.
Other interactions retain direct derivative quadrature.

See van 't Wout et al. (2022), doi:10.1016/j.camwa.2021.11.021, for the operator
identities; their hypersingular operator has the opposite sign to `H` here.
"""
function _fluid_derivative_operators(op, target, source, S, D, correction; regular = true)
    if regular && target === source && correction.method === :dim
        if _fluid_regular_range(op.k, _fluid_quadrature_size(source))
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
`equilibrate=true` scales rows and columns by their maximum absolute entry before
factorization, then recovers the physical traces. This scaling does not change the equations.
Both formulations require positive finite wavenumber and material contrasts.
For density interpolation at `k*radius <= 1` and `2k*rms_radius <= 1`, self-interaction
normal derivatives use Calderón identities with the pressure operators. The enclosing
radius measures distances from the mean node position; the RMS radius weights squared
distances by surface quadrature area. Other interactions
use direct derivative quadrature.

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
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    formulation in (:muller, :cbie) ||
        throw(ArgumentError("fluid formulation must be :muller or :cbie"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    d̂ = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    g = boundary.density_contrast
    h = boundary.soundspeed_contrast
    isfinite(g) && g > 0 && isfinite(h) && h > 0 ||
        throw(ArgumentError("fluid density and soundspeed contrasts must be finite and positive"))
    k_int = k / boundary.soundspeed_contrast

    op_ext = Inti.Helmholtz(; k = k, dim = 3)
    op_int = Inti.Helmholtz(; k = k_int, dim = 3)
    S_ext, D_ext = _fluid_layer_operators(op_ext, quad, quad, correction;
        regular = formulation === :muller)
    S_int, D_int = _fluid_layer_operators(op_int, quad, quad, correction;
        regular = formulation === :muller)
    n = length(quad)

    p_inc = ComplexF64[cis(k * dot(d̂, q.coords)) for q in quad]
    dpdn_inc = ComplexF64[im * k * dot(d̂, q.normal) * p_inc[i]
                          for (i, q) in enumerate(quad)]

    Id = Matrix{ComplexF64}(I, n, n)
    derivative_evaluation = nothing
    if formulation == :muller
        K_ext, H_ext, exterior = _fluid_derivative_operators(
            op_ext, quad, quad, S_ext, D_ext, correction)
        K_int, H_int, interior = _fluid_derivative_operators(
            op_int, quad, quad, S_int, D_int, correction)
        derivative_evaluation = (; exterior, interior)
        A = [Id-D_ext+D_int S_ext-g*S_int; -H_ext+H_int/g Id+K_ext-K_int]
        b = [p_inc; dpdn_inc]
    else
        off_p, off_d, off_pi, off_di = 0, n, 2n, 3n
        A = zeros(ComplexF64, 4n, 4n)
        b = zeros(ComplexF64, 4n)
        rows = 1:n
        A[rows, (off_p + 1):(off_p + n)] = 0.5 .* Id .- D_ext
        A[rows, (off_d + 1):(off_d + n)] = S_ext
        rows = (n + 1):(2n)
        A[rows, (off_pi + 1):(off_pi + n)] = 0.5 .* Id .+ D_int
        A[rows, (off_di + 1):(off_di + n)] = -S_int
        rows = (2n + 1):(3n)
        A[rows, (off_p + 1):(off_p + n)] = Id
        A[rows, (off_pi + 1):(off_pi + n)] = -Id
        b[rows] = -p_inc
        rows = (3n + 1):(4n)
        A[rows, (off_d + 1):(off_d + n)] = Id
        A[rows, (off_di + 1):(off_di + n)] = -Id ./ g
        b[rows] = -dpdn_inc
    end

    if equilibrate
        row_norms = vec(maximum(abs, A; dims = 2))
        row_norms = ifelse.(iszero.(row_norms), 1.0, row_norms)
        scaled_A = A ./ row_norms
        col_norms = vec(maximum(abs, scaled_A; dims = 1))
        col_norms = ifelse.(iszero.(col_norms), 1.0, col_norms)
        scaled_A ./= transpose(col_norms)
        scaled_b = b ./ row_norms
        scaled_x = scaled_A \ scaled_b
        x = scaled_x ./ col_norms
    else
        scaled_A, scaled_b = A, b
        x = scaled_x = A \ b
    end
    p_scat = x[1:n]
    dpdn_scat = x[(n + 1):(2n)]
    if formulation == :muller
        p_scat = p_scat - p_inc
        dpdn_scat = dpdn_scat - dpdn_inc
    end

    if return_diagnostics
        scaled_report = _linear_residual(scaled_A, scaled_x, scaled_b)
        compute_condition = size(A, 1) <= condition_limit
        condition_number = compute_condition ? cond(A) : nothing
        scaled_condition_number = compute_condition ?
                                  (equilibrate ? cond(scaled_A) : condition_number) :
                                  nothing
        diagnostics = merge(_linear_residual(A, x, b),
            (
                method = :direct, converged = nothing, iterations = nothing,
                formulation, derivative_evaluation, equilibrate, condition_limit,
                condition_number, scaled_condition_number,
                conditioning = compute_condition ? :svd : :not_computed,
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
