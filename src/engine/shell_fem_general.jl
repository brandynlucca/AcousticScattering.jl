# General (Fourier-mode m >= 0) axisymmetric elastic-shell FEM operator: a full through-thickness
# 2D linear-elasticity solid on a triangular mesh (Ferrite.jl), unlike shell_fem.jl's 1D m=0-only theory.

using Ferrite: Ferrite

# numpy.gradient's default behavior (edge_order=1, unit index spacing)
function _np_gradient(f::AbstractVector{<:Real})
    n = length(f)
    n >= 2 || throw(ArgumentError("_np_gradient needs at least 2 points"))
    g = similar(f, Float64)
    g[1] = f[2] - f[1]
    g[n] = f[n] - f[n - 1]
    for i in 2:(n - 1)
        g[i] = (f[i + 1] - f[i - 1]) / 2
    end
    return g
end

"""
    face_normals_from_curve(ρ, z; prefer_radial_positive)

Outward unit normals of a meridian curve `(ρ[i], z[i])`, estimated via
finite differences of the curve itself (not assuming any particular
parametrization). If `prefer_radial_positive` and the normals' `ρ`-
component is predominantly negative, the whole set is flipped.
"""
function face_normals_from_curve(
        ρ::AbstractVector{<:Real}, z::AbstractVector{<:Real}; prefer_radial_positive::Bool)
    dρ = _np_gradient(ρ)
    dz = _np_gradient(z)
    nz = dρ
    nρ = -dz
    norm_ = sqrt.(nz .^ 2 .+ nρ .^ 2)
    norm_[norm_ .<= 1e-14] .= 1.0
    nz = nz ./ norm_
    nρ = nρ ./ norm_
    if prefer_radial_positive && maximum(nρ) < 0.0
        nz = -nz
        nρ = -nρ
    end
    return nz, nρ
end

"""
    GeneralShellMesh

Triangulated through-thickness meridional cross-section mesh for the
general shell FEM: a `Ferrite.Grid` spanning `n_t` layers from the inner to
outer confocal surface at `n_eta` meridional stations, plus the
outer/inner boundary curves' geometry (coordinates and outward normals)
and the global node indices lying on each.
"""
struct GeneralShellMesh
    grid::Ferrite.Grid
    eta::Vector{Float64}
    outer_ids::Vector{Int}
    inner_ids::Vector{Int}
    outer_ρ::Vector{Float64}
    outer_z::Vector{Float64}
    outer_normal_ρ::Vector{Float64}
    outer_normal_z::Vector{Float64}
    inner_ρ::Vector{Float64}
    inner_z::Vector{Float64}
    inner_normal_ρ::Vector{Float64}
    inner_normal_z::Vector{Float64}
end

"""
    build_structured_shell_strip(geometry, n_eta, n_t; pole_offset=1e-3, eta_override=nothing)

Structured triangular mesh of the shell's through-thickness meridional
cross-section: `n_t` layers (inner to outer confocal surface) at
`n_eta` meridional stations (`uniform_eta_grid`, or `eta_override` if
given). Two triangles per structured quad cell.
"""
function build_structured_shell_strip(
        geometry::ProlateShellGeometry, n_eta::Integer, n_t::Integer;
        pole_offset::Real = 1e-3, eta_override::Union{Nothing, AbstractVector{<:Real}} = nothing)
    n_t >= 2 || throw(ArgumentError("n_t must be at least 2 to resolve shell thickness"))
    eta = eta_override === nothing ? uniform_eta_grid(n_eta; pole_offset = pole_offset) :
          collect(Float64, eta_override)
    n = length(eta)
    a = geometry.semimajor_length
    b = geometry.semiminor_length

    # Use the outer semi-axes and confocal inner surface specified by ProlateShellGeometry.
    sin_theta = sqrt.(max.(0.0, 1.0 .- eta .^ 2))
    z_outer = a .* eta
    ρ_outer = b .* sin_theta
    z_inner = geometry.semimajor_inner .* eta
    ρ_inner = geometry.semiminor_inner .* sin_theta

    onz, onρ = eta ./ a, sin_theta ./ b
    outer_norm = hypot.(onz, onρ)
    onz, onρ = onz ./ outer_norm, onρ ./ outer_norm
    inz, inρ = -eta ./ geometry.semimajor_inner, -sin_theta ./ geometry.semiminor_inner
    inner_norm = hypot.(inz, inρ)
    inz, inρ = inz ./ inner_norm, inρ ./ inner_norm

    t_vals = range(0.0, 1.0; length = n_t)
    node_ids = Matrix{Int}(undef, n_t, n)
    nodes = Vector{Ferrite.Node{2, Float64}}(undef, n_t * n)
    idx = 0
    for (it, t) in enumerate(t_vals)
        z = (1 - t) .* z_inner .+ t .* z_outer
        ρ = (1 - t) .* ρ_inner .+ t .* ρ_outer
        for j in 1:n
            idx += 1
            nodes[idx] = Ferrite.Node((z[j], ρ[j]))
            node_ids[it, j] = idx
        end
    end

    # Ferrite requires counter-clockwise triangle vertices for a positive Jacobian.
    ccw(a, b, c) = begin
        pa, pb, pc = nodes[a].x, nodes[b].x, nodes[c].x
        area2 = (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1])
        area2 >= 0 ? (a, b, c) : (a, c, b)
    end

    cells = Ferrite.Triangle[]
    for it in 1:(n_t - 1), j in 1:(n - 1)

        n00, n01 = node_ids[it, j], node_ids[it, j + 1]
        n10, n11 = node_ids[it + 1, j], node_ids[it + 1, j + 1]
        push!(cells, Ferrite.Triangle(ccw(n00, n10, n11)))
        push!(cells, Ferrite.Triangle(ccw(n00, n11, n01)))
    end
    grid = Ferrite.Grid(cells, nodes)

    return GeneralShellMesh(grid, eta, node_ids[n_t, :], node_ids[1, :],
        ρ_outer, z_outer, onρ, onz, ρ_inner, z_inner, inρ, inz)
end

"""
    build_structured_spherical_shell(outer_radius, thickness, n_eta, n_t; pole_offset=1e-3, eta_override=nothing)

Structured triangular mesh of a spherical shell's through-thickness
meridional cross-section, in the same [`GeneralShellMesh`](@ref) shape as
[`build_structured_shell_strip`](@ref), a sphere is the `a = b` limit of a
prolate spheroid, but that limit is a genuine coordinate singularity of
[`ProlateShellGeometry`](@ref)'s confocal parametrization (`focal_radius =
√(a²-b²) → 0`, so `ProlateShellGeometry` refuses to construct one), so this
uses the sphere's own, much simpler parametrization directly instead of
approaching the `a → b` limit numerically: `η = cos θ ∈ (-1, 1)`,
`z = r η`, `ρ = r √(1-η²)` at any layer radius `r`, with an *exact*
radially-outward normal at every layer. `outer_radius` is the
*outer* shell surface's radius (matching [`ProlateShellGeometry`](@ref)'s
and [`ElasticLayer`](@ref)'s convention); the inner surface is at
`outer_radius - thickness`.
"""
function build_structured_spherical_shell(
        outer_radius::Real, thickness::Real, n_eta::Integer, n_t::Integer;
        pole_offset::Real = 1e-3, eta_override::Union{Nothing, AbstractVector{<:Real}} = nothing)
    n_t >= 2 || throw(ArgumentError("n_t must be at least 2 to resolve shell thickness"))
    0 < thickness < outer_radius ||
        throw(ArgumentError("thickness must lie in (0, outer_radius)"))
    eta = eta_override === nothing ? uniform_eta_grid(n_eta; pole_offset = pole_offset) :
          collect(Float64, eta_override)
    n = length(eta)
    sin_theta = sqrt.(max.(0.0, 1.0 .- eta .^ 2))

    r_outer = outer_radius
    r_inner = outer_radius - thickness
    onρ, onz = sin_theta, eta
    inρ, inz = -onρ, -onz

    t_vals = range(0.0, 1.0; length = n_t)
    node_ids = Matrix{Int}(undef, n_t, n)
    nodes = Vector{Ferrite.Node{2, Float64}}(undef, n_t * n)
    idx = 0
    for (it, t) in enumerate(t_vals)
        r = (1 - t) * r_inner + t * r_outer
        z = r .* eta
        ρ = max.(1e-9, r .* sin_theta)
        for j in 1:n
            idx += 1
            nodes[idx] = Ferrite.Node((z[j], ρ[j]))
            node_ids[it, j] = idx
        end
    end

    ccw(a, b, c) = begin
        pa, pb, pc = nodes[a].x, nodes[b].x, nodes[c].x
        area2 = (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1])
        area2 >= 0 ? (a, b, c) : (a, c, b)
    end

    cells = Ferrite.Triangle[]
    for it in 1:(n_t - 1), j in 1:(n - 1)

        n00, n01 = node_ids[it, j], node_ids[it, j + 1]
        n10, n11 = node_ids[it + 1, j], node_ids[it + 1, j + 1]
        push!(cells, Ferrite.Triangle(ccw(n00, n10, n11)))
        push!(cells, Ferrite.Triangle(ccw(n00, n11, n01)))
    end
    grid = Ferrite.Grid(cells, nodes)

    z_outer = r_outer .* eta
    ρ_outer = max.(1e-9, r_outer .* sin_theta)
    z_inner = r_inner .* eta
    ρ_inner = max.(1e-9, r_inner .* sin_theta)

    return GeneralShellMesh(grid, eta, node_ids[n_t, :], node_ids[1, :],
        ρ_outer, z_outer, onρ, onz, ρ_inner, z_inner, inρ, inz)
end

const _GENERAL_SHELL_IP = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()^3
const _GENERAL_SHELL_QR = Ferrite.QuadratureRule{Ferrite.RefTriangle}(2)

# Global mesh-node index -> its 3 global DOF indices (axial, radial, circumferential). NOT simply
# `3(node-1) .+ (1,2,3)`, Ferrite assigns DOFs in cell-traversal order, not raw grid-node order.
function _node_to_dofs(dh::Ferrite.DofHandler, grid::Ferrite.Grid)
    n_nodes = Ferrite.getnnodes(grid)
    node_dofs = Vector{NTuple{3, Int}}(undef, n_nodes)
    filled = falses(n_nodes)
    for cellid in 1:Ferrite.getncells(grid)
        verts = Ferrite.vertices(grid.cells[cellid])
        dofs = Ferrite.celldofs(dh, cellid)
        for (local_v, node) in enumerate(verts)
            filled[node] && continue
            base = 3 * (local_v - 1)
            node_dofs[node] = (dofs[base + 1], dofs[base + 2], dofs[base + 3])
            filled[node] = true
        end
    end
    all(filled) || throw(ErrorException("some mesh nodes are not referenced by any cell"))
    return node_dofs
end

"""
    GeneralShellOperators

Per-Fourier-mode assembled operators from [`assemble_shell_modal_operators`](@ref):
the (possibly `m = 0`-reduced) dynamic-stiffness matrix, the retained global
DOF indices, and the outer/inner boundary displacement-extraction (`B`) and
pressure-load (`Q`) matrices, already restricted to the retained DOFs.
"""
struct GeneralShellOperators
    dynamic_matrix::SparseMatrixCSC{ComplexF64, Int}
    keep::Vector{Int}
    B_out::SparseMatrixCSC{ComplexF64, Int}
    B_in::SparseMatrixCSC{ComplexF64, Int}
    Q_out::SparseMatrixCSC{ComplexF64, Int}
    Q_in::SparseMatrixCSC{ComplexF64, Int}
end

"""
    assemble_shell_modal_operators(mesh::GeneralShellMesh, m, omega, rho_shell, youngs_modulus, poisson)

Assemble Fourier-mode-`m`'s dynamic-stiffness matrix `K - ω²M` for the
general (full through-thickness, solid-elasticity) shell FEM on `mesh`
(see the module preamble for the strain-displacement relations), plus the
boundary displacement-extraction and pressure-load coupling matrices on
both the outer and inner confocal surfaces. At `m = 0` the
circumferential displacement DOF decouples entirely and is dropped from
the system (`keep` excludes it).
"""
function assemble_shell_modal_operators(mesh::GeneralShellMesh, m::Integer, omega::Real,
        rho_shell::Real, youngs_modulus::Real, poisson::Real)
    mu = youngs_modulus / (2 * (1 + poisson))
    lam = youngs_modulus * poisson / ((1 + poisson) * (1 - 2poisson))
    mc = ComplexF64(m)

    cv = Ferrite.CellValues(_GENERAL_SHELL_QR, _GENERAL_SHELL_IP)
    dh = Ferrite.DofHandler(mesh.grid)
    Ferrite.add!(dh, :u, _GENERAL_SHELL_IP)
    Ferrite.close!(dh)
    n = Ferrite.ndofs(dh)

    n_entries = Ferrite.getncells(mesh.grid) * Ferrite.getnbasefunctions(cv)^2
    rows = Vector{Int}(undef, n_entries)
    columns = Vector{Int}(undef, n_entries)
    values = Vector{ComplexF64}(undef, n_entries)
    entry = 0

    for cellid in 1:Ferrite.getncells(mesh.grid)
        coords = Ferrite.getcoordinates(mesh.grid, cellid)
        Ferrite.reinit!(cv, coords)
        dofs = Ferrite.celldofs(dh, cellid)
        nlocal = Ferrite.getnbasefunctions(cv)
        Ke = zeros(ComplexF64, nlocal, nlocal)
        Me = zeros(ComplexF64, nlocal, nlocal)

        for q in 1:Ferrite.getnquadpoints(cv)
            xq = Ferrite.spatial_coordinate(cv, q, coords)
            r = max(xq[2], 1e-8)
            dOmega = Ferrite.getdetJdV(cv, q)

            for j in 1:nlocal
                uj = Ferrite.shape_value(cv, q, j)
                guj = Ferrite.shape_gradient(cv, q, j)
                exx_u = guj[1, 1]
                err_u = guj[2, 2]
                eph_u = uj[2] / r + im * mc * uj[3] / r
                gxr_u = guj[1, 2] + guj[2, 1]
                gxp_u = guj[3, 1] + im * mc * uj[1] / r
                grp_u = guj[3, 2] - uj[3] / r + im * mc * uj[2] / r
                tr_u = exx_u + err_u + eph_u

                for i in 1:nlocal
                    vi = Ferrite.shape_value(cv, q, i)
                    gvi = Ferrite.shape_gradient(cv, q, i)
                    exx_v = conj(gvi[1, 1])
                    err_v = conj(gvi[2, 2])
                    eph_v = conj(vi[2] / r + im * mc * vi[3] / r)
                    gxr_v = conj(gvi[1, 2] + gvi[2, 1])
                    gxp_v = conj(gvi[3, 1] + im * mc * vi[1] / r)
                    grp_v = conj(gvi[3, 2] - vi[3] / r + im * mc * vi[2] / r)
                    tr_v = exx_v + err_v + eph_v

                    normal = lam * tr_u * tr_v +
                             2mu * (exx_u * exx_v + err_u * err_v + eph_u * eph_v)
                    shear = mu * (gxr_u * gxr_v + gxp_u * gxp_v + grp_u * grp_v)
                    Ke[i, j] += (2pi * r) * (normal + shear) * dOmega

                    mass_term = uj[1] * conj(vi[1]) + uj[2] * conj(vi[2]) +
                                uj[3] * conj(vi[3])
                    Me[i, j] += (2pi * rho_shell * r) * mass_term * dOmega
                end
            end
        end

        for a in 1:nlocal, b in 1:nlocal

            entry += 1
            rows[entry] = dofs[a]
            columns[entry] = dofs[b]
            values[entry] = Ke[a, b] - omega^2 * Me[a, b]
        end
    end

    A_full = sparse(rows, columns, values, n, n)
    node_dofs = _node_to_dofs(dh, mesh.grid)

    keep = collect(1:n)
    if m == 0
        fixed = [node_dofs[node][3] for node in 1:Ferrite.getnnodes(mesh.grid)]
        keep = setdiff(keep, fixed)
    end
    A = A_full[keep, keep]

    B_out_full = build_boundary_displacement_matrix(
        n, node_dofs, mesh.outer_ids, mesh.outer_normal_z, mesh.outer_normal_ρ)
    B_in_full = build_boundary_displacement_matrix(
        n, node_dofs, mesh.inner_ids, mesh.inner_normal_z, mesh.inner_normal_ρ)
    Q_out_full = build_boundary_pressure_load_matrix(
        n, node_dofs, mesh.outer_ids, mesh.outer_z,
        mesh.outer_ρ, mesh.outer_normal_z, mesh.outer_normal_ρ)
    Q_in_full = build_boundary_pressure_load_matrix(
        n, node_dofs, mesh.inner_ids, mesh.inner_z,
        mesh.inner_ρ, mesh.inner_normal_z, mesh.inner_normal_ρ)

    return GeneralShellOperators(A, keep, B_out_full[:, keep], B_in_full[:, keep],
        Q_out_full[keep, :], Q_in_full[keep, :])
end

"""
Boundary displacement-extraction matrix: row `j` picks out the (axial,
radial) displacement DOFs at boundary node `boundary_ids[j]` and projects
onto that node's outward normal, giving the normal displacement pointwise.
"""
function build_boundary_displacement_matrix(
        n_state::Integer, node_dofs::AbstractVector{<:NTuple{3, Integer}},
        boundary_ids::AbstractVector{<:Integer},
        normal_z::AbstractVector{<:Real}, normal_ρ::AbstractVector{<:Real})
    n_bnd = length(boundary_ids)
    B = spzeros(ComplexF64, n_bnd, n_state)
    for j in 1:n_bnd
        dof_z, dof_ρ, _ = node_dofs[boundary_ids[j]]
        B[j, dof_z] = normal_z[j]
        B[j, dof_ρ] = normal_ρ[j]
    end
    return B
end

"""
Boundary pressure-load matrix: consistent nodal-force assembly of a unit
pressure load along the boundary curve (piecewise-linear between
consecutive boundary nodes, 3-point Gauss-Legendre per edge, `2πρ`
axisymmetric line-element weighting), projected onto the (axial, radial)
displacement DOFs via the *local* (per-edge-interpolated) outward normal.
"""
function build_boundary_pressure_load_matrix(
        n_state::Integer, node_dofs::AbstractVector{<:NTuple{3, Integer}},
        boundary_ids::AbstractVector{<:Integer},
        boundary_z::AbstractVector{<:Real}, boundary_ρ::AbstractVector{<:Real},
        normal_z::AbstractVector{<:Real}, normal_ρ::AbstractVector{<:Real})
    n_bnd = length(boundary_ids)
    Q = spzeros(ComplexF64, n_state, n_bnd)
    t_leg, w_leg = Ferrite.getpoints(Ferrite.QuadratureRule{Ferrite.RefLine}(3)),
    Ferrite.getweights(Ferrite.QuadratureRule{Ferrite.RefLine}(3))
    t_nodes = [0.5 * (t[1] + 1.0) for t in t_leg]
    w_nodes = [0.5 * w for w in w_leg]

    for e in 1:(n_bnd - 1)
        za, ρa = boundary_z[e], boundary_ρ[e]
        zb, ρb = boundary_z[e + 1], boundary_ρ[e + 1]
        seg = hypot(zb - za, ρb - ρa)
        seg <= 0 && continue
        dof_a = node_dofs[boundary_ids[e]]
        dof_b = node_dofs[boundary_ids[e + 1]]
        na = (normal_z[e], normal_ρ[e])
        nb = (normal_z[e + 1], normal_ρ[e + 1])

        for (tq, wq) in zip(t_nodes, w_nodes)
            n1 = 1 - tq
            n2 = tq
            ρq = max(1e-9, n1 * ρa + n2 * ρb)
            nqz = n1 * na[1] + n2 * nb[1]
            nqρ = n1 * na[2] + n2 * nb[2]
            nnorm = hypot(nqz, nqρ)
            nnorm <= 0 && continue
            nqz /= nnorm
            nqρ /= nnorm
            weight = 2pi * ρq * seg * wq

            Q[dof_a[1], e] += weight * n1 * nqz * n1
            Q[dof_a[2], e] += weight * n1 * nqρ * n1
            Q[dof_b[1], e] += weight * n2 * nqz * n1
            Q[dof_b[2], e] += weight * n2 * nqρ * n1

            Q[dof_a[1], e + 1] += weight * n1 * nqz * n2
            Q[dof_a[2], e + 1] += weight * n1 * nqρ * n2
            Q[dof_b[1], e + 1] += weight * n2 * nqz * n2
            Q[dof_b[2], e + 1] += weight * n2 * nqρ * n2
        end
    end
    return Q
end
