function _pml_operator(pml, x::SVector{3, Float64})
    if pml.kind === :spherical
        r = norm(x)
        gamma, lambda = _pml_stretch(r, pml.R, pml.thickness, pml.sigma0)
        er = x / r
        projector = er * er'
        return (lambda^2 / gamma) * projector + gamma * (I - projector), gamma * lambda^2
    end
    rotation = SMatrix{3, 3, Float64}(pml.rotation)
    xb = rotation' * x
    xi, eta = _spheroidal_coordinates(pml.kind, pml.c, xb)
    gradient = ForwardDiff.gradient(y -> _spheroidal_coordinates(pml.kind, pml.c, y)[1], xb)
    e_xi = gradient / norm(gradient)
    s = (xi - pml.xi_interface) / (pml.xi_outer - pml.xi_interface)
    xi_derivative = 1 + im * pml.sigma0 * s^2
    stretched = xi + im * pml.sigma0 * (pml.xi_outer - pml.xi_interface) * s^3 / 3
    sign = pml.kind === :prolate ? -1 : 1
    s_eta = sqrt((stretched^2 + sign * eta^2) / (xi^2 + sign * eta^2))
    s_phi = sqrt((stretched^2 + sign) / (xi^2 + sign))
    s_xi = xi_derivative * s_eta / s_phi
    azimuthal = SVector{3, Float64}(-xb[2], xb[1], 0.0)
    tensor = (e_xi * e_xi') / s_xi^2
    if norm(azimuthal) > 1e-10 * max(norm(xb), pml.c)
        e_phi = azimuthal / norm(azimuthal)
        e_eta = cross(e_phi, e_xi)
        tensor += (e_eta * e_eta') / s_eta^2 + (e_phi * e_phi') / s_phi^2
    else
        tensor += (I - e_xi * e_xi') / s_eta^2
    end
    d = s_xi * s_eta * s_phi
    return d * rotation * tensor * rotation', d
end

# Nodes carry the pressure dof `node` and the displacement dofs `n_nodes + 3(node - 1) + component`.
_pressure_dof(node) = node
_displacement_dof(n_nodes, node, component) = n_nodes + 3 * (node - 1) + component

function _facet_loop(body, grid, list)
    fv = Ferrite.FacetValues(_VOLUME_QR_FACET, _VOLUME_IP, _VOLUME_GIP)
    for (cell_id, local_facet) in list
        cell = grid.cells[cell_id]
        coords = [grid.nodes[i].x for i in cell.nodes]
        Ferrite.reinit!(fv, cell, coords, local_facet)
        nodes = collect(cell.nodes)
        for q in 1:Ferrite.getnquadpoints(fv)
            shapes = [Ferrite.shape_value(fv, q, i) for i in eachindex(nodes)]
            body(nodes, shapes, Ferrite.spatial_coordinate(fv, q, coords),
                Ferrite.getnormal(fv, q), Ferrite.getdetJdV(fv, q))
        end
    end
end

# Node numbers of the facets in a (cell, local facet) list.
function _facet_nodes(grid, list)
    nodes = Set{Int}()
    for (cell_id, local_facet) in list
        cell = grid.cells[cell_id]
        for local_dof in Ferrite.facetdof_indices(_VOLUME_IP)[local_facet]
            push!(nodes, cell.nodes[local_dof])
        end
    end
    return sort!(collect(nodes))
end

# Dot product of a vector with a three-component vector or tensor.
_dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# Pointwise plane-wave `pinc(x)`/`gradinc(x)`, shared across solvers, matches to rounding, not bit-for-bit.
function _plane_wave_incident(k, direction)
    pinc = x -> cis(k * _dot3(direction, x))
    gradinc = x -> (im * k) .* direction .* pinc(x)
    return pinc, gradinc
end

# Unit vector of the incident wave direction in the solution frame.
function _volume_direction(incidence_angle, incidence_azimuth)
    return [sin(incidence_angle) * cos(incidence_azimuth),
        sin(incidence_angle) * sin(incidence_azimuth), cos(incidence_angle)]
end

# Stiffness-like and squared-wavenumber-like parts of the global matrix, which add to the system matrix.
struct _VolumeMatrices
    stiff::SparseMatrixCSC{ComplexF64, Int32}
    mass::SparseMatrixCSC{ComplexF64, Int32}
end

# Sparse matrix `stiff + factor * mass`, so that `factor = 1` is the system matrix and `1 + i` its complex-shifted preconditioner.
_volume_sparse(m::_VolumeMatrices, factor) = m.stiff + factor * m.mass

# Number of cells whose element matrices are held as triplets before they are reduced to a sparse matrix.
const _VOLUME_ASSEMBLY_CHUNK = 40_000

# Reference values of the shape functions at the cell quadrature points.
function _volume_reference_values()
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    nb = Ferrite.getnbasefunctions(cv)
    return [Ferrite.shape_value(cv, q, i) for q in 1:Ferrite.getnquadpoints(cv), i in 1:nb]
end

# Element matrices of the fluid, PML and elastic cells, assembled over threads with the squared-wavenumber terms kept apart in `mass`.
# Fluid coefficients are sampled in the weak form of div(grad(p)/rho) + k_local^2*p/rho = 0.
function _volume_matrices(grid, labels, ids, facets, k, model, pml)
    n_nodes = Ferrite.getnnodes(grid)
    ncells = length(grid.cells)
    nb = Ferrite.getnbasefunctions(_VOLUME_IP)
    N = 4n_nodes
    counts = [labels[c] == 0 ? 0 : (labels[c] == _REGION_SOLID ? 9nb^2 : nb^2)
              for c in 1:ncells]
    reference = _volume_reference_values()
    nq = size(reference, 1)
    nthreads = Threads.nthreads()
    empty = SparseMatrixCSC{ComplexF64, Int32}(
        N, N, ones(Int32, N + 1), Int32[], ComplexF64[])
    stiff_total, mass_total = empty, empty
    for chunk in Iterators.partition(1:ncells, _VOLUME_ASSEMBLY_CHUNK)
        chunk_offsets = cumsum(counts[chunk]) .- counts[chunk]
        total = sum(counts[chunk])
        rows, cols = Vector{Int32}(undef, total), Vector{Int32}(undef, total)
        stiff, mass = zeros(ComplexF64, total), zeros(ComplexF64, total)
        Threads.@threads :static for t in 1:nthreads
            cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
            G = zeros(nb, 3)
            S, M = zeros(ComplexF64, nb, nb), zeros(ComplexF64, nb, nb)
            B, MB = zeros(ComplexF64, 3nb, 3nb), zeros(3nb, 3nb)
            D = zeros(nb, 3)
            for local_id in t:nthreads:length(chunk)
                cell_id = chunk[local_id]
                label = labels[cell_id]
                label == 0 && continue
                cell = grid.cells[cell_id]
                coords = [grid.nodes[i].x for i in cell.nodes]
                Ferrite.reinit!(cv, cell, coords)
                offset = chunk_offsets[local_id]
                if label == _REGION_SOLID
                    rho, lame, mu, reduced = model.solid isa Vector ?
                                             model.solid[ids[cell_id]] : model.solid
                    fill!(B, 0.0)
                    fill!(MB, 0.0)
                    fill!(D, 0.0)
                    volume = 0.0
                    for q in 1:nq
                        dV = Ferrite.getdetJdV(cv, q)
                        volume += dV
                        for i in 1:nb
                            g = Ferrite.shape_gradient(cv, q, i)
                            G[i, 1], G[i, 2], G[i, 3] = g[1], g[2], g[3]
                            for a in 1:3
                                D[i, a] += G[i, a] * dV
                            end
                        end
                        for j in 1:nb, i in 1:nb

                            gg = G[i, 1] * G[j, 1] + G[i, 2] * G[j, 2] + G[i, 3] * G[j, 3]
                            for a in 1:3, b in 1:3

                                v = mu * G[i, b] * G[j, a]
                                reduced || (v += lame * G[i, a] * G[j, b])
                                a == b && (v += mu * gg)
                                B[3(i - 1) + a, 3(j - 1) + b] += v * dV
                            end
                            nn = reference[q, i] * reference[q, j] * dV
                            for a in 1:3
                                MB[3(i - 1) + a, 3(j - 1) + a] -= rho * k^2 * nn
                            end
                        end
                    end
                    if reduced
                        # A nearly incompressible solid takes its dilatation as one value per element, which avoids locking.
                        for j in 1:nb, b in 1:3, i in 1:nb, a in 1:3
                            B[3(i - 1) + a, 3(j - 1) + b] += lame * D[i, a] * D[j, b] /
                                                             volume
                        end
                    end
                    for j in 1:nb, b in 1:3, i in 1:nb, a in 1:3
                        ii, jj = 3(i - 1) + a, 3(j - 1) + b
                        p = offset + (3(j - 1) + b - 1) * 3nb + ii
                        rows[p] = _displacement_dof(n_nodes, cell.nodes[i], a)
                        cols[p] = _displacement_dof(n_nodes, cell.nodes[j], b)
                        stiff[p] = B[ii, jj]
                        mass[p] = MB[ii, jj]
                    end
                else
                    medium = if label != _REGION_INTERIOR
                        (1.0, k)
                    elseif model.interior isa Vector
                        model.interior[ids[cell_id]]
                    else
                        model.interior
                    end
                    fill!(S, 0)
                    fill!(M, 0)
                    for q in 1:nq
                        density, wavenumber = _volume_fluid_parameters(medium, cv, q, coords)
                        dV = Ferrite.getdetJdV(cv, q)
                        for i in 1:nb
                            g = Ferrite.shape_gradient(cv, q, i)
                            G[i, 1], G[i, 2], G[i, 3] = g[1], g[2], g[3]
                        end
                        if label == _REGION_PML
                            x = Ferrite.spatial_coordinate(cv, q, coords)
                            A, d = _pml_operator(pml, SVector{3, Float64}(x[1], x[2], x[3]))
                            for j in 1:nb
                                tx = A[1, 1] * G[j, 1] + A[1, 2] * G[j, 2] +
                                     A[1, 3] * G[j, 3]
                                ty = A[2, 1] * G[j, 1] + A[2, 2] * G[j, 2] +
                                     A[2, 3] * G[j, 3]
                                tz = A[3, 1] * G[j, 1] + A[3, 2] * G[j, 2] +
                                     A[3, 3] * G[j, 3]
                                for i in 1:j
                                    S[i, j] += dV *
                                               (G[i, 1] * tx + G[i, 2] * ty + G[i, 3] * tz)
                                    M[i, j] -= wavenumber^2 * d * reference[q, i] *
                                               reference[q, j] * dV
                                end
                            end
                        else
                            w = dV / density
                            for j in 1:nb, i in 1:j

                                S[i, j] += w * (G[i, 1] * G[j, 1] + G[i, 2] * G[j, 2] +
                                            G[i, 3] * G[j, 3])
                                M[i, j] -= wavenumber^2 * w * reference[q, i] *
                                           reference[q, j]
                            end
                        end
                    end
                    for j in 1:nb, i in 1:nb

                        lo, hi = min(i, j), max(i, j)
                        p = offset + (j - 1) * nb + i
                        rows[p] = _pressure_dof(cell.nodes[i])
                        cols[p] = _pressure_dof(cell.nodes[j])
                        stiff[p] = S[lo, hi]
                        mass[p] = M[lo, hi]
                    end
                end
            end
        end
        stiff_total += sparse(rows, cols, stiff, N, N)
        mass_total += sparse(rows, cols, mass, N, N)
    end
    # Elastic body: continuity of normal displacement and traction, with n pointing out of the solid.
    extra_rows, extra_cols = Int32[], Int32[]
    extra_stiff, extra_mass = ComplexF64[], ComplexF64[]
    if model.kind in (:solid, :shell, :regions)
        for list in (facets[:solid_fluid], facets[:solid_interior])
            _facet_loop(grid, list) do nodes, N, x, n, dS
                for (i, ni) in enumerate(nodes), (j, nj) in enumerate(nodes), c in 1:3
                    push!(extra_rows, _pressure_dof(ni))
                    push!(extra_cols, _displacement_dof(n_nodes, nj, c))
                    push!(extra_stiff, 0)
                    push!(extra_mass, k^2 * n[c] * N[i] * N[j] * dS)
                    push!(extra_rows, _displacement_dof(n_nodes, ni, c))
                    push!(extra_cols, _pressure_dof(nj))
                    push!(extra_stiff, n[c] * N[i] * N[j] * dS)
                    push!(extra_mass, 0)
                end
            end
        end
    end
    stiff_total += sparse(extra_rows, extra_cols, extra_stiff, N, N)
    mass_total += sparse(extra_rows, extra_cols, extra_mass, N, N)
    return _VolumeMatrices(stiff_total, mass_total)
end

# Load of a general incident field, given pointwise `pinc(x)`/`gradinc(x)`.
function _volume_load(grid, labels, ids, facets, model, n_nodes, pinc, gradinc)
    load = zeros(ComplexF64, 4n_nodes)
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    nb = Ferrite.getnbasefunctions(cv)
    # Total-field source of the incident wave inside a fluid body.
    for (cell_id, cell) in enumerate(grid.cells)
        labels[cell_id] == _REGION_INTERIOR || continue
        medium = model.interior isa Vector ? model.interior[ids[cell_id]] : model.interior
        coords = [grid.nodes[i].x for i in cell.nodes]
        Ferrite.reinit!(cv, cell, coords)
        for q in 1:Ferrite.getnquadpoints(cv)
            dV = Ferrite.getdetJdV(cv, q)
            x = Ferrite.spatial_coordinate(cv, q, coords)
            density, wavenumber = _volume_fluid_parameters(medium, x)
            p, dp = pinc(x), gradinc(x)
            for i in 1:nb
                g = Ferrite.shape_gradient(cv, q, i)
                gradient = _dot3(dp, g)
                load[_pressure_dof(cell.nodes[i])] -= (gradient -
                                                       wavenumber^2 * p *
                                                       Ferrite.shape_value(cv, q, i)) * dV /
                                                      density
            end
        end
    end
    # Rigid body: the normal derivative of the total pressure vanishes, n pointing out of the body.
    if model.kind === :rigid
        _facet_loop(grid, facets[:fluid_bare]) do nodes, N, x, n, dS
            for (i, node) in enumerate(nodes)
                load[_pressure_dof(node)] -= _dot3(gradinc(x), n) * N[i] * dS
            end
        end
    end
    # A fluid body: the exterior boundary term of the incident wave, with n pointing out of the body.
    if model.kind in (:fluid, :regions)
        _facet_loop(grid, facets[:fluid_interior]) do nodes, N, x, n, dS
            for (i, node) in enumerate(nodes)
                load[_pressure_dof(node)] += _dot3(gradinc(x), n) * N[i] * dS
            end
        end
    end
    if model.kind in (:solid, :shell, :regions)
        for (list, is_exterior) in ((facets[:solid_fluid], true), (
            facets[:solid_interior], false))
            _facet_loop(grid, list) do nodes, N, x, n, dS
                p = pinc(x)
                for (i, ni) in enumerate(nodes)
                    is_exterior &&
                        (load[_pressure_dof(ni)] += _dot3(gradinc(x), n) * N[i] * dS)
                    for c in 1:3
                        load[_displacement_dof(n_nodes, ni, c)] -= p * n[c] * N[i] * dS
                    end
                end
            end
        end
    end
    return load
end

function _harmonic_norms(L)
    logfact = cumsum(vcat(0.0, log.(1:(2L + 1))))
    return [l >= m ? sqrt((2l + 1) / (4pi) * exp(logfact[l - m + 1] - logfact[l + m + 1])) :
            0.0 for l in 0:L, m in 0:L]
end

# Spherical harmonics Y_lm with m >= 0 up to degree L at one point, in the order (l, m) with m = 0:l.
function _harmonics_upto(L, cos_theta, phi, norms)
    sin_theta = sqrt(max(1 - cos_theta^2, 0.0))
    P = zeros(L + 1, L + 1)
    for m in 0:L
        pmm = m == 0 ? 1.0 : (-1)^m * prod(2i - 1 for i in 1:m; init = 1) * sin_theta^m
        P[m + 1, m + 1] = pmm
        m < L && (P[m + 2, m + 1] = cos_theta * (2m + 1) * pmm)
        for l in (m + 2):L
            P[l + 1, m + 1] = ((2l - 1) * cos_theta * P[l, m + 1] -
                               (l + m - 1) * P[l - 1, m + 1]) / (l - m)
        end
    end
    return [norms[l + 1, m + 1] * P[l + 1, m + 1] * cis(m * phi) for l in 0:L for m in 0:l]
end

# Factors of the exact Dirichlet-to-Neumann operator on the sphere of radius R over the boundary nodes of `list`, for degrees up to L.
function _dtn_factors(grid, list, k, R, L)
    boundary = Dict(node => index for (index, node) in enumerate(_facet_nodes(grid, list)))
    norms = _harmonic_norms(L)
    labels = [(l, m) for l in 0:L for m in 0:l]
    Cr = zeros(length(labels), length(boundary))
    Ci = zeros(length(labels), length(boundary))
    _facet_loop(grid, list) do nodes, N, x, n, dS
        Y = _harmonics_upto(L, x[3] / norm(x), atan(x[2], x[1]), norms)
        for (i, node) in enumerate(nodes)
            haskey(boundary, node) || continue
            for h in eachindex(Y)
                Cr[h, boundary[node]] += real(Y[h]) * N[i] * dS
                Ci[h, boundary[node]] -= imag(Y[h]) * N[i] * dS
            end
        end
    end
    weights = [(m == 0 ? 1.0 : 2.0) * k * hsd(l, k * R) / hs(l, k * R) / R^2
               for (l, m) in labels]
    nodes = Vector{Int}(undef, length(boundary))
    for (node, index) in boundary
        nodes[index] = node
    end
    return (; nodes, Cr, Ci, weights)
end

# The operator applied to values `x` on the boundary nodes.
function _dtn_apply(f, x)
    return f.Cr' * (f.weights .* (f.Cr * x)) + f.Ci' * (f.weights .* (f.Ci * x))
end

# Dense global matrix of the Dirichlet-to-Neumann operator.
function _dtn_matrix(f, n_nodes)
    T = f.Cr' * (f.weights .* f.Cr) + f.Ci' * (f.weights .* f.Ci)
    return sparse(
        repeat(f.nodes, outer = length(f.nodes)), repeat(f.nodes, inner = length(f.nodes)),
        vec(T), 4n_nodes, 4n_nodes)
end

# Mass matrix of the pressure on the facets of `list`.
function _boundary_mass(grid, list, n_nodes)
    rows, cols, values = Int[], Int[], Float64[]
    _facet_loop(grid, list) do nodes, N, x, n, dS
        for (i, ni) in enumerate(nodes), (j, nj) in enumerate(nodes)

            push!(rows, _pressure_dof(ni))
            push!(cols, _pressure_dof(nj))
            push!(values, N[i] * N[j] * dS)
        end
    end
    return sparse(rows, cols, values, 4n_nodes, 4n_nodes)
end

# Coefficients of the scattered pressure on the extraction sphere against conj(Y_lm) and against (-1)^m Y_lm, for the positive and negative orders.
function _volume_coefficients(solution, grid, list, n_nodes, L)
    norms = _harmonic_norms(L)
    labels = [(l, m) for l in 0:L for m in 0:l]
    positive = zeros(ComplexF64, length(labels))
    negative = zeros(ComplexF64, length(labels))
    _facet_loop(grid, list) do nodes, N, x, n, dS
        Y = _harmonics_upto(L, x[3] / norm(x), atan(x[2], x[1]), norms)
        pressure = sum(solution[_pressure_dof(nodes[i])] * N[i] for i in eachindex(nodes))
        for (h, (l, m)) in enumerate(labels)
            positive[h] += pressure * conj(Y[h]) * dS
            negative[h] += pressure * (-1)^m * Y[h] * dS
        end
    end
    return positive, negative
end
