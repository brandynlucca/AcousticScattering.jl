# Full-3D finite elements on order-2 curved tetrahedra (Gmsh meshes, Ferrite integrals) with a PML or exact spherical DtN exterior closure.

const _VOLUME_IP = Ferrite.Lagrange{Ferrite.RefTetrahedron, 2}()
const _VOLUME_GIP = Ferrite.geometric_interpolation(Ferrite.QuadraticTetrahedron)
const _VOLUME_QR_CELL = Ferrite.QuadratureRule{Ferrite.RefTetrahedron}(5)
const _VOLUME_QR_FACET = Ferrite.FacetQuadratureRule{Ferrite.RefTetrahedron}(6)
const _VOLUME_EDGES = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))

# Region labels of a cell.
const _REGION_SOLID = 1
const _REGION_FLUID = 2
const _REGION_PML = 3
const _REGION_INTERIOR = 4

# Whether every kept element of the grid has a positive Jacobian at all quadrature points.
function _volume_valid_grid(grid, labels)
    valid = Threads.Atomic{Bool}(true)
    nthreads = Threads.nthreads()
    Threads.@threads :static for t in 1:nthreads
        cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
        for cell_id in t:nthreads:length(grid.cells)
            labels[cell_id] == 0 && continue
            cell = grid.cells[cell_id]
            try
                Ferrite.reinit!(cv, cell, [grid.nodes[i].x for i in cell.nodes])
            catch
                valid[] = false
            end
        end
    end
    return valid[]
end

_body_level(x, semi_axes) = (x[1]^2 + x[2]^2) / semi_axes[1]^2 + x[3]^2 / semi_axes[2]^2

# Target element size of the region containing a point, read by the Gmsh size callback.
const _MESH_REGION_SIZE = Ref{Any}(nothing)
_mesh_size_callback(dim, tag, x, y, z, lc) = _MESH_REGION_SIZE[](x, y, z, lc)

# Order-2 tetrahedral mesh of the posed bodies inside the domain spheroid, returning the grid, the cell labels (0 if dropped) and region indices.
function _volume_mesh(
        bodies, interface_axes, domain_axes, domain_rotation, sizes, classify, region_size)
    local coords, conn
    gmsh.initialize(String[], false)
    try
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("volume_fem")
        gmsh.option.setNumber("Mesh.MeshSizeMax", sizes.fluid)
        gmsh.option.setNumber("Mesh.MeshSizeMin",
            min(minimum(b.size for b in bodies), sizes.fluid, sizes.interface, sizes.domain) /
            2)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", get(sizes, :optimize, 0))
        outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
        gmsh.model.occ.dilate([(3, outer_sphere)], 0, 0, 0, domain_axes[1], domain_axes[1],
            domain_axes[2])
        domain_tools = [(3, outer_sphere)]
        if interface_axes !== nothing
            tag = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
            gmsh.model.occ.dilate(
                [(3, tag)], 0, 0, 0, interface_axes[1], interface_axes[1],
                interface_axes[2])
            push!(domain_tools, (3, tag))
        end
        _gmsh_rotate!(domain_tools, domain_rotation)
        tools = Tuple{Int, Int}[]
        for body in bodies
            tag = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
            gmsh.model.occ.dilate(
                [(3, tag)], 0, 0, 0, body.axes[1], body.axes[1], body.axes[2])
            _gmsh_rotate!([(3, tag)], body.rotation)
            gmsh.model.occ.translate([(3, tag)], body.center...)
            push!(tools, (3, tag))
        end
        gmsh.model.occ.fragment(domain_tools[1:1], vcat(domain_tools[2:end], tools))
        gmsh.model.occ.synchronize()
        groups = [Float64[] for _ in bodies]
        interface_surfaces, domain_surfaces = Float64[], Float64[]
        for (dim, tag) in gmsh.model.getEntities(2)
            lower, upper = gmsh.model.getParametrizationBounds(dim, tag)
            point = gmsh.model.getValue(dim, tag, (lower .+ upper) ./ 2)
            owner = findfirst(bodies) do body
                abs(_body_level(body.rotation' * (point - body.center), body.axes) - 1) <
                1e-3
            end
            if owner !== nothing
                push!(groups[owner], tag)
            elseif interface_axes !== nothing &&
                   abs(_body_level(domain_rotation' * point, interface_axes) - 1) < 1e-3
                push!(interface_surfaces, tag)
            else
                push!(domain_surfaces, tag)
            end
        end
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        entries = [(groups[i], bodies[i].size,
                       0.5 * maximum(bodies[i].axes) + bodies[i].size)
                   for i in eachindex(bodies)]
        push!(entries,
            (interface_surfaces, sizes.interface,
                1.01 * sizes.interface + 2 * (sizes.fluid - sizes.interface)),
            (domain_surfaces, sizes.domain,
                1.01 * sizes.domain + 2 * (sizes.fluid - sizes.domain)))
        fields = Float64[]
        for (surfaces, size, reach) in entries
            isempty(surfaces) && continue
            distance = 2length(fields) + 1
            gmsh.model.mesh.field.add("Distance", distance)
            gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", surfaces)
            gmsh.model.mesh.field.setNumber(distance, "Sampling", 200)
            gmsh.model.mesh.field.add("Threshold", distance + 1)
            gmsh.model.mesh.field.setNumber(distance + 1, "InField", distance)
            gmsh.model.mesh.field.setNumber(distance + 1, "SizeMin", size)
            # A thin feature only needs to grade up to a nearby scale; the global MeshSizeMax ceiling
            # still applies everywhere once a point falls outside every field's own reach.
            gmsh.model.mesh.field.setNumber(distance + 1, "SizeMax", min(sizes.fluid, 8size))
            gmsh.model.mesh.field.setNumber(distance + 1, "DistMin", size)
            gmsh.model.mesh.field.setNumber(distance + 1, "DistMax", reach)
            push!(fields, distance + 1)
        end
        combined = 2length(fields) + 1
        gmsh.model.mesh.field.add("Min", combined)
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)
        _MESH_REGION_SIZE[] = (x, y, z, lc) -> min(lc, region_size([x, y, z]))
        gmsh.model.mesh.setSizeCallback(_mesh_size_callback)
        gmsh.model.mesh.generate(3)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        index = Dict(Int(t) => i for (i, t) in enumerate(node_tags))
        coords = reshape(node_coords, 3, :)
        _, element_nodes = gmsh.model.mesh.getElementsByType(11)
        conn = reshape([index[Int(t)] for t in element_nodes], 10, :)
    finally
        _MESH_REGION_SIZE[] = nothing
        gmsh.finalize()
    end
    cells = Vector{Ferrite.QuadraticTetrahedron}(undef, size(conn, 2))
    Threads.@threads for j in axes(conn, 2)
        cells[j] = Ferrite.QuadraticTetrahedron(_ferrite_tet10(coords, view(conn, :, j)))
    end
    grid = Ferrite.Grid(cells, [Ferrite.Node(Tuple(coords[:, i])) for i in axes(coords, 2)])
    classified = map(axes(conn, 2)) do j
        centre = sum(coords[:, i] for i in conn[1:4, j]) / 4
        label, id = classify(centre)
        label != _REGION_FLUID && return (label, id)
        interface_axes === nothing && return (_REGION_FLUID, 0)
        outside = _body_level(domain_rotation' * centre, interface_axes) >= 1
        return (outside ? _REGION_PML : _REGION_FLUID, 0)
    end
    return grid, first.(classified), last.(classified)
end

# Ferrite node order of a Gmsh second-order tetrahedron, whose last two edge nodes are swapped and which is reversed if inverted.
function _ferrite_tet10(coords, c)
    ordered = (c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[10], c[9])
    a, b, d = ordered[1], ordered[2], ordered[3]
    e = ordered[4]
    u = (coords[1, b] - coords[1, a], coords[2, b] - coords[2, a],
        coords[3, b] - coords[3, a])
    v = (coords[1, d] - coords[1, a], coords[2, d] - coords[2, a],
        coords[3, d] - coords[3, a])
    w = (coords[1, e] - coords[1, a], coords[2, e] - coords[2, a],
        coords[3, e] - coords[3, a])
    determinant = u[1] * (v[2] * w[3] - v[3] * w[2]) - u[2] * (v[1] * w[3] - v[3] * w[1]) +
                  u[3] * (v[1] * w[2] - v[2] * w[1])
    determinant < 0 && return (ordered[1], ordered[3], ordered[2], ordered[4], ordered[7],
        ordered[6], ordered[5], ordered[8], ordered[10], ordered[9])
    return ordered
end

# Rotate every entity in `tools` about the origin by the matrix `rotation`, as an active rotation about its axis.
function _gmsh_rotate!(tools, rotation)
    isempty(tools) && return
    angle = acos(clamp((sum(diag(rotation)) - 1) / 2, -1.0, 1.0))
    angle < 1e-12 && return
    axis = [rotation[3, 2] - rotation[2, 3], rotation[1, 3] - rotation[3, 1],
        rotation[2, 1] - rotation[1, 2]]
    if norm(axis) < 1e-12
        # A half turn has no antisymmetric part, so read the axis from the symmetric part.
        symmetric = (rotation + I) / 2
        axis = symmetric[:, argmax([symmetric[i, i] for i in 1:3])]
    end
    axis ./= norm(axis)
    gmsh.model.occ.rotate(tools, 0, 0, 0, axis[1], axis[2], axis[3], angle)
end

# Facets keyed by their vertex triple, mapped to the (cell, local facet) pairs that own them.
function _volume_facet_owners(grid)
    owners = Dict{NTuple{3, Int}, Vector{Tuple{Int, Int}}}()
    for (cid, cell) in enumerate(grid.cells)
        for (lf, nodes) in enumerate(Ferrite.facets(cell))
            key = Tuple(sort(collect(nodes[1:3])))
            push!(get!(owners, key, Tuple{Int, Int}[]), (cid, lf))
        end
    end
    return owners
end

# (cell, local facet) lists, each owned by the cell named first in its label pair.
function _volume_facets(grid, labels, on_outer)
    owners = _volume_facet_owners(grid)
    lists = Dict(name => Tuple{Int, Int}[]
    for name in (:solid_fluid, :solid_interior, :fluid_interior, :pml_interface,
        :outer, :fluid_bare, :interior_bare, :solid_bare))
    bare_name(label) = label == _REGION_FLUID ? :fluid_bare :
                       label == _REGION_INTERIOR ? :interior_bare :
                       label == _REGION_SOLID ? :solid_bare : nothing
    for (key, list) in owners
        if length(list) == 2
            (c1, f1), (c2, f2) = list
            l1, l2 = labels[c1], labels[c2]
            # A dropped body leaves the neighbouring cell with a bare facet.
            l1 == 0 && l2 != 0 && push!(lists[bare_name(l2)], (c2, f2))
            l2 == 0 && l1 != 0 && push!(lists[bare_name(l1)], (c1, f1))
            for (owner, neighbour, name) in (
                (_REGION_SOLID, _REGION_FLUID, :solid_fluid),
                (_REGION_SOLID, _REGION_INTERIOR, :solid_interior),
                (_REGION_INTERIOR, _REGION_FLUID, :fluid_interior),
                (_REGION_FLUID, _REGION_PML, :pml_interface))
                (l1, l2) == (owner, neighbour) && push!(lists[name], (c1, f1))
                (l2, l1) == (owner, neighbour) && push!(lists[name], (c2, f2))
            end
        else
            cell, facet = list[1]
            points = [collect(grid.nodes[i].x) for i in key]
            if all(on_outer, points)
                push!(lists[:outer], (cell, facet))
            elseif labels[cell] != 0
                push!(lists[bare_name(labels[cell])], (cell, facet))
            end
        end
    end
    return lists
end

function _pml_stretch(r, R, thickness, sigma0)
    s = (r - R) / thickness
    gamma = 1 + im * sigma0 * s^2
    stretched = r + im * sigma0 * thickness * s^3 / 3
    return gamma, stretched / r
end

# Prolate or oblate spheroidal coordinates (xi, eta) with focal half-distance c of a point in the body frame.
function _spheroidal_coordinates(kind, c, x)
    rho2 = x[1]^2 + x[2]^2
    if kind === :prolate
        near = sqrt(rho2 + (x[3] - c)^2)
        far = sqrt(rho2 + (x[3] + c)^2)
        return (near + far) / (2c), (far - near) / (2c)
    end
    half = (c^2 - rho2 - x[3]^2) / 2
    xi = sqrt((-half + sqrt(half^2 + c^2 * x[3]^2)) / c^2)
    return xi, x[3] / (c * xi)
end

# Level function of the nested surfaces of a domain, the radius for a spherical one and the spheroidal coordinate xi otherwise.
function _domain_level(domain, x)
    domain.kind === :spherical && return norm(x)
    return _spheroidal_coordinates(domain.kind, domain.c, domain.rotation' * x)[1]
end

# Tensor A and scalar d of the PML equation div(A grad p) + k^2 d p = 0 at a point of the solution frame.
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
                    density, wavenumber = if label != _REGION_INTERIOR
                        (1.0, k)
                    elseif model.interior isa Vector
                        model.interior[ids[cell_id]]
                    else
                        model.interior
                    end
                    fill!(S, 0)
                    fill!(M, 0)
                    for q in 1:nq
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
        density, wavenumber = model.interior isa Vector ? model.interior[ids[cell_id]] :
                              model.interior
        coords = [grid.nodes[i].x for i in cell.nodes]
        Ferrite.reinit!(cv, cell, coords)
        for q in 1:Ferrite.getnquadpoints(cv)
            dV = Ferrite.getdetJdV(cv, q)
            x = Ferrite.spatial_coordinate(cv, q, coords)
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

# Result of `fem(...; method = :volume)`, holding harmonic coefficients of the scattered pressure or the volume extraction data `shell`.
struct _VolumeFEMData
    positive::Vector{ComplexF64}
    negative::Vector{ComplexF64}
    R::Float64
    L::Int
    shell::Union{Nothing, NamedTuple}
    rotation::Matrix{Float64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    pinc::Any
    diagnostics::NamedTuple
    system::Any
    solution::Vector{ComplexF64}
end

function _volume_amplitude(d::_VolumeFEMData, k::Real; angle::Real, azimuth::Real)
    direction = d.rotation *
                [sin(angle) * cos(azimuth), sin(angle) * sin(azimuth), cos(angle)]
    d.shell === nothing || return _volume_extraction_amplitude(d.shell, k, direction)
    return _volume_far_field(d.positive, d.negative, k, d.R, d.L, direction)
end

# Smooth step from 0 at t = 0 to 1 at t = 1 with two continuous derivatives.
_smoothstep(t) = t <= 0 ? zero(t) : t >= 1 ? one(t) : t^3 * (10 + t * (6t - 15))

# Solution-independent data of the volume far-field integral, on the fluid cells between the levels where the cutoff chi rises from 0 to 1.
function _volume_extraction_setup(grid, labels, domain, lower, upper)
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    chi(x) = _smoothstep((_domain_level(domain, x) - lower) / (upper - lower))
    reference = _volume_reference_values()
    nq = size(reference, 1)
    cells = Int[]
    points, dVs, laplacians, gradients = Vector{Float64}[], Float64[], Float64[],
    Vector{Float64}[]
    for (cell_id, cell) in enumerate(grid.cells)
        labels[cell_id] == _REGION_FLUID || continue
        coords = [grid.nodes[i].x for i in cell.nodes]
        levels = [_domain_level(domain, collect(x)) for x in coords]
        (minimum(levels) < upper && maximum(levels) > lower) || continue
        Ferrite.reinit!(cv, cell, coords)
        push!(cells, cell_id)
        for q in 1:nq
            x = collect(Ferrite.spatial_coordinate(cv, q, coords))
            push!(points, x)
            push!(dVs, Ferrite.getdetJdV(cv, q))
            push!(laplacians, sum(diag(ForwardDiff.hessian(chi, x))))
            push!(gradients, ForwardDiff.gradient(chi, x))
        end
    end
    nodes = reduce(hcat, [collect(grid.cells[c].nodes) for c in cells])
    return (; nodes, reference, dVs, points = reduce(hcat, points), laplacians,
        gradients = reduce(hcat, gradients))
end

# The extraction data with the quadrature weights p * dV of one pressure solution.
function _volume_extraction(setup, solution)
    nq = size(setup.reference, 1)
    weights = zeros(ComplexF64, length(setup.dVs))
    for c in axes(setup.nodes, 2), q in 1:nq

        pressure = zero(ComplexF64)
        for i in axes(setup.nodes, 1)
            pressure += setup.reference[q, i] * solution[_pressure_dof(setup.nodes[i, c])]
        end
        weights[(c - 1) * nq + q] = pressure * setup.dVs[(c - 1) * nq + q]
    end
    return (; weights, setup.points, setup.laplacians, setup.gradients)
end

# Far-field amplitude in the unit direction `direction` of the solution frame from the volume extraction data.
function _volume_extraction_amplitude(data, k, direction)
    phase = cis.(-k .* (direction' * data.points)[:])
    factor = data.laplacians .- 2im * k .* (direction' * data.gradients)[:]
    return sum(data.weights .* phase .* factor) / 4pi
end

# Far-field amplitude at the unit vector `direction` of the solution frame.
function _volume_far_field(positive, negative, k, R, L, direction)
    norms = _harmonic_norms(L)
    Y = _harmonics_upto(L, direction[3], atan(direction[2], direction[1]), norms)
    total = zero(ComplexF64)
    for (h, (l, m)) in enumerate([(l, m) for l in 0:L for m in 0:l])
        weight = (-im)^(l + 1) / (R^2 * hs(l, k * R))
        total += weight * positive[h] * Y[h]
        m > 0 && (total += weight * negative[h] * (-1)^m * conj(Y[h]))
    end
    return total / k
end

# Number of active dofs above which `solver = :auto` iterates.
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
struct _VolumeSystem
    grid::Any
    labels::Vector{Int}
    ids::Vector{Int}
    facets::Any
    n_nodes::Int
    k::Float64
    model::Any
    closure::Symbol
    L::Int
    R::Float64
    rotation::Matrix{Float64}
    free::Vector{Int}
    fixed::Vector{Int}
    soft::BitVector
    fixed_matrix::SparseMatrixCSC{ComplexF64, <:Integer}
    matrix::SparseMatrixCSC{ComplexF64, <:Integer}
    solver::_VolumeLinearSolver
    extraction::Any
    diagnostics::NamedTuple
end

# Scattered-pressure solution for a plane wave incident from the polar and azimuthal angles, in the frame of the bodies.
function _volume_solution(system::_VolumeSystem, incidence_angle::Real, incidence_azimuth::Real)
    direction = _volume_direction(incidence_angle, incidence_azimuth)
    pinc, gradinc = _plane_wave_incident(system.k, direction)
    return _volume_solution(system, pinc, gradinc, incidence_angle, incidence_azimuth;
        illumination = :plane_wave)
end

# General incident field. Angle args are only kept as default-observation metadata.
function _volume_solution(system::_VolumeSystem, pinc, gradinc,
        incidence_angle::Real, incidence_azimuth::Real; illumination::Symbol = :prescribed)
    n_nodes = system.n_nodes
    load = _volume_load(system.grid, system.labels, system.ids, system.facets,
        system.model, n_nodes, pinc, gradinc)
    values = zeros(ComplexF64, length(system.fixed))
    for (i, dof) in enumerate(system.fixed)
        system.soft[i] && (values[i] = -pinc(system.grid.nodes[dof].x))
    end
    rhs = load[system.free]
    isempty(values) || (rhs -= system.fixed_matrix * values)
    x, info = _solve!(system.solver, rhs)
    solution = zeros(ComplexF64, 4n_nodes)
    solution[system.fixed] = values
    solution[system.free] = x
    residual = norm(_apply(system.solver, x) - rhs) / max(norm(rhs), eps())
    positive, negative, shell = if system.closure === :dtn || system.extraction === nothing
        list = system.closure === :dtn ? system.facets[:outer] :
               system.facets[:pml_interface]
        (_volume_coefficients(solution, system.grid, list, n_nodes, system.L)..., nothing)
    else
        (ComplexF64[], ComplexF64[], _volume_extraction(system.extraction, solution))
    end
    diagnostics = merge(system.diagnostics, (;
        residual, info.solver, info.iterations, illumination))
    return _VolumeFEMData(positive, negative, system.R, system.L, shell, system.rotation,
        Float64(incidence_angle), Float64(incidence_azimuth), pinc, diagnostics, system, solution)
end
