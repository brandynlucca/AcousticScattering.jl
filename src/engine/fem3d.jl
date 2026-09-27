# Full-3D finite elements on order-2 curved tetrahedra (Gmsh meshes, Ferrite element integrals). The scattered pressure is solved in
# the fluid regions and the displacement in an elastic body, with a spherical PML or an exact spherical Dirichlet-to-Neumann exterior closure.

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

# Rotation taking the body frame to the frame in which the incident wave travels along +z.
function _volume_rotation(incidence_angle::Real, incidence_azimuth::Real)
    cz, sz = cos(incidence_azimuth), sin(incidence_azimuth)
    cy, sy = cos(incidence_angle), sin(incidence_angle)
    Rz = [cz sz 0; -sz cz 0; 0 0 1]
    Ry = [cy 0 -sy; 0 1 0; sy 0 cy]
    return Ry * Rz
end

# Whether every element of the grid has a positive Jacobian at all quadrature points.
function _volume_valid_grid(grid)
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    for cell in grid.cells
        try
            Ferrite.reinit!(cv, cell, [grid.nodes[i].x for i in cell.nodes])
        catch
            return false
        end
    end
    return true
end

_body_level(x, semi_axes) = (x[1]^2 + x[2]^2) / semi_axes[1]^2 + x[3]^2 / semi_axes[2]^2

# Target element size of the region containing a point, read by the Gmsh size callback.
const _MESH_REGION_SIZE = Ref{Any}(nothing)
_mesh_size_callback(dim, tag, x, y, z, lc) = _MESH_REGION_SIZE[](x, y, z, lc)

# Order-2 tetrahedral mesh of the bodies, each a spheroid of `axes`, `center` and `rotation` in the solution frame with its own
# surface `size`, inside the spheroid `domain_axes` rotated by `domain_rotation`. When `interface_axes` is not `nothing` the fluid and
# PML regions split at that spheroid. `classify` maps a cell centroid to its region label and index, and `region_size` maps a point
# to the element size wanted inside its region. Returns the grid, the labels (0 for dropped cells) and the region indices.
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
        gmsh.option.setNumber("Mesh.HighOrderOptimize", get(sizes, :optimize, 1))
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
            gmsh.model.mesh.field.setNumber(distance + 1, "SizeMax", sizes.fluid)
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
    function order_cell(c)
        vertices, mids = c[1:4], c[5:10]
        placed = map(_VOLUME_EDGES) do (i, j)
            target = (coords[:, vertices[i]] + coords[:, vertices[j]]) / 2
            mids[argmin([norm(coords[:, m] - target) for m in mids])]
        end
        ordered = vcat(vertices, collect(placed))
        edges = [coords[:, ordered[i]] - coords[:, ordered[1]] for i in 2:4]
        det(hcat(edges...)) < 0 && (ordered = ordered[[1, 3, 2, 4, 7, 6, 5, 8, 10, 9]])
        return ordered
    end
    cells = [Ferrite.QuadraticTetrahedron(Tuple(order_cell(conn[:, j])))
             for j in axes(conn, 2)]
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
function _pml_operator(pml, x)
    if pml.kind === :spherical
        r = norm(x)
        gamma, lambda = _pml_stretch(r, pml.R, pml.thickness, pml.sigma0)
        er = x / r
        return (lambda^2 / gamma) * (er * er') + gamma * (I - er * er'), gamma * lambda^2
    end
    xb = pml.rotation' * x
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
    azimuthal = [-xb[2], xb[1], 0.0]
    tensor = e_xi * e_xi' / s_xi^2
    if norm(azimuthal) > 1e-10 * max(norm(xb), pml.c)
        e_phi = azimuthal / norm(azimuthal)
        e_eta = cross(e_phi, e_xi)
        tensor += e_eta * e_eta' / s_eta^2 + e_phi * e_phi' / s_phi^2
    else
        tensor += (I - e_xi * e_xi') / s_eta^2
    end
    d = s_xi * s_eta * s_phi
    return d * pml.rotation * tensor * pml.rotation', d
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

_volume_incident(x, k) = cis(k * x[3])

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

# Global sparse system and load for the scattered pressure and, in an elastic body, the displacement.
function _volume_system(grid, labels, ids, facets, k, model, pml)
    n_nodes = Ferrite.getnnodes(grid)
    rows, cols, values = Int[], Int[], ComplexF64[]
    mass_rows, mass_cols, mass_values = Int[], Int[], ComplexF64[]
    load = zeros(ComplexF64, 4n_nodes)
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    nb = Ferrite.getnbasefunctions(cv)
    # The terms proportional to the squared wavenumber are also collected in `mass`, which the iterative solver shifts.
    function push_block!(dofs, block, mass_block)
        for (i, di) in enumerate(dofs), (j, dj) in enumerate(dofs)

            push!(rows, di)
            push!(cols, dj)
            push!(values, block[i, j] + mass_block[i, j])
            push!(mass_rows, di)
            push!(mass_cols, dj)
            push!(mass_values, mass_block[i, j])
        end
    end
    for (cell_id, cell) in enumerate(grid.cells)
        label = labels[cell_id]
        label == 0 && continue
        coords = [grid.nodes[i].x for i in cell.nodes]
        Ferrite.reinit!(cv, cell, coords)
        nodes = collect(cell.nodes)
        if label == _REGION_SOLID
            rho, lame, mu = model.solid isa Vector ? model.solid[ids[cell_id]] : model.solid
            block = zeros(ComplexF64, 3nb, 3nb)
            mass_block = zeros(ComplexF64, 3nb, 3nb)
            for q in 1:Ferrite.getnquadpoints(cv)
                dV = Ferrite.getdetJdV(cv, q)
                G = [Ferrite.shape_gradient(cv, q, i)[c] for i in 1:nb, c in 1:3]
                N = [Ferrite.shape_value(cv, q, i) for i in 1:nb]
                for a in 1:3, b in 1:3

                    part = (lame * G[:, a] * G[:, b]' .+ mu * G[:, b] * G[:, a]') .* dV
                    a == b && (part = part .+ (mu .* (G * G')) .* dV)
                    for i in 1:nb, j in 1:nb

                        block[3(i - 1) + a, 3(j - 1) + b] += part[i, j]
                        a == b &&
                            (mass_block[3(i - 1) + a, 3(j - 1) + b] -= rho * k^2 * N[i] *
                                                                       N[j] * dV)
                    end
                end
            end
            push_block!(
                [_displacement_dof(n_nodes, nodes[i], c) for i in 1:nb for c in 1:3], block,
                mass_block)
        else
            density, wavenumber = if label != _REGION_INTERIOR
                (1.0, k)
            elseif model.interior isa Vector
                model.interior[ids[cell_id]]
            else
                model.interior
            end
            block = zeros(ComplexF64, nb, nb)
            mass_block = zeros(ComplexF64, nb, nb)
            source = zeros(ComplexF64, nb)
            for q in 1:Ferrite.getnquadpoints(cv)
                dV = Ferrite.getdetJdV(cv, q)
                G = [Ferrite.shape_gradient(cv, q, i)[c] for i in 1:nb, c in 1:3]
                N = [Ferrite.shape_value(cv, q, i) for i in 1:nb]
                if label == _REGION_PML
                    x = collect(Ferrite.spatial_coordinate(cv, q, coords))
                    tensor, d = _pml_operator(pml, x)
                    block .+= (G * tensor * G') .* dV
                    mass_block .-= wavenumber^2 * d .* (N * N') .* dV
                else
                    block .+= (G * G') .* dV ./ density
                    mass_block .-= wavenumber^2 .* (N * N') .* dV ./ density
                    if label == _REGION_INTERIOR
                        # Total-field source of the incident wave inside a fluid body.
                        x = Ferrite.spatial_coordinate(cv, q, coords)
                        pinc = _volume_incident(x, k)
                        for i in 1:nb
                            gradient = im * k * pinc * G[i, 3]
                            source[i] -= (gradient - wavenumber^2 * pinc * N[i]) * dV /
                                         density
                        end
                    end
                end
            end
            push_block!([_pressure_dof(n) for n in nodes], block, mass_block)
            for (i, n) in enumerate(nodes)
                load[_pressure_dof(n)] += source[i]
            end
        end
    end
    # Rigid body: the normal derivative of the total pressure vanishes, n pointing out of the body.
    if model.kind === :rigid
        _facet_loop(grid, facets[:fluid_bare]) do nodes, N, x, n, dS
            for (i, node) in enumerate(nodes)
                load[_pressure_dof(node)] -= im * k * n[3] * _volume_incident(x, k) * N[i] *
                                             dS
            end
        end
    end
    # A fluid body: the exterior boundary term of the incident wave, with n pointing out of the body.
    if model.kind in (:fluid, :regions)
        _facet_loop(grid, facets[:fluid_interior]) do nodes, N, x, n, dS
            for (i, node) in enumerate(nodes)
                load[_pressure_dof(node)] += im * k * n[3] * _volume_incident(x, k) * N[i] *
                                             dS
            end
        end
    end
    # Elastic body: continuity of normal displacement and traction, with n pointing out of the solid.
    if model.kind in (:solid, :shell, :regions)
        for (list, is_exterior) in ((facets[:solid_fluid], true), (
            facets[:solid_interior], false))
            _facet_loop(grid, list) do nodes, N, x, n, dS
                pinc = _volume_incident(x, k)
                for (i, ni) in enumerate(nodes)
                    is_exterior &&
                        (load[_pressure_dof(ni)] += im * k * n[3] * pinc * N[i] * dS)
                    for c in 1:3
                        load[_displacement_dof(n_nodes, ni, c)] -= pinc * n[c] * N[i] * dS
                    end
                    for (j, nj) in enumerate(nodes), c in 1:3

                        for (r, cl, v) in ((rows, cols, values), (
                            mass_rows, mass_cols, mass_values))
                            push!(r, _pressure_dof(ni))
                            push!(cl, _displacement_dof(n_nodes, nj, c))
                            push!(v, k^2 * n[c] * N[i] * N[j] * dS)
                        end
                        push!(rows, _displacement_dof(n_nodes, ni, c))
                        push!(cols, _pressure_dof(nj))
                        push!(values, n[c] * N[i] * N[j] * dS)
                    end
                end
            end
        end
    end
    return sparse(rows, cols, values, 4n_nodes, 4n_nodes), load, n_nodes,
    sparse(mass_rows, mass_cols, mass_values, 4n_nodes, 4n_nodes)
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

# Exact Dirichlet-to-Neumann operator on the sphere of radius R over the boundary nodes of `list`, for degrees up to L.
function _dtn_matrix(grid, list, n_nodes, k, R, L)
    boundary = Dict{Int, Int}()
    _facet_loop(grid, list) do nodes, N, x, n, dS
        for node in nodes
            haskey(boundary, node) || (boundary[node] = length(boundary) + 1)
        end
    end
    norms = _harmonic_norms(L)
    labels = [(l, m) for l in 0:L for m in 0:l]
    C = zeros(ComplexF64, length(labels), length(boundary))
    _facet_loop(grid, list) do nodes, N, x, n, dS
        Y = _harmonics_upto(L, x[3] / norm(x), atan(x[2], x[1]), norms)
        for (i, node) in enumerate(nodes), h in eachindex(Y)

            C[h, boundary[node]] += conj(Y[h]) * N[i] * dS
        end
    end
    coefficient = [k * hsd(l, k * R) / hs(l, k * R) for (l, m) in labels]
    weight = [m == 0 ? 1.0 : 2.0 for (l, m) in labels]
    Cr, Ci = real.(C), imag.(C)
    T = (Cr' * ((coefficient .* weight) .* Cr) + Ci' * ((coefficient .* weight) .* Ci)) /
        R^2
    nodes = Vector{Int}(undef, length(boundary))
    for (node, index) in boundary
        nodes[index] = node
    end
    return sparse(
        repeat(nodes, outer = length(nodes)), repeat(nodes, inner = length(nodes)),
        vec(T), 4n_nodes, 4n_nodes)
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

# Smooth step from 0 at t = 0 to 1 at t = 1 with two continuous derivatives.
_smoothstep(t) = t <= 0 ? zero(t) : t >= 1 ? one(t) : t^3 * (10 + t * (6t - 15))

# Data of the volume far-field integral. The cutoff chi rises from 0 at level `lower` to 1 at level `upper` and the fluid cells
# between those levels supply the quadrature points of p * (Laplacian(chi) - 2ik n . grad(chi)) exp(-ik n . x).
function _volume_extraction(solution, grid, labels, domain, lower, upper)
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    chi(x) = _smoothstep((_domain_level(domain, x) - lower) / (upper - lower))
    points, weights, laplacians, gradients = Vector{Float64}[], ComplexF64[], Float64[],
    Vector{Float64}[]
    for (cell_id, cell) in enumerate(grid.cells)
        labels[cell_id] == _REGION_FLUID || continue
        coords = [grid.nodes[i].x for i in cell.nodes]
        levels = [_domain_level(domain, collect(x)) for x in coords]
        (minimum(levels) < upper && maximum(levels) > lower) || continue
        Ferrite.reinit!(cv, cell, coords)
        for q in 1:Ferrite.getnquadpoints(cv)
            x = collect(Ferrite.spatial_coordinate(cv, q, coords))
            pressure = sum(solution[_pressure_dof(cell.nodes[i])] *
                           Ferrite.shape_value(cv, q, i) for i in eachindex(cell.nodes))
            push!(points, x)
            push!(weights, pressure * Ferrite.getdetJdV(cv, q))
            push!(laplacians, sum(diag(ForwardDiff.hessian(chi, x))))
            push!(gradients, ForwardDiff.gradient(chi, x))
        end
    end
    return (; points = reduce(hcat, points), weights, laplacians,
        gradients = reduce(hcat, gradients))
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

# GMRES on `matrix` preconditioned by an incomplete LU factorization of the complex-shifted matrix `shifted`.
function _shifted_gmres(matrix, shifted, rhs, ilu_tolerance, tolerance)
    factor = IncompleteLU.ilu(shifted; τ = ilu_tolerance)
    solution, history = IterativeSolvers.gmres(matrix, rhs; Pr = factor, restart = 100,
        maxiter = 500, reltol = tolerance, log = true)
    return solution, history.isconverged, history.iters
end

# Solve on the active dofs with prescribed values on `fixed`, a dictionary from dof to value. The iterative solver preconditions with
# the shifted matrix returned by `shifted()` and falls back to the direct solver if it does not converge.
function _volume_solve(K, load, grid, labels, n_nodes, fixed; solver = :direct,
        shifted = nothing, ilu_tolerance = 1e-3, tolerance = 1e-8)
    active = falses(4n_nodes)
    for (cell_id, cell) in enumerate(grid.cells)
        labels[cell_id] == 0 && continue
        for node in cell.nodes
            if labels[cell_id] == _REGION_SOLID
                for c in 1:3
                    active[_displacement_dof(n_nodes, node, c)] = true
                end
            else
                active[_pressure_dof(node)] = true
            end
        end
    end
    fixed_dofs = collect(keys(fixed))
    fixed_values = ComplexF64[fixed[d] for d in fixed_dofs]
    is_fixed = falses(4n_nodes)
    is_fixed[fixed_dofs] .= true
    free = findall(active .& .!is_fixed)
    rhs = load[free]
    isempty(fixed_dofs) || (rhs -= K[free, fixed_dofs] * fixed_values)
    solution = zeros(ComplexF64, 4n_nodes)
    solution[fixed_dofs] = fixed_values
    matrix = K[free, free]
    iterate = solver === :iterative ||
              (solver === :auto && shifted !== nothing &&
               length(free) >= _VOLUME_ITERATIVE_DOFS)
    used, iterations = :direct, 0
    if iterate
        shifted_matrix = shifted()[free, free]
        for scale in (1.0, 0.2)
            x, converged, iterations = _shifted_gmres(
                matrix, shifted_matrix, rhs, scale * ilu_tolerance, tolerance)
            if converged
                solution[free] = x
                used = :iterative
                break
            end
        end
        used === :iterative ||
            @warn "the iterative volume solve did not converge, using the direct solver"
    end
    used === :direct && (solution[free] = matrix \ rhs)
    residual = norm(matrix * solution[free] - rhs) / max(norm(rhs), eps())
    return solution, length(free), residual, (; solver = used, iterations)
end

# Result of `fem(...; method = :volume)`. Either spherical-harmonic coefficients of the scattered pressure on the sphere of radius `R`,
# or the quadrature data `shell` of the volume far-field integral.
struct _VolumeFEMData
    positive::Vector{ComplexF64}
    negative::Vector{ComplexF64}
    R::Float64
    L::Int
    shell::Union{Nothing, NamedTuple}
    rotation::Matrix{Float64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    diagnostics::NamedTuple
end

function _volume_amplitude(d::_VolumeFEMData, k::Real; angle::Real, azimuth::Real)
    direction = d.rotation *
                [sin(angle) * cos(azimuth), sin(angle) * sin(azimuth), cos(angle)]
    d.shell === nothing || return _volume_extraction_amplitude(d.shell, k, direction)
    return _volume_far_field(d.positive, d.negative, k, d.R, d.L, direction)
end
