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
            catch err
                # Only a rejected Jacobian warrants remeshing. Propagate programming
                # errors instead of reporting them as inverted elements.
                err isa ArgumentError && startswith(err.msg, "det(J) is not positive:") ||
                    rethrow()
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
        # Repair the straight tetrahedra before projecting quadratic edge nodes.
        gmsh.option.setNumber("Mesh.OptimizeNetgen", get(sizes, :optimize_linear, false))
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
