# Supplied, oriented triangular surfaces for the full-3D solver.

struct _SurfaceGeometry <: AbstractBody
    nodes::Matrix{Float64}
    connectivity::Vector{Vector{Int}}
    labels::Vector{Vector{Pair{Int, String}}}
    units::Symbol
    input_units::Symbol
    orientation::Symbol
    provenance::String
    validation::NamedTuple
end

function _surface_scale(units::Symbol)
    units === :m && return 1.0
    units === :cm && return 0.01
    units === :mm && return 0.001
    throw(ArgumentError("units must be :m, :cm or :mm"))
end

function _surface_session(f)
    gmsh.isInitialized() == 0 ||
        throw(ArgumentError("mesh requires its own Gmsh session; finalize the active session first"))
    try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        return f()
    finally
        gmsh.finalize()
    end
end

"""
    mesh(; semiaxes, resolution=0.5, center=(0,0,0),
        rotation=(axis=(0,0,1), angle=0), tip_ratio=1,
        mesh_order=3, qorder=5, validation=(;))

Construct a closed ellipsoid with `semiaxes=(x,y,z)` and `center` in metres.
`rotation` gives a nonzero axis and a right-handed angle in radians, applied before
translation. Mesh a unit sphere, then stretch, rotate and translate its nodes.
`resolution` is the dimensionless target edge length on that sphere; physical edge
lengths vary with stretching. `tip_ratio` in `(0,1]` reduces the target size at the
local x poles, with size proportional to `1-(1-tip_ratio)*x^4` on the unit sphere.
`mesh_order` (1, 2 or 3) and `qorder` control geometry and quadrature independently.

# Examples
```julia
body = mesh(; semiaxes=(0.1, 0.018, 0.025), tip_ratio=0.4)
bladder = mesh(; semiaxes=(0.025, 0.006, 0.009), center=(0.01, 0.003, 0),
    rotation=(axis=(0,0,1), angle=deg2rad(10)), tip_ratio=0.4)
```
"""
function mesh(; semiaxes, resolution::Real = 0.5, center = (0, 0, 0),
        rotation = (axis = (0, 0, 1), angle = 0), tip_ratio::Real = 1,
        mesh_order::Integer = 3, qorder::Integer = 5, validation::NamedTuple = (;))
    length(semiaxes) == 3 && all(x -> isfinite(x) && x > 0, semiaxes) ||
        throw(ArgumentError("semiaxes must contain three finite positive lengths"))
    length(center) == 3 && all(isfinite, center) ||
        throw(ArgumentError("center must contain three finite coordinates"))
    isfinite(resolution) && resolution > 0 ||
        throw(ArgumentError("resolution must be finite and positive"))
    isfinite(tip_ratio) && 0 < tip_ratio <= 1 ||
        throw(ArgumentError("tip_ratio must be in (0,1]"))
    mesh_order in (1, 2, 3) || throw(ArgumentError("mesh_order must be 1, 2 or 3"))
    length(rotation.axis) == 3 && all(isfinite, rotation.axis) &&
    norm(rotation.axis) > 0 && isfinite(rotation.angle) ||
        throw(ArgumentError("rotation needs a finite nonzero axis and a finite angle"))
    axis = Float64.(collect(rotation.axis)) / norm(rotation.axis)
    x, y, z = axis
    cross_matrix = [0 -z y; z 0 -x; -y x 0]
    s, c = sincos(rotation.angle)
    # Map the tip-graded sphere's z axis to the ellipsoid's local x axis.
    transform = (c * I + (1 - c) * axis * transpose(axis) + s * cross_matrix) *
                Diagonal(Float64.(collect(semiaxes))) * [0 0 1; 1 0 0; 0 1 0]
    affine = [transform Float64.(collect(center)); 0 0 0 1]
    return mesh(; qorder, validation, provenance = "ellipsoid") do g
        g.model.add("ellipsoid")
        g.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        g.model.occ.synchronize()
        g.option.setNumber("Mesh.MeshSizeMin", tip_ratio * resolution)
        g.option.setNumber("Mesh.MeshSizeMax", resolution)
        g.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        g.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        field = g.model.mesh.field.add("MathEval")
        g.model.mesh.field.setString(field, "F", "$(resolution)*($(tip_ratio)+$(1-tip_ratio)*(1-z^4))")
        g.model.mesh.field.setAsBackgroundMesh(field)
        g.model.mesh.generate(2)
        g.model.mesh.setOrder(mesh_order)
        g.model.mesh.affineTransform(vec(permutedims(affine)))
    end
end

"""
    mesh(path::AbstractString; units=:m, qorder=4)
    mesh(generate::Function; units=:m, qorder=4, provenance="generated Gmsh model")
    mesh(nodes, triangles; units=:m, qorder=4, labels=nothing, provenance="supplied triangles")

Construct a full-3D [`Mesh`](@ref) from a Gmsh file, a function `generate(gmsh)` that builds
and meshes a Gmsh model, or coordinate/connectivity matrices. Each column of `nodes` is a
3D point; each column of `triangles` contains 3, 6 or 10 one-based node indices in Gmsh's
linear, quadratic or cubic triangle ordering. `labels` optionally gives one positive integer
physical-group tag per triangle. Gmsh input preserves all surface physical tags and names.

Supply one connected, closed surface with outward-oriented triangles. Coordinates are
converted from `units` (`:m`, `:cm`, `:mm`) to metres and retain their supplied Cartesian
frame. No remeshing, hole filling, node merging
or orientation repair is performed. Unsupported elements, open/nonmanifold surfaces,
inconsistent normals, inward orientation and detected intersections raise `ArgumentError`.
Curved elements use outward-rounded Bernstein bounds and adaptive subdivision. Acceptance
requires positive Jacobians projected onto each element's corner plane, injective elements,
and separation except at topologically shared edges/vertices. Strongly curved valid elements
whose corner-plane projection folds are outside this acceptance criterion.

`validation=(maxdepth=20, maxwork=200000)` sets refinement and work limits. An unresolved
case raises `ArgumentError`; neither exhausted limits nor small distances imply validity.
Increasing the limits can resolve conservative bounds, but cannot repair invalid geometry.

The returned `m.body` records `nodes` in metres, `connectivity` in Gmsh ordering, per-element
`labels` (tag/name pairs), `units`, `input_units`, `orientation`, `provenance` and `validation`.
`m.resolution` is the maximum corner-edge length in metres. Pass `m` directly to [`bem`](@ref).
The Gmsh constructors require that no other Gmsh session is active.
"""
function mesh(path::AbstractString; units::Symbol = :m, qorder::Integer = 4, validation::NamedTuple = (;))
    isfile(path) || throw(ArgumentError("Gmsh mesh file does not exist: $path"))
    return _surface_session() do
        gmsh.open(abspath(path))
        _surface_from_gmsh(; units, qorder, provenance = abspath(path), validation)
    end
end

function mesh(generate::Function; units::Symbol = :m, qorder::Integer = 4,
        provenance::AbstractString = "generated Gmsh model", validation::NamedTuple = (;))
    return _surface_session() do
        generate(gmsh)
        _surface_from_gmsh(; units, qorder, provenance, validation)
    end
end

function mesh(nodes::AbstractMatrix{<:Real}, triangles::AbstractMatrix{<:Integer};
        units::Symbol = :m, qorder::Integer = 4,
        labels::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        provenance::AbstractString = "supplied triangles", validation::NamedTuple = (;))
    size(nodes, 1) == 3 || throw(ArgumentError("nodes must have three rows"))
    size(triangles, 1) in (3, 6, 10) ||
        throw(ArgumentError("triangles must have 3, 6 or 10 rows in Gmsh ordering"))
    all(isfinite, nodes) || throw(ArgumentError("node coordinates must be finite"))
    isempty(triangles) && throw(ArgumentError("surface must contain triangles"))
    all(i -> 1 <= i <= size(nodes, 2), triangles) ||
        throw(ArgumentError("triangle connectivity contains an invalid node index"))
    groups = labels === nothing ? ones(Int, size(triangles, 2)) : Int.(labels)
    length(groups) == size(triangles, 2) && all(>(0), groups) ||
        throw(ArgumentError("labels must contain one positive physical tag per triangle"))
    _surface_scale(units)
    return mesh(; units, qorder, provenance, validation) do g
        g.model.add("supplied surface")
        order = size(triangles, 1) == 3 ? 1 : size(triangles, 1) == 6 ? 2 : 3
        element_type = g.model.mesh.getElementType("Triangle", order)
        entities = Dict(tag => g.model.addDiscreteEntity(2) for tag in unique(groups))
        g.model.mesh.addNodes(2, first(values(entities)), collect(axes(nodes, 2)), vec(nodes))
        for (tag, entity) in entities
            ids = findall(==(tag), groups)
            g.model.mesh.addElementsByType(entity, element_type, ids, vec(triangles[:, ids]))
            g.model.addPhysicalGroup(2, [entity], tag)
        end
    end
end

function _surface_from_gmsh(; units, qorder, provenance, validation)
    scale = _surface_scale(units)
    qorder > 0 || throw(ArgumentError("qorder must be positive"))
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    nodes = scale .* reshape(coordinates, 3, :)
    all(isfinite, nodes) || throw(ArgumentError("node coordinates must be finite"))
    local_tags = Dict(tag => i for (i, tag) in enumerate(tags))
    connectivity = Vector{Int}[]
    labels = Vector{Pair{Int, String}}[]
    for (_, entity) in gmsh.model.getEntities(2)
        physical = [Int(tag) => gmsh.model.getPhysicalName(2, tag)
                    for tag in gmsh.model.getPhysicalGroupsForEntity(2, entity)]
        types, _, element_nodes = gmsh.model.mesh.getElements(2, entity)
        for (type, indices) in zip(types, element_nodes)
            name, _, order, count, _, _ = gmsh.model.mesh.getElementProperties(type)
            startswith(name, "Triangle") && order in (1, 2, 3) && count in (3, 6, 10) ||
                throw(ArgumentError("unsupported surface element: $name"))
            for column in eachcol(reshape(indices, Int(count), :))
                push!(connectivity, [local_tags[tag] for tag in column])
                push!(labels, copy(physical))
            end
        end
    end
    resolution = _validate_surface_topology(nodes, connectivity)
    imported = Inti.import_mesh(; dim = 3)
    for i in eachindex(imported.nodes)
        imported.nodes[i] *= scale
    end
    surface = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(imported))
    surface_mesh = view(imported, surface)
    geometry_report = _validate_surface_patches(surface_mesh; validation...)
    quad = Inti.Quadrature(surface_mesh; qorder)
    all(q -> all(isfinite, q.normal) && isfinite(q.weight), quad) ||
        throw(ArgumentError("surface has a singular quadrature Jacobian"))
    report = (; closed = true, connected = true, manifold = true,
        geometry_report..., quadrature_order = qorder)
    body = _SurfaceGeometry(nodes, connectivity, labels, :m, units, :outward,
        String(provenance), report)
    return Mesh(quad, body, :full, resolution)
end

function _validate_surface_topology(nodes, connectivity)
    isempty(connectivity) && throw(ArgumentError("surface must contain triangles"))
    edges = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    vertex_faces = Dict{Int, Vector{Int}}()
    points = [SVector{3}(p) for p in eachcol(nodes)]
    extent = maximum(norm(p - first(points)) for p in points)
    tolerance = 100eps(Float64) * extent^2
    resolution = 0.0
    volume = 0.0
    origin = points[first(first(connectivity))]
    for (i, triangle) in enumerate(connectivity)
        length(unique(triangle)) == length(triangle) ||
            throw(ArgumentError("triangle $i repeats a node"))
        a, b, c = triangle[1:3]
        for v in (a, b, c)
            push!(get!(vertex_faces, v, Int[]), i)
        end
        normal = cross(points[b] - points[a], points[c] - points[a])
        norm(normal) > tolerance || throw(ArgumentError("triangle $i is degenerate"))
        volume += dot(points[a] - origin, cross(points[b] - origin, points[c] - origin)) / 6
        for (u, v) in ((a, b), (b, c), (c, a))
            edge = minmax(u, v)
            push!(get!(edges, edge, Tuple{Int, Int}[]), (i, u < v ? 1 : -1))
            resolution = max(resolution, norm(points[u] - points[v]))
        end
    end
    neighbors = [Int[] for _ in connectivity]
    for (edge, incident) in edges
        length(incident) == 2 ||
            throw(ArgumentError("edge $edge has $(length(incident)) incident triangles; expected two"))
        (i, s), (j, t) = incident
        s == -t || throw(ArgumentError("triangles $i and $j have inconsistent orientation"))
        push!(neighbors[i], j)
        push!(neighbors[j], i)
        _validate_surface_edge(connectivity[i], connectivity[j], edge)
    end
    visited = falses(length(connectivity))
    stack = [1]
    while !isempty(stack)
        i = pop!(stack)
        visited[i] && continue
        visited[i] = true
        append!(stack, neighbors[i])
    end
    all(visited) ||
        throw(ArgumentError("supply one connected surface; nested/disjoint regions need a multi-region solve"))
    for (v, faces) in vertex_faces
        incident = Set(faces)
        fan = Set([first(incident)])
        stack = [first(incident)]
        while !isempty(stack)
            i = pop!(stack)
            for j in neighbors[i]
                if j in incident && !(j in fan)
                    push!(fan, j)
                    push!(stack, j)
                end
            end
        end
        length(fan) == length(incident) || throw(ArgumentError("vertex $v is nonmanifold"))
    end
    volume > 100eps(Float64) * extent^3 ||
        throw(ArgumentError("surface must enclose positive volume with outward orientation"))
    return resolution
end

function _surface_edge_nodes(triangle, edge)
    order = length(triangle) == 3 ? 1 : length(triangle) == 6 ? 2 : 3
    for (j, (u, v)) in enumerate(((1, 2), (2, 3), (3, 1)))
        minmax(triangle[u], triangle[v]) == edge || continue
        middle = triangle[(4 + (j - 1) * (order - 1)):(3 + j * (order - 1))]
        return triangle[u] < triangle[v] ? middle : reverse(middle)
    end
end

function _validate_surface_edge(a, b, edge)
    _surface_edge_nodes(a, edge) == _surface_edge_nodes(b, edge) ||
        throw(ArgumentError("nonconforming curved nodes at edge $edge"))
end

"""
    bem(surface::Mesh, boundary, k; incidence_angle=π/2, incidence_azimuth=0, kwargs...)

Solve scattering on a supplied full-3D `surface` with rigid, pressure-release or homogeneous
fluid/gas material. The stored quadrature and outward normals are used directly. Wavenumber
`k` is in inverse metres, including when the mesh input used centimetres or millimetres.
Solver options are the same as `bem(body, boundary, k; method=:full)`; mesh resolution and
quadrature are selected when constructing `surface`. Returns a [`BEMSolution`](@ref).
"""
function bem(surface::Mesh{<:Inti.Quadrature},
        boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0, kwargs...)
    p, q, quad, report = solve_full_bem(boundary, k, surface.data;
        incidence_angle, incidence_azimuth, return_diagnostics = true, kwargs...)
    report = merge(report, (; meshsize = surface.resolution))
    if surface.body isa _SurfaceGeometry
        report = merge(report, (; geometry = surface.body.validation,
            provenance = surface.body.provenance))
    end
    data = _FullBEMSurfaceData(quad, p, q, incidence_angle, incidence_azimuth, report)
    return BEMSolution(surface.body, boundary, Float64(k), :full, data)
end
