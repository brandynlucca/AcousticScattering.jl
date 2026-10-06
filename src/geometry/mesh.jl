# --- Shared geometry helpers (bem/mfs) --------------------------------------

_characteristic_radius(body::Sphere) = body.radius
_characteristic_radius(body::Spheroid) = max(body.a, body.b)
_characteristic_radius(body::Cylinder) = max(body.radius, body.length / 2)

function _axisymmetric_default_panels(body::AbstractBody, k::Real)
    bem_panel_count(k, _characteristic_radius(body))
end

_axisymmetric_mesh(body::Sphere, n::Integer) = sphere_mesh(body.radius, n)
_axisymmetric_mesh(body::Spheroid, n::Integer) = spheroid_mesh(body.a, body.b, n)
_axisymmetric_mesh(body::Cylinder, n::Integer) = cylinder_mesh(body.radius, body.length, n)

# --- Generic mesh interface --------------------------------------------------

"""
    Mesh

A discretized surface mesh, returned by [`mesh`](@ref). The single public mesh type, whether the
underlying representation is an axisymmetric meridian curve (`method=:axisymmetric`) or a full 3D
triangulated surface (`method=:full`). `body`/`method`/`resolution` record what `mesh(...)` was
called with. `resolution` is the maximum corner-edge length in meters. Inspect the surface with
`coordinates`, `normals`, `elements`, and `element_count`.
"""
struct Mesh{D}
    data::D
    body::AbstractBody
    method::Symbol
    resolution::Float64
end

"""
    mesh(body::AbstractBody; resolution=nothing, k=nothing, method=:axisymmetric,
         qorder=4, mesh_order=2)

Build a [`Mesh`](@ref) for `body`, the same construction `bem`/`mfs` use internally. Exactly one
of `resolution` or `k` must be given. `resolution` sets panel count (`:axisymmetric`) or target
element edge length in m (`:full`) directly. `k` in 1/m derives a wavenumber-appropriate default.
For full surfaces, `mesh_order` (1, 2 or 3) controls triangle geometry and `qorder` controls
quadrature. Supplied surfaces use `mesh(path)`, `mesh(generate)` or `mesh(nodes, triangles)`.
"""
function mesh(body::AbstractBody; resolution::Union{Nothing, Real} = nothing,
        k::Union{Nothing, Real} = nothing, method::Symbol = :axisymmetric,
        qorder::Integer = 4, mesh_order::Integer = 2)
    (resolution === nothing) == (k === nothing) &&
        throw(ArgumentError("mesh(...) needs exactly one of `resolution` or `k`"))
    if method === :axisymmetric
        body isa Cylinder && _isbent(body) &&
            throw(ArgumentError(
                "a bent Cylinder requires mesh(...; method=:full)"))
        n = resolution === nothing ? _axisymmetric_default_panels(body, k) : Int(resolution)
        return Mesh(_axisymmetric_mesh(body, n), body, method, Float64(n))
    end
    if method === :full
        body isa Union{Sphere, Spheroid, Cylinder} ||
            throw(ArgumentError("mesh(...; method=:full) has no mesh generator for $(typeof(body)) in this package yet"))
        meshsize = resolution === nothing ? bem3d_elements_per_wavelength(k) :
                   Float64(resolution)
        if body isa Cylinder
            surface = _cylinder_full_mesh(body; meshsize, qorder, mesh_order)
            return Mesh(surface.data, body, method, meshsize)
        end
        quad = body isa Sphere ?
               gmsh_sphere_mesh(body.radius; meshsize, qorder, mesh_order) :
               gmsh_spheroid_mesh(body.a, body.b; meshsize, qorder, mesh_order)
        return Mesh(quad, body, method, meshsize)
    end
    throw(ArgumentError("mesh(...) supports method=:axisymmetric or :full, got $method"))
end

"""
    coordinates(mesh::Mesh)

Element midpoint/quadrature-node positions of `mesh`: `(rho, x)` radial/axial tuples
for `method=:axisymmetric`, Cartesian `(x,y,z)` 3-vectors for `method=:full`.
"""
coordinates(m::Mesh{MeridianMesh}) = [(p.rhom, p.zm) for p in panels(m.data)]
coordinates(m::Mesh{<:Inti.Quadrature}) = [q.coords for q in m.data]

"""
    normals(mesh::Mesh)

Outward unit normal at each element of `mesh`, same shape convention as [`coordinates`](@ref).
"""
normals(m::Mesh{MeridianMesh}) = [(p.nrho, p.nz) for p in panels(m.data)]
normals(m::Mesh{<:Inti.Quadrature}) = [q.normal for q in m.data]

"""
    elements(mesh::Mesh)

The discretization elements of `mesh`: a `Vector{Panel}` for `method=:axisymmetric`, or the
quadrature nodes themselves for `method=:full`.
"""
elements(m::Mesh{MeridianMesh}) = panels(m.data)
elements(m::Mesh{<:Inti.Quadrature}) = collect(m.data)

"""
    element_count(mesh::Mesh)

Number of discretization elements in `mesh` (replaces the former `npanels`/`length(quad)` calls).
"""
element_count(m::Mesh{MeridianMesh}) = npanels(m.data)
element_count(m::Mesh{<:Inti.Quadrature}) = length(m.data)
