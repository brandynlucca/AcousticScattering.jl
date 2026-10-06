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
    return _canonical_surface_mesh("sphere"; meshsize, qorder, mesh_order) do
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, radius)
    end
end

"""
    gmsh_spheroid_mesh(a, b; meshsize, qorder=4, mesh_order=2)

Full 3D triangulated surface quadrature for a prolate/oblate spheroid with
semi-axis `a` in m along the x axis of symmetry and equatorial semi-axis `b`
in m, matching [`Spheroid`](@ref)'s convention (prolate if `a > b`, oblate
if `a < b`), via Gmsh (a unit sphere, non-uniformly scaled with OpenCASCADE's
`dilate`). `mesh_order=2` uses curved quadratic triangles; `mesh_order=1` uses
flat triangles and `mesh_order=3` uses cubic triangles. `qorder` controls integration separately.
"""
function gmsh_spheroid_mesh(a::Real, b::Real; meshsize::Real, qorder::Integer = 4,
        mesh_order::Integer = 2)
    return _canonical_surface_mesh("spheroid"; meshsize, qorder, mesh_order) do
        tag = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, tag)], 0.0, 0.0, 0.0, b, b, a)
    end
end

function _canonical_surface_mesh(build, name; meshsize, qorder, mesh_order)
    meshsize > 0 || throw(ArgumentError("meshsize must be positive, got $meshsize"))
    mesh_order in (1, 2, 3) || throw(ArgumentError("mesh_order must be 1, 2 or 3"))
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add(name)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        build()
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(mesh_order)
        gmsh.model.mesh.affineTransform([0.0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    surface = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
    return Inti.Quadrature(view(msh, surface); qorder)
end
