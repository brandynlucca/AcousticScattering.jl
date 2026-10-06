function _validate_cylinder_surface(body::Cylinder)
    a, L, R, d = body.radius, body.length, body.radius_curvature, body.endcap_depth
    all(isfinite, (a, L, d)) || throw(ArgumentError("cylinder dimensions must be finite"))
    R > a || throw(ArgumentError("radius_curvature must exceed the cylinder radius"))
    if isfinite(R)
        L / R < 2π ||
            throw(ArgumentError("the cylinder centerline must span less than a full circle"))
        L / R > π && (R - a) * sin(L / (2R)) <= d &&
            throw(ArgumentError(
                "the end caps overlap or cannot be separated; reduce the bend or endcap_depth"))
    end
    return nothing
end

function _bend_cylinder_node(p, body::Cylinder)
    R = body.radius_curvature
    isinf(R) && return p
    x, y, z = p
    s = clamp(z, -body.length / 2, body.length / 2)
    t = z - s
    γ = s / R
    sn, cs = sincos(γ)
    return [2R * sin(γ / 2)^2 + x * cs + t * sn, y, (R - x) * sn + t * cs]
end

function _grade_cylinder_node(p, body::Cylinder)
    x, y, z = p
    r = hypot(x, y)
    a, h = body.radius, body.length / 2
    scale = iszero(r) ? 1.0 : 0.05 + 0.95a * sinpi(r / (2a)) / r
    return [scale * x, scale * y, 0.05z + 0.95h * sinpi(z / (2h))]
end

function _cylinder_full_mesh(body::Cylinder; meshsize::Real, qorder::Integer = 4,
        mesh_order::Integer = 2)
    _validate_cylinder_surface(body)
    isfinite(meshsize) && meshsize > 0 ||
        throw(ArgumentError("meshsize must be finite and positive"))
    mesh_order in (1, 2, 3) || throw(ArgumentError("mesh_order must be 1, 2 or 3"))
    unit = 2body.radius
    isfinite(unit) || throw(ArgumentError("cylinder diameter must be finite"))
    geometry = Cylinder(0.5, body.length / unit;
        radius_curvature = body.radius_curvature / unit, endcap_depth = body.endcap_depth /
                                                                        unit)
    return mesh(; qorder, provenance = "closed circular-arc cylinder") do g
        g.model.add("closed cylinder")
        a, L, d = geometry.radius, geometry.length, geometry.endcap_depth
        spacing = iszero(d) ? min(meshsize / unit, a / 2, L / 4) : meshsize / unit
        g.option.setNumber("Mesh.MeshSizeMin", spacing)
        g.option.setNumber("Mesh.MeshSizeMax", spacing)
        volume = g.model.occ.addCylinder(0.0, 0.0, -L / 2, 0.0, 0.0, L, a)
        if d > 0
            caps = Tuple{Int, Int}[]
            for z in (-L / 2, L / 2)
                cap = g.model.occ.addSphere(0.0, 0.0, z, a)
                g.model.occ.dilate([(3, cap)], 0.0, 0.0, z, 1.0, 1.0, d / a)
                push!(caps, (3, cap))
            end
            g.model.occ.fuse([(3, volume)], caps)
        end
        g.model.occ.synchronize()
        g.model.mesh.generate(2)
        g.model.mesh.setOrder(mesh_order)
        tags, xyz, _ = g.model.mesh.getNodes()
        for (tag, p) in zip(tags, eachcol(reshape(xyz, 3, :)))
            point = iszero(d) ? _grade_cylinder_node(p, geometry) : p
            bent = _bend_cylinder_node(point, geometry)
            # Rotate the axial template to x while preserving triangle orientation.
            g.model.mesh.setNode(tag, unit .* bent[[3, 1, 2]], Float64[])
        end
    end
end
