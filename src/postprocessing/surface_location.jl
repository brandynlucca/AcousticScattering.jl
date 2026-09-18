"""
Locate a point relative to a closed polynomial surface by oriented ray crossings.
Bernstein hulls exclude patches and keep the curved boundary homotopy away from the
ray before replacing a patch by its corner triangle. Ambiguous patches subdivide;
alternate rays handle tangencies and shared edges. Points within the terminal
coordinate tolerance are treated as surface points; exhausted bounds throw.
"""
function _surface_location(patches, point)
    scale = maximum(maximum(max(abs(t.lo), abs(t.hi)) for t in v)
    for patch in patches for v in patch.net)
    tolerance = 512eps(scale)
    degrees = unique(patch.degree for patch in patches)
    splits = Dict(p => _surface_bernstein_split(p) for p in degrees)
    for direction in (SVector(1.0, 0.371, 0.529), SVector(0.219, 1.0, -0.413),
        SVector(-0.617, 0.293, 1.0))
        axis = direction/norm(direction)
        u = cross(axis, SVector(0.0, 0.0, 1.0))
        u /= norm(u)
        v = cross(axis, u)
        state = _SurfaceValidation(splits, Dict{Tuple{Int, Int}, Int}(),
            maximum(maximum(p.corners) for p in patches), 48, 50000, 0, 0)
        winding = 0
        resolved = true
        for patch in patches
            count = _surface_ray_crossings(patch, point, (u, v, axis), tolerance, state, 0)
            count === :on && return :on
            if count === nothing
                resolved = false
                break
            end
            winding += count
        end
        resolved && winding in (0, 1) && return winding == 1 ? :inside : :outside
    end
    throw(ArgumentError("point location is unresolved on the curved surface"))
end

function _surface_ray_crossings(patch, point, frame, tolerance, state, depth)
    state.work += 1
    state.work > state.maxwork && return nothing
    shifted = [v - _SurfaceBound.(SVector{3, Float64}(point)) for v in patch.net]
    projected = [SVector(dot(frame[1], v), dot(frame[2], v), dot(frame[3], v))
                 for v in shifted]
    bounds = [_surface_hull([v[d] for v in projected]) for d in 1:3]
    any(b -> b.lo > 0 || b.hi < 0, bounds[1:2]) && return 0
    bounds[3].hi < -tolerance && return 0
    if all(b -> b.lo >= -tolerance && b.hi <= tolerance, bounds)
        return :on
    end
    if bounds[3].lo > tolerance && _surface_ray_edges_clear(projected, patch.degree)
        vertices = projected[[1, patch.degree + 1, length(projected)]]
        a, b, c = vertices
        sides = (a[1]*b[2] - a[2]*b[1], b[1]*c[2] - b[2]*c[1],
            c[1]*a[2] - c[2]*a[1])
        all(s -> s.lo > 0, sides) && return 1
        all(s -> s.hi < 0, sides) && return -1
        any(s -> s.lo > 0, sides) && any(s -> s.hi < 0, sides) && return 0
    end
    depth >= state.maxdepth && return nothing
    count = 0
    for child in _surface_children(patch, state)
        next = _surface_ray_crossings(child, point, frame, tolerance, state, depth + 1)
        next in (:on, nothing) && return next
        count += next
    end
    return count
end

function _surface_ray_edges_clear(net, degree)
    indices = _surface_indices(degree)
    for axis in 1:3
        edge = [net[i] for (i, powers) in enumerate(indices) if powers[axis] == 0]
        clear = any(1:2) do d
            bound = _surface_hull([v[d] for v in edge])
            bound.lo > 0 || bound.hi < 0
        end
        clear && continue
        chord = _surface_mid.(last(edge) - first(edge))
        bound = _surface_hull([-chord[2]*v[1] + chord[1]*v[2] for v in edge])
        (bound.lo > 0 || bound.hi < 0) || return false
    end
    return true
end
