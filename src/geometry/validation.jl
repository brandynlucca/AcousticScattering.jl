# Outward-rounded Bernstein bounds for supplied polynomial triangles.

struct _SurfaceBound <: Real
    lo::Float64
    hi::Float64
end

_SurfaceBound(x::Float64) = _SurfaceBound(x, x)
_SurfaceBound(x::Integer) = _SurfaceBound(Float64(x))
function _SurfaceBound(x::Rational)
    y = Float64(x)
    Rational{BigInt}(y) == x && return _SurfaceBound(y)
    return _SurfaceBound(prevfloat(y), nextfloat(y))
end
Base.zero(::Type{_SurfaceBound}) = _SurfaceBound(0.0)
Base.zero(::_SurfaceBound) = zero(_SurfaceBound)
Base.conj(x::_SurfaceBound) = x
Base.iszero(x::_SurfaceBound) = iszero(x.lo) && iszero(x.hi)
Base.:-(x::_SurfaceBound) = _SurfaceBound(-x.hi, -x.lo)
function Base.:+(x::_SurfaceBound, y::_SurfaceBound)
    iszero(x) && return y
    iszero(y) && return x
    return _SurfaceBound(prevfloat(x.lo + y.lo), nextfloat(x.hi + y.hi))
end
Base.:-(x::_SurfaceBound, y::_SurfaceBound) = x + (-y)
function Base.:*(x::_SurfaceBound, y::_SurfaceBound)
    (iszero(x) || iszero(y)) && return zero(_SurfaceBound)
    values = (x.lo * y.lo, x.lo * y.hi, x.hi * y.lo, x.hi * y.hi)
    any(isnan, values) && return _SurfaceBound(-Inf, Inf)
    return _SurfaceBound(prevfloat(minimum(values)), nextfloat(maximum(values)))
end
Base.:*(x::Real, y::_SurfaceBound) = _SurfaceBound(x) * y
Base.:*(x::_SurfaceBound, y::Real) = x * _SurfaceBound(y)

_surface_mid(x::_SurfaceBound) = x.lo / 2 + x.hi / 2
_surface_indices(n) = [(i, j, n - i - j) for j in 0:n for i in 0:(n - j)]
_surface_multinomial(t) = factorial(sum(t)) ÷ prod(factorial, t)

function _surface_bernstein_conversion(n)
    indices = _surface_indices(n)
    matrix = [_surface_multinomial(b) * prod((a[d] // n)^b[d] for d in 1:3)
              for a in indices, b in indices]
    return _SurfaceBound.(inv(matrix))
end

function _surface_bernstein_split(n)
    indices = _surface_indices(n)
    u, v, w = (1//1, 0//1, 0//1), (0//1, 1//1, 0//1), (0//1, 0//1, 1//1)
    uv, uw, vw = (u .+ v) ./ 2, (u .+ w) ./ 2, (v .+ w) ./ 2
    triangles = ((uw, vw, w), (u, uv, uw), (uv, v, vw), (vw, uw, uv))
    return map(triangles) do triangle
        matrix = zeros(Rational{Int}, length(indices), length(indices))
        for (row, powers) in enumerate(indices)
            net = [Rational{Int}.(collect(1:length(indices)) .== i)
                   for i in eachindex(indices)]
            degree = n
            for axis in 1:3, _ in 1:powers[axis]

                lookup = Dict(t => i for (i, t) in enumerate(_surface_indices(degree)))
                net = [sum(triangle[axis][d] .*
                           net[lookup[ntuple(k -> t[k] + (k == d), 3)]]
                       for d in 1:3) for t in _surface_indices(degree - 1)]
                degree -= 1
            end
            matrix[row, :] = only(net)
        end
        _SurfaceBound.(matrix)
    end
end

function _surface_transform(matrix, values)
    return [sum(matrix[i, j] * values[j] for j in eachindex(values))
            for i in axes(matrix, 1)]
end

struct _SurfacePatch
    net::Vector{SVector{3, _SurfaceBound}}
    degree::Int
    corners::NTuple{3, Int}
    face::Int
end

mutable struct _SurfaceValidation
    splits::Dict{Int, NTuple{4, Matrix{_SurfaceBound}}}
    midpoints::Dict{Tuple{Int, Int}, Int}
    next_vertex::Int
    maxdepth::Int
    maxwork::Int
    work::Int
    depth::Int
end

function _surface_budget!(state, depth, what)
    state.work += 1
    state.depth = max(state.depth, depth)
    state.work <= state.maxwork ||
        throw(ArgumentError("surface validation unresolved: work limit exceeded during $what"))
    depth <= state.maxdepth ||
        throw(ArgumentError("surface validation unresolved: refinement limit exceeded during $what"))
end

function _surface_children(patch, state)
    a, b, c = patch.corners
    midpoint(u, v) = get!(state.midpoints, minmax(u, v)) do
        state.next_vertex += 1
    end
    ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
    corners = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (bc, ca, ab))
    return [_SurfacePatch(_surface_transform(matrix, patch.net), patch.degree, corner, patch.face)
            for (matrix, corner) in zip(state.splits[patch.degree], corners)]
end

function _surface_derivatives(patch)
    p = patch.degree
    lookup = Dict(t => i for (i, t) in enumerate(_surface_indices(p)))
    du = [p * (patch.net[lookup[(i + 1, j, k)]] - patch.net[lookup[(i, j, k + 1)]])
          for (i, j, k) in _surface_indices(p - 1)]
    dv = [p * (patch.net[lookup[(i, j + 1, k)]] - patch.net[lookup[(i, j, k + 1)]])
          for (i, j, k) in _surface_indices(p - 1)]
    return du, dv
end

function _surface_patch_vertices(patch)
    p = patch.degree
    return patch.net[[1, p + 1, length(patch.net)]]
end

function _surface_projection(patch)
    a, b, c = map(x -> _surface_mid.(x), _surface_patch_vertices(patch))
    scale = max(norm(b - a), norm(c - a))
    u, v = (b - a) / scale, (c - a) / scale
    normal = cross(u, v)
    area = dot(normal, normal)
    area > 0 && isfinite(area) ||
        throw(ArgumentError("surface validation unresolved: singular corner-plane projection"))
    return transpose(hcat(cross(v, normal) / area / scale, cross(normal, u) / area / scale))
end

function _surface_jacobian(patch)
    projection = _surface_projection(patch)
    normal = cross(_SurfaceBound.(projection[1, :]), _SurfaceBound.(projection[2, :]))
    du, dv = _surface_derivatives(patch)
    indices = _surface_indices(patch.degree - 1)
    target = _surface_indices(2patch.degree - 2)
    lookup = Dict(t => i for (i, t) in enumerate(target))
    coefficients = fill(zero(_SurfaceBound), length(target))
    for (i, u) in enumerate(indices), (j, v) in enumerate(indices)

        t = u .+ v
        weight = (_surface_multinomial(u) * _surface_multinomial(v)) //
                 _surface_multinomial(t)
        coefficients[lookup[t]] += _SurfaceBound(weight) * dot(normal, cross(du[i], dv[j]))
    end
    return coefficients
end

function _surface_positive_jacobian(coefficients, degree, state, face, depth = 0)
    _surface_budget!(state, depth, "Jacobian bounds for triangle $face")
    all(x -> x.lo > 0, coefficients) && return
    any(i -> coefficients[i].hi <= 0, (1, degree + 1, length(coefficients))) &&
        throw(ArgumentError("triangle $face has a nonpositive projected Jacobian on its corner plane"))
    for matrix in state.splits[degree]
        _surface_positive_jacobian(
            _surface_transform(matrix, coefficients), degree, state, face, depth + 1)
    end
end

function _surface_hull(values)
    return _SurfaceBound(minimum(x -> x.lo, values), maximum(x -> x.hi, values))
end

"""Certify injectivity by a uniformly positive symmetric part of a projected derivative."""
function _surface_injective(patch)
    projection = try
        _surface_projection(patch)
    catch error
        error isa ArgumentError || rethrow()
        return false
    end
    all(isfinite, projection) || return false
    du, dv = _surface_derivatives(patch)
    jac = [_surface_hull([sum(projection[i, d] * t[d] for d in 1:3) for t in column])
           for i in 1:2, column in (du, dv)]
    off = 0.5 * (jac[1, 2] + jac[2, 1])
    return jac[1, 1].lo > 0 && jac[2, 2].lo > 0 &&
           (jac[1, 1] * jac[2, 2] - off * off).lo > 0
end

"""
Certify a simple projected boundary. Together with an everywhere positive projected
Jacobian, the planar degree theorem then gives injectivity on the closed triangle.
"""
function _surface_boundary_injective(patch)
    projection = _surface_projection(patch)
    net = [SVector(sum(projection[1, d] * x[d] for d in 1:3),
               sum(projection[2, d] * x[d] for d in 1:3), zero(_SurfaceBound))
           for x in patch.net]
    p = patch.degree
    lookup = Dict(t => i for (i, t) in enumerate(_surface_indices(p)))
    curves = ([net[lookup[(i, 0, p - i)]] for i in 0:p],
        [net[lookup[(p - i, i, 0)]] for i in 0:p],
        [net[lookup[(0, p - i, i)]] for i in 0:p])
    for curve in curves
        chord = _SurfaceBound.(_surface_mid.(last(curve) - first(curve)))
        all(dot(chord, curve[i + 1] - curve[i]).lo > 0 for i in 1:p) || return false
    end
    for i in 1:3
        left, right = curves[i], curves[mod1(i + 1, 3)]
        rays = vcat([x - last(left) for x in left[1:(end - 1)]],
            [first(right) - x for x in right[2:end]])
        _surface_positive_direction(rays) || return false
    end
    return true
end

function _surface_separated(a, b, axis)
    all(isfinite, axis) && norm(axis) > 0 || return false
    direction = _SurfaceBound.(axis / norm(axis))
    left = _surface_hull([dot(direction, x) for x in a.net])
    right = _surface_hull([dot(direction, x) for x in b.net])
    return left.hi < right.lo || right.hi < left.lo
end

function _surface_boxes_separated(a, b)
    return any(1:3) do d
        maximum(x -> x[d].hi, a.net) < minimum(x -> x[d].lo, b.net) ||
            maximum(x -> x[d].hi, b.net) < minimum(x -> x[d].lo, a.net)
    end
end

function _surface_positive_direction(rays)
    directions = [_surface_mid.(x) for x in rays]
    any(x -> !all(isfinite, x) || iszero(norm(x)), directions) && return false
    directions = [x / norm(x) for x in directions]
    axis = sum(directions)
    for _ in 1:64
        all(x -> dot(_SurfaceBound.(axis), x).lo > 0, rays) && return true
        worst = argmin(x -> dot(axis, x), directions)
        axis += worst
    end
    distinct = SVector{3, Float64}[]
    for direction in directions
        any(x -> norm(x - direction) < 1e-8, distinct) || push!(distinct, direction)
    end
    function separates(candidate)
        norm(candidate) > 0 && all(isfinite, candidate) || return false
        candidate /= norm(candidate)
        all(x -> dot(candidate, x) > 0, directions) || return false
        return all(x -> dot(_SurfaceBound.(candidate), x).lo > 0, rays)
    end
    for i in eachindex(distinct)
        separates(distinct[i]) && return true
        for j in (i + 1):length(distinct)
            separates(distinct[i] + distinct[j]) && return true
            for k in (j + 1):length(distinct)
                candidate = cross(distinct[i] - distinct[j], distinct[i] - distinct[k])
                separates(candidate) && return true
                separates(-candidate) && return true
            end
        end
    end
    return false
end

function _surface_vertex_contact(a, b, vertex)
    ia, ib = findfirst(==(vertex), a.corners), findfirst(==(vertex), b.corners)
    va, vb = _surface_patch_vertices(a)[ia], _surface_patch_vertices(b)[ib]
    ai, bi = (1, a.degree + 1, length(a.net))[ia], (1, b.degree + 1, length(b.net))[ib]
    rays = vcat([x - va for (i, x) in enumerate(a.net) if i != ai],
        [vb - x for (i, x) in enumerate(b.net) if i != bi])
    return _surface_positive_direction(rays)
end

"""
Bound the common edge tangent and `(P(t,s)-P(t,0))/s` in Duffy coordinates.
Factoring the boundary parameter algebraically avoids dividing by a small distance.
"""
function _surface_edge_data(patch, edge)
    p = patch.degree
    start, stop = findfirst(==(edge[1]), patch.corners),
    findfirst(==(edge[2]), patch.corners)
    apex = only(setdiff(1:3, (start, stop)))
    barycentric_axis = (3, 1, 2)
    permutation = barycentric_axis[[stop, apex, start]]
    lookup = Dict(t => i for (i, t) in enumerate(_surface_indices(p)))
    layer(k) = [patch.net[lookup[ntuple(d -> (i, k, p - i - k)[findfirst(==(d), permutation)], 3)]]
                for i in 0:(p - k)]
    boundary = layer(0)
    tangent = [p * (boundary[i + 1] - boundary[i]) for i in 1:p]
    transverse = SVector{3, _SurfaceBound}[]
    for k in 1:p
        curve = layer(k)
        for degree in (p - k):(p - 1)
            curve = vcat([first(curve)],
                [(i // (degree + 1)) * curve[i] + (1 - i // (degree + 1)) * curve[i + 1]
                 for i in 1:degree], [last(curve)])
        end
        append!(transverse, [(p // k) * (curve[i] - boundary[i]) for i in 1:(p + 1)])
    end
    return tangent, transverse
end

"""
Certify that both transverse quotients remain on opposite sides of every edge secant.
Their cross products with the edge tangent must share a strictly positive direction;
this excludes intersections off the common curve, including arbitrarily close to it.
"""
function _surface_edge_contact(a, b, edge)
    tangent, left = _surface_edge_data(a, edge)
    _, right = _surface_edge_data(b, edge)
    chord = sum(_surface_mid.(x) for x in tangent)
    all(x -> dot(_SurfaceBound.(chord), x).lo > 0, tangent) || return false
    rays = vcat([cross(v, x) for v in tangent for x in left],
        [cross(v, -x) for v in tangent for x in right])
    return _surface_positive_direction(rays)
end

function _surface_pair(a, b, state, depth = 0)
    _surface_budget!(state, depth, "separation of triangles $(a.face) and $(b.face)")
    _surface_boxes_separated(a, b) && return
    shared = intersect(a.corners, b.corners)
    length(shared) == 1 && _surface_vertex_contact(a, b, only(shared)) && return
    length(shared) == 2 && _surface_edge_contact(a, b, shared) && return
    av, bv = map(p -> map(x -> _surface_mid.(x), _surface_patch_vertices(p)), (a, b))
    ae, be = (av[2] - av[1], av[3] - av[1]), (bv[2] - bv[1], bv[3] - bv[1])
    for axis in (cross(ae...), cross(be...), (cross(x, y) for x in ae for y in be)...)
        _surface_separated(a, b, axis) && return
    end
    if !isempty(shared)
        for child_a in _surface_children(a, state), child_b in _surface_children(b, state)

            _surface_pair(child_a, child_b, state, depth + 1)
        end
    else
        diameter(p) = sum((maximum(x -> x[d].hi, p.net) - minimum(x -> x[d].lo, p.net))^2
        for d in 1:3)
        large, small = diameter(a) >= diameter(b) ? (a, b) : (b, a)
        for child in _surface_children(large, state)
            _surface_pair(child, small, state, depth + 1)
        end
    end
end

function _surface_self(patch, state, depth = 0)
    _surface_budget!(state, depth, "injectivity of triangle $(patch.face)")
    _surface_injective(patch) && return
    children = _surface_children(patch, state)
    for child in children
        _surface_self(child, state, depth + 1)
    end
    for i in 1:4, j in (i + 1):4

        _surface_pair(children[i], children[j], state, depth + 1)
    end
end

function _validate_surface_patches(msh; maxdepth::Integer = 20, maxwork::Integer = 200000)
    0 <= maxdepth <= 24 ||
        throw(ArgumentError("validation maxdepth must be between 0 and 24"))
    maxwork > 0 || throw(ArgumentError("validation maxwork must be positive"))
    state = _SurfaceValidation(Dict(n => _surface_bernstein_split(n) for n in 0:4),
        Dict{Tuple{Int, Int}, Int}(), length(Inti.nodes(msh)), maxdepth, maxwork, 0, 0)
    patches = _SurfacePatch[]
    for E in Inti.element_types(msh)
        p = Inti.order(E)
        conversion = _surface_bernstein_conversion(p)
        for (index, el) in enumerate(Inti.elements(msh, E))
            indices = Inti.connectivity(msh, E)[:, index]
            values = [_SurfaceBound.(Inti.nodes(msh)[i]) for i in indices]
            net = _surface_transform(conversion, values)
            corners = Tuple(indices[collect(Inti.vertices_idxs(E))])
            patch = _SurfacePatch(net, p, corners, length(patches) + 1)
            _surface_positive_jacobian(_surface_jacobian(patch), 2p - 2, state, patch.face)
            _surface_boundary_injective(patch) || _surface_self(patch, state)
            push!(patches, patch)
        end
    end
    lower = [minimum(x -> x[1].lo, p.net) for p in patches]
    upper = [maximum(x -> x[1].hi, p.net) for p in patches]
    order = sortperm(lower)
    for (position, i) in enumerate(order)
        for next in (position + 1):length(order)
            j = order[next]
            lower[j] > upper[i] && break
            _surface_pair(patches[i], patches[j], state)
        end
    end
    return (; intersection_check = :adaptive_bernstein, jacobian_check = :bernstein_bounds,
        arithmetic = :outward_rounded, maxdepth, maxwork, depth = state.depth, work = state.work)
end
