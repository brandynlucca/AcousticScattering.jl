# Shared coordinate validation and output shape for field queries.
function _field_points(evaluate, point::Tuple{Real, Real, Real})
    only(evaluate([point]))
end

function _field_points(evaluate, point::AbstractVector{<:Real})
    length(point) == 3 || throw(ArgumentError("a point must have three coordinates"))
    return only(evaluate([Tuple(point)]))
end

function _field_points(evaluate, points::AbstractMatrix{<:Real})
    size(points, 1) == 3 || throw(ArgumentError("point matrices must have three rows"))
    return evaluate([Tuple(point) for point in eachcol(points)])
end

function _field_points(evaluate, points::AbstractArray)
    coordinates = map(_field_coordinates, points)
    return reshape(evaluate(vec(coordinates)), size(points))
end

function _field_coordinates(point)
    point isa Union{Tuple{Real, Real, Real}, AbstractVector{<:Real}} &&
    length(point) == 3 ||
        throw(ArgumentError("each point must have three real coordinates"))
    return Tuple(point)
end

function _field_points(evaluate, points)
    throw(ArgumentError("points must be a Cartesian point, an array of points or a 3-row matrix"))
end
