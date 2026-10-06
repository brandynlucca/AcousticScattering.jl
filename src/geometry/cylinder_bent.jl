# Bent-cylinder surface points, normals and quadrature in Cartesian coordinates.

function _bent_cylinder_point(s::Real, φ::Real, radius::Real, radius_curvature::Real)
    γ = s / radius_curvature
    sγ, cγ = sincos(γ)
    cφ, sφ = sincos(φ)
    x = (radius_curvature + radius * cφ) * sγ
    y = radius_curvature * (1 - cγ) - radius * cφ * cγ
    z = -(radius * sφ)
    return (x, y, z)
end

function _bent_cylinder_normal(s::Real, φ::Real, radius_curvature::Real)
    γ = s / radius_curvature
    sγ, cγ = sincos(γ)
    cφ, sφ = sincos(φ)
    # Outward normal = (r(s,φ) - r_c(s))/radius, r_c the centerline (`_bent_cylinder_point` at radius=0).
    return (cφ * sγ, -cφ * cγ, -sφ)
end

"""
    bent_cylinder_mfs_points(radius, length, radius_curvature, n_s, n_φ)

Collocation points, outward normals, and surface-element areas for a
uniformly bent finite cylinder (see [`bent_cylinder_kirchhoff_form_function`](@ref)
for the geometry), on an `n_s × n_φ` grid over `s ∈ [-length/2, length/2]`
(midpoint rule) and `φ ∈ [0, 2π)` (uniform, periodic). Returns
`(points, normals, areas)`, each a `Vector` of length `n_s*n_φ`.
"""
function bent_cylinder_mfs_points(
        radius::Real, length::Real, radius_curvature::Real, n_s::Integer, n_φ::Integer)
    ds = length / n_s
    dφ = 2π / n_φ
    points = Vector{NTuple{3, Float64}}(undef, n_s * n_φ)
    normals = Vector{NTuple{3, Float64}}(undef, n_s * n_φ)
    areas = Vector{Float64}(undef, n_s * n_φ)
    idx = 1
    for i in 1:n_s
        s = -length / 2 + (i - 0.5) * ds
        for j in 1:n_φ
            φ = (j - 0.5) * dφ
            points[idx] = _bent_cylinder_point(s, φ, radius, radius_curvature)
            normals[idx] = _bent_cylinder_normal(s, φ, radius_curvature)
            areas[idx] = radius * (1 + (radius / radius_curvature) * cos(φ)) * ds * dφ
            idx += 1
        end
    end
    return points, normals, areas
end
