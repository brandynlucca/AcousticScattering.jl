"""
    Irregular(r, order; npoints=8*order+16)
    Irregular(a, rc, rs)

A body of revolution about the z axis, for smooth irregular shapes between the canonical
([`Sphere`](@ref), [`Spheroid`](@ref)) and general numerical (`bem`/`mfs`) bodies. Solve with
[`fourier`](@ref).

The first form Fourier-fits a harmonic `order` series to a meridian profile function `r(theta)` in m,
the radial distance from the origin at polar angle `theta`, sampled on `theta in [0,pi]` with
`npoints` quadrature nodes. The profile is extended evenly about the axis, so the fit is a cosine
series and both poles lie on the axis.

The second form supplies the Fourier series `R(theta) = a + sum_n rc[n]*cos(n*theta)` (Eq. (21))
directly. `a` is the mean radius and `rc[n]` the deviation coefficients for harmonic order
`n = 1:length(rc)`. Sine coefficients `rs` must be zero, since any sine term moves the poles off
the axis of symmetry.
"""
struct Irregular <: AbstractBody
    a::Float64
    rc::Vector{Float64}
    rs::Vector{Float64}
    function Irregular(a::Real, rc::AbstractVector{<:Real}, rs::AbstractVector{<:Real})
        length(rc) == length(rs) ||
            throw(ArgumentError("rc and rs must have the same length"))
        isfinite(a) && a > 0 ||
            throw(ArgumentError("a must be finite and positive, got $a"))
        all(iszero, rs) || throw(ArgumentError(
            "rs must be zero, sine terms move the poles off the axis of symmetry"))
        return new(Float64(a), Float64.(rc), Float64.(rs))
    end
end

function Irregular(r, order::Integer; npoints::Integer = 8 * order + 16)
    order >= 0 || throw(ArgumentError("order must be nonnegative, got $order"))
    nodes, weights = _full_period_quadrature(npoints)
    values = r.(min.(nodes, 2π .- nodes))
    a = sum(weights .* values) / (2π)
    rc = [sum(weights .* values .* cos.(n .* nodes)) / π for n in 1:order]
    return Irregular(a, rc, zeros(order))
end

"""
    profile_radius(body, theta)

Evaluate an [`Irregular`](@ref)'s meridian profile `R(theta)` in m (Eq. (21)) at polar angle
`theta` in rad.
"""
function profile_radius(body::Irregular, theta::Real)
    r = body.a
    for n in eachindex(body.rc)
        r += body.rc[n] * cos(n * theta) + body.rs[n] * sin(n * theta)
    end
    return r
end

function _full_period_quadrature(npoints::Integer)
    nodes, weights = gauss(npoints, 0, 2π)
    return nodes, weights
end
