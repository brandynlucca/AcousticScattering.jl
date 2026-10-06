"""
    Spheroid(a, b)

Spheroid of revolution with semi-axis `a` [m] along the x axis of symmetry
and equatorial semi-axis `b` [m]. Prolate if `a > b`, oblate if `a < b`.
"""
struct Spheroid <: AbstractBody
    a::Float64
    b::Float64
    kind::Symbol
    xi0::Float64
    q::Float64

    function Spheroid(a::Real, b::Real)
        a == b &&
            throw(ArgumentError("Spheroid requires a ≠ b; use the sphere modal series (solvers/modal/sphere.jl) for a sphere"))
        a > 0 && b > 0 || throw(ArgumentError("Spheroid semi-axes must be positive"))
        if a > b
            kind = :prolate
            q = sqrt(a^2 - b^2)
        else
            kind = :oblate
            q = sqrt(b^2 - a^2)
        end
        xi0 = a / q
        return new(Float64(a), Float64(b), kind, xi0, q)
    end
end
