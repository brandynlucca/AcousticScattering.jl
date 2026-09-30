# Baffled circular piston, Rayleigh integral, unit amplitude (Kinsler and Frey, Zemanek 1971). Piston centered at the origin, axis +z.
using QuadGK: quadgk, gauss

"""
    _piston_pressure_axial(k, a, z)

Exact on-axis pressure of a baffled circular piston of radius `a` [m] at axial distance `z`
[m], exterior wavenumber `k` [1/m]. Closed form of [`_piston_pressure`](@ref) restricted to
the axis, `p(z) = exp(ikz) - exp(ik*sqrt(z^2+a^2))`, used to check the general quadrature.
"""
function _piston_pressure_axial(k::Real, a::Real, z::Real)
    return cis(k * z) - cis(k * hypot(z, a))
end

"""
    _piston_pressure(k, a, x; rtol=1e-6)

Pressure at `x = (x, y, z)` [m] radiated by a baffled circular piston of radius `a` [m] in
the `z=0` plane, exterior wavenumber `k` [1/m], from the Rayleigh integral
`(k/2πi) ∫∫_disk exp(ik|x-x'|)/|x-x'| dS'`. On the `z` axis this reduces exactly to
[`_piston_pressure_axial`](@ref). Evaluate an equivalent single boundary integral,
with its removable singularity canceled analytically. This is the full Rayleigh
field, not a far-field approximation.
"""
function _piston_pressure(k::Real, a::Real, x; rtol::Real = 1e-6)
    x1, x2, x3 = x[1], x[2], x[3]
    iszero(x3) && x1^2+x2^2<=a^2 &&
        throw(ArgumentError("pressure evaluation on the piston aperture is not supported"))
    z=abs(x3)
    # Planar divergence theorem: the radial primitive of exp(ikR)/R is
    # (exp(ikR)-exp(ik|z|))/(ik). Rationalize R-|z| and use sinc to
    # remove cancellation at projected points on the aperture rim.
    angular = phi -> begin
        s, c=sincos(phi)
        distance2=(x1-a*c)^2+(x2-a*s)^2
        r=sqrt(distance2+z^2)
        delta=distance2/(r+z)
        numerator=a*(a-x1*c-x2*s)
        -im*k*numerator/(r+z)*cis(k*(z+delta/2))*sinc(k*delta/(2pi))
    end
    return quadgk(angular, 0.0, 2pi; rtol)[1]/(2pi)
end

"""
    _piston_gradient(k, a, x; rtol=1e-6)

Gradient of [`_piston_pressure`](@ref) at `x`, by automatic differentiation of the same
quadrature (the field is smooth off the piston's own plane).
"""
function _piston_gradient(k::Real, a::Real, x; rtol::Real = 1e-6)
    return ForwardDiff.gradient(
        y -> real(_piston_pressure(k, a, y; rtol)), collect(x)) +
           im .*
           ForwardDiff.gradient(y -> imag(_piston_pressure(k, a, y; rtol)), collect(x))
end

"""
    AbstractTransducer

Supertype of sound sources with a `_transducer_field(t, k)` method giving pointwise
`pinc(x)`/`gradinc(x)`. See [`Transducer`](@ref) and [`TankTransducer`](@ref).
"""
abstract type AbstractTransducer end

"""
    Transducer(position, axis, radius)

A baffled circular piston transmitter, aperture `radius` [m], face centered at `position` [m]
with outward normal `axis` (need not be unit length). Evaluate its field with
[`pressure`](@ref) or supply it with `transducer=...` to a supported solver.
"""
struct Transducer <: AbstractTransducer
    position::NTuple{3, Float64}
    axis::NTuple{3, Float64}
    radius::Float64
    function Transducer(position, axis, radius::Real)
        isfinite(radius) && radius > 0 ||
            throw(ArgumentError("radius must be finite and positive"))
        all(isfinite, position) && length(position) == 3 ||
            throw(ArgumentError("position must be a finite three-vector"))
        length(axis)==3 || throw(ArgumentError("axis must have three components"))
        n = norm(collect(Float64.(axis)))
        isfinite(n) && n > 0 || throw(ArgumentError("axis must be a finite nonzero vector"))
        return new(Tuple(Float64.(position)), Tuple(Float64.(axis) ./ n), Float64(radius))
    end
end

"""
    _transducer_field(t::Transducer, k; rtol=1e-6)

Pointwise `pinc(x)`/`gradinc(x)` callables for `t`'s radiated field at exterior wavenumber `k`
[1/m], in the frame `t.position`/`t.axis` are given in. Rotates and translates into the
piston's own frame ([`_piston_pressure`](@ref)/[`_piston_gradient`](@ref), origin-centered,
axis `+z`), then rotates the gradient back.
"""
function _transducer_field(t::Transducer, k::Real; rtol::Real = 1e-6,
        approximation::Symbol = :rayleigh)
    isfinite(k) && k>0 || throw(ArgumentError("k must be finite and positive"))
    isfinite(rtol) && rtol>0 || throw(ArgumentError("rtol must be finite and positive"))
    approximation in (:rayleigh, :farfield) ||
        throw(ArgumentError("approximation must be :rayleigh or :farfield"))
    R = _axis_rotation(collect(t.axis))
    Rt = permutedims(R)
    pos = collect(t.position)
    to_local(x) = Rt * (collect(x) - pos)
    local_pressure = approximation===:rayleigh ?
                     x->_piston_pressure(k, t.radius, x; rtol) :
                     x->_piston_farfield(k, t.radius, x)
    pinc = x -> local_pressure(to_local(x))
    gradinc = if approximation===:rayleigh
        x -> R * _piston_gradient(k, t.radius, to_local(x); rtol)
    else
        x -> R*(ForwardDiff.gradient(y->real(local_pressure(y)), to_local(x)) +
                im*ForwardDiff.gradient(y->imag(local_pressure(y)), to_local(x)))
    end
    return pinc, gradinc
end

function _piston_farfield(k, a, x)
    r=norm(x)
    r>0 || throw(ArgumentError("far-field evaluation requires nonzero source distance"))
    s2=(k*a)^2*(x[1]^2+x[2]^2)/r^2
    # Entire series at the axis avoids differentiating sqrt(0) or J1(z)/z.
    directivity=s2<1e-6 ? 1-s2/8+s2^2/192-s2^3/9216 :
                2besselj(1, sqrt(s2))/sqrt(s2)
    return (-im*k*a^2/2)*cis(k*r)/r*directivity
end

"""
    pressure(transducer::AbstractTransducer, k, points; rtol=1e-6,
        approximation=:rayleigh)

Evaluate a piston or first-order tank-image source field at exterior wavenumber `k`
[rad/m]. Accept a Cartesian point, an array of points, or a 3-by-N matrix. Coordinates
are in meters in the transducer frame's global coordinates. Pressure follows the
existing unit piston-drive convention; physical gain belongs to the response function.

`:rayleigh` evaluates the aperture integral, including Fresnel structure. The explicit
`:farfield` approximation uses `-im*k*a^2/2 * exp(im*k*r)/r * 2J1(k*a*sin(theta))/(k*a*sin(theta))`.
Use it only when both aperture size and `k*a^2` are small relative to range; it is
never selected automatically during a scattering solve. Source-plane singular points
are excluded. The Rayleigh field is the free-space continuation used for image sums;
the physical baffled-piston model describes the front half-space.
"""
function pressure(transducer::AbstractTransducer, k::Real, points; kwargs...)
    pinc, _=_transducer_field(transducer, k; kwargs...)
    evaluate=x->begin
        length(x)==3 && all(isfinite, x) ||
            throw(ArgumentError("points must be finite three-vectors"))
        pinc(x)
    end
    if points isa Tuple{Vararg{Real, 3}} || points isa AbstractVector{<:Real}
        return evaluate(points)
    elseif points isa AbstractMatrix{<:Real}
        size(points, 1)==3 || throw(ArgumentError("point matrices must have three rows"))
        return [evaluate(view(points, :, i)) for i in axes(points, 2)]
    end
    return map(evaluate, points)
end
