# Spherical Bessel/Hankel functions and Legendre polynomials (DLMF §10, §14).

"""
    js(n, x)

Spherical Bessel function of the first kind, jₙ(x) = √(π/2x) Jₙ₊₁/₂(x).
"""
function js(n::Integer, x::Number)
    iszero(x) && return n == 0 ? one(float(x)) : zero(float(x))
    return sqrt(oftype(float(real(x)), pi) / (2x)) * besselj(n + 0.5, x)
end

"""
    ys(n, x)

Spherical Bessel function of the second kind (Neumann function),
yₙ(x) = √(π/2x) Yₙ₊₁/₂(x). Singular at x = 0.
"""
function ys(n::Integer, x::Number)
    return sqrt(oftype(float(real(x)), pi) / (2x)) * bessely(n + 0.5, x)
end

"""
    hs(n, x)

Spherical Hankel function of the first kind, h⁽¹⁾ₙ(x) = jₙ(x) + i·yₙ(x),
representing an outgoing wave under the e^{-iωt} time convention.
"""
hs(n::Integer, x::Number) = js(n, x) + im * ys(n, x)

# DLMF 10.51.2: f'_n(x) = (n/x) f_n(x) - f_{n+1}(x)
_besselderiv(f::F, n::Integer, x::Number) where {F} = (n / x) * f(n, x) - f(n + 1, x)

"Derivative of [`js`](@ref) with respect to its argument."
jsd(n::Integer, x::Number) = _besselderiv(js, n, x)
"Derivative of [`ys`](@ref) with respect to its argument."
ysd(n::Integer, x::Number) = _besselderiv(ys, n, x)
"Derivative of [`hs`](@ref) with respect to its argument."
hsd(n::Integer, x::Number) = _besselderiv(hs, n, x)

# f''_n(x) = -(n/x²) f_n(x) + (n/x) f'_n(x) - f'_{n+1}(x)
function _besselderiv2(f::F, n::Integer, x::Number) where {F}
    -(n / x^2) * f(n, x) + (n / x) * _besselderiv(f, n, x) - _besselderiv(f, n + 1, x)
end

"Second derivative of [`js`](@ref) with respect to its argument."
jsdd(n::Integer, x::Number) = _besselderiv2(js, n, x)
"Second derivative of [`ys`](@ref) with respect to its argument."
ysdd(n::Integer, x::Number) = _besselderiv2(ys, n, x)

"""
    legendre_p(n, x)

Legendre polynomial Pₙ(x) via the standard three-term recurrence.
"""
function legendre_p(n::Integer, x::Number)
    n < 0 && throw(ArgumentError("n must be nonnegative, got $n"))
    n == 0 && return one(float(x))
    p0, p1 = one(float(x)), float(x)
    for k in 2:n
        p0, p1 = p1, ((2k - 1) * x * p1 - (k - 1) * p0) / k
    end
    return p1
end

"""
    legendre_p(l, m, x)

Associated Legendre function Pₗᵐ(x) (Condon-Shortley phase included),
via the standard stable three-term recurrence in `l` starting from the
closed-form diagonal `Pₘᵐ(x) = (-1)ᵐ(2m-1)!!(1-x²)^{m/2}` (DLMF 14.7.7,
14.10.3/14.10.4). Used by the finite cylinder's oblique-incidence
meridian FEM (`cylinder_meridian_fem.jl`) to project a general
(non-axisymmetric) azimuthal Fourier mode's surface trace onto the exact
spherical DtN operator's eigenbasis. The `m=0` case is exactly
[`legendre_p(l, x)`](@ref).
"""
function legendre_p(l::Integer, m::Integer, x::Number)
    l < 0 && throw(ArgumentError("l must be nonnegative, got $l"))
    m < 0 && throw(ArgumentError("m must be nonnegative, got $m"))
    m > l && return zero(float(x))

    pmm = one(float(x))
    if m > 0
        somx2 = sqrt(one(x) - x^2)
        fact = one(x)
        for _ in 1:m
            pmm *= -fact * somx2
            fact += 2
        end
    end
    l == m && return pmm

    pmmp1 = x * (2m + 1) * pmm
    l == m + 1 && return pmmp1

    pll = zero(float(x))
    plm2, plm1 = pmm, pmmp1
    for ll in (m + 2):l
        pll = ((2ll - 1) * x * plm1 - (ll + m - 1) * plm2) / (ll - m)
        plm2, plm1 = plm1, pll
    end
    return pll
end

# (l+m)!/(l-m)! as a product of 2m reciprocals, avoiding factorial overflow.
function _legendre_norm_ratio(l::Integer, m::Integer)
    r = 1.0
    for kk in (l - m + 1):(l + m)
        r /= kk
    end
    return r
end

"""
    neumann_factor(m)

Neumann factor ν_m used in cylindrical/spheroidal modal-series expansions:
ν₀ = 1, νₘ = 2 for m > 0.
"""
neumann_factor(m::Integer) = m == 0 ? 1 : 2
