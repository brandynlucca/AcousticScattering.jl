# Cartesian tuple operations and outgoing three-dimensional Helmholtz kernels.
_dot3(a::NTuple{3, <:Real}, b::NTuple{3, <:Real}) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
_sub3(a::NTuple{3, <:Real}, b::NTuple{3, <:Real}) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
_norm3(a::NTuple{3, <:Real}) = sqrt(_dot3(a, a))

function _green3d(k::Real, x::NTuple{3, <:Real}, y::NTuple{3, <:Real})
    cis(k * _norm3(_sub3(x, y))) / (4π * _norm3(_sub3(x, y)))
end
function _dgreen3d_dn(k::Real, x::NTuple{3, <:Real}, nx::NTuple{3, <:Real}, y::NTuple{
        3, <:Real})
    r = _norm3(_sub3(x, y))
    G = cis(k * r) / (4π * r)
    return (im * k - 1 / r) * G * _dot3(nx, _sub3(x, y)) / r
end
