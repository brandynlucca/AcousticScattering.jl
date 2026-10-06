# Close-ring Helmholtz kernels: integrate the first three even Taylor terms
# analytically in azimuth, leaving a smooth remainder for fixed quadrature.
# Laplace Fourier coefficients are toroidal harmonics Q_{m-1/2}(1+d²/2).
# With R=sqrt(rho*rho2), d=distance/R and s=r/R, q_m=∫₀^π cos(mφ)/s dφ.
# Multiplication by s² gives the adjacent-mode differences for the s and s³
# moments. The final factor 1/(2π) includes the full-ring symmetry and 1/(4π).
# See Helsing and Karlsson, JCP 272 (2014), doi:10.1016/j.jcp.2014.04.053,
# for the kernel-splitting approach. The existing adaptive rule remains fallback.

# AGM in the complementary modulus preserves small separations even when
# 1 - (d/hypot(d,2))² rounds to one.
function _ring_elliptic_complement(b::Float64)
    a, total, weight = 1.0, (1 - b * b) / 2, 1.0
    for _ in 1:20
        c = (a - b) / 2
        total += weight * c * c
        an = (a + b) / 2
        b = sqrt(a * b)
        a = an
        weight *= 2
        abs(c) <= eps(a) && break
    end
    K = π / (2a)
    return K, K * (1 - total)
end

@inline function _ring_laplace_modes!(q, d)
    dsq, root = d * d, hypot(d, 2.0)
    K, E = _ring_elliptic_complement(d / root)
    if (length(q) - 1) * d > 2
        # Q is the minimal solution away from the singular limit. Evaluate
        # continued-fraction ratios backwards, then normalize with Q_{-1/2}.
        stop = length(q) + ceil(Int, 20 / (2asinh(d / 2)))
        ratio = 0.0
        @inbounds for m in stop:-1:1
            ratio = (m - 0.5) / (2m * (1 + dsq / 2) - (m + 0.5) * ratio)
            m < length(q) && (q[m + 1] = ratio)
        end
        q[1] = 2K / root
        @inbounds for j in 2:length(q)
            q[j] *= q[j - 1]
        end
        return E
    end
    q[1] = 2K / root
    q[2] = q[1] + dsq / 2 * q[1] - root * E
    @inbounds for m in 1:(length(q) - 2)
        q[m + 2] = (2m * (q[m + 1] + dsq / 2 * q[m + 1]) -
                    (m - 0.5) * q[m]) / (m + 0.5)
    end
    return E
end

@inline _ring_q(q, m) = @inbounds q[abs(m) + 1]
@inline function _ring_r1(q, dsq, m)
    return (2 + dsq) * _ring_q(q, m) - _ring_q(q, m - 1) - _ring_q(q, m + 1)
end
@inline function _ring_r3(q, dsq, m)
    return (2 + dsq) * _ring_r1(q, dsq, m) -
           _ring_r1(q, dsq, m - 1) - _ring_r1(q, dsq, m + 1)
end

function _ring_split_rule(order, modes, width)
    nodes, weights = gauss(order)
    angles = (π / 2) .* (nodes .+ 1)
    table = [_cos_multiples(phi, first(modes), length(modes), width) for phi in angles]
    return (; half_sines = sin.(angles ./ 2), weights = (π / 2) .* weights, table)
end

function _ring_split_workspace(modes, width)
    (first(modes) < 0 || last(modes) > 64) && return nothing
    order = 16cld(last(modes) + 24, 16)
    return (; q = zeros(last(modes) + 4),
        coarse = _ring_split_rule(order, modes, width),
        fine = _ring_split_rule(order + 32, modes, width))
end

# Stable evaluation near zero avoids subtracting nearly equal Taylor terms.
@inline function _ring_split_remainder(x)
    if abs(x) < 1
        y = x * x
        v = -x^6 / 720 * (1 -
             y / 56 * (1 -
              y / 90 * (1 -
               y / 132 *
               (1 - y / 182 * (1 - y / 240 * (1 - y / 306 * (1 - y / 380)))))))
        k = x^6 / 144 * (1 -
             y / 40 * (1 -
              y / 70 * (1 -
               y / 108 *
               (1 - y / 154 * (1 - y / 208 * (1 - y / 270 * (1 - y / 340)))))))
        ik = x^3 / 3 * (1 -
              y / 10 * (1 -
               y / 28 * (1 -
                y / 54 *
                (1 -
                 y / 88 * (1 - y / 130 * (1 - y / 180 * (1 - y / 238 *
                                                             (1 - y / 304))))))))
        return complex(k, ik), complex(v, sin(x))
    end
    s, c = sincos(x)
    return complex(c + x * s - 1 - x * x / 2 + x^4 / 8, s - x * c),
    complex(c - 1 + x * x / 2 - x^4 / 24, s)
end

@inline function _ring_split_base(q, E, d, lambda, p, t, modes, ::Val{N}) where {N}
    # projection/R = p - 4t*sin²(φ/2) = (p+t*d²) - t*s².
    dsq, a = d * d, p + t * d * d
    K, V = MVector{N, ComplexF64}(undef), MVector{N, ComplexF64}(undef)
    @inbounds for c in 1:N
        if c > length(modes)
            K[c] = V[c] = 0
            continue
        end
        m = first(modes) + c - 1
        qm, r1, r3 = _ring_q(q, m), _ring_r1(q, dsq, m), _ring_r3(q, dsq, m)
        # H = d² ∫cos(mφ)/(d²+4sin²(φ/2))^(3/2)dφ stays finite at d→0.
        H = m == 0 ? 2E / hypot(d, 2.0) :
            (m - 0.5) * (_ring_q(q, m - 1) - qm - dsq / 2 * qm) / (1 + dsq / 4)
        V[c] = qm - lambda^2 / 2 * r1 + lambda^4 / 24 * r3
        K[c] = (p / dsq + t) * H - t * qm +
               lambda^2 / 2 * (a * qm - t * r1) - lambda^4 / 8 * (a * r1 - t * r3)
    end
    return vcat(SVector(K), SVector(V))
end

@inline function _ring_split_integral(
        base, d, lambda, p, t, inverse_radius, rule, ::Val{N}) where {N}
    total = base
    @inbounds for j in eachindex(rule.weights)
        s = rule.half_sines[j]
        r = hypot(d, 2s)
        kr, vr = _ring_split_remainder(lambda * r)
        kr *= (p - 4t * s * s) / r^3 * rule.weights[j]
        vr *= rule.weights[j] / r
        total += vcat(kr * rule.table[j], vr * rule.table[j])
    end
    factors = SVector{2N, Float64}(ntuple(
        i -> (i <= N ? inverse_radius^2 : inverse_radius) / (2π), Val(2N)))
    return total .* factors
end

# A concrete status/result pair and explicit inlining help inference through
# nested quadrature callers. Unspecialized calls can box results and arguments.
@inline function _ring_split_KV(k, rho, z, rho2, z2, normal, projection,
        modes, rtol, width::Val{N}, scratch) where {N}
    rejected = (false, zero(SVector{2N, ComplexF64}))
    scratch === nothing && return rejected
    radius = sqrt(rho * rho2)
    distance = hypot(rho - rho2, z - z2)
    isfinite(radius) && radius > 0 && distance >= _RING_DISTANCE_FLOOR || return rejected
    d, lambda = distance / radius, k * radius
    0 < d <= 0.5 && abs(lambda) <= 8 || return rejected
    inverse_radius = inv(radius)
    isfinite(inverse_radius^2) && inverse_radius^2 > 0 || return rejected
    p, t = projection / radius, normal * rho / (2radius)
    E = _ring_laplace_modes!(scratch.q, d)
    base = _ring_split_base(scratch.q, E, d, lambda, p, t, modes, width)
    fine = _ring_split_integral(base, d, lambda, p, t, inverse_radius, scratch.fine, width)
    all(isfinite, fine) || return rejected
    target = max(_QUAD_ATOL, rtol * norm(fine))
    # Estimate roundoff in the recurrence and subtracted moments;
    # quadrature refinement alone cannot detect their shared arithmetic error.
    qbound = (last(modes) + 3) * d <= 2 ? scratch.q[1] :
             _ring_q(scratch.q, max(first(modes) - 3, 0))
    roundoff = 8eps(Float64) * (last(modes) + 4)^2 * qbound *
               (1 + lambda^4) *
               max(inverse_radius,
                   (abs(p) / d^2 + abs(t)) * inverse_radius^2)
    roundoff += 64eps(Float64) * (1 + lambda^4) *
                max(inverse_radius, (abs(p) + 4abs(t)) * inverse_radius^2)
    roundoff <= 0.05target || return rejected
    coarse = _ring_split_integral(
        base, d, lambda, p, t, inverse_radius, scratch.coarse, width)
    norm(fine - coarse) <= 0.05target || return rejected
    return true, fine
end
