# Independent spherical quadrature of public far-field values. Composite
# Simpson in cos(theta), periodic trapezoidal in azimuth, normalized by 4pi.
function layered_angular_average(solution; n = 16, transform = identity)
    total = zero(ComplexF64)
    for j in 0:n, l in 0:(n - 1)

        weight = j in (0, n) ? 1 : isodd(j) ? 4 : 2
        value = scattering_amplitude(solution;
            angle = acos(-1 + 2j / n), azimuth = 2pi * l / n)
        total += weight * transform(value)
    end
    return total / (3n^2)
end
