# Outgoing exp(ikr)/(4πr) sources, in the full-3D boundary solver frame.
function _point_source_amplitude(k, coefficients, sources, beta, alpha, direction)
    q = direction === nothing ? -_bem3d_incidence_direction(beta, alpha) : direction
    length(q) == 3 && all(isfinite, q) && norm(q) > 0 ||
        throw(ArgumentError("direction must be a finite nonzero three-vector"))
    qhat = Tuple(q / norm(q))
    return sum(c * cis(-k * _dot3(qhat, y))
    for (c, y) in zip(coefficients, sources)) / (4π)
end

# Pressure and density-scaled normal-derivative continuity at one fluid interface.
function _fluid_mfs_interface!(matrix, rhs, quad, exterior, interior, boundary, k,
        pinc, gradinc; pressure_offset = 0, velocity_offset = length(quad),
        interior_offset = length(exterior))
    ki = k / boundary.soundspeed_contrast
    invrho = inv(boundary.density_contrast)
    Threads.@threads for i in eachindex(quad)
        point, normal = Tuple(quad[i].coords), Tuple(quad[i].normal)
        rowp, rowv = pressure_offset + i, velocity_offset + i
        rhs[rowp] = -pinc(point)
        rhs[rowv] = -_dot3(gradinc(point), normal)
        for j in eachindex(exterior)
            matrix[rowp, j] = _green3d(k, point, exterior[j])
            matrix[rowv, j] = _dgreen3d_dn(k, point, normal, exterior[j])
        end
        for j in eachindex(interior)
            matrix[rowp, interior_offset + j] = -_green3d(ki, point, interior[j])
            matrix[rowv, interior_offset + j] = -invrho *
                                                _dgreen3d_dn(ki, point, normal, interior[j])
        end
    end
    return matrix, rhs
end
