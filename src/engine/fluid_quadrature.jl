function _fluid_quadrature_size(source)
    center = sum(q.coords for q in source)/sum(1 for _ in source)
    radius = maximum(q -> norm(q.coords-center), source)
    rms = sqrt(sum(q.weight*sum(abs2, q.coords-center) for q in source) /
               sum(q.weight for q in source))
    return (; center, radius, rms)
end

_fluid_regular_range(k, bounds) = k*bounds.radius <= 1 && 2k*bounds.rms <= 1

"""
    _regular_helmholtz_traces(point, center, radius, k, degree)

Pressure and normal traces of regular solid spherical harmonics multiplied by
the normalized spherical Bessel series. Cartesian recurrences avoid polar-axis
singularities; the radial series is used only for arguments of magnitude at most two.
See NIST DLMF 10.53.1 and 14.10.3 for the radial series and degree recurrence.
"""
function _regular_helmholtz_traces(point, center, radius, k, degree)
    v = (point.coords - center) / radius
    x, y, z = v
    t = dot(v, v)
    xy = x + im*y
    diagonal = one(ComplexF64)
    gradient = zero(SVector{3, ComplexF64})
    values, fluxes = ComplexF64[], ComplexF64[]
    for m in 0:degree
        if m > 0
            gradient = -(2m-1)*(xy*gradient + diagonal*SVector(1.0, im, 0.0))
            diagonal *= -(2m-1)*xy
        end
        previous, previous_gradient = zero(diagonal), zero(gradient)
        current, current_gradient = diagonal, gradient
        for l in m:degree
            if l > m
                next = ((2l-1)*z*current-(l+m-1)*t*previous)/(l-m)
                next_gradient = ((2l-1)*(z*current_gradient+current*SVector(0.0, 0.0, 1.0)) -
                                 (l+m-1)*(t*previous_gradient+2v*previous))/(l-m)
                previous, previous_gradient = current, current_gradient
                current, current_gradient = next, next_gradient
            end
            coefficient, radial, derivative = 1.0, 1.0, 0.0
            for p in 1:20
                coefficient *= -(k*radius)^2/(2p*(2l+2p+1))
                radial += coefficient*t^p
                derivative += p*coefficient*t^(p-1)
            end
            normalization = sqrt((2l+1)/(4pi) / prod((l - m + 1):(l + m); init = 1.0))
            value = normalization*current*radial
            flux = normalization*sum(point.normal .*
                                     (radial*current_gradient+2v*current*derivative))/radius
            push!(values, value)
            push!(fluxes, flux)
            if m > 0
                push!(values, conj(value))
                push!(fluxes, conj(flux))
            end
        end
    end
    return values, fluxes
end

"""
    _fluid_layer_operators(op, target, source, correction; derivative=false, regular=true)

Dense fluid layer quadrature. At low frequency, correct nearby interactions using
regular Helmholtz solutions and a scaled minimum-norm fit to each element's traces.
The complete harmonic space has at most twice the element's quadrature-node count.
The Green-identity correction follows Faria, Pérez-Arancibia and Bonnet (2021),
doi:10.1016/j.cma.2021.113703. Regular waves require `k*radius <= 1` and
`2k*rms_radius <= 1`, with the RMS radius weighted by surface quadrature area.
"""
function _fluid_layer_operators(
        op, target, source, correction; derivative = false, regular = true)
    bounds = _fluid_quadrature_size(source)
    center, radius = bounds.center, bounds.radius
    eligible = regular && correction.method === :dim &&
               _fluid_regular_range(op.k, bounds) &&
               op.k*maximum(q -> norm(q.coords-center), target) <= 2
    eligible || return Inti.single_double_layer(; op, target, source, derivative,
        compression = (method = :none,), correction)
    S, D = Inti.single_double_layer(; op, target, source, derivative,
        compression = (method = :none,), correction = (method = :none,))
    degree = floor(Int, sqrt(2minimum(size(tags, 1)
    for tags in values(source.etype2qtags))))-1
    traces = [_regular_helmholtz_traces(q, center, radius, op.k, degree) for q in source]
    pressure = reduce(vcat, transpose.(first.(traces)))
    flux = reduce(vcat, transpose.(last.(traces)))
    location = target === source ? :on : correction.target_location
    location in (:on, :inside, :outside) ||
        throw(ArgumentError("target_location must be :on, :inside or :outside"))
    multiplier = location === :on ? -0.5 : location === :inside ? -1.0 : 0.0
    defect = multiplier .* reduce(vcat,
        [transpose(_regular_helmholtz_traces(
             q, center, radius, op.k, degree)[derivative ? 2 : 1]) for q in target])
    mul!(defect, S, flux, 1, 1)
    mul!(defect, D, pressure, -1, 1)
    near = Inti.etype_to_nearest_points(target, source; maxdist = get(correction, :maxdist, Inf))
    for (element, tags) in source.etype2qtags
        nq, ne = size(tags)
        for e in 1:ne
            rows = near[element][e]
            isempty(rows) && continue
            columns = tags[:, e]
            traces = [pressure[columns, :]; flux[columns, :]]
            scales = max.(vec(sqrt.(sum(abs2, traces; dims = 2))), eps(Float64))
            factor = svd(transpose(traces ./ scales))
            keep = factor.S .> 1e-13*first(factor.S)
            weights = (factor.V[:, keep] *
                       ((factor.U[:, keep]' * transpose(defect[rows, :])) ./
                        factor.S[keep])) ./ scales
            S[rows, columns] .-= transpose(weights[(nq + 1):2nq, :])
            D[rows, columns] .+= transpose(weights[1:nq, :])
        end
    end
    return S, D
end
