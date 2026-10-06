# Result of `fem(...; method = :volume)`, holding harmonic coefficients of the scattered pressure or the volume extraction data `shell`.
struct _VolumeFEMData
    positive::Vector{ComplexF64}
    negative::Vector{ComplexF64}
    R::Float64
    L::Int
    shell::Union{Nothing, NamedTuple}
    rotation::Matrix{Float64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    pinc::Any
    diagnostics::NamedTuple
    system::Any
    solution::Vector{ComplexF64}
end

function _volume_amplitude(d::_VolumeFEMData, k::Real; angle::Real, azimuth::Real)
    direction = d.rotation *
                [sin(angle) * cos(azimuth), sin(angle) * sin(azimuth), cos(angle)]
    d.shell === nothing || return _volume_extraction_amplitude(d.shell, k, direction)
    return _volume_far_field(d.positive, d.negative, k, d.R, d.L, direction)
end

# Smooth step from 0 at t = 0 to 1 at t = 1 with two continuous derivatives.
_smoothstep(t) = t <= 0 ? zero(t) : t >= 1 ? one(t) : t^3 * (10 + t * (6t - 15))

# Solution-independent data of the volume far-field integral, on the fluid cells between the levels where the cutoff chi rises from 0 to 1.
function _volume_extraction_setup(grid, labels, domain, lower, upper)
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    chi(x) = _smoothstep((_domain_level(domain, x) - lower) / (upper - lower))
    reference = _volume_reference_values()
    nq = size(reference, 1)
    cells = Int[]
    points, dVs, laplacians, gradients = Vector{Float64}[], Float64[], Float64[],
    Vector{Float64}[]
    for (cell_id, cell) in enumerate(grid.cells)
        labels[cell_id] == _REGION_FLUID || continue
        coords = [grid.nodes[i].x for i in cell.nodes]
        levels = [_domain_level(domain, collect(x)) for x in coords]
        (minimum(levels) < upper && maximum(levels) > lower) || continue
        Ferrite.reinit!(cv, cell, coords)
        push!(cells, cell_id)
        for q in 1:nq
            x = collect(Ferrite.spatial_coordinate(cv, q, coords))
            push!(points, x)
            push!(dVs, Ferrite.getdetJdV(cv, q))
            push!(laplacians, sum(diag(ForwardDiff.hessian(chi, x))))
            push!(gradients, ForwardDiff.gradient(chi, x))
        end
    end
    nodes = reduce(hcat, [collect(grid.cells[c].nodes) for c in cells])
    return (; nodes, reference, dVs, points = reduce(hcat, points), laplacians,
        gradients = reduce(hcat, gradients))
end

# The extraction data with the quadrature weights p * dV of one pressure solution.
function _volume_extraction(setup, solution)
    nq = size(setup.reference, 1)
    weights = zeros(ComplexF64, length(setup.dVs))
    for c in axes(setup.nodes, 2), q in 1:nq

        pressure = zero(ComplexF64)
        for i in axes(setup.nodes, 1)
            pressure += setup.reference[q, i] * solution[_pressure_dof(setup.nodes[i, c])]
        end
        weights[(c - 1) * nq + q] = pressure * setup.dVs[(c - 1) * nq + q]
    end
    return (; weights, setup.points, setup.laplacians, setup.gradients)
end

# Far-field amplitude in the unit direction `direction` of the solution frame from the volume extraction data.
function _volume_extraction_amplitude(data, k, direction)
    phase = cis.(-k .* (direction' * data.points)[:])
    factor = data.laplacians .- 2im * k .* (direction' * data.gradients)[:]
    return sum(data.weights .* phase .* factor) / 4pi
end

# Far-field amplitude at the unit vector `direction` of the solution frame.
function _volume_far_field(positive, negative, k, R, L, direction)
    norms = _harmonic_norms(L)
    Y = _harmonics_upto(L, direction[3], atan(direction[2], direction[1]), norms)
    total = zero(ComplexF64)
    for (h, (l, m)) in enumerate([(l, m) for l in 0:L for m in 0:l])
        weight = (-im)^(l + 1) / (R^2 * hs(l, k * R))
        total += weight * positive[h] * Y[h]
        m > 0 && (total += weight * negative[h] * (-1)^m * conj(Y[h]))
    end
    return total / k
end

# Number of active dofs above which `solver = :auto` iterates.
