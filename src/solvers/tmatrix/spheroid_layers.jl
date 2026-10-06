# Galerkin projection of confocal fluid/elastic interface conditions onto
# spheroidal angular modes. The full mixed transition is assembled per m.
function _mixed_spheroid_basis(layer::FluidLayer, m, nmax, k, body, xi, nodes, mode)
    kinds = mode === :both ? (1, 3) : mode === :regular ? (1,) : (3,)
    pressure = Matrix{ComplexF64}(undef, length(nodes), 0)
    radial = similar(pressure)
    c = k * body.q / layer.soundspeed_contrast
    hxi = [_surface_metric(body.q, xi, eta, body.kind)[1] for eta in nodes]
    for kind in kinds
        p = zeros(ComplexF64, length(nodes), nmax-m+1)
        u = similar(p)
        for (j, n) in enumerate(m:nmax)
            R = SpheroidalWaves.rmn(m, n, c, [xi]; spheroid = body.kind, kind)
            S = SpheroidalWaves.smn(m, n, c, nodes; spheroid = body.kind, normalize = true).value
            p[:, j] .= R.value[1] .* S
            u[:, j] .= R.derivative[1] .* S ./
                       (layer.density_contrast * k^2 .* hxi)
        end
        pressure = hcat(pressure, p)
        radial = hcat(radial, u)
    end
    return (; kind = :fluid, p = pressure, un = radial)
end

function _mixed_spheroid_basis(layer::ElasticLayer, m, nmax, k, body, xi, nodes, mode)
    material = (layer.density_contrast, layer.speed_longitudinal_contrast,
        layer.speed_transversal_contrast)
    regular = _elastic_surface_samples(m, nmax, body.q, xi, nodes, k, material,
        body.kind; radial_kind = 1)[2]
    values = if mode === :both
        singular = _elastic_surface_samples(m, nmax, body.q, xi, nodes, k,
            material, body.kind; radial_kind = 2)[2]
        cat(ComplexF64.(regular), ComplexF64.(regular .+ im .* singular); dims = 2)
    else
        ComplexF64.(regular)
    end
    field(i) = permutedims(values[i, :, :], (2, 1))
    return (; kind = :elastic, un = field(1), ut = field(2), up = field(3),
        tn = field(4), tt = field(5), tp = field(6))
end

function _mixed_spheroid_conditions(outer, inner, m)
    if inner === nothing
        return outer === :fluid ? ((:p, 1, nothing, 0, :normal),) :
               m == 0 ? ((:tn, 1, nothing, 0, :normal),
            (:tt, 1, nothing, 0, :meridional)) :
               ((:tn, 1, nothing, 0, :normal), (:tt, 1, nothing, 0, :meridional),
            (:tp, 1, nothing, 0, :azimuthal))
    elseif outer === :fluid && inner === :fluid
        return ((:p, 1, :p, -1, :normal), (:un, 1, :un, -1, :normal))
    elseif outer === :elastic && inner === :elastic
        return m == 0 ?
               ((:un, 1, :un, -1, :normal), (:ut, 1, :ut, -1, :meridional),
            (:tn, 1, :tn, -1, :normal), (:tt, 1, :tt, -1, :meridional)) :
               ((:un, 1, :un, -1, :normal), (:ut, 1, :ut, -1, :meridional),
            (:up, 1, :up, -1, :azimuthal), (:tn, 1, :tn, -1, :normal),
            (:tt, 1, :tt, -1, :meridional), (:tp, 1, :tp, -1, :azimuthal))
    elseif outer === :fluid
        return m == 0 ?
               ((:un, 1, :un, -1, :normal), (:p, 1, :tn, 1, :normal),
            (nothing, 0, :tt, 1, :meridional)) :
               ((:un, 1, :un, -1, :normal), (:p, 1, :tn, 1, :normal),
            (nothing, 0, :tt, 1, :meridional), (nothing, 0, :tp, 1, :azimuthal))
    else
        return m == 0 ?
               ((:un, 1, :un, -1, :normal), (:tn, 1, :p, 1, :normal),
            (:tt, 1, nothing, 0, :meridional)) :
               ((:un, 1, :un, -1, :normal), (:tn, 1, :p, 1, :normal),
            (:tt, 1, nothing, 0, :meridional), (:tp, 1, nothing, 0, :azimuthal))
    end
end

function _mixed_spheroid_transition(body, bc, k, m, nmax; nquad = 2nmax+24, rtol = 1e-13)
    layers, ratios = _layered_materials(bc.material)
    xis = [body.xi0;
           [_confocal_inner_xi(body, r) for r in [ratios[2:end]; bc.radius_ratio]]]
    nodes, weights = gauss(nquad)
    ne = nmax-m+1
    ranges = UnitRange{Int}[]
    start = ne+1
    for layer in layers
        nb = layer isa FluidLayer ? ne : length(_elastic_basis_list(m, nmax))
        push!(ranges, start:(start + 2nb - 1))
        start += 2nb
    end
    core_range = bc.interior isa FluidInterior ? (start:(start + ne - 1)) : (1:0)
    nunknown = isempty(core_range) ? start-1 : last(core_range)
    exterior = FluidLayer(1.0, 1.0)
    core = bc.interior isa FluidInterior ?
           FluidLayer(
        bc.interior.density_contrast, bc.interior.soundspeed_contrast) : nothing
    tests = Dict{Symbol, Matrix{Float64}}()
    for (symbol, degrees) in ((:normal, m:nmax),
        (:meridional, (m == 0 ? 1 : m):nmax), (:azimuthal, m:nmax))
        rows = [SpheroidalWaves.smn(m, n, k*body.q, nodes;
                    spheroid = body.kind, normalize = true) for n in degrees]
        tests[symbol] = reduce(vcat,
            [reshape(symbol === :meridional ?
                     row.derivative : row.value, 1, :) for row in rows])
    end
    blocks = Matrix{ComplexF64}[]
    rhsblocks = Matrix{ComplexF64}[]
    for i in 1:length(xis)
        xi = xis[i]
        outer = i == 1 ? exterior : layers[i - 1]
        inner = i <= length(layers) ? layers[i] : core
        outerbasis = _mixed_spheroid_basis(outer, m, nmax, k, body, xi, nodes,
            i == 1 ? :outgoing : :both)
        innerbasis = inner === nothing ? nothing :
                     _mixed_spheroid_basis(inner, m, nmax, k,
            body, xi, nodes, i == length(xis) ? :regular : :both)
        incident = i == 1 ?
                   _mixed_spheroid_basis(exterior, m, nmax, k, body, xi, nodes, :regular) :
                   nothing
        outercols = i == 1 ? (1:ne) : ranges[i - 1]
        innercols = i <= length(layers) ? ranges[i] : core_range
        element = [_surface_metric(body.q, xi, eta, body.kind)[2]
                   for eta in nodes]
        for (of, os, inf, is, testkind) in _mixed_spheroid_conditions(outerbasis.kind,
            innerbasis === nothing ? nothing : innerbasis.kind, m)
            Q = tests[testkind] .* (weights .* element)'
            block = zeros(ComplexF64, size(Q, 1), nunknown)
            rb = zeros(ComplexF64, size(Q, 1), ne)
            if of !== nothing
                block[:, outercols] .= os .* (Q * getproperty(outerbasis, of))
                incident === nothing ||
                    (rb .= -os .* (Q * getproperty(incident, of)))
            end
            inf === nothing ||
                (block[:, innercols] .= is .* (Q * getproperty(innerbasis, inf)))
            scale = maximum(abs, block)
            scale > 0 || throw(ArgumentError("mixed spheroid interface block is singular"))
            block ./= scale
            rb ./= scale
            push!(blocks, block)
            push!(rhsblocks, rb)
        end
    end
    A = vcat(blocks...)
    B = vcat(rhsblocks...)
    size(A, 1) == size(A, 2) || error("projected system is not square: $(size(A))")
    colscale = vec(maximum(abs, A; dims = 1))
    all(>(0), colscale) ||
        throw(ArgumentError("mixed spheroid interface columns are singular"))
    C = A ./ colscale'
    singular = svdvals(C)
    m == 0 && singular[end] / singular[1] < 1e-14 &&
        throw(ArgumentError("mixed spheroid transition is ill-conditioned at this truncation; use volume FEM"))
    X = (pinv(C; rtol) * B) ./ colscale
    T = X[1:ne, :]
    all(isfinite, T) || throw(ArgumentError("mixed spheroid transition is nonfinite"))
    return T
end

"""
    form_function(boundary::Shelled{<:LayeredMaterial}, k, body::Spheroid;
        incidence_angle=0, scatter_angle=π-incidence_angle,
        m_max=4, n_max=8, check=true)

Projected spheroidal transition matrix for confocal fluid and elastic layers.
Currently bounded to prolate aspect ratio at most 1.25, `k*body.a ≤ 1.5`,
`m_max ≤ 4`, and `n_max ≤ 8` in double precision. A reduced-degree check
rejects amplitudes that change by more than 3%. Use full-3D volume FEM beyond
these bounds.
"""
function form_function(boundary::Shelled{<:LayeredMaterial}, k::Real,
        body::Spheroid;
        incidence_angle::Real = 0.0, incidence_azimuth::Real = 0.0,
        scatter_angle::Real = π - incidence_angle,
        scatter_azimuth::Real = incidence_azimuth + π,
        m_max::Integer = 4, n_max::Integer = 8,
        n_quad::Integer = 2n_max + 24, check::Bool = true)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    body.kind === :prolate || throw(ArgumentError(
        "mixed-layer spheroidal T-matrix currently supports prolate bodies only; use volume FEM"))
    body.a / body.b <= 1.25 || throw(ArgumentError(
        "mixed-layer spheroidal T-matrix currently requires aspect ratio ≤ 1.25; use volume FEM"))
    k * body.a <= 1.5 || throw(ArgumentError(
        "mixed-layer spheroidal T-matrix currently requires ka ≤ 1.5; use volume FEM"))
    0 <= m_max <= 4 && 4 <= n_max <= 8 && m_max <= n_max ||
        throw(ArgumentError("mixed-layer spheroidal T-matrix requires 0 ≤ m_max ≤ 4 and 4 ≤ n_max ≤ 8"))
    n_quad >= 2n_max + 8 ||
        throw(ArgumentError("n_quad must be at least 2n_max + 8"))
    layers, _ = _layered_materials(boundary.material)
    all(layer -> layer isa Union{FluidLayer, ElasticLayer}, layers) ||
        throw(ArgumentError("mixed-layer spheroidal T-matrix requires FluidLayer or ElasticLayer"))
    any(layer -> layer isa FluidLayer, layers) &&
    any(layer -> layer isa ElasticLayer, layers) ||
        throw(ArgumentError("mixed-layer spheroidal T-matrix requires both fluid and elastic layers"))
    all(
        layer -> !(layer isa ElasticLayer) ||
                 layer.interior_coupling === :generalized, layers) ||
        throw(ArgumentError("mixed-layer spheroidal T-matrix requires generalized elastic coupling"))
    function amplitude(order)
        transition = _memoized_transition(m -> _mixed_spheroid_transition(
            body, boundary, k, m, order; nquad = n_quad))
        return _elastic_spheroid_amplitude(transition, k, body,
            incidence_angle, incidence_azimuth, scatter_angle,
            scatter_azimuth, m_max, order)
    end
    result = amplitude(n_max)
    if check && n_max >= 6
        reduced = amplitude(n_max - 2)
        relative_change = abs(result - reduced) / max(abs(result), 1e-12)
        relative_change <= 0.03 || throw(ArgumentError(
            "mixed-layer spheroidal T-matrix changed by $(round(100relative_change; sigdigits=3))% when n_max was reduced by 2; use volume FEM"))
    end
    return result
end
