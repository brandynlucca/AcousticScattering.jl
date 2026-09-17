const _EDGE_VERTICES = (SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0))

_edge_coordinate(u, opposite) = opposite == 1 ? 1-u[1]-u[2] : u[opposite - 1]

function _edge_patches(source; enrich = false)
    owners = Dict{Tuple, Vector{Any}}()
    entries = Any[]
    for (E, tags) in source.etype2qtags
        E <: Inti.LagrangeElement{Inti.ReferenceTriangle} && 1 <= Inti.order(E) <= 3 ||
            throw(ArgumentError("edge quadrature requires linear, quadratic or cubic triangles"))
        elements = Inti.elements(source.mesh, E)
        orientations = Inti.orientation(source.mesh, E)
        rule = source.etype2qrule[E]
        for e in eachindex(elements)
            el = elements[e]
            center = el(SVector(1/3, 1/3))
            push!(entries,
                (; el, orientation = orientations[e],
                    columns = tags[:, e], refs = Inti.qcoords(rule),
                    basis = Inti.lagrange_basis(rule), center,
                    radius = maximum(norm(v-center) for v in Inti.vals(el))))
            for opposite in 1:3
                a, b = _EDGE_VERTICES[mod1(opposite+1, 3)],
                _EDGE_VERTICES[mod1(opposite+2, 3)]
                endpoints = sort([Tuple(el(a)), Tuple(el(b))])
                u = (a+b)/2
                normal = Inti._normal(Inti.jacobian(el, u), orientations[e])
                push!(get!(owners, Tuple(endpoints), Any[]),
                    (; index = length(entries), opposite, normal, inward = center-el(u)))
            end
        end
    end
    edges = Dict{Int, Tuple{Int, Float64}}()
    for pair in values(owners)
        length(pair) == 2 ||
            throw(ArgumentError("edge quadrature requires a closed conforming surface"))
        a, b = pair
        cosine = clamp(dot(a.normal, b.normal), -1.0, 1.0)
        cosine < cos(pi/6) || continue
        convex = dot(a.normal, b.inward)+dot(b.normal, a.inward) < 0
        angle = pi+(convex ? 1 : -1)*acos(cosine)
        exponent = pi/angle-1
        for face in pair
            haskey(edges, face.index) &&
                throw(ArgumentError("refine triangles meeting more than one sharp edge"))
            edges[face.index] = (face.opposite, exponent)
        end
    end
    return map(eachindex(entries)) do i
        opposite, exponent = get(edges, i, (3, 0.0))
        entry = entries[i]
        scales = [_edge_coordinate(u, opposite)^(-exponent) for u in entry.refs]
        patch = merge(entry, (; opposite, exponent, scales))
        enrich && !iszero(exponent) ? _edge_enriched_basis(patch) : patch
    end
end

function _edge_enriched_basis(patch)
    degree = round(Int, sqrt(2length(patch.refs)+0.25)-1.5)
    (degree+1)*(degree+2) == 2length(patch.refs) ||
        throw(ArgumentError("edge interpolation requires a triangular polynomial node count"))
    candidates = sort(vec([i*(patch.exponent+1)+j for i in 0:degree, j in 0:degree]))
    exponents = Float64[]
    for power in candidates
        (isempty(exponents) ||
         !isapprox(power, last(exponents); rtol = 1e-6, atol = 1e-8)) &&
            push!(exponents, power)
    end
    a, b = _EDGE_VERTICES[mod1(patch.opposite+1, 3)],
    _EDGE_VERTICES[mod1(patch.opposite+2, 3)]
    powers = u -> begin
        x = max(0.0, _edge_coordinate(u, patch.opposite))
        y = dot(u-a, b-a)/sum(abs2, b-a)
        [x^exponents[i + 1]*y^j for i in 0:degree for j in 0:(degree - i)]
    end
    matrix = reduce(vcat, transpose.(powers.(patch.refs)))
    cond(matrix) < 1e12 ||
        throw(ArgumentError("edge interpolation is ill-conditioned; refine the mesh"))
    inverse = inv(matrix)
    return merge(patch, (; basis = u -> transpose(inverse)*powers(u)))
end

function _edge_power_difference(x, a, b)
    iszero(x) && return iszero(a) ? -1/(b-a) : 0.0
    return x^a*expm1((b-a)*log(x))/(b-a)
end

function _edge_sample(patch, uv)
    (; el, opposite, exponent, scales, basis) = patch
    a, b = _EDGE_VERTICES[mod1(opposite+1, 3)], _EDGE_VERTICES[mod1(opposite+2, 3)]
    c = _EDGE_VERTICES[opposite]
    t, s = uv
    power = get(patch, :power, 3)
    u = (1-t^power)*((1-s)*a+s*b)+t^power*c
    weight = power*t^(power*(1+exponent)-1)*(1-t^power)*Inti._integration_measure(Inti.jacobian(el, u))
    return el(u), weight*basis(u) .* scales
end

function _edge_integral(f, options)
    domain = Inti.HAdaptiveIntegration.Rectangle((0.0, 0.0), (1.0, 1.0))
    value, error = Inti.HAdaptiveIntegration.integrate(f, domain; options...)
    error <= max(options.atol, options.rtol*norm(value, Inf)) ||
        throw(ArgumentError("edge quadrature did not converge; refine the surface mesh"))
    return value
end

function _edge_self_single_layer(k, patch, anchor, options, normal = nothing)
    (; el, opposite, exponent, scales, basis) = patch
    J0 = Inti.jacobian(el, anchor)
    power = get(patch, :power, 3)
    value = zeros(ComplexF64, length(scales))
    for side in 1:3
        a, b = _EDGE_VERTICES[side]-anchor, _EDGE_VERTICES[mod1(side+1, 3)]-anchor
        area = abs(a[1]*b[2]-a[2]*b[1])
        value += _edge_integral(options) do uv
            v, w = uv
            s = 1-(1-v)^power
            t = w^power/(w^power+(1-w)^power)
            measure = power^2*(1-v)^(power-1)*w^(power-1)*(1-w)^(power-1)/(w^power+(1-w)^power)^2
            d = (1-t)*a+t*b
            Jh, Je = Inti.jacobian(el, anchor+d/2), Inti.jacobian(el, anchor+d)
            J1, J2 = 4Jh-3J0-Je, 4(Je-2Jh+J0)
            jac = J0+s*J1+s^2*J2/2
            distance = norm((J0+s*J1/2+s^2*J2/6)*d)
            edge_distance = (1-v)^power*_edge_coordinate(anchor, opposite) +
                            s*((1-t)*_edge_coordinate(_EDGE_VERTICES[side], opposite) +
                               t*_edge_coordinate(_EDGE_VERTICES[mod1(side+1, 3)], opposite))
            weight = area*measure*Inti._integration_measure(jac)*edge_distance^exponent
            kernel = normal === nothing ? cis(k*s*distance)/(4pi*distance) :
                     normal === :source ?
                     cis(k*s*distance) * (1-im*k*s*distance) *
                     dot(Inti._normal(jac, patch.orientation), (J1/2+s*J2/3)*d)/(4pi*distance^3) :
                     cis(k*s*distance) * (1-im*k*s*distance) *
                     dot(normal, (J1/2+s*J2/6)*d)/(4pi*distance^3)
            weight*kernel*basis(anchor+s*d) .* scales
        end
    end
    return value
end

function _edge_nearest_reference(patch, x)
    u = SVector(1/3, 1/3)
    for _ in 1:10
        step = Inti.jacobian(patch.el, u) \ (patch.el(u)-x)
        u -= step
        norm(step) < 1e-12 && break
    end
    candidates = all(>=(0), (u[1], u[2], 1-sum(u))) ? [u] : SVector{2, Float64}[]
    for side in 1:3
        a, b = _EDGE_VERTICES[side], _EDGE_VERTICES[mod1(side+1, 3)]
        t = 0.5
        for _ in 1:10
            v = a+t*(b-a)
            tangent = Inti.jacobian(patch.el, v)*(b-a)
            t = clamp(t-dot(patch.el(v)-x, tangent)/sum(abs2, tangent), 0.0, 1.0)
        end
        push!(candidates, a+t*(b-a))
    end
    return candidates[argmin(norm(patch.el(u)-x) for u in candidates)]
end

function _edge_regular_single_layer(k, patch, x, options, normal = nothing)
    return _edge_integral(options) do uv
        y, weights = _edge_sample(patch, uv)
        r = norm(x-y)
        kernel = normal === :source ?
                 (1-im*k*r)*cis(k*r)*dot(_edge_sample_normal(patch, uv), x-y)/(4pi*r^3) :
                 normal === nothing ? cis(k*r)/(4pi*r) :
                 (im*k*r-1)*cis(k*r)*dot(normal, x-y)/(4pi*r^3)
        kernel*weights
    end
end

"""
    _edge_near_single_layer(k, patch, x, options; double_layer=false)

Integrate a neighboring panel by splitting about its nearest reference point.
Panels separated by more than one fifth of their bounding radius use the regular rule.
The angular map resolves stretched triangles in the local surface metric; the
hyperbolic radial map resolves the target separation. Endpoint grading retains
integrable corner-flux singularities. Complementary coordinates are evaluated
separately to avoid cancellation at the rim.
"""
function _edge_near_single_layer(k, patch, x, options; double_layer = false)
    anchor = _edge_nearest_reference(patch, x)
    J0 = Inti.jacobian(patch.el, anchor)
    offset = x-patch.el(anchor)
    norm(offset) > 0.2patch.radius &&
        return _edge_regular_single_layer(
            k, patch, x, options, double_layer ? :source : nothing)
    power = get(patch, :power, 3)
    value = zeros(ComplexF64, length(patch.refs))
    for side in 1:3
        va, vb = _EDGE_VERTICES[side], _EDGE_VERTICES[mod1(side+1, 3)]
        a, b = va-anchor, vb-anchor
        area = side == 2 ? 1-sum(anchor) : anchor[side == 1 ? 2 : 1]
        iszero(area) && continue
        tangent, delta = J0*a, J0*(b-a)
        center = -dot(tangent, delta)/sum(abs2, delta)
        height = norm(tangent+center*delta)/norm(delta)
        lo, hi = atan(-center/height), atan((1-center)/height)
        value += _edge_integral(options) do uv
            v, w = uv
            radial_denominator = v^power+(1-v)^power
            radial, radial_complement = v^power/radial_denominator,
            (1-v)^power/radial_denominator
            angular_denominator = w^power+(1-w)^power
            t, tb = w^power/angular_denominator, (1-w)^power/angular_denominator
            measure = power*w^(power-1)*(1-w)^(power-1)/angular_denominator^2
            if 0 < center < 1
                phase = lo+(hi-lo)*t
                ta = height*sin((hi-lo)*t)/(cos(phase)*cos(lo))
                tb = height*sin((hi-lo)*tb)/(cos(phase)*cos(hi))
                measure *= height*(hi-lo)/cos(phase)^2
                t = ta
            end
            d = tb*a+t*b
            scale = max(norm(offset)/norm(J0*d), eps(Float64))
            span = asinh(1/scale)
            complement = 2scale*cosh(span*(1+radial)/2)*sinh(span*radial_complement/2)
            s = radial < 0.5 ? scale*sinh(span*radial) : 1-complement
            measure *= power*v^(power-1)*(1-v)^(power-1)/radial_denominator^2*scale*span*cosh(span*radial)
            Jh, Je = Inti.jacobian(patch.el, anchor+d/2), Inti.jacobian(patch.el, anchor+d)
            J1, J2 = 4Jh-3J0-Je, 4(Je-2Jh+J0)
            jac = J0+s*J1+s^2*J2/2
            separation = offset-s*(J0+s*J1/2+s^2*J2/6)*d
            r = norm(separation)
            lambda = complement*_edge_coordinate(anchor, patch.opposite) +
                     s*(tb*_edge_coordinate(va, patch.opposite)+t*_edge_coordinate(vb, patch.opposite))
            weight = area*measure*s*Inti._integration_measure(jac)*lambda^patch.exponent
            kernel = double_layer ?
                     (1-im*k*r)*cis(k*r)*dot(Inti._normal(jac, patch.orientation), separation)/(4pi*r^3) :
                     cis(k*r)/(4pi*r)
            weight*kernel*patch.basis(anchor+s*d) .* patch.scales
        end
    end
    return value
end

function _edge_correct_single_layer!(
        S, k, target, source, patch, options, derivative; double_layer = false)
    weighted = !iszero(patch.exponent) || get(patch, :correct_far, false)
    weighted && _edge_far_layer!(S, k, target, patch, derivative, double_layer)
    for i in eachindex(target)
        x = target[i].coords
        self = target === source ? findfirst(==(i), patch.columns) : nothing
        value = if self !== nothing
            normal = double_layer ? :source : derivative ? target[i].normal : nothing
            _edge_self_single_layer(k, patch, patch.refs[self], options, normal)
        elseif target === source && get(patch, :resolve_near, false) &&
               norm(x-patch.center) <= 2patch.radius && !derivative
            _edge_near_single_layer(k, patch, x, options; double_layer)
        elseif norm(x-patch.center) <= 2patch.radius
            normal = double_layer ? :source : derivative ? target[i].normal : nothing
            _edge_regular_single_layer(k, patch, x, options, normal)
        else
            continue
        end
        S[i, patch.columns] = value
    end
    return S
end

function _edge_far_layer!(S, k, target, patch, derivative, double_layer)
    rows = findall(i -> norm(target[i].coords-patch.center) > 2patch.radius, eachindex(target))
    isempty(rows) && return S
    gx, gw = Inti.GaussLegendre(get(patch, :far_order, 12))()
    refs = [SVector(t[1], s[1]) for t in gx for s in gx]
    samples = _edge_sample.(Ref(patch), refs)
    normals = double_layer ? _edge_sample_normal.(Ref(patch), refs) : nothing
    weights = ComplexF64.(reduce(vcat, transpose.(last.(samples))))
    weights .*= [a*b for a in gw for b in gw]
    for first in 1:256:length(rows)
        block = rows[first:min(first + 255, length(rows))]
        kernels = Matrix{ComplexF64}(undef, length(block), length(samples))
        for j in eachindex(samples), (i, row) in enumerate(block)

            separation = target[row].coords-samples[j][1]
            r = norm(separation)
            kernels[i, j] = double_layer ?
                            (1-im*k*r)*cis(k*r)*dot(normals[j], separation)/(4pi*r^3) :
                            derivative ?
                            (im*k*r-1)*cis(k*r)*dot(target[row].normal, separation)/(4pi*r^3) :
                            cis(k*r)/(4pi*r)
        end
        S[block, patch.columns] = kernels*weights
    end
    return S
end

"""
    _edge_single_layer(op, target, source, correction)

Single-layer quadrature for a surface with sharp edges. On triangles
adjoining a normal jump greater than 30 degrees, interpolate the density after factoring
out `lambda^(pi/exterior_angle-1)`, where `lambda` vanishes at that edge. Each triangle
may adjoin at most one such edge. Smooth triangles retain polynomial interpolation.

Adaptive self and neighboring-element integrals use graded coordinate maps. The self
map cancels the Green-function singularity without subtracting nearby physical points;
its polynomial geometry expansion is exact through cubic elements. The radial singularity
mapping follows Duffy (1982), doi:10.1137/0719090. Error estimates are checked explicitly.
With `derivative=true`, evaluate its target-normal derivative. `enrich=true` includes
successive corner and integer powers in the density interpolation for the rigid equation.
With `double_layer=true`, use the source-normal derivative instead. Supplied `patches`
can define material-dependent interpolation and grading for fluid transmission.
"""
function _edge_single_layer(
        op, target, source, correction; derivative = false, enrich = false,
        double_layer = false, patches = _edge_patches(source; enrich))
    options = _edge_options(correction)
    kernel = double_layer ? Inti.DoubleLayerKernel(op) :
             derivative ? Inti.AdjointDoubleLayerKernel(op) : Inti.SingleLayerKernel(op)
    S = Matrix(Inti.IntegralOperator(kernel, target, source))
    Threads.@threads for i in eachindex(patches)
        _edge_correct_single_layer!(
            S, op.k, target, source, patches[i], options, derivative; double_layer)
    end
    return S
end

function _edge_options(correction)
    options = (; rtol = get(correction, :rtol, 1e-7), atol = get(correction, :atol, 1e-10),
        maxsubdiv = get(correction, :maxsubdiv, 16384))
    all(x -> x isa Real && isfinite(x) && x > 0, (options.rtol, options.atol)) ||
        throw(ArgumentError("edge quadrature tolerances must be finite and positive"))
    options.maxsubdiv isa Integer && options.maxsubdiv > 0 ||
        throw(ArgumentError("edge quadrature maxsubdiv must be a positive integer"))
    return options
end

function _edge_fluid_operators(op, quad, correction, pressure, flux)
    S = _edge_single_layer(op, quad, quad, correction; patches = flux)
    D = _edge_single_layer(
        op, quad, quad, correction; patches = pressure, double_layer = true)
    return S, D
end

function _uses_edge_quadrature(data)
    get(get(data.diagnostics, :correction, (;)), :method, :dim) === :edge
end

function _edge_total_flux(data, k)
    direction = _bem3d_incidence_direction(data.incidence_angle, data.incidence_azimuth)
    return data.dpdn_scat + [im*k*dot(direction, q.normal)*cis(k*dot(direction, q.coords))
            for q in data.quad]
end

"""
Interpolate fluid pressure and flux using the scalar transmission corner exponents.
The characteristic equation is the scalar Laplace/Helmholtz transmission relation in
Schmidt (2009), Integral methods for conical diffraction, equation (6.9).
The acoustic coefficient ratio is the inverse density ratio; the exponent set is
unchanged by inversion. Wavenumber affects higher terms, not the leading edge powers.

Factor out the leading flux singularity and combine successive radial corner powers
and their integer shifts with tangential polynomials. Divided differences of nearby
radial powers avoid cancellation when the material contrast is weak.
"""
function _edge_fluid_patches(source, contrast)
    pairs = map(_edge_patches(source)) do patch
        patch = merge(patch, (; resolve_near = true))
        count = length(patch.refs)
        regular = merge(patch, (; exponent = 0.0, scales = ones(SVector{count, Float64})))
        (iszero(patch.exponent) || contrast == 1) && return (regular, regular)
        angle = pi/(patch.exponent+1)
        ratio = abs((contrast-1)/(contrast+1))
        degree = round(Int, sqrt(2length(patch.refs)+0.25)-1.5)
        roots = Float64[]
        for j in 1:(degree + 1), sign in (-1, 1)

            lo, hi = j-0.5, j+0.5
            f = t -> sinpi(t)+sign*ratio*sin((angle-pi)*t)
            for _ in 1:50
                mid = (lo+hi)/2
                if signbit(f(mid)) == signbit(f(lo))
                    lo = mid
                else
                    hi = mid
                end
            end
            root = (lo+hi)/2
            any(x -> isapprox(x, root; rtol = 1e-8), roots) || push!(roots, root)
        end
        sort!(roots)
        exponent = first(roots)-1
        scales = map(u -> _edge_coordinate(u, patch.opposite)^(-exponent), patch.refs)
        pressure = _edge_fluid_basis(
            merge(regular, (;
                correct_far = true, power = 8, far_order = 24)), roots)
        flux = _edge_fluid_basis(
            merge(patch, (;
                exponent, scales, power = 8, far_order = 24)), roots .- 1)
        (pressure, flux)
    end
    return first.(pairs), last.(pairs)
end

function _edge_fluid_basis(patch, roots)
    count = length(patch.refs)
    degree = round(Int, sqrt(2count+0.25)-1.5)
    (degree+1)*(degree+2) == 2count ||
        throw(ArgumentError("fluid edge interpolation requires a triangular polynomial node count"))
    integers = iszero(patch.exponent) ? [0; collect(2:degree)] : collect(1:degree)
    candidates = sort([Float64.(integers); [root+j for root in roots for j in 0:degree]])
    exponents = Float64[]
    for power in candidates
        (isempty(exponents) ||
         !isapprox(power, last(exponents); rtol = 1e-8, atol = 1e-10)) &&
            push!(exponents, power)
    end
    radial_powers = Tuple(exponents[1:(degree + 1)] .- patch.exponent)
    intervals = Tuple((i == 1 ? -Inf : radial_powers[i - 1], radial_powers[i])
    for i in eachindex(radial_powers))
    terms = Tuple((i+1, j) for i in 0:degree for j in 0:(degree - i))
    coordinates = _edge_fluid_coordinates(patch)
    powers = u -> begin
        lambda, factor, y = coordinates(u)
        x = lambda*factor
        radial = map(intervals) do (lo, hi)
            hi-lo < 0.1 ? _edge_power_difference(x, lo, hi) : x^hi
        end
        factor^patch.exponent*SVector(map(term -> radial[first(term)]*y^last(term), terms))
    end
    matrix = reduce(vcat, transpose.(powers.(patch.refs)))
    cond(matrix) < 1e12 ||
        throw(ArgumentError("fluid edge interpolation is ill-conditioned; change the quadrature order"))
    inverse = SMatrix{count, count}(inv(matrix))
    return merge(patch, (; basis = u -> transpose(inverse)*powers(u)))
end

function _edge_cubic_coefficients(values)
    a, d = first(values), last(values)
    b = 3values[2]-1.5values[3]-5a/6+d/3
    c = 3values[3]-1.5values[2]+a/3-5d/6
    return (a, 3(b-a), 3(a-2b+c), -a+3b-3c+d)
end

_edge_polynomial(c, t) = c[1]+t*(c[2]+t*(c[3]+t*c[4]))

"""
Local physical coordinates about a curved edge. Match the tangential projection
to a point on the edge, then measure the remaining displacement. Polynomial divided
differences retain the limiting radial scale without subtracting nearly equal points.
The geometry expansion is exact for linear, quadratic and cubic triangles.
"""
function _edge_fluid_coordinates(patch)
    opposite = patch.opposite
    a, b = _EDGE_VERTICES[mod1(opposite+1, 3)], _EDGE_VERTICES[mod1(opposite+2, 3)]
    c = _EDGE_VERTICES[opposite]
    samples = (0.0, 1/3, 2/3, 1.0)
    origin = patch.el(a)
    curve = _edge_cubic_coefficients(map(t -> patch.el((1-t)*a+t*b)-origin, samples))
    derivatives = map(samples) do t
        anchor = (1-t)*a+t*b
        direction = c-anchor
        J0 = Inti.jacobian(patch.el, anchor)
        Jh = Inti.jacobian(patch.el, anchor+direction/2)
        Je = Inti.jacobian(patch.el, c)
        (J0*direction, (4Jh-3J0-Je)*direction/2, 2(Je-2Jh+J0)*direction/3)
    end
    transverse = ntuple(i -> _edge_cubic_coefficients(map(v -> v[i], derivatives)), 3)
    edge_direction = curve[2]+curve[3]+3curve[4]/4
    tangent = edge_direction/norm(edge_direction)
    tangent_scale = norm(patch.el(b)-origin)/2
    height = patch.el(c)-origin-_edge_polynomial(curve, 0.5)
    radial_scale = norm(height-dot(height, tangent)*tangent)
    radial_scale > 0 ||
        throw(ArgumentError("edge coordinates require a nondegenerate triangle"))
    return u -> begin
        lambda = clamp(_edge_coordinate(u, opposite), 0.0, 1.0)
        s = lambda < 1 ?
            clamp(_edge_coordinate(u, mod1(opposite+2, 3))/(1-lambda), 0.0, 1.0) : 0.5
        w = _edge_polynomial(transverse[1], s) + lambda*_edge_polynomial(transverse[2], s) +
            lambda^2*_edge_polynomial(transverse[3], s)
        e1, e2, e3 = curve[2]+2s*curve[3]+3s^2*curve[4], curve[3]+3s*curve[4], curve[4]
        c1, c2, c3 = dot(e1, tangent), dot(e2, tangent), dot(e3, tangent)
        along = dot(w, tangent)
        offset = along/c1
        for _ in 1:8
            t = lambda*offset
            step = (offset*(c1+t*(c2+t*c3))-along)/(c1+t*(2c2+3t*c3))
            offset -= step
            abs(step) <= 8eps(Float64)*(1+abs(offset)) && break
        end
        t = lambda*offset
        abs(offset*(c1+t*(c2+t*c3))-along) <= 1e-12*max(abs(c1), abs(along)) ||
            throw(ArgumentError("edge coordinates did not converge; refine the surface mesh"))
        factor = norm(w-offset*(e1+t*(e2+t*e3)))/radial_scale
        factor > 0 ||
            throw(ArgumentError("edge coordinates require a regular radial direction"))
        edge_offset = (s-0.5)*(curve[2]+(s+0.5)*curve[3]+(s^2+s/2+0.25)*curve[4])
        y = (dot(edge_offset, tangent)+lambda*along)/tangent_scale
        (lambda, factor, y)
    end
end

function _edge_fluid_pressure(data, k, points, regions, boundary)
    direction = _bem3d_incidence_direction(data.incidence_angle, data.incidence_azimuth)
    p = data.p_scat + [cis(k*dot(direction, q.coords)) for q in data.quad]
    q = _edge_total_flux(data, k)
    pressure_patches, patches = _edge_fluid_patches(data.quad, boundary.density_contrast)
    patches = [merge(patch, (; correct_far = true)) for patch in patches]
    correction = merge(data.diagnostics.correction,
        (; rtol = get(data.diagnostics.correction, :rtol, 1e-10),
            atol = get(data.diagnostics.correction, :atol, 1e-13),
            maxsubdiv = get(data.diagnostics.correction, :maxsubdiv, 65536)))
    values = zeros(ComplexF64, length(points))
    for region in (:exterior, :interior)
        indices = findall(==(region), regions)
        isempty(indices) && continue
        inside = region === :interior
        wave = inside ? k/boundary.soundspeed_contrast : k
        for first in 1:256:length(indices)
            rows = indices[first:min(first + 255, length(indices))]
            targets = [(; coords = SVector{3, Float64}(points[i]),
                           normal = zero(SVector{3, Float64})) for i in rows]
            S = _edge_single_layer(Inti.Helmholtz(; k = wave, dim = 3), targets,
                data.quad, correction; patches)
            Dp = zeros(ComplexF64, length(rows))
            Threads.@threads for j in eachindex(rows)
                Dp[j] = _edge_double_pressure(
                    wave, SVector{3, Float64}(points[rows[j]]), pressure_patches, p, inside,
                    correction)
            end
            values[rows] = inside ? boundary.density_contrast*S*q-Dp : Dp-S*q
        end
    end
    return values
end

function _edge_pressure_anchor(patches, pressure, x)
    distance, value = Inf, 0.0im
    for patch in patches
        norm(x-patch.center)-patch.radius > distance && continue
        candidate = _edge_nearest_reference(patch, x)
        separation = norm(patch.el(candidate)-x)
        if separation < distance
            distance = separation
            value = sum(patch.basis(candidate) .* pressure[patch.columns])
        end
    end
    return value
end

function _edge_double_remainder(z)
    abs(z) >= 0.01 && return (1-im*z)*cis(z)-1
    term, value = (im*z)^2/2, zero(ComplexF64)
    for n in 2:12
        value -= (n-1)*term
        term *= im*z/(n+1)
    end
    return value
end

function _edge_double_pressure(k, x, patches, pressure, inside, correction)
    anchor = _edge_pressure_anchor(patches, pressure, x)
    options = _edge_options(correction)
    value = inside ? -anchor : zero(ComplexF64)
    for patch in patches
        gx, gw = Inti.GaussLegendre(get(patch, :far_order, 12))()
        density = pressure[patch.columns]
        shifted = density .- anchor
        integrand = uv -> begin
            y, weights = _edge_sample(patch, uv)
            r = norm(x-y)
            normal = _edge_sample_normal(patch, uv)
            difference = dot(weights, shifted) +
                         _edge_double_remainder(k*r)*dot(weights, density)
            dot(normal, x-y)/(4pi*r^3)*difference
        end
        value += norm(x-patch.center) <= 2patch.radius ?
                 _edge_integral(integrand, options) :
                 sum(gw[i]*gw[j]*integrand(SVector(gx[i][1], gx[j][1]))
        for i in eachindex(gx), j in eachindex(gx))
    end
    return value
end

function _edge_pressure(data, k, points)
    enrich = data.single_layer_density !== nothing
    density = enrich ? data.single_layer_density : -_edge_total_flux(data, k)
    values = zeros(ComplexF64, length(points))
    for first in 1:256:length(points)
        rows = first:min(first + 255, length(points))
        targets = [(; coords = SVector{3, Float64}(points[i]),
                       normal = zero(SVector{3, Float64})) for i in rows]
        S = _edge_single_layer(Inti.Helmholtz(; k, dim = 3), targets,
            data.quad, data.diagnostics.correction; enrich)
        values[rows] = S*density
    end
    return values
end

function _edge_far_field(data, k, direction)
    enrich = data.single_layer_density !== nothing
    flux = enrich ? data.single_layer_density : -_edge_total_flux(data, k)
    return _edge_far_integral(
        data.quad, k, direction, flux, _edge_patches(data.quad; enrich))
end

function _edge_fluid_far_field(data, k, direction, boundary)
    incoming = _bem3d_incidence_direction(data.incidence_angle, data.incidence_azimuth)
    p = data.p_scat + [cis(k*dot(incoming, q.coords)) for q in data.quad]
    q = _edge_total_flux(data, k)
    pressure, flux = _edge_fluid_patches(data.quad, boundary.density_contrast)
    return -im*k*_edge_far_integral(data.quad, k, direction, p, pressure; normal = true) -
           _edge_far_integral(data.quad, k, direction, q, flux)
end

function _edge_far_integral(quad, k, direction, flux, patches; normal = false)
    total = sum(q.weight * flux[i] * cis(-k*dot(direction, q.coords)) *
                (normal ? dot(direction, q.normal) : 1) for (i, q) in enumerate(quad))
    for patch in patches
        iszero(patch.exponent) && !get(patch, :correct_far, false) && continue
        gx, gw = Inti.GaussLegendre(get(patch, :far_order, 12))()
        for j in patch.columns
            q = quad[j]
            total -= q.weight * flux[j] * cis(-k*dot(direction, q.coords)) *
                     (normal ? dot(direction, q.normal) : 1)
        end
        for it in eachindex(gx), is in eachindex(gx)

            y, weights = _edge_sample(patch, SVector(gx[it][1], gx[is][1]))
            factor = normal ?
                     dot(direction, _edge_sample_normal(patch, SVector(gx[it][1], gx[is][1]))) :
                     1
            total += factor*gw[it]*gw[is]*cis(-k*dot(direction, y))*sum(weights .*
                                                                        flux[patch.columns])
        end
    end
    return total/(4pi)
end

function _edge_sample_normal(patch, uv)
    t, s = uv
    a, b = _EDGE_VERTICES[mod1(patch.opposite+1, 3)],
    _EDGE_VERTICES[mod1(patch.opposite+2, 3)]
    power = get(patch, :power, 3)
    u = (1-t^power)*((1-s)*a+s*b)+t^power*_EDGE_VERTICES[patch.opposite]
    return Inti._normal(Inti.jacobian(patch.el, u), patch.orientation)
end
