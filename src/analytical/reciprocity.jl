# Electroacoustic reciprocity received signal (Foldy and Primakoff 1945, Auld 1979), full-3D
# closed-surface MFS and full-3D BEM.

"""
    received_pressure(source::AbstractTransducer, receiver::Transducer, k;
        quadrature=(16,64), rtol=1e-6)
    received_pressure(solution, receiver::AbstractTransducer; kwargs...)
    received_pressure(field::ScatteringField, receiver::AbstractTransducer; rtol=1e-6)

Complex pressure averaged coherently over a uniform circular receiver aperture.
The source overload integrates its Rayleigh field, including source images when
`source` is a `TankTransducer`; `quadrature` gives radial Gauss and azimuthal orders.
It includes direct transmission and the explicitly supplied source-wall paths.

The solution overload returns target-scattered pressure from `received_signal`
divided by `2im*k*pi*receiver.radius^2`, the reciprocity normalization for this
package's piston field. Receiver images add outgoing target-wall paths. Its other
keywords are those of `received_signal`. Both overloads therefore share pressure
normalization and may be added coherently before time synthesis. A receiver does
not emit an additional field. These values exclude electrical gain, housing/backing
scattering and calibration to volts; refine aperture/surface quadrature separately.
`ScatteringField` uses the exact piston aperture integral of its monopole/dipole
sources, including the receiver's supplied environmental fields. A
`ScatteringTransducer` on either target leg adds the explicitly prepared line/wall
response; it does not iterate a closed target-environment feedback system.
"""
function received_pressure(source::AbstractTransducer, receiver::Transducer, k::Real;
        quadrature = (16, 64), rtol::Real = 1e-6)
    length(quadrature)==2 && all(n->n isa Integer && n>0, quadrature) ||
        throw(ArgumentError("quadrature must contain two positive integer orders"))
    radial, angular=quadrature
    nodes, weights=gauss(radial, 0.0, 1.0)
    rotation=_axis_rotation(collect(receiver.axis))
    field=_transducer_field(source, k; rtol)[1]
    total=0.0im
    for (u, w) in zip(nodes, weights), j in 0:(angular - 1)

        phi=2pi*(j+0.5)/angular
        rho=receiver.radius*sqrt(u)
        point=ntuple(d->receiver.position[d]+rho*(rotation[d, 1]*cos(phi)+rotation[d, 2]*sin(phi)), 3)
        total+=(w/angular)*field(point)
    end
    return total
end

function received_pressure(
        sol::Union{MFSSolution{_FullMFSSurfaceData},
            BEMSolution{_FullBEMSurfaceData}, BEMSolution{_RegionBEMData}},
        receiver::AbstractTransducer; kwargs...)
    aperture=_transducer_aperture(receiver)
    return received_signal(sol, receiver; kwargs...)/(2im*sol.k*pi*aperture.radius^2)
end

# A piston radiates -2im*k times its aperture integral of G. Apply the same
# identity to monopoles and source-coordinate dipoles without receiver meshing.
function received_pressure(field::ScatteringField, receiver::AbstractTransducer;
        rtol::Real = 1e-6)
    aperture=_transducer_aperture(receiver)
    p, g=_transducer_field(receiver, field.k; rtol)
    totals=zeros(ComplexF64, length(field.points))
    Threads.@threads for j in eachindex(field.points)
        x=field.points[j]
        totals[j]=field.monopoles[j]*p(x) +
                  sum(field.dipoles[j] .* g(x))
    end
    sum(totals)/(-2im*field.k*pi*aperture.radius^2)
end

function _solution_scattering_field(
        sol::Union{MFSSolution{_FullMFSSurfaceData},
            BEMSolution{_FullBEMSurfaceData}, BEMSolution{_RegionBEMData}},
        k)
    k==sol.k || throw(ArgumentError("solution and scattering frequencies differ"))
    points=SVector{3, Float64}[]
    mono=ComplexF64[]
    dip=SVector{3, ComplexF64}[]
    for quad in _reciprocity_quadratures(sol, nothing)
        p, dp=_reciprocity_traces(sol, quad)
        for (i, q) in enumerate(quad)
            push!(points, q.coords)
            push!(mono, -q.weight*dp[i])
            push!(dip, q.weight*p[i]*q.normal)
        end
    end
    ScatteringField(Float64(k), points, mono, dip, nothing)
end
function _scattering_incident(
        sol::Union{MFSSolution{_FullMFSSurfaceData},
            BEMSolution{_FullBEMSurfaceData}, BEMSolution{_RegionBEMData}},
        k)
    IncidentField(_solution_scattering_field(sol, k))
end
function _scattering_pair(
        sol::Union{MFSSolution{_FullMFSSurfaceData},
            BEMSolution{_FullBEMSurfaceData}, BEMSolution{_RegionBEMData}},
        k)
    field=_solution_scattering_field(sol, k)
    x->_scattering_value(field, x, true, true)
end
_field_excluded(::Nothing, x) = false

"""
    received_signal(sol, receiver::AbstractTransducer; surface=nothing, rtol=1e-6)

Reciprocity received signal for `receiver`, given a scattering solution `sol` obtained under a
transmitting [`Transducer`](@ref) or [`TankTransducer`](@ref) (`mfs(...; transducer=...)`).
Supports full-3D closed-surface MFS, single-interface BEM and coupled-fluid BEM.
Integrates `p_rx*∂p_s/∂n - p_s*∂p_rx/∂n` over the target's exterior surface(s).
Pass `surface=mesh(...)` to use a separate enclosing surface; both meshes need
quadrature refinement for surface-independence checks. The receiver aperture and
all nonzero receiver images must lie outside the integration surface. Alternative
surfaces are not supported for edge-weighted BEM.
`p_s`/`∂p_s/∂n` are
the solved scattered field and its normal derivative there, `p_rx`/`∂p_rx/∂n` are `receiver`'s
own free-space radiated field, evaluated as if it were transmitting. Independent of the choice
of enclosing surface (any closed surface separating the target from both transducers gives the
same value), and reduces to `-4π*C_rx*(exp(ikR)/R)*f(x̂_rx)` as the receiver recedes to distance `R`
in direction `x̂_rx`, `f` the bistatic amplitude toward the receiver and `C_rx` the receiver's
own on-axis far-field radiation factor (checked directly in the module tests).
The result is the complex acoustic reciprocity integral, not voltage: electroacoustic
gain and units belong to the separately supplied transfer functions. No complex
conjugation is applied to the receiving field. `TankTransducer` includes first-order
receiver images; the target solve must independently use the desired transmitter images.
"""
function received_signal(
        sol::Union{MFSSolution{_FullMFSSurfaceData},
            BEMSolution{_FullBEMSurfaceData}, BEMSolution{_RegionBEMData}},
        receiver::AbstractTransducer; surface = nothing, rtol::Real = 1e-6)
    isfinite(rtol) && rtol > 0 || throw(ArgumentError("rtol must be finite and positive"))
    quads = _reciprocity_quadratures(sol, surface)
    pinc_rx, gradinc_rx = _transducer_field(receiver, sol.k; rtol)
    total = zero(ComplexF64)
    for quad in quads
        _check_receiver_exclusion(quad, receiver, sol.k)
        p, dp = _reciprocity_traces(sol, quad)
        if receiver isa ScatteringTransducer
            pair=_scattering_pair(receiver, sol.k; rtol)
            terms=zeros(ComplexF64, length(quad))
            Threads.@threads for i in eachindex(terms)
                q=quad[i]
                pr, gr=pair(q.coords)
                terms[i]=q.weight*(pr*dp[i]-p[i]*_dot3(gr, q.normal))
            end
            total+=sum(terms)
        else
            for (i, q) in enumerate(quad)
                total += q.weight*(pinc_rx(q.coords)*dp[i] -
                                   p[i]*_dot3(gradinc_rx(q.coords), q.normal))
            end
        end
    end
    return total
end

function _reciprocity_surfaces(sol::Union{
        MFSSolution{_FullMFSSurfaceData}, BEMSolution{_FullBEMSurfaceData}})
    [Mesh(sol.data.quad, sol.body, :full, 0.0)]
end
function _reciprocity_surfaces(sol::BEMSolution{_RegionBEMData})
    [part.surface for part in sol.data.interfaces if part.exterior == 0]
end

function _reciprocity_quadratures(sol, surface)
    targets = _reciprocity_surfaces(sol)
    surface === nothing && return [target.data for target in targets]
    surface isa Mesh{<:Inti.Quadrature} && surface.method === :full ||
        throw(ArgumentError("surface must be a closed full-3D mesh"))
    length(targets)==1 && surface.data===only(targets).data && return [surface.data]
    _validate_regions([surface; targets], [0; ones(Int, length(targets))])
    return [surface.data]
end

function _check_receiver_exclusion(quad, receiver::Transducer, k)
    _region_contains(quad, SVector(receiver.position)) &&
        throw(ArgumentError("the receiver must lie outside the integration surface"))
    for patch in _region_patches(quad, 0)
        distance2 = sum(1:3) do d
            lo = minimum(p[d].lo for p in patch.net)
            hi = maximum(p[d].hi for p in patch.net)
            max(lo-receiver.position[d], 0, receiver.position[d]-hi)^2
        end
        distance2 > receiver.radius^2 || throw(ArgumentError(
            "receiver aperture is not separated from the integration surface; refine the surface or increase separation"))
    end
    return nothing
end
function _check_receiver_exclusion(quad, receiver::TankTransducer, k)
    _check_receiver_exclusion(quad, receiver.source, k)
    for wall in receiver.walls
        iszero(_wall_reflection(wall, receiver.source, k, receiver.reference_point)) ||
            _check_receiver_exclusion(quad, _image(wall, receiver.source), k)
    end
    return nothing
end

function _check_receiver_exclusion(quad, receiver::ScatteringTransducer, k)
    _check_receiver_exclusion(quad, receiver.source, k)
    patches=_region_patches(quad, 0)
    bounds=ntuple(
        d->(minimum(p[d].lo for patch in patches for p in patch.net),
            maximum(p[d].hi for patch in patches for p in patch.net)),
        3)
    for field in receiver.fields
        field.k==k || throw(ArgumentError("receiver field frequency does not match"))
        for x in field.points
            all(d->bounds[d][1]<=x[d]<=bounds[d][2], 1:3) || continue
            _surface_location(patches, x)===:outside || throw(ArgumentError(
                "receiver scattering sources must be outside the integration surface"))
        end
    end
    nothing
end

function _reciprocity_traces(sol::MFSSolution{_FullMFSSurfaceData}, quad)
    data = sol.data
    k = sol.k
    pressure, derivative = zeros(ComplexF64, length(quad)), zeros(ComplexF64, length(quad))
    for (i, q) in enumerate(quad)
        x = Tuple(q.coords)
        n = Tuple(q.normal)
        p_s = sum(c * _green3d(k, x, y) for (c, y) in zip(data.coefficients, data.sources))
        dpdn_s = sum(c * _dgreen3d_dn(k, x, n, y)
        for (c, y) in zip(data.coefficients, data.sources))
        pressure[i], derivative[i] = p_s, dpdn_s
    end
    return pressure, derivative
end

function _reciprocity_traces(sol::BEMSolution{_FullBEMSurfaceData}, quad)
    data = sol.data
    quad === data.quad && return data.p_scat, data.dpdn_scat
    _uses_edge_quadrature(data) && throw(ArgumentError(
        "an alternative reciprocity surface is not supported for edge-weighted BEM"))
    return _layer_surface_traces(data.quad, data.p_scat, data.dpdn_scat, sol.k, quad)
end

function _reciprocity_traces(sol::BEMSolution{_RegionBEMData}, quad)
    p, dp = zeros(ComplexF64, length(quad)), zeros(ComplexF64, length(quad))
    for part in sol.data.interfaces
        part.exterior == 0 || continue
        source = part.surface.data
        pi, dpi = _incident_traces(source, sol.k, sol.data.incidence_angle,
            sol.data.incidence_azimuth, sol.data.incident)
        ps, dps = part.pressure-pi, part.normal_derivative_exterior-dpi
        if source === quad
            return ps, dps
        end
        values, derivatives = _layer_surface_traces(source, ps, dps, sol.k, quad)
        p .+= values
        dp .+= derivatives
    end
    return p, dp
end

# Differentiate the exterior representation D*p-S*dp analytically. This regular
# quadrature is for separated enclosing surfaces; refine both surfaces to check it.
function _layer_surface_traces(source, p, dp, k, target)
    values, derivatives = zeros(ComplexF64, length(target)),
    zeros(ComplexF64, length(target))
    for (i, t) in enumerate(target), (j, s) in enumerate(source)

        separation = t.coords-s.coords
        r = norm(separation)
        r > 0 || throw(ArgumentError("integration and target surfaces must be separated"))
        u = separation/r
        G = cis(k*r)/(4pi*r)
        a = im*k-1/r
        b = -k^2-3im*k/r+3/r^2
        un = dot(u, s.normal)
        values[i] += s.weight*G*(-a*un*p[j]-dp[j])
        derivatives[i] += s.weight*G*(-(b*un*dot(t.normal, u) +
                                        (a/r)*dot(t.normal, s.normal))*p[j]-a*dot(t.normal, u)*dp[j])
    end
    return values, derivatives
end
