# Separated-source quadrature for thin fluid-equivalent filaments and reciprocal
# Kirchhoff reflection from a cylindrical sidewall. No cavity or contact solve.

"""
    ThinFilament(vertices, radius; density_ratio, sound_speed_ratio)

Piecewise straight fluid-equivalent monofilament, coordinates/radius in meters.
The small-radius model retains local monopole and anisotropic dipole response,
and phase along the complete line. Density and sound speed are relative to water.
It neglects elastic shear, tension, end corrections, knots and line self-scattering.
Use only when both exterior and interior `k*radius <= 0.3`, segments are slender,
and sources/observers are separated from the filament. This is not a calibrated
elastic nylon model. Break the line into separate objects around a target;
penetrating a target or resolving its attachment requires a coupled mesh model.
"""
struct ThinFilament
    vertices::Vector{SVector{3, Float64}}
    radius::Float64
    density_ratio::Float64
    sound_speed_ratio::Float64
    function ThinFilament(vertices, radius; density_ratio, sound_speed_ratio)
        length(vertices)>=2 ||
            throw(ArgumentError("a filament needs at least two vertices"))
        all(p->length(p)==3 && all(isfinite, p), vertices) ||
            throw(ArgumentError("vertices must be finite three-vectors"))
        all(x->x isa Real && isfinite(x) && x>0,
            (radius, density_ratio, sound_speed_ratio)) ||
            throw(ArgumentError("radius and material ratios must be finite and positive"))
        points=[SVector{3, Float64}(p) for p in vertices]
        all(i->norm(points[i + 1]-points[i])>2radius, 1:(length(points) - 1)) ||
            throw(ArgumentError("each filament segment must be longer than its diameter"))
        new(points, radius, density_ratio, sound_speed_ratio)
    end
end

"""
    ScatteringField

Fixed-frequency outgoing monopole/dipole quadrature produced by
[`filament_scattering`](@ref) or [`sidewall_scattering`](@ref). Evaluate with
`pressure(field, points)` or obtain point pressure/gradient through
`IncidentField(field)`. The Green function is `exp(im*k*r)/(4pi*r)`; the field
uses full spherical propagation at every distance. Surface/filament interiors
are excluded. Refine quadrature and observer separation independently.
"""
struct ScatteringField{G}
    k::Float64
    points::Vector{SVector{3, Float64}}
    monopoles::Vector{ComplexF64}
    dipoles::Vector{SVector{3, ComplexF64}}
    geometry::G
end

function _filament_distance(line::ThinFilament, x)
    minimum(1:(length(line.vertices) - 1)) do i
        a, b=line.vertices[i], line.vertices[i + 1]
        v=b-a
        norm(x-a-clamp(dot(x-a, v)/dot(v, v), 0.0, 1.0)*v)
    end
end
_field_excluded(line::ThinFilament, x) = _filament_distance(line, x)<=line.radius
function _field_excluded(tank::Tank, x)
    cx, cy=sum(tank.bounds.x)/2, sum(tank.bounds.y)/2
    r=hypot(x[1]-cx, x[2]-cy)
    abs(r-tank.radius)<=64eps(max(tank.radius, norm(x))) &&
        tank.bounds.z[1]<=x[3]<=tank.bounds.z[2]
end

function _scattering_value(f::ScatteringField, point, gradient::Bool = false, both::Bool = false)
    x=SVector{3}(point)
    all(isfinite, x) || throw(ArgumentError("points must be finite"))
    _field_excluded(f.geometry, x) && throw(ArgumentError(
        "scattering quadrature requires points outside the filament or off the sidewall"))
    value=0.0im
    grad=zero(SVector{3, ComplexF64})
    @inbounds for j in eachindex(f.points)
        v=x-f.points[j]
        r=norm(v)
        r>0 || throw(ArgumentError("observer coincides with a scattering source"))
        u=v/r
        G=cis(f.k*r)/(4pi*r)
        a=im*f.k-1/r
        # dipoles multiply the source-coordinate gradient of G, -G*a*u.
        ud=sum(u .* f.dipoles[j])
        value+=G*(f.monopoles[j]-a*ud)
        if gradient
            b=-f.k^2-3im*f.k/r+3/r^2
            grad+=G*(a*u*f.monopoles[j]-b*u*ud-(a/r)*f.dipoles[j])
        end
    end
    both ? (value, grad) : gradient ? grad : value
end

function IncidentField(f::ScatteringField)
    IncidentField(
        x->_scattering_value(f, x), x->_scattering_value(f, x, true))
end

function pressure(f::ScatteringField, points)
    if points isa Tuple{Vararg{Real, 3}} || points isa AbstractVector{<:Real}
        return _scattering_value(f, points)
    elseif points isa AbstractMatrix{<:Real}
        size(points, 1)==3 || throw(ArgumentError("point matrices need three rows"))
        return [_scattering_value(f, view(points, :, i)) for i in axes(points, 2)]
    end
    values=Array{ComplexF64}(undef, size(points))
    Threads.@threads for i in eachindex(points)
        values[i]=_scattering_value(f, points[i])
    end
    values
end

function _scattering_incident(incident::_PointIncidentField, k)
    incident
end
function _scattering_incident(source::AbstractTransducer, k)
    p, g=_transducer_field(source, k)
    IncidentField(p, g)
end
function _scattering_incident(f::ScatteringField, k)
    k==f.k || throw(ArgumentError("scattering field frequency does not match"))
    IncidentField(f)
end

function _scattering_pair(incident, k)
    field=_scattering_incident(incident, k)
    x->(field.pressure(x), SVector{3, ComplexF64}(field.gradient(x)))
end
function _scattering_pair(field::ScatteringField, k)
    field.k==k || throw(ArgumentError("scattering field frequency does not match"))
    x->_scattering_value(field, x, true, true)
end
function _scattering_pair(source::AbstractTransducer, k; kwargs...)
    p, g=_transducer_field(source, k; kwargs...)
    x->(p(x), SVector{3, ComplexF64}(g(x)))
end

"""
    filament_scattering(line::ThinFilament, k, incident;
        panels_per_wavelength=4, order=4)

Leading small-radius scattering driven by a transducer, `IncidentField`,
`ScatteringField`, or supported full-3D boundary solution's scattered field.
Integrates `k^2*(1/(g*h^2)-1)*G*p + grad_y(G) dot A*grad(p)` over line volume,
where `A` has transverse eigenvalue `2*(g-1)/(g+1)` and axial eigenvalue
`1-1/g`. These are fluid-cylinder quasistatic polarizabilities, not a row of
independent spheres. No conjugation is used. Sources must be external to the line.
"""
function filament_scattering(line::ThinFilament, k::Real, incident;
        panels_per_wavelength::Real = 4, order::Integer = 4)
    isfinite(k) && k>0 || throw(ArgumentError("k must be finite and positive"))
    max(k*line.radius, k*line.radius/line.sound_speed_ratio)<=0.3 ||
        throw(ArgumentError("filament model requires exterior and interior k*radius <= 0.3"))
    isfinite(panels_per_wavelength) && panels_per_wavelength>0 && order>0 ||
        throw(ArgumentError("quadrature controls must be positive"))
    pair=_scattering_pair(incident, k)
    nodes, weights=gauss(order, 0.0, 1.0)
    points=SVector{3, Float64}[]
    mono=ComplexF64[]
    dip=SVector{3, ComplexF64}[]
    g, h=line.density_ratio, line.sound_speed_ratio
    transverse=2(g-1)/(g+1)
    axial=1-1/g
    for i in 1:(length(line.vertices) - 1)
        a, b=line.vertices[i], line.vertices[i + 1]
        segment_length=norm(b-a)
        tangent=(b-a)/segment_length
        panels=max(1, ceil(Int, segment_length*k*panels_per_wavelength/(2pi)))
        for panel in 0:(panels - 1), (t, w) in zip(nodes, weights)

            x=a+((panel+t)/panels)*(b-a)
            volume=pi*line.radius^2*segment_length*w/panels
            p, gradient=pair(x)
            all(isfinite, gradient) && isfinite(p) ||
                throw(ArgumentError("incident field must be finite on the filament"))
            push!(points, x)
            push!(mono, volume*k^2*(1/(g*h^2)-1)*p)
            push!(dip, volume*(transverse*gradient +
                               (axial-transverse)*sum(tangent .* gradient)*tangent))
        end
    end
    ScatteringField(Float64(k), points, mono, dip, line)
end

"""
    sidewall_scattering(tank::Tank, k, incident; reflection=1,
        reference_point=nothing, quadrature=(nphi,nz))

Reciprocal Kirchhoff single-reflection approximation on the actual cylindrical
sidewall, outward normal from water into wall. Integrates
`-R*(G*dn(p_inc) + dn_y(G)*p_inc)` over its curved surface. This symmetric
Kirchhoff kernel reproduces the infinite-plane image and preserves source/receiver
reciprocity for constant `R`; it is a high-frequency tangent-plane approximation,
not a boundary-condition solve or a cavity resonance model. `reflection` is a
constant complex pressure coefficient with magnitude <= 1, or `R(k,cosine)`.
The latter requires an explicit `reference_point` to freeze each wall node's
incidence angle. Use the same point for transmitter and receiver to preserve
reciprocity. This neglects the actual angular distribution of the incident field.
Top/bottom are not integrated. Use a `TankTransducer` incident field to include
incoming top/bottom images explicitly; repeated sidewall scattering is absent.

`quadrature` is periodic midpoint order in azimuth and Gauss order in height.
The default grows with `k`; refine it particularly near the wall and caustics.
No dense wall-to-wall matrix is formed. Valid pressure/gradient of the incident
field are required at every wall node. Evaluation exactly on the wall is excluded.
"""
function sidewall_scattering(tank::Tank, k::Real, incident;
        reflection = 1, reference_point = nothing, quadrature = nothing)
    tank.radius===nothing &&
        throw(ArgumentError("sidewall scattering requires a cylindrical tank"))
    isfinite(k) && k>0 || throw(ArgumentError("k must be finite and positive"))
    if reflection isa Number
        isfinite(reflection) && abs(reflection)<=1 ||
            throw(ArgumentError("reflection must be finite with magnitude <= 1"))
    else
        applicable(reflection, k, 0.5) && reference_point!==nothing &&
        length(reference_point)==3 && all(isfinite, reference_point) ||
            throw(ArgumentError("R(k,cosine) needs an explicit finite reference_point"))
    end
    height=tank.bounds.z[2]-tank.bounds.z[1]
    counts=quadrature===nothing ?
           (max(32, ceil(Int, 4k*tank.radius+32)), max(24, ceil(Int, k*height+24))) :
           quadrature
    length(counts)==2 && all(n->n isa Integer && n>0, counts) ||
        throw(ArgumentError("quadrature must contain two positive integer orders"))
    nphi, nz=counts
    zs, ws=gauss(nz, tank.bounds.z...)
    pair=_scattering_pair(incident, k)
    points=Vector{SVector{3, Float64}}(undef, nphi*nz)
    mono=Vector{ComplexF64}(undef, length(points))
    dip=Vector{SVector{3, ComplexF64}}(undef, length(points))
    cx, cy=sum(tank.bounds.x)/2, sum(tank.bounds.y)/2
    Threads.@threads for iz in 1:nz
        for j in 1:nphi
            sn, cs=sincos(2pi*(j-0.5)/nphi)
            normal=SVector(cs, sn, 0.0)
            x=SVector(cx+tank.radius*cs, cy+tank.radius*sn, zs[iz])
            index=(iz-1)*nphi+j
            points[index]=x
            coefficient=if reflection isa Number
                reflection
            else
                ray=x-SVector{3}(reference_point)
                norm(ray)>0 || throw(ArgumentError("reference point lies on a wall node"))
                reflection(k, clamp(abs(dot(ray, normal))/norm(ray), 0.0, 1.0))
            end
            coefficient isa Number && isfinite(coefficient) &&
            abs(coefficient)<=1+64eps() ||
                throw(ArgumentError("wall coefficients must be finite with magnitude <= 1"))
            if iszero(coefficient)
                mono[index]=0
                dip[index]=zero(SVector{3, ComplexF64})
            else
                p, gradient=pair(x)
                all(isfinite, gradient) && isfinite(p) ||
                    throw(ArgumentError("incident field must be finite on the wall"))
                weight=-coefficient*ws[iz]*2pi*tank.radius/nphi
                mono[index]=weight*sum(normal .* gradient)
                dip[index]=weight*p*normal
            end
        end
    end
    ScatteringField(Float64(k), points, mono, dip, tank)
end

"""
    boundary_scattering(tank::ProfiledTank, k, incident; boundary=:sidewall,
        reflection=1, reference_point=nothing, quadrature=nothing)

Reciprocal Kirchhoff single-reflection field on a profiled sidewall or bottom.
Uses the same kernel as [`sidewall_scattering`](@ref), with actual surface
positions, outward water normals and area Jacobians. `quadrature=(nphi,nv)`
uses periodic azimuth and Gauss height/radius. Refine both orders for small
features, nearby observers and caustics. Shallow profile features are geometric
approximations; this does not resolve exact edge diffraction, shadowing, elastic
wall motion or repeated tank scattering. Angle-dependent reflection uses the
same explicit frozen reference-point convention as `sidewall_scattering`.
"""
function boundary_scattering(tank::ProfiledTank, k::Real, incident;
        boundary::Symbol = :sidewall, reflection = 1, reference_point = nothing, quadrature = nothing)
    isfinite(k) && k>0 || throw(ArgumentError("k must be finite and positive"))
    if reflection isa Number
        isfinite(reflection) && abs(reflection)<=1 ||
            throw(ArgumentError("invalid reflection coefficient"))
    else
        applicable(reflection, k, 0.5) && reference_point!==nothing &&
        length(reference_point)==3 && all(isfinite, reference_point) ||
            throw(ArgumentError("R(k,cosine) requires a finite reference_point"))
    end
    height=tank.bounds.z[2]-tank.bounds.z[1]
    counts=quadrature===nothing ?
           (max(32, ceil(Int, 4k*tank.radius+32)),
        max(24, ceil(Int, k*(boundary===:sidewall ? height : tank.radius)+32))) : quadrature
    quad=_tank_boundary_quadrature(tank, boundary, counts)
    pair=_scattering_pair(incident, k)
    mono=Vector{ComplexF64}(undef, length(quad.points))
    dip=Vector{SVector{3, ComplexF64}}(undef, length(quad.points))
    Threads.@threads for i in eachindex(quad.points)
        x, normal=quad.points[i], quad.normals[i]
        coefficient=if reflection isa Number
            reflection
        else
            ray=x-SVector{3}(reference_point)
            norm(ray)>0 || throw(ArgumentError("reference point lies on the boundary"))
            reflection(k, clamp(abs(dot(ray, normal))/norm(ray), 0.0, 1.0))
        end
        coefficient isa Number && isfinite(coefficient) && abs(coefficient)<=1+64eps() ||
            throw(ArgumentError("wall coefficients must be finite with magnitude <= 1"))
        if iszero(coefficient)
            mono[i]=0
            dip[i]=zero(SVector{3, ComplexF64})
        else
            p, g=pair(x)
            isfinite(p) && all(isfinite, g) ||
                throw(ArgumentError("incident field must be finite on the boundary"))
            weight=-coefficient*quad.areas[i]
            mono[i]=weight*sum(normal .* g)
            dip[i]=weight*p*normal
        end
    end
    ScatteringField(Float64(k), quad.points, mono, dip, tank)
end

function sidewall_scattering(tank::ProfiledTank, k::Real, incident; kwargs...)
    boundary_scattering(tank, k, incident; boundary = :sidewall, kwargs...)
end

"""
    fluid_wall_reflection(k, cosine; thickness, density_ratio, sound_speed_ratio,
        backing_density_ratio, backing_sound_speed_ratio)

Pressure reflection coefficient of a lossless planar fluid layer backed by a
second fluid. Ratios are relative to the incident water; `k` is its wavenumber,
`cosine` the incident normal-angle cosine, and thickness is in meters. Retains
refraction, evanescence and repeated reflection within the layer. Uses the
`exp(-im*omega*t)` convention. Thickness zero recovers the water/backing interface.

A polyethylene-like fluid layer is only a surrogate for a plastic tank wall:
this function includes no elastic shear, flexural shell modes or damping.
With `sidewall_scattering`, use a shared explicit `reference_point` to freeze
the incidence-angle coefficient over each surface node for Tx and Rx. That
preserves reciprocal field propagation but approximates the true angular response.
"""
function fluid_wall_reflection(k::Real, cosine::Real; thickness::Real,
        density_ratio::Real, sound_speed_ratio::Real,
        backing_density_ratio::Real, backing_sound_speed_ratio::Real)
    isfinite(k) && k>0 && isfinite(cosine) && 0<=cosine<=1 &&
    isfinite(thickness) && thickness>=0 ||
        throw(ArgumentError("require k>0, cosine in [0,1], and thickness>=0"))
    all(x->isfinite(x) && x>0,
        (density_ratio, sound_speed_ratio,
            backing_density_ratio, backing_sound_speed_ratio)) ||
        throw(ArgumentError("material ratios must be finite and positive"))
    s2=1-cosine^2
    q1=k*sqrt(complex(1/sound_speed_ratio^2-s2))
    beta1=q1/(k*density_ratio)
    beta2=sqrt(complex(1/backing_sound_speed_ratio^2-s2))/backing_density_ratio
    if thickness==0
        return iszero(cosine+beta2) ?
               complex((backing_density_ratio-1)/(backing_density_ratio+1)) :
               (cosine-beta2)/(cosine+beta2)
    end
    if iszero(cosine)
        return sound_speed_ratio==1 && backing_sound_speed_ratio==1 ?
               complex((backing_density_ratio-1)/(backing_density_ratio+1)) : -1.0+0im
    end
    if abs(q1*thickness)<1e-6
        ss=thickness*sinc(q1*thickness/pi)
        pp=cos(q1*thickness)-im*beta2*k*density_ratio*ss
        vv=beta2*cos(q1*thickness)-im*q1^2/(k*density_ratio)*ss
        return (cosine*pp-vv)/(cosine*pp+vv)
    end
    r01=(cosine-beta1)/(cosine+beta1)
    r12=(beta1-beta2)/(beta1+beta2)
    phase=exp(2im*q1*thickness)
    (r01+r12*phase)/(1+r01*r12*phase)
end

"""
    ScatteringTransducer(source, fields...)

Original piston/tank-image source plus prepared fixed-frequency scattering fields.
Use as `transducer=` or a reciprocal receiver. This includes the supplied wall/line
interaction on the incident or receiving leg; it does not iterate target-wall or
target-filament feedback. Rebuild fields when frequency changes.
"""
struct ScatteringTransducer{T <: AbstractTransducer, F <: Tuple} <: AbstractTransducer
    source::T
    fields::F
end
function ScatteringTransducer(source::AbstractTransducer, fields::ScatteringField...)
    ScatteringTransducer(source, fields)
end
_transducer_aperture(t::Transducer) = t
_transducer_aperture(t) = throw(ArgumentError("a piston-based transducer is required"))
_transducer_aperture(t::TankTransducer) = t.source
_transducer_aperture(t::ScatteringTransducer) = _transducer_aperture(t.source)
function _transducer_field(t::ScatteringTransducer, k::Real; kwargs...)
    all(f->f.k==k, t.fields) ||
        throw(ArgumentError("prepared fields have a different wavenumber"))
    p, g=_transducer_field(t.source, k; kwargs...)
    (x->p(x)+sum(f->_scattering_value(f, x), t.fields; init = 0.0im),
        x->g(x)+sum(f->_scattering_value(f, x, true), t.fields; init = zero(SVector{
            3, ComplexF64})))
end

function _scattering_pair(t::ScatteringTransducer, k; kwargs...)
    all(f->f.k==k, t.fields) ||
        throw(ArgumentError("prepared fields have a different wavenumber"))
    source_pair=_scattering_pair(t.source, k; kwargs...)
    x->begin
        p, g=source_pair(x)
        for field in t.fields
            pi, gi=_scattering_value(field, x, true, true)
            p+=pi
            g+=gi
        end
        p, g
    end
end
