# Incident fields beyond a plane wave, for solvers built on a spherical-harmonic expansion
# (Sapozhnikov and Bailey, J. Acoust. Soc. Am. 133, 661-676, 2013).

"""
    IncidentField

Supertype of incident excitations for solvers that accept an `incident` keyword. See
[`PlaneWave`](@ref), [`SphericalWave`](@ref) and [`BesselBeam`](@ref).
"""
abstract type IncidentField end

struct _PointIncidentField{P, G, S} <: IncidentField
    pressure::P
    gradient::G
    source::S
end

"""
    IncidentField(pressure, gradient)

Prescribe a harmonic incident pressure and its Cartesian gradient at a fixed
wavenumber. Both callables accept a point in meters in the solver's body frame;
`gradient(x)` returns three components per meter. The field must satisfy the
exterior Helmholtz equation near the target, with all sources outside it.
Both callbacks must be safe for concurrent calls during assembly and sampling.
Pass as `incident=...` to full-3D BEM, closed-surface MFS or volume FEM.
Changing frequency requires callables for the new wavenumber. This constructor
does not project a general field into modal or axisymmetric solver bases.
"""
IncidentField(pressure, gradient) = _PointIncidentField(pressure, gradient, nothing)

function _resolve_incident(k, beta, alpha; incident = nothing, transducer = nothing)
    incident !== nothing && transducer !== nothing &&
        throw(ArgumentError("supply incident or transducer, not both"))
    if transducer !== nothing
        p, gradient = _transducer_field(transducer, k)
        return _PointIncidentField(p, gradient, transducer)
    end
    incident === nothing && return nothing
    incident isa PlaneWave && return nothing
    incident isa _PointIncidentField && return incident
    throw(ArgumentError("this solver accepts IncidentField(pressure, gradient) or PlaneWave()"))
end

function _incident_traces(quad, k, beta, alpha, incident)
    if incident === nothing
        direction = _bem3d_incidence_direction(beta, alpha)
        p = ComplexF64[cis(k * dot(direction, q.coords)) for q in quad]
        dp = ComplexF64[im*k*dot(direction, q.normal)*p[i] for (i, q) in enumerate(quad)]
    else
        p = ComplexF64[incident.pressure(q.coords) for q in quad]
        dp = ComplexF64[_incident_normal(incident, q.coords, q.normal) for q in quad]
    end
    all(isfinite, p) && all(isfinite, dp) ||
        throw(ArgumentError("incident traces must be finite on the target"))
    return p, dp
end

function _incident_normal(incident::_PointIncidentField, x, normal)
    gradient = incident.gradient(x)
    length(gradient) == 3 && all(isfinite, gradient) ||
        throw(ArgumentError("incident gradient must have three finite components"))
    return sum(gradient[i]*normal[i] for i in 1:3)
end

function _data_incident_pressure(data, k, x)
    data.incident === nothing || return data.incident.pressure(x)
    direction = _bem3d_incidence_direction(data.incidence_angle, data.incidence_azimuth)
    return cis(k*dot(direction, x))
end

function _data_incident_normal(data, k, x, normal)
    data.incident === nothing || return _incident_normal(data.incident, x, normal)
    direction = _bem3d_incidence_direction(data.incidence_angle, data.incidence_azimuth)
    return im*k*dot(direction, normal)*cis(k*dot(direction, x))
end

"""
    PlaneWave()

A unit-amplitude plane wave, `exp(ikz)` along its own propagation axis. The default incident
field wherever an `incident` keyword exists.
"""
struct PlaneWave <: IncidentField end

"""
    SphericalWave(range)

A unit-amplitude point-source spherical wave, `exp(ikR)/R`, from a source `range` [m] from the
expansion origin, on the observation axis.
"""
struct SphericalWave <: IncidentField
    range::Float64
    function SphericalWave(range::Real)
        isfinite(range) && range > 0 ||
            throw(ArgumentError("range must be finite and positive"))
        return new(Float64(range))
    end
end

"""
    BesselBeam(angle)

A unit-amplitude, non-diffracting zeroth-order Bessel beam on the observation axis, with
half-conical angle `angle` [rad] between its plane-wave components and that axis (Gong et al.,
LIP2016 proceedings; Sapozhnikov and Bailey, 2013, Sec. III.B). `angle = 0` is a plane wave.
"""
struct BesselBeam <: IncidentField
    angle::Float64
    function BesselBeam(angle::Real)
        isfinite(angle) ? new(Float64(angle)) :
        throw(ArgumentError("angle must be finite"))
    end
end

# Q_l/(2l+1) of the incident field's spherical-harmonic expansion, Sapozhnikov and Bailey
# Eqs. (9), (10) and the zeroth-order case of (62), for an expansion about the origin with the
# field's own axis as the pole.
_incident_coefficient(::PlaneWave, l::Integer, ::Real) = im^l
function _incident_coefficient(w::SphericalWave, l::Integer, k::Real)
    im * k * (-1)^l * hs(l, k * w.range)
end
function _incident_coefficient(b::BesselBeam, l::Integer, ::Real)
    im^l * legendre_p(l, cos(b.angle))
end
