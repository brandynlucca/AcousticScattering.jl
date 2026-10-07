# Incident fields beyond a plane wave, for solvers built on a spherical-harmonic expansion
# (Sapozhnikov and Bailey, J. Acoust. Soc. Am. 133, 661-676, 2013).

"""
    IncidentField

Supertype of incident excitations for solvers that accept an `incident` keyword. See
[`PlaneWave`](@ref), [`SphericalWave`](@ref) and [`BesselBeam`](@ref).
Full-3D BEM, closed-surface MFS and volume FEM evaluate [`incident_pressure`](@ref)
and [`incident_gradient`](@ref). Sphere modal uses [`incident_coefficient`](@ref)
for the three built-in axisymmetric fields. Acoustic spheroid modal and single
elastic spheroid/shell T-matrix paths project pointwise Cauchy data. Other solver
bases do not yet project general incident fields. Explicit directions use Cartesian solver coordinates;
numerical observation defaults still follow the solve's incidence-angle keywords.
"""
abstract type IncidentField end

struct _PointIncidentField{P, G} <: IncidentField
    pressure::P
    gradient::G
end

"""
    IncidentField(pressure, gradient)

Define a harmonic incident pressure and its Cartesian gradient at a fixed
wavenumber. Both callables accept a point in meters in the solver's body frame;
`gradient(x)` returns three components per meter. The field must satisfy the
exterior Helmholtz equation near the target, with all sources outside it.
Both callbacks must be safe for concurrent calls during assembly and sampling.
Pass as `incident=...` to full-3D BEM, closed-surface MFS, volume FEM,
acoustic spheroid `modal`, or single elastic spheroid/shell `tmatrix`.
Changing frequency requires callables for the new wavenumber. Sphere modal,
axisymmetric numerical and mixed-layer T-matrix paths do not project arbitrary callbacks.
"""
IncidentField(pressure, gradient) = _PointIncidentField(pressure, gradient)

function _resolve_incident(k, beta, alpha; incident = nothing,
        plane_direction = _bem3d_incidence_direction(beta, alpha))
    incident === nothing && return nothing
    incident isa PlaneWave && incident.direction === nothing &&
        incident.amplitude == 1 && return nothing
    incident isa _PointIncidentField && return incident
    incident isa IncidentField || throw(ArgumentError("incident must be an IncidentField"))
    field = if incident isa PlaneWave && incident.direction === nothing
        PlaneWave(; direction = plane_direction, amplitude = incident.amplitude)
    else
        incident
    end
    return IncidentField(x -> incident_pressure(field, k, x),
        x -> incident_gradient(field, k, x))
end

function _incident_callbacks(k, beta, alpha; incident = nothing)
    field = _resolve_incident(k, beta, alpha; incident)
    if field === nothing
        direction = Tuple(_bem3d_incidence_direction(beta, alpha))
        pressure, gradient = _plane_wave_incident(k, direction)
        return field, pressure, gradient
    end
    return field, field.pressure, field.gradient
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
    PlaneWave(; direction=nothing, amplitude=1, phase=0)

A plane wave `amplitude * exp(i*phase) * exp(i*k*dot(direction,x))` under
`exp(-i*omega*t)`. A supplied Cartesian direction is normalized. `direction=nothing`
uses the solver's incidence angles, or +x for sphere modal and standalone field queries.
`phase` is in radians at the origin. `PlaneWave()` preserves the default unit wave.
"""
struct PlaneWave <: IncidentField
    direction::Union{Nothing, SVector{3, Float64}}
    amplitude::ComplexF64
    function PlaneWave(; direction = nothing, amplitude::Number = 1, phase::Real = 0)
        new(direction === nothing ? nothing : _incident_direction(direction),
            _incident_amplitude(amplitude, phase))
    end
end

function _incident_direction(direction)
    length(direction) == 3 && all(x -> x isa Real && isfinite(x), direction) ||
        throw(ArgumentError("direction must have three finite real components"))
    d = SVector{3, Float64}(direction)
    scale = maximum(abs, d)
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("direction must be nonzero and finite"))
    d = d / scale
    return d / norm(d)
end

function _incident_amplitude(amplitude, phase)
    isfinite(amplitude) && isfinite(phase) ||
        throw(ArgumentError("amplitude and phase must be finite"))
    a, phi = ComplexF64(amplitude), Float64(phase)
    isfinite(a) && isfinite(phi) ||
        throw(ArgumentError("amplitude and phase must be representable as Float64"))
    value = a * cis(phi)
    isfinite(value) || throw(ArgumentError("phase-adjusted amplitude must be finite"))
    return value
end

"""
    SphericalWave(range; direction=(1,0,0), amplitude=1, phase=0)

An outgoing point-source wave `amplitude * exp(i*phase) * exp(i*k*R)/R`,
where `R = norm(x + range*direction)`. The source lies at `-range*direction`
in meters; the normalized direction is its propagation axis at the origin.
The amplitude has pressure-times-length units if pressure is dimensional;
it is not normalized to unit pressure at the origin. Sphere modal requires
`range > radius`. Phase is in radians, and there is no `1/(4pi)` factor.
"""
struct SphericalWave <: IncidentField
    range::Float64
    direction::SVector{3, Float64}
    amplitude::ComplexF64
    function SphericalWave(range::Real; direction = (1, 0, 0),
            amplitude::Number = 1, phase::Real = 0)
        isfinite(Float64(range)) && Float64(range) > 0 ||
            throw(ArgumentError("range must be finite and positive"))
        return new(Float64(range), _incident_direction(direction),
            _incident_amplitude(amplitude, phase))
    end
end

"""
    BesselBeam(angle; direction=(1,0,0), amplitude=1, phase=0)

A non-diffracting zeroth-order Bessel beam along the normalized Cartesian `direction`, with
half-conical angle `angle` [rad] between its plane-wave components and that axis (Gong et al.,
LIP2016 proceedings; Sapozhnikov and Bailey, 2013, Sec. III.B). `angle = 0` is a plane wave.
Its pressure is `amplitude * exp(i*phase) * J0(k*sin(angle)*rho) * exp(i*k*cos(angle)*s)`,
where `s=dot(direction,x)` and `rho=norm(x-s*direction)`. Phase is in radians at the origin.
"""
struct BesselBeam <: IncidentField
    angle::Float64
    direction::SVector{3, Float64}
    amplitude::ComplexF64
    function BesselBeam(angle::Real; direction = (1, 0, 0),
            amplitude::Number = 1, phase::Real = 0)
        isfinite(Float64(angle)) || throw(ArgumentError("angle must be finite"))
        return new(Float64(angle), _incident_direction(direction),
            _incident_amplitude(amplitude, phase))
    end
end

# Q_l/(2l+1) of the incident field's spherical-harmonic expansion, Sapozhnikov and Bailey
# Eqs. (9), (10) and the zeroth-order case of (62), for an expansion about the origin with the
# field's own axis as the pole.
_scale_incident(w, value) = w.amplitude == 1 ? value : w.amplitude * value
_incident_coefficient(w::PlaneWave, l::Integer, ::Real) = _scale_incident(w, im^l)
function _incident_coefficient(w::SphericalWave, l::Integer, k::Real)
    _scale_incident(w, im * k * (-1)^l * hs(l, k * w.range))
end
function _incident_coefficient(b::BesselBeam, l::Integer, ::Real)
    _scale_incident(b, im^l * legendre_p(l, cos(b.angle)))
end

const _AxisIncidentField = Union{PlaneWave, SphericalWave, BesselBeam}
function _incident_axis(w::_AxisIncidentField)
    w.direction === nothing ? SVector(1.0, 0.0, 0.0) : w.direction
end

function _incident_point(k, point)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    length(point) == 3 && all(x -> x isa Real && isfinite(x), point) ||
        throw(ArgumentError("point must have three finite real coordinates"))
    x = SVector{3, Float64}(point)
    all(isfinite, x) ||
        throw(ArgumentError("point coordinates must be representable as Float64"))
    return x
end

"""
    incident_pressure(field::IncidentField, k, point)

Evaluate the complex incident pressure at one Cartesian point [m],
with exterior wavenumber `k` [rad/m], under `exp(-i*omega*t)`. Built-in fields
can be reused at different frequencies. Callable `IncidentField(p, grad)` fields
are already tied to their construction frequency; `k` cannot retune those callbacks.
Implement this and [`incident_gradient`](@ref) for custom pointwise fields.
"""
function incident_pressure(w::PlaneWave, k::Real, point)
    x = _incident_point(k, point)
    return w.amplitude * cis(k * dot(_incident_axis(w), x))
end
function incident_pressure(w::SphericalWave, k::Real, point)
    x = _incident_point(k, point)
    r = norm(x + w.range * w.direction)
    r > 0 || throw(ArgumentError("spherical incident field is singular at its source"))
    return w.amplitude * cis(k * r) / r
end
function incident_pressure(w::BesselBeam, k::Real, point)
    x = _incident_point(k, point)
    s = dot(w.direction, x)
    rho = norm(x - s * w.direction)
    return w.amplitude * besselj(0, k * sin(w.angle) * rho) * cis(k * cos(w.angle) * s)
end
function incident_pressure(w::_PointIncidentField, k::Real, point)
    x = _incident_point(k, point)
    value = w.pressure(x)
    value isa Number && isfinite(value) ||
        throw(ArgumentError("incident pressure must be finite"))
    return value
end

"""
    incident_gradient(field::IncidentField, k, point)

Cartesian gradient of [`incident_pressure`](@ref), in pressure units per meter.
Returns three complex components, with no complex conjugation. Custom fields
must provide a source-free Helmholtz pressure and its matching gradient near
the target; callbacks must be safe for concurrent assembly and sampling.
"""
function incident_gradient(w::PlaneWave, k::Real, point)
    return (im * k * incident_pressure(w, k, point)) * _incident_axis(w)
end
function incident_gradient(w::SphericalWave, k::Real, point)
    x = _incident_point(k, point)
    delta = x + w.range * w.direction
    r = norm(delta)
    return incident_pressure(w, k, point) * (im * k - 1 / r) * delta / r
end
function incident_gradient(w::BesselBeam, k::Real, point)
    x = _incident_point(k, point)
    s = dot(w.direction, x)
    transverse = x - s * w.direction
    rho = norm(transverse)
    kt, kz = k * sin(w.angle), k * cos(w.angle)
    radial = iszero(rho) ? zero(transverse) : -kt * besselj(1, kt * rho) * transverse / rho
    return w.amplitude * cis(kz * s) *
           (radial + im * kz * besselj(0, kt * rho) * w.direction)
end
function incident_gradient(w::_PointIncidentField, k::Real, point)
    x = _incident_point(k, point)
    value = w.gradient(x)
    length(value) == 3 && all(isfinite, value) ||
        throw(ArgumentError("incident gradient must have three finite components"))
    return SVector{3, ComplexF64}(value)
end

"""
    incident_coefficient(field::IncidentField, degree, k)

Regular spherical coefficient `Q_degree/(2degree+1)` about the origin and
the built-in field's own axis: `p = sum((2l+1)*c_l*j_l(k*r)*P_l(cos(theta)))`.
Includes complex amplitude and phase. For a spherical source the expansion
converges only for `r < range`. Available for `PlaneWave`, `SphericalWave` and
zeroth-order `BesselBeam`; arbitrary callbacks are not automatically projected.
"""
function incident_coefficient(w::_AxisIncidentField, degree::Integer, k::Real)
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    return _incident_coefficient(w, degree, k)
end
function incident_coefficient(w::IncidentField, degree::Integer, k::Real)
    throw(ArgumentError("spherical expansion is unavailable for this incident field"))
end
