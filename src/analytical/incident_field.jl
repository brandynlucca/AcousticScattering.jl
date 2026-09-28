# Incident fields beyond a plane wave, for solvers built on a spherical-harmonic expansion
# (Sapozhnikov and Bailey, J. Acoust. Soc. Am. 133, 661-676, 2013).

"""
    IncidentField

Supertype of incident excitations for solvers that accept an `incident` keyword. See
[`PlaneWave`](@ref), [`SphericalWave`](@ref) and [`BesselBeam`](@ref).
"""
abstract type IncidentField end

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
