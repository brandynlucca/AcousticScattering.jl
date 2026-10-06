"""
    SpatialFluid(density_contrast, soundspeed_contrast; min_soundspeed_contrast)

Lossless spatially varying fluid for `fem(...; method=:volume)`. Each contrast
is a positive real constant or a callable `x -> value`, relative to the
homogeneous exterior fluid. `x` contains Cartesian coordinates in meters in
the volume FEM solution frame, the same frame as region `centers` and `pressure`
queries, with the default symmetry axis along `+z`. Profiles do not move or
rotate automatically with a region.

The positive lower bound `min_soundspeed_contrast` sets the shortest wavelength
used for meshing. Sampled speeds below this bound are rejected. Callbacks must
be deterministic and safe to evaluate concurrently. Resolve variation of the
profiles with `h`/`h_body`; represent discontinuities by separate regions.

Supports a single `Sphere`/`Spheroid` or the coupled-region volume FEM interface.
The exterior and its radiation closure remain homogeneous. This material is
not supported by `free_surface`, which requires a reflected material profile.
"""
struct SpatialFluid{D, C} <: AbstractBoundaryCondition
    density_contrast::D
    soundspeed_contrast::C
    min_soundspeed_contrast::Float64

    function SpatialFluid(density, speed; min_soundspeed_contrast::Real)
        lower = Float64(min_soundspeed_contrast)
        isfinite(lower) && lower > 0 || throw(ArgumentError(
            "min_soundspeed_contrast must be finite and positive"))
        for (name, value) in (("density_contrast", density), ("soundspeed_contrast", speed))
            value isa Number && _spatial_fluid_positive(value, name)
        end
        speed isa Real && speed < lower &&
            throw(ArgumentError(
                "soundspeed_contrast is below min_soundspeed_contrast"))
        return new{typeof(density), typeof(speed)}(density, speed, lower)
    end
end

function _spatial_fluid_positive(value, name)
    value isa Real && isfinite(value) && value > 0 || throw(ArgumentError(
        "SpatialFluid $name must evaluate to a finite positive real number"))
    result = Float64(value)
    isfinite(result) && result > 0 || throw(ArgumentError(
        "SpatialFluid $name must be representable as a finite positive Float64"))
    return result
end

_spatial_fluid_value(value::Real, x) = value
_spatial_fluid_value(value, x) = value(x)

struct _SpatialFluidCoefficients{M}
    material::M
    k::Float64
end

_volume_fluid_parameters(material::Tuple, x) = material
function _volume_fluid_parameters(coefficients::_SpatialFluidCoefficients, x)
    material = coefficients.material
    density = _spatial_fluid_positive(
        _spatial_fluid_value(material.density_contrast, x), "density_contrast")
    speed = _spatial_fluid_positive(
        _spatial_fluid_value(material.soundspeed_contrast, x), "soundspeed_contrast")
    speed >= material.min_soundspeed_contrast || throw(ArgumentError(
        "SpatialFluid soundspeed_contrast at $x is below min_soundspeed_contrast"))
    return density, coefficients.k / speed
end

# Constant media avoid calculating coordinates in the assembly loop.
_volume_fluid_parameters(material::Tuple, cv, q, coords) = material
function _volume_fluid_parameters(material::_SpatialFluidCoefficients, cv, q, coords)
    _volume_fluid_parameters(material, Ferrite.spatial_coordinate(cv, q, coords))
end
