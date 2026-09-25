const _MAGNITUDE_COLORMAP = :viridis
const _PHASE_COLORMAP = Symbol("cyclic_mygbm_30-90_c48_n256")
const _DEFAULT_DYNAMIC_RANGE_DB = 40.0

function _angle_label(units::Symbol)
    units === :deg ? "Angle (deg)" :
    units === :rad ? "Angle (rad)" :
    throw(ArgumentError("angle_units must be :deg or :rad, got $units"))
end

function _convert_angle(values, units::Symbol)
    units === :deg ? rad2deg.(values) :
    units === :rad ? values :
    throw(ArgumentError("angle_units must be :deg or :rad, got $units"))
end

"""
Default color range for a target-strength colormap: the finite maximum down to
`_DEFAULT_DYNAMIC_RANGE_DB` below it, so a single very deep null (or an actual `-Inf`) doesn't
wash out the color scale across the rest of the pattern. Purely a display default, the underlying
sampled data stays unclamped; override with an explicit `colorrange` keyword.
"""
function _default_colorrange(ts::AbstractArray)
    finite_ts = filter(isfinite, vec(ts))
    isempty(finite_ts) && return (-1.0, 1.0)
    hi = maximum(finite_ts)
    lo = max(minimum(finite_ts), hi - _DEFAULT_DYNAMIC_RANGE_DB)
    return (lo, hi)
end

# Colorbar label for a plotted pressure `field`, of the scattered or the total pressure.
function _pressure_label(field::Symbol; scattered::Bool = false)
    if scattered
        field === :pressure_phase && return L"\arg(p_\mathrm{scatter})~(\mathrm{rad})"
        field === :pressure_real &&
            return L"\mathrm{Re}(p_\mathrm{scatter} / p_\mathrm{incident})~(\mathrm{dimensionless})"
        field === :pressure_imag &&
            return L"\mathrm{Im}(p_\mathrm{scatter} / p_\mathrm{incident})~(\mathrm{dimensionless})"
        return L"|p_\mathrm{scatter}| / |p_\mathrm{incident}|~(\mathrm{dimensionless})"
    end
    field === :pressure_phase && return L"\arg(p)~(\mathrm{rad})"
    field === :pressure_real &&
        return L"\mathrm{Re}(p / p_\mathrm{incident})~(\mathrm{dimensionless})"
    field === :pressure_imag &&
        return L"\mathrm{Im}(p / p_\mathrm{incident})~(\mathrm{dimensionless})"
    return L"|p| / |p_\mathrm{incident}|~(\mathrm{dimensionless})"
end
