# Acoustic plane-wave spheroids retain modal coefficients for other receive
# directions. Other modal results still store their selected observation angle.
function target_strength(sol::ModalSolution; kwargs...)
    sol.data isa _SpheroidModalData &&
        return target_strength(scattering_amplitude(sol; kwargs...))
    isempty(kwargs) || throw(ArgumentError(
        "target_strength(::ModalSolution) takes no keywords: the observation angle was already " *
        "fixed when modal(...) was called. Call modal(...) again with a different `angle`/" *
        "`incidence_angle` to get a different result."))
    return target_strength(sol.f)
end
function scattering_amplitude(sol::ModalSolution; kwargs...)
    if sol.data isa _SpheroidModalData
        isempty(kwargs) && return sol.f
        all(k -> k in (:angle, :azimuth), keys(kwargs)) || throw(ArgumentError(
            "retained spheroid modal solutions accept observation `angle` and `azimuth`"))
        angle = get(kwargs, :angle, sol.data.scatter_angle)
        azimuth = get(kwargs, :azimuth, sol.data.scatter_azimuth)
        return only(_spheroid_modal_amplitudes(sol, [angle], [azimuth]))
    end
    isempty(kwargs) || throw(ArgumentError(
        "scattering_amplitude(::ModalSolution) takes no keywords: the observation angle was " *
        "already fixed when modal(...) was called. Call modal(...) again with a different " *
        "`angle`/`incidence_angle` to get a different result."))
    return sol.f
end
