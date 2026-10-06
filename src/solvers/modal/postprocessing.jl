# `modal`/`kirchhoff` bake the observation angle in at solve time (a formula re-evaluation, not
# reusable surface state), so these take no keywords — rejected explicitly with a message pointing
# at the actual fix (re-solve at a different angle), rather than silently ignored or a raw MethodError.
function target_strength(sol::ModalSolution; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "target_strength(::ModalSolution) takes no keywords: the observation angle was already " *
        "fixed when modal(...) was called. Call modal(...) again with a different `angle`/" *
        "`incidence_angle` to get a different result."))
    return target_strength(sol.f)
end
function scattering_amplitude(sol::ModalSolution; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "scattering_amplitude(::ModalSolution) takes no keywords: the observation angle was " *
        "already fixed when modal(...) was called. Call modal(...) again with a different " *
        "`angle`/`incidence_angle` to get a different result."))
    return sol.f
end
