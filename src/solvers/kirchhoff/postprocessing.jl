function target_strength(sol::KirchhoffSolution; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "target_strength(::KirchhoffSolution) takes no keywords: the observation angle was " *
        "already fixed when kirchhoff(...) was called. Call kirchhoff(...) again with a " *
        "different `angle`/`incidence_angle` to get a different result."))
    return target_strength(sol.f)
end
function scattering_amplitude(sol::KirchhoffSolution; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "scattering_amplitude(::KirchhoffSolution) takes no keywords: the observation angle was " *
        "already fixed when kirchhoff(...) was called. Call kirchhoff(...) again with a " *
        "different `angle`/`incidence_angle` to get a different result."))
    return sol.f
end
