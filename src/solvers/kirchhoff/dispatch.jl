# --- `kirchhoff`: high-frequency physical optics ----------------------------

"""
    kirchhoff(body::AbstractBody, boundary::AbstractBoundaryCondition, k; incidence_angle=π/2)

High-frequency (physical-optics) result, returns a [`KirchhoffSolution`](@ref). Same `body`-type
dispatch and bend-auto-detection as [`modal`](@ref). Post-process with [`target_strength`](@ref)`(sol)`
or [`scattering_amplitude`](@ref)`(sol)`.
"""
function kirchhoff(body::Sphere, boundary::AbstractBoundaryCondition, k::Real)
    # Same closed form as `kirchhoff_target_strength`, replicated here for the complex amplitude.
    x = k * body.radius
    Rc = reflection_coefficient(boundary)
    f = Rc * body.radius * (cis(-2x) / 2 - im * (cis(-2x) - 1) / (4x))
    return KirchhoffSolution(body, boundary, k, f)
end

function kirchhoff(body::Sphere,
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}}, k::Real)
    # `reflection_coefficient` for a fluid shell needs (k, a), its own frequency/thickness-dependent
    # reflection, not a single constant, so this can't share the generic `AbstractBoundaryCondition` method above.
    x = k * body.radius
    Rc = reflection_coefficient(boundary, k, body.radius)
    f = Rc * body.radius * (cis(-2x) / 2 - im * (cis(-2x) - 1) / (4x))
    return KirchhoffSolution(body, boundary, k, f)
end

function kirchhoff(body::Spheroid, boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2)
    f = kirchhoff_form_function(boundary, k, body; angle = incidence_angle)
    return KirchhoffSolution(body, boundary, k, f)
end

function kirchhoff(body::Cylinder, boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2)
    f = if _isbent(body)
        bent_cylinder_kirchhoff_form_function(boundary, k, body.radius, body.length,
            body.radius_curvature; aspect_angle = incidence_angle)
    else
        kirchhoff_form_function(
            boundary, k, body.radius, body.length; angle = incidence_angle)
    end
    return KirchhoffSolution(body, boundary, k, f)
end
