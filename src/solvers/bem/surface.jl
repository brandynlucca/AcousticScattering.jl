"""
    bem(surface::Mesh, boundary, k; incidence_angle=π/2, incidence_azimuth=0, kwargs...)

Solve scattering on a supplied full-3D `surface` with rigid, pressure-release or homogeneous
fluid/gas material. The stored quadrature and outward normals are used directly. Wavenumber
`k` is in inverse meters, including when the mesh input used centimeters or millimeters.
Solver options are the same as `bem(body, boundary, k; method=:full)`; mesh resolution and
quadrature are selected when constructing `surface`. Returns a [`BEMSolution`](@ref).
"""
function bem(surface::Mesh{<:Inti.Quadrature},
        boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        incident = nothing, kwargs...)
    incident = _resolve_incident(
        k, incidence_angle, incidence_azimuth; incident)
    density = Ref{Union{Nothing, Vector{ComplexF64}}}(nothing)
    capture = boundary isa Rigid ? (; _density = density) : (;)
    p, q, quad, report = solve_full_bem(boundary, k, surface.data;
        incidence_angle, incidence_azimuth, incident, return_diagnostics = true, capture..., kwargs...)
    return _full_bem_solution(surface, boundary, k, p, q, quad, report,
        incidence_angle, incidence_azimuth; density = density[], incident)
end

function _full_bem_solution(surface, boundary, k, p, q, quad, report,
        incidence_angle, incidence_azimuth; density = nothing, incident = nothing)
    report = merge(report,
        (; meshsize = surface.resolution,
            illumination = incident === nothing ? :plane_wave : :prescribed))
    if surface.body isa _SurfaceGeometry
        report = merge(report, (; geometry = surface.body.validation,
            provenance = surface.body.provenance))
    end
    data = _FullBEMSurfaceData(
        quad, p, q, incidence_angle, incidence_azimuth, report, density, incident)
    return BEMSolution(surface.body, boundary, Float64(k), :full, data)
end
