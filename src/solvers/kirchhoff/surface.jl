"""
    kirchhoff(surface::Mesh, boundary, k; incidence_angle=π/2, incidence_azimuth=0)

Kirchhoff physical-optics backscatter on a supplied full-3D surface, using
its quadrature nodes, outward normals and endcaps. For incident propagation
`d`, integrate only source-facing points with `dot(n,d) < 0`:

    f = im*k*Rc/(2pi) * integral(dot(n,d)*exp(2im*k*dot(d,y)), surface).

Here `Rc` is the normal-incidence pressure reflection coefficient. The time
convention is `exp(-im*omega*t)`, with outgoing scattered pressure
`f*exp(im*k*r)/r`. Matching canonical meshes reproduce the analytic-shape
Kirchhoff amplitudes to quadrature/geometry accuracy. Local normal selection
does not test visibility: a concave surface may include self-shadowed points.
Returns a [`KirchhoffSolution`](@ref).
"""

function kirchhoff(surface::Mesh{<:Inti.Quadrature}, boundary::AbstractBoundaryCondition,
        k::Real; incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0)
    Rc = reflection_coefficient(boundary)
    direction = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    total = zero(ComplexF64)
    for q in surface.data
        nd = dot(q.normal, direction)
        nd < 0 || continue
        total += nd * cis(2k * dot(direction, q.coords)) * q.weight
    end
    f = im * Rc * (k / (2π)) * total
    return KirchhoffSolution(surface.body, boundary, Float64(k), f)
end
