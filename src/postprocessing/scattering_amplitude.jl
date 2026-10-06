# --- `target_strength`/`scattering_amplitude` on `AbstractSolution`s -----------------------
# `scattering_amplitude` is the solution-level replacement for the low-level, unexported
# `form_function(boundary, k, ...)` family: the complex amplitude [m], pre-dB-conversion.

"""
    scattering_amplitude(solution; kwargs...)

Return the complex far-field scattering amplitude in meters. Axisymmetric BEM/MFS and
structural shell FEM accept observation `angle` and `azimuth` [rad] in body coordinates;
their defaults are backscatter, `angle = pi - incidence_angle`, `azimuth = pi`.
Full BEM accepts a unit-vector `direction`, defaulting to the negative incident direction.
For prescribed `incident` fields, numerical defaults use the solve's incidence-angle
keywords, not the field's Cartesian axis; specify the observation explicitly.
Cartesian body length is x, width is y and height/depth is z. Polar angles are from +x;
azimuth is from +y toward +z, so `(angle,azimuth)=(pi/2,0)` points along +y.
Bent MFS returns backscatter only. Modal/Kirchhoff observation is fixed at solve time and
post-processing keywords throw `ArgumentError`. Supported radial spheres return complex
backscatter amplitude without observation keywords. Sphere meridian and elastic
cylinder radial FEM also return complex backscatter. Cylinder/spheroid meridian
and volume FEM accept observation `angle` and `azimuth`. Any scalar-only FEM result
rejects complex amplitude queries; use `target_strength` for those results.

`TMatrixSolution` from `tmatrix(...; method=:farfield)` accepts `incidence_angle`,
`incidence_azimuth`, observation `angle` and `azimuth` without additional FEM solves.
With no overrides, it uses the solve's directions. If either incidence keyword is
supplied, omitted observation angles select backscatter at that incidence.
Other T-matrix paths retain their solve-time directions and reject these keywords.

# Examples
```julia
solution = modal(Sphere(0.01), Rigid(), 100.0)
amplitude = scattering_amplitude(solution)
```
"""
function scattering_amplitude end
