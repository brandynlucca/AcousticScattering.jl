# --- Solutions: one concrete type per dispatcher, so a result's own type tells you which solver
# produced it. Internal `data` shapes are reused across sub-methods (e.g. axisymmetric bem/mfs)
# purely as an implementation detail; that reuse never surfaces as a shared public type. -----------

"""
    AbstractSolution

Supertype of results returned by [`modal`](@ref), [`tmatrix`](@ref), [`kirchhoff`](@ref), [`fem`](@ref),
[`bem`](@ref) and [`mfs`](@ref). Query [`target_strength`](@ref),
[`scattering_amplitude`](@ref) and [`diagnostics`](@ref) where available.
Construct solutions through their solver rather than their internal storage fields.

Evaluate a field by calling `solution(points; quantity=:pressure, kwargs...)`.
The default delegates to [`pressure`](@ref), including its `field` and `region`
keywords and incident-pressure normalization. Point inputs have the same shapes
as `pressure`.

Full-3D volume FEM also supports `quantity=:displacement`, `:velocity`, or
`:stress` strictly inside elastic or `ViscousLayer` regions. These return complex
Cartesian vectors [m or m/s] or 3-by-3 Cauchy stress tensors [Pa], respectively,
in the solution frame, using `exp(-iωt)` and tension-positive stress. The local
material determines the constitutive law. Scalar acoustic fluid and exterior
regions do not support these mechanical queries.

Displacement and velocity require `density_exterior` [kg/m³] and
`soundspeed_exterior` [m/s]. Mechanical queries accept `pressure_amplitude=1`
[Pa] per unit incident-pressure normalization. Stress needs no exterior scales;
if a sound speed is supplied in a viscous region, it must match the solve.
Velocity is `-im*k*soundspeed_exterior` times displacement. See
[Material fields in volume FEM](@ref fem-material-fields) for scaling and limits.
"""
abstract type AbstractSolution end
