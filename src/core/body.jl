# Shared geometry supertype.
"""
    AbstractBody

Supertype of scattering geometries. Construct a [`Sphere`](@ref), [`Spheroid`](@ref),
[`Cylinder`](@ref) or [`Shell`](@ref), then pass it to a solver with a boundary condition
and exterior wavenumber. Geometry dimensions are in meters.
"""
abstract type AbstractBody end
