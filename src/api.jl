# Unified, short-named public API. Dispatches to the numerical implementations elsewhere in this
# package, no numerics live here. Six dispatchers: `modal`, `kirchhoff`, `fem`, `bem`, `mfs`, `shell`.

# --- Geometry: the `AbstractBody` hierarchy -----------------------------------------

"""
    Sphere(radius)

A sphere of the given `radius` [m]. `radius` alone determines its acoustic
behavior once paired with a boundary condition and wavenumber, incidence
angle is physically meaningless for a sphere, so methods dispatching on
`Sphere` never take an `incidence_angle` keyword.
"""
struct Sphere <: AbstractBody
    radius::Float64
    function Sphere(radius::Real)
        (radius > 0 || throw(ArgumentError("Sphere radius must be positive"));
            new(Float64(radius)))
    end
end

"""
    Cylinder(radius, length; radius_curvature=Inf, endcap_depth=0.0)

A finite circular cylinder of the given `radius` and `length` [m].

- `radius_curvature` [m]: `Inf` (default) is a straight cylinder; a finite
  value uniformly bends it into an arc of that radius of curvature (in the
  plane of incidence, see [`modal`](@ref)/[`kirchhoff`](@ref)/[`mfs`](@ref)
  for how each method family responds to this). Bending near-broadside is
  well-supported (`modal`, `kirchhoff`); `mfs` supports any angle via a
  genuinely different, non-axisymmetric solve; `bem` has no bent-cylinder
  implementation at all yet and raises an error rather than silently
  ignoring the curvature.
- `endcap_depth` [m]: `0.0` (default) uses flat end caps; a positive value
  caps the cylinder with quarter-prolate-spheroid domes of that depth
  instead, matching the cylinder's radius at the join so the surface
  normal stays continuous there. Only [`mfs`](@ref) uses this (flat caps
  give a sharp-corner normal discontinuity that breaks the method of
  fundamental solutions specifically); `bem`/`fem` on a `Cylinder` always
  use the flat-cap mesh regardless of this field.
"""
struct Cylinder <: AbstractBody
    radius::Float64
    length::Float64
    radius_curvature::Float64
    endcap_depth::Float64
    function Cylinder(radius::Real, length::Real; radius_curvature::Real = Inf, endcap_depth::Real = 0.0)
        radius > 0 && length > 0 ||
            throw(ArgumentError("Cylinder radius/length must be positive"))
        radius_curvature > 0 ||
            throw(ArgumentError("radius_curvature must be positive (Inf for straight)"))
        endcap_depth >= 0 || throw(ArgumentError("endcap_depth must be nonnegative"))
        return new(Float64(radius), Float64(length), Float64(radius_curvature), Float64(endcap_depth))
    end
end

_isbent(body::Cylinder) = !isinf(body.radius_curvature)
_iscapped(body::Cylinder) = body.endcap_depth > 0

"""
    Shell(body, thickness)

A thin shell of the given `thickness` [m] over the outer surface of
`body` (a [`Sphere`](@ref) or [`Spheroid`](@ref), the only two currently
supported by [`shell`](@ref)'s underlying FEM implementations). Replaces
the previous separate `ProlateShellGeometry` (spheroid-only) and bare
`outer_radius`/`thickness` scalar pair (sphere-only), which base
geometry a shell has is now just `typeof(body)`, not a separate argument.
"""
struct Shell <: AbstractBody
    body::AbstractBody
    thickness::Float64
    function Shell(body::AbstractBody, thickness::Real)
        body isa Union{Sphere, Spheroid} ||
            throw(ArgumentError("Shell only supports a Sphere or Spheroid base body"))
        thickness > 0 || throw(ArgumentError("Shell thickness must be positive"))
        return new(body, Float64(thickness))
    end
end

# --- Solutions: one concrete type per dispatcher, so a result's own type tells you which solver
# produced it. Internal `data` shapes are reused across sub-methods (e.g. axisymmetric bem/mfs)
# purely as an implementation detail; that reuse never surfaces as a shared public type. -----------

abstract type AbstractSolution end

# Reusable axisymmetric surface-field data (bem/mfs axisymmetric solves), an internal component
# type shared by `BEMSolution`/`MFSSolution`, never exposed as its own public type.
struct _AxisymmetricSurfaceData
    mesh::MeridianMesh
    p_scat_modes::Vector{Vector{ComplexF64}}
    dpdn_scat_modes::Vector{Vector{ComplexF64}}
    p_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    dpdn_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
end

# Full 3D (non-axisymmetric) BEM surface-field data, `bem(...; method=:full)` only.
struct _FullBEMSurfaceData
    quad::Any
    p_scat::Vector{ComplexF64}
    dpdn_scat::Vector{ComplexF64}
    incidence_angle::Float64
    incidence_azimuth::Float64
end

# Non-axisymmetric bent-cylinder MFS surface-field data, `mfs` on a bent `Cylinder` only.
struct _BentMFSSurfaceData
    p_scat::Vector{ComplexF64}
    dpdn_scat::Vector{ComplexF64}
    points::Vector{NTuple{3, Float64}}
    normals::Vector{NTuple{3, Float64}}
    areas::Vector{Float64}
    incidence_angle::Float64
end

# A `fem` result that's already a fully-computed target strength (today's radial/meridian FEM
# implementations return only the final scalar, not reusable surface data).
struct _ScalarFEMData
    ts::Float64
end

# A `fem` result on a `Shell` body: exterior (and, for a fluid-filled interior, interior) surface
# pressure/normal-derivative data plus the shell's own mechanical state, mirroring
# `solve_shell_fluid_coupled`/`solve_shell_fluid_filled_coupled`/
# `solve_general_shell_fluid_filled_coupled`'s return tuples. `p_int_modes`/`ps_int`/
# `dpdn_int_modes` are `nothing` for a vacuum/air-backed shell (no interior fluid coupling).
struct _ShellFEMSurfaceData
    ps_ext::Vector{Panel}
    p_ext_modes::Vector{Vector{ComplexF64}}
    dpdn_ext_modes::Vector{Vector{ComplexF64}}
    ps_int::Union{Nothing, Vector{Panel}}
    p_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    dpdn_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    shell_state::Any
end

"""
    ModalSolution

Result of [`modal`](@ref): the exact modal-series complex scattering amplitude `f` [m] at the
angle(s) `modal` was called with. Post-process with [`target_strength`](@ref) or
[`form_function`](@ref).
"""
struct ModalSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
end

"""
    KirchhoffSolution

Result of [`kirchhoff`](@ref): the high-frequency (physical-optics) complex scattering amplitude
`f` [m] at the angle(s) `kirchhoff` was called with. Post-process with [`target_strength`](@ref) or
[`form_function`](@ref).
"""
struct KirchhoffSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
end

"""
    FEMSolution

Result of [`fem`](@ref): either an already-computed target strength (`method=:radial`/`:meridian`
on `Sphere`/`Cylinder`/`Spheroid`, today's radial/meridian FEM implementations don't expose
reusable surface data) or, for a `Shell` body, full exterior/interior surface pressure/normal-
derivative data plus the shell's own mechanical state (mirroring the elastic-shell/fluid coupling
solve this folds in). Post-process with [`target_strength`](@ref)`(sol; angle, azimuth)` — the
`angle`/`azimuth` keywords only affect the `Shell`-body case, which is genuinely bistatic-queryable;
the scalar case ignores them (the angle was already consumed when `fem` was called).
"""
struct FEMSolution{D} <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    method::Symbol
    data::D
end

"""
    BEMSolution

Result of [`bem`](@ref): axisymmetric (`method=:axisymmetric`, `Sphere`/`Spheroid`/straight
`Cylinder`) or full 3D (`method=:full`, `Sphere`/`Spheroid` only) boundary-element surface data.
Post-process with [`target_strength`](@ref)`(sol; angle, azimuth)` (axisymmetric) or
[`target_strength`](@ref)`(sol; direction)` (full 3D).
"""
struct BEMSolution{D} <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    method::Symbol
    data::D
end

"""
    MFSSolution

Result of [`mfs`](@ref): axisymmetric (`Sphere`/`Spheroid`/straight `Cylinder`) or non-axisymmetric
(bent `Cylinder`, a genuinely different 3D point-source solve, not a correction on the axisymmetric
one) method-of-fundamental-solutions surface data. Post-process with
[`target_strength`](@ref)`(sol; angle, azimuth)` (axisymmetric) or [`target_strength`](@ref)`(sol)`
(bent).
"""
struct MFSSolution{D} <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    data::D
end

# --- `modal`: exact modal series --------------------------------------------

"""
    modal(body::AbstractBody, boundary::AbstractBoundaryCondition, k; incidence_angle=π/2, kwargs...)

Exact modal-series result, returns a [`ModalSolution`](@ref). Dispatches on `body`'s concrete type,
`Sphere` (backscatter is angle-independent, so no `incidence_angle`
keyword), `Spheroid`, or `Cylinder` (straight: plain finite-cylinder
modal series; bent, i.e. finite `radius_curvature`: automatically applies
the Fresnel bend-coherence correction, formerly `bcms_target_strength`, see [`Cylinder`](@ref)).
Post-process with [`target_strength`](@ref)`(sol)` [dB re 1 m²] or [`form_function`](@ref)`(sol)`
[m] (the complex scattering amplitude).
"""
function modal(body::Sphere, boundary::AbstractBoundaryCondition, k::Real; kwargs...)
    f = form_function(boundary, k, body.radius; kwargs...)
    return ModalSolution(body, boundary, k, f)
end

function modal(body::Spheroid, boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, kwargs...)
    f = form_function(boundary, k, body; incidence_angle = incidence_angle, kwargs...)
    return ModalSolution(body, boundary, k, f)
end

function modal(body::Cylinder,
        boundary::Union{Rigid, PressureRelease, FluidFilled,
            Shelled{ElasticLayer, FluidInterior}, SolidElastic},
        k::Real;
        incidence_angle::Real = π / 2,
        m_max::Integer = _default_mode_count(k * sin(incidence_angle) * body.radius), kwargs...)
    f_straight = form_function(boundary, k, body.radius, body.length;
        aspect_angle = incidence_angle, m_max = m_max, kwargs...)
    f = if _isbent(body)
        # Mirrors `bcms_target_strength`'s Fresnel bend-coherence correction (bent_cylinder.jl).
        Lebc = equivalent_length_fresnel(k, body.length, body.radius_curvature)
        Lebc * f_straight / body.length
    else
        f_straight
    end
    return ModalSolution(body, boundary, k, f)
end

# --- `kirchhoff`: high-frequency physical optics ----------------------------

"""
    kirchhoff(body::AbstractBody, boundary::AbstractBoundaryCondition, k; incidence_angle=π/2)

High-frequency (physical-optics) result, returns a [`KirchhoffSolution`](@ref). Same `body`-type
dispatch and bend-auto-detection as [`modal`](@ref); post-process with [`target_strength`](@ref)`(sol)`
or [`form_function`](@ref)`(sol)`.
"""
function kirchhoff(body::Sphere, boundary::AbstractBoundaryCondition, k::Real)
    # Same closed form as `kirchhoff_target_strength`, replicated here for the complex amplitude.
    x = k * body.radius
    Rc = reflection_coefficient(boundary)
    f = Rc * body.radius * (-im * cis(2x) / 2 + (cis(2x) - 1) / (4x))
    return KirchhoffSolution(body, boundary, k, f)
end

function kirchhoff(body::Sphere,
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}}, k::Real)
    # `reflection_coefficient` for a fluid shell needs (k, a), its own frequency/thickness-dependent
    # reflection, not a single constant, so this can't share the generic `AbstractBoundaryCondition` method above.
    x = k * body.radius
    Rc = reflection_coefficient(boundary, k, body.radius)
    f = Rc * body.radius * (-im * cis(2x) / 2 + (cis(2x) - 1) / (4x))
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

# --- `fem`: radial FEM+DtN and meridian FEM ---------------------------------

"""
    fem(body::AbstractBody, boundary::AbstractBoundaryCondition, k; method=:radial, R=1.2*characteristic_radius, incidence_angle=π/2, adaptive=false, kwargs...)

Finite-element result, returns a [`FEMSolution`](@ref); post-process with [`target_strength`](@ref)`(sol)`
[dB re 1 m²]. `method`:
- `:radial`, radial FEM + exact Dirichlet-to-Neumann closure (`Sphere`
  only; angle-independent, no `incidence_angle` keyword; also covers
  `SolidElastic`/`Shelled{ElasticLayer}`/`Shelled{FluidLayer}`
  boundaries on a `Sphere`, and `SolidElastic`/`Shelled{ElasticLayer}` on a
  `Cylinder`). `adaptive=true` (only for `Rigid`/`PressureRelease`/
  `FluidFilled`) self-checks by doubling mesh resolution until
  `target_tol` is met, instead of a fixed `n_elements`.
- `:meridian`, genuine 2D (ρ,z) meridian FEM, angular part discretized.
  `Sphere` supports only axial incidence (no `incidence_angle` keyword,
  matching the underlying method's own scope); `Cylinder`/`Spheroid`
  support general `incidence_angle` (default broadside).
`R` [m] is the Dirichlet-to-Neumann truncation radius, default `1.2` times
the body's own characteristic radius. See the `fem(shell::Shell, ...)` method for the
elastic-shell/fluid-coupling case (`method=:thin`/`:general`).
"""
function fem(body::Sphere, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        method::Symbol = :radial, R::Real = 1.2body.radius, adaptive::Bool = false, kwargs...)
    if method === :radial
        ts = adaptive ?
             radial_fem_target_strength_adaptive(boundary, k, body.radius, R; kwargs...) :
             radial_fem_target_strength(boundary, k, body.radius, R; kwargs...)
        return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
    end
    if method === :meridian
        adaptive &&
            throw(ArgumentError("fem(::Sphere, ...; method=:meridian) has no adaptive variant"))
        ts = meridian_fem_target_strength(boundary, k, body.radius, R; kwargs...)
        return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
    end
    throw(ArgumentError("fem(::Sphere, ...) supports method=:radial or :meridian, got $method"))
end

function fem(
        body::Sphere, boundary::SolidElastic, k::Real; method::Symbol = :radial, kwargs...)
    method === :radial ||
        throw(ArgumentError("fem(::Sphere, ::SolidElastic, ...) only supports method=:radial"))
    ts = solid_elastic_sphere_radial_fem_target_strength(k, body.radius;
        density_contrast = boundary.density_contrast,
        speed_longitudinal_contrast = boundary.speed_longitudinal_contrast,
        speed_transversal_contrast = boundary.speed_transversal_contrast, kwargs...)
    return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
end

function fem(
        body::Sphere, boundary::Shelled{ElasticLayer, FluidInterior},
        k::Real; method::Symbol = :radial, kwargs...)
    method === :radial ||
        throw(ArgumentError("fem(::Sphere, ::Shelled{ElasticLayer,FluidInterior}, ...) only supports method=:radial"))
    ts = elastic_shell_sphere_radial_fem_target_strength(k, body.radius;
        density_shell_contrast = boundary.material.density_contrast,
        speed_longitudinal_contrast = boundary.material.speed_longitudinal_contrast,
        speed_transversal_contrast = boundary.material.speed_transversal_contrast,
        radius_ratio = boundary.radius_ratio,
        density_interior_contrast = boundary.interior.density_contrast,
        soundspeed_interior_contrast = boundary.interior.soundspeed_contrast, kwargs...)
    return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
end

function fem(body::Sphere,
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}},
        k::Real; method::Symbol = :radial, kwargs...)
    method === :radial ||
        throw(ArgumentError("fem(::Sphere, ::Shelled{FluidLayer}, ...) only supports method=:radial"))
    ts = fluid_shell_sphere_radial_fem_target_strength(boundary, k, body.radius; kwargs...)
    return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
end

function fem(body::Cylinder, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        method::Symbol = :meridian, R::Real = 1.2body.radius, incidence_angle::Real = π / 2,
        m_max::Integer = _default_mode_count(k * body.radius), kwargs...)
    method === :meridian ||
        throw(ArgumentError("fem(::Cylinder, ::Union{Rigid,PressureRelease,FluidFilled}, ...) only supports method=:meridian"))
    ts = cylinder_meridian_fem_target_strength(
        boundary, k, body.radius, body.length, R, incidence_angle; m_max = m_max, kwargs...)
    return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
end

function fem(body::Cylinder,
        boundary::Union{SolidElastic, Shelled{ElasticLayer, FluidInterior}}, k::Real;
        method::Symbol = :radial, incidence_angle::Real = π / 2, kwargs...)
    method === :radial ||
        throw(ArgumentError("fem(::Cylinder, ::Union{SolidElastic,Shelled{ElasticLayer,FluidInterior}}, ...) only supports method=:radial"))
    ts = elastic_cylinder_radial_fem_target_strength(
        boundary, k, body.radius, body.length; aspect_angle = incidence_angle, kwargs...)
    return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
end

function fem(body::Spheroid, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        method::Symbol = :meridian, R::Real = 1.2max(body.a, body.b), incidence_angle::Real = π /
                                                                                              2,
        m_max::Integer = _default_mode_count(k * max(body.a, body.b)), kwargs...)
    method === :meridian ||
        throw(ArgumentError("fem(::Spheroid, ...) only supports method=:meridian"))
    ts = spheroid_meridian_fem_target_strength(
        boundary, k, body.a, body.b, R, incidence_angle; m_max = m_max, kwargs...)
    return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts))
end

# --- Shared geometry helpers (bem/mfs) --------------------------------------

_characteristic_radius(body::Sphere) = body.radius
_characteristic_radius(body::Spheroid) = max(body.a, body.b)
_characteristic_radius(body::Cylinder) = max(body.radius, body.length / 2)

function _axisymmetric_default_panels(body::AbstractBody, k::Real)
    bem_panel_count(k, _characteristic_radius(body))
end

_axisymmetric_mesh(body::Sphere, n::Integer) = sphere_mesh(body.radius, n)
_axisymmetric_mesh(body::Spheroid, n::Integer) = spheroid_mesh(body.a, body.b, n)
_axisymmetric_mesh(body::Cylinder, n::Integer) = cylinder_mesh(body.radius, body.length, n)

# --- Generic mesh interface --------------------------------------------------

"""
    Mesh

A discretized surface mesh, returned by [`mesh`](@ref) — the single public mesh type, whether the
underlying representation is an axisymmetric meridian curve (`method=:axisymmetric`) or a full 3D
triangulated surface (`method=:full`); a result's type never depends on which one produced it.
`body`/`method`/`resolution` record what `mesh(...)` was actually called with (`resolution` is
always the concrete value used, even when derived from `k` rather than passed directly) — the
dimension, coordinate system, and element type are already fully determined by `method` together
with `typeof(mesh).parameters[1]` (`MeridianMesh` for `:axisymmetric`, an `Inti.Quadrature` for
`:full`), so aren't duplicated as separate fields. Inspect with [`coordinates`](@ref),
[`normals`](@ref), [`elements`](@ref), and [`element_count`](@ref).
"""
struct Mesh{D}
    data::D
    body::AbstractBody
    method::Symbol
    resolution::Float64
end

"""
    mesh(body::AbstractBody; resolution=nothing, k=nothing, method=:axisymmetric)

Build a [`Mesh`](@ref) for `body`, consolidating `sphere_mesh`/`spheroid_mesh`/`cylinder_mesh`
(`method=:axisymmetric`, default) and `gmsh_sphere_mesh`/`gmsh_spheroid_mesh` (`method=:full`,
`Sphere`/`Spheroid` only) behind one name and one return type, the same construction `bem`/`mfs` do
internally. Exactly one of `resolution` or `k` must be given: `resolution` sets panel count
(`:axisymmetric`) or target element edge length [m] (`:full`) directly; `k` [1/m] derives a
wavenumber-appropriate default via [`bem_panel_count`](@ref)/`bem3d_elements_per_wavelength`.
"""
function mesh(body::AbstractBody; resolution::Union{Nothing, Real} = nothing,
        k::Union{Nothing, Real} = nothing, method::Symbol = :axisymmetric)
    (resolution === nothing) == (k === nothing) &&
        throw(ArgumentError("mesh(...) needs exactly one of `resolution` or `k`"))
    if method === :axisymmetric
        n = resolution === nothing ? _axisymmetric_default_panels(body, k) : Int(resolution)
        return Mesh(_axisymmetric_mesh(body, n), body, method, Float64(n))
    end
    if method === :full
        body isa Union{Sphere, Spheroid} ||
            throw(ArgumentError("mesh(...; method=:full) has no mesh generator for $(typeof(body)) in this package yet"))
        meshsize = resolution === nothing ? bem3d_elements_per_wavelength(k) :
                   Float64(resolution)
        quad = body isa Sphere ? gmsh_sphere_mesh(body.radius; meshsize = meshsize) :
               gmsh_spheroid_mesh(body.a, body.b; meshsize = meshsize)
        return Mesh(quad, body, method, meshsize)
    end
    throw(ArgumentError("mesh(...) supports method=:axisymmetric or :full, got $method"))
end

"""
    coordinates(mesh::Mesh)

Element midpoint/quadrature-node positions of `mesh`, `(rho, z)` tuples for `method=:axisymmetric`,
3-vectors for `method=:full`.
"""
coordinates(m::Mesh{MeridianMesh}) = [(p.rhom, p.zm) for p in panels(m.data)]
coordinates(m::Mesh{<:Inti.Quadrature}) = [q.coords for q in m.data]

"""
    normals(mesh::Mesh)

Outward unit normal at each element of `mesh`, same shape convention as [`coordinates`](@ref).
"""
normals(m::Mesh{MeridianMesh}) = [(p.nrho, p.nz) for p in panels(m.data)]
normals(m::Mesh{<:Inti.Quadrature}) = [q.normal for q in m.data]

"""
    elements(mesh::Mesh)

The discretization elements of `mesh`: a `Vector{Panel}` for `method=:axisymmetric`, or the
quadrature nodes themselves for `method=:full`.
"""
elements(m::Mesh{MeridianMesh}) = panels(m.data)
elements(m::Mesh{<:Inti.Quadrature}) = collect(m.data)

"""
    element_count(mesh::Mesh)

Number of discretization elements in `mesh` (replaces the former `npanels`/`length(quad)` calls).
"""
element_count(m::Mesh{MeridianMesh}) = npanels(m.data)
element_count(m::Mesh{<:Inti.Quadrature}) = length(m.data)

# --- `bem`: axisymmetric and full 3D boundary element method ---------------

"""
    bem(body::AbstractBody, boundary::AbstractBoundaryCondition, k; method=:axisymmetric, incidence_angle=π/2, kwargs...)

Boundary-element solve, returns a [`BEMSolution`](@ref)
(`method=:axisymmetric`, `Sphere`/`Spheroid`/straight `Cylinder`, or
`method=:full`, `Sphere`/`Spheroid` only). Post-
process with [`target_strength`](@ref)`(sol; angle, azimuth)` (axisymmetric) or
[`target_strength`](@ref)`(sol; direction)` (full 3D). A bent
`Cylinder` (finite `radius_curvature`) has no BEM implementation in this
package (axisymmetric or full) and raises an error rather than silently
ignoring the curvature, see [`mfs`](@ref) for the bent-cylinder case.
"""
function bem(body::Union{Sphere, Spheroid, Cylinder},
        boundary::AbstractBoundaryCondition, k::Real;
        method::Symbol = :axisymmetric, incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)), kwargs...)
    if method === :full
        return _bem_full(body, boundary, k; incidence_angle = incidence_angle,
            incidence_azimuth = incidence_azimuth, kwargs...)
    end
    method === :axisymmetric ||
        throw(ArgumentError("bem(...) supports method=:axisymmetric or :full, got $method"))
    body isa Cylinder && _isbent(body) &&
        throw(ArgumentError(
            "bem(::Cylinder, ...) has no bent-cylinder implementation (axisymmetric or full) in this package yet"))
    mesh = _axisymmetric_mesh(body, n)
    iszero(incidence_angle) && return _bem_axial(body, boundary, k, mesh; kwargs...)
    return _bem_oblique(body, boundary, k, mesh, incidence_angle; m_max = m_max, kwargs...)
end

"""
    bem(body::Sphere, boundary::Shelled{FluidLayer}, k; n=default, kwargs...)

Two-surface (outer + inner mesh) fluid-shell BEM, `boundary`'s own
`radius_ratio` field gives the inner surface's radius, `body.radius *
boundary.radius_ratio`. Axial incidence only (the underlying
`engine/shell_bem.jl` solve has no oblique variant).
"""
function bem(body::Sphere,
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}}, k::Real;
        n::Integer = _axisymmetric_default_panels(body, k), kwargs...)
    mesh_outer = sphere_mesh(body.radius, n)
    mesh_inner = sphere_mesh(body.radius * boundary.radius_ratio, n)
    p_scat, dpdn_scat, _ = solve_axial(boundary, k, mesh_outer, mesh_inner; kwargs...)
    data = _AxisymmetricSurfaceData(mesh_outer, [p_scat], [dpdn_scat], nothing, nothing)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_axial(body::AbstractBody, boundary::Union{Rigid, PressureRelease},
        k::Real, mesh::MeridianMesh; kwargs...)
    p_scat, dpdn_scat, _ = solve_axial(boundary, k, mesh; kwargs...)
    data = _AxisymmetricSurfaceData(mesh, [p_scat], [dpdn_scat], nothing, nothing)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_axial(
        body::AbstractBody, boundary::FluidFilled, k::Real, mesh::MeridianMesh; kwargs...)
    p_scat, dpdn_scat, _, p_int, dpdn_int = solve_axial(boundary, k, mesh; kwargs...)
    data = _AxisymmetricSurfaceData(mesh, [p_scat], [dpdn_scat], [p_int], [dpdn_int])
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_oblique(
        body::AbstractBody, boundary::Union{Rigid, PressureRelease, FluidFilled},
        k::Real, mesh::MeridianMesh,
        incidence_angle::Real; m_max::Integer, kwargs...)
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique(
        boundary, k, mesh, incidence_angle; m_max = m_max, kwargs...)
    data = _AxisymmetricSurfaceData(mesh, p_scat_modes, dpdn_scat_modes, nothing, nothing)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_full(body::Union{Sphere, Spheroid},
        boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        meshsize::Real = bem3d_elements_per_wavelength(k), kwargs...)
    quad = body isa Sphere ? gmsh_sphere_mesh(body.radius; meshsize = meshsize) :
           gmsh_spheroid_mesh(body.a, body.b; meshsize = meshsize)
    p_scat, dpdn_scat, _ = solve_full_bem(
        boundary, k, quad; incidence_angle = incidence_angle,
        incidence_azimuth = incidence_azimuth, kwargs...)
    data = _FullBEMSurfaceData(quad, p_scat, dpdn_scat, incidence_angle, incidence_azimuth)
    return BEMSolution(body, boundary, k, :full, data)
end
function _bem_full(body::Cylinder, boundary, k::Real; kwargs...)
    throw(ArgumentError(
        "bem(::Cylinder, ...; method=:full) has no full-3D mesh generator for Cylinder in this package yet"))
end

# --- `mfs`: axisymmetric and bent-cylinder method of fundamental solutions --

"""
    mfs(body::AbstractBody, boundary::AbstractBoundaryCondition, k; incidence_angle=π/2, offset=0.3*characteristic_radius, kwargs...)

Method-of-fundamental-solutions solve, returns an [`MFSSolution`](@ref),
axisymmetric for `Sphere`/`Spheroid`/straight `Cylinder`, or the genuinely
non-axisymmetric 3D point-source solve for a bent `Cylinder` (finite
`radius_curvature`, not a correction on the axisymmetric one).
A `Cylinder` with `endcap_depth > 0` uses the smooth-quarter-spheroid-cap
mesh instead of flat caps (see [`Cylinder`](@ref)), MFS specifically
needs this, unlike `bem`/`fem`, since flat caps give a sharp-corner
normal discontinuity that breaks source placement.
"""
function mfs(body::Union{Sphere, Spheroid, Cylinder},
        boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, offset::Real = 0.3_characteristic_radius(body),
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)), kwargs...)
    if body isa Cylinder && _isbent(body)
        return _mfs_bent(body, boundary, k; incidence_angle = incidence_angle,
            offset = offset, kwargs...)
    end
    mesh = (body isa Cylinder && _iscapped(body)) ?
           cylinder_spheroidal_endcap_mesh(body.radius, body.length, body.endcap_depth, n) :
           _axisymmetric_mesh(body, n)
    iszero(incidence_angle) &&
        return _mfs_axial(body, boundary, k, mesh; offset = offset, kwargs...)
    return _mfs_oblique(
        body, boundary, k, mesh, incidence_angle; offset = offset, m_max = m_max, kwargs...)
end

function _mfs_axial(body::AbstractBody, boundary::Union{Rigid, PressureRelease},
        k::Real, mesh::MeridianMesh; offset::Real, kwargs...)
    p_scat, dpdn_scat, _ = solve_axial_mfs(boundary, k, mesh; offset = offset, kwargs...)
    data = _AxisymmetricSurfaceData(mesh, [p_scat], [dpdn_scat], nothing, nothing)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_axial(
        body::AbstractBody, boundary::FluidFilled, k::Real, mesh::MeridianMesh; offset::Real,
        offset_ext::Real = offset, offset_int::Real = offset, kwargs...)
    p_scat, dpdn_scat, _, p_int, dpdn_int = solve_axial_mfs(
        boundary, k, mesh; offset_ext = offset_ext, offset_int = offset_int, kwargs...)
    data = _AxisymmetricSurfaceData(mesh, [p_scat], [dpdn_scat], [p_int], [dpdn_int])
    return MFSSolution(body, boundary, k, data)
end

function _mfs_oblique(body::AbstractBody, boundary::Union{Rigid, PressureRelease},
        k::Real, mesh::MeridianMesh, incidence_angle::Real;
        offset::Real, m_max::Integer, kwargs...)
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique_mfs(
        boundary, k, mesh, incidence_angle; m_max = m_max, offset = offset, kwargs...)
    data = _AxisymmetricSurfaceData(mesh, p_scat_modes, dpdn_scat_modes, nothing, nothing)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_oblique(body::AbstractBody, boundary::FluidFilled, k::Real,
        mesh::MeridianMesh, incidence_angle::Real;
        offset::Real, m_max::Integer, offset_ext::Real = offset, offset_int::Real = offset, kwargs...)
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique_mfs(
        boundary, k, mesh, incidence_angle;
        m_max = m_max, offset_ext = offset_ext, offset_int = offset_int, kwargs...)
    data = _AxisymmetricSurfaceData(mesh, p_scat_modes, dpdn_scat_modes, nothing, nothing)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_bent(body::Cylinder, boundary::Union{Rigid, PressureRelease}, k::Real;
        incidence_angle::Real = π / 2, offset::Real, n_s::Integer = 40, n_φ::Integer = 32, kwargs...)
    p_scat, dpdn_scat, points, normals, areas = solve_bent_cylinder_mfs(
        boundary, k, body.radius, body.length, body.radius_curvature;
        aspect_angle = incidence_angle, offset = offset, n_s = n_s, n_φ = n_φ, kwargs...)
    data = _BentMFSSurfaceData(p_scat, dpdn_scat, points, normals, areas, incidence_angle)
    return MFSSolution(body, boundary, k, data)
end

# --- `fem` on a `Shell` body: thin (Hayek & Boisvert) and general shell-fluid coupling -----

function _prolate_shell_geometry(s::Shell)
    s.body isa Spheroid ||
        throw(ArgumentError("this shell method requires a Spheroid-based Shell (got $(typeof(s.body)))"))
    return ProlateShellGeometry(s.body.a, s.body.b, s.thickness)
end

"""
    fem(shell::Shell, boundary::Shelled, ext_density, ext_soundspeed, int_density, int_soundspeed, k;
        method=:general, incidence_angle=π/2, kwargs...)

Elastic-shell/fluid coupling, returns a [`FEMSolution`](@ref); post-process with
[`target_strength`](@ref)`(sol; angle, azimuth)`. `boundary` is built via
[`Shelled`](@ref)`(poisson, density, youngs_modulus)`. `k`'s
own medium is the exterior fluid; `ext_density`/`ext_soundspeed` are that
fluid's absolute density [kg/m³] and sound speed [m/s] (not contrasts, to
match the underlying frequency-domain solve), `int_density`/
`int_soundspeed` the interior's, pass `int_density=0` for a vacuum/air-
backed shell (no interior coupling).

`method=:thin`, Hayek & Boisvert 1D midsurface theory. Only
`shell.body isa Spheroid` (prolate) is supported, and only axial
incidence (`incidence_angle` must be `0.0`) since the theory has no
`m > 0` degrees of freedom at all, not a missing feature, a hard
theoretical limit.

`method=:general` (default), full through-thickness 2D solid-elasticity
shell FEM, general incidence, `shell.body isa Union{Sphere,Spheroid}`,
always fluid-filled (pass a small `int_density`/`int_soundspeed` rather
than `0` if you want a near-vacuum limit, no dedicated vacuum-backed
path exists for this method).
"""
function fem(s::Shell, boundary::Shelled{ElasticFEMLayer, Nothing},
        ext_density::Real, ext_soundspeed::Real,
        int_density::Real, int_soundspeed::Real, k::Real;
        method::Symbol = :general, incidence_angle::Real = π / 2,
        m_max::Integer = _default_mode_count(k * _characteristic_radius(s.body)),
        n_eta::Integer = 65, n_t::Integer = 3, kwargs...)
    material = boundary.material
    freq_hz = k * ext_soundspeed / (2π)
    if method === :thin
        iszero(incidence_angle) || throw(ArgumentError(
            "fem(::Shell, ...; method=:thin) only supports axial incidence (Hayek & Boisvert has no m>0 " *
            "degrees of freedom), use method=:general for oblique incidence"))
        geometry = _prolate_shell_geometry(s)
        if iszero(int_density)
            p_scat, dpdn_scat, ps, shell_state, _ = solve_shell_fluid_coupled(
                geometry, material, ext_density, ext_soundspeed,
                freq_hz; n_eta = n_eta, kwargs...)
            data = _ShellFEMSurfaceData(
                ps, [p_scat], [dpdn_scat], nothing, nothing, nothing, shell_state)
            return FEMSolution(s, boundary, k, method, data)
        end
        p_ext, dpdn_ext, ps_ext, p_int, dpdn_int, ps_int,
        shell_state, _ = solve_shell_fluid_filled_coupled(
            geometry, material, ext_density, ext_soundspeed, int_density,
            int_soundspeed, freq_hz; n_eta = n_eta, kwargs...)
        data = _ShellFEMSurfaceData(
            ps_ext, [p_ext], [dpdn_ext], ps_int, [p_int], [dpdn_int], shell_state)
        return FEMSolution(s, boundary, k, method, data)
    end
    method === :general ||
        throw(ArgumentError("fem(::Shell, ...) supports method=:thin or :general, got $method"))
    p_scat_modes, dpdn_scat_modes, ps = if s.body isa Spheroid
        solve_general_shell_fluid_filled_coupled(
            _prolate_shell_geometry(s), material.density,
            material.youngs_modulus, material.poisson,
            ext_density, ext_soundspeed, int_density, int_soundspeed, freq_hz,
            incidence_angle; m_max = m_max, n_eta = n_eta, n_t = n_t, kwargs...)
    else
        mesh = build_structured_spherical_shell(s.body.radius, s.thickness, n_eta, n_t)
        solve_general_shell_fluid_filled_coupled(
            mesh, material.density, material.youngs_modulus, material.poisson,
            ext_density, ext_soundspeed, int_density, int_soundspeed,
            freq_hz, incidence_angle; m_max = m_max, kwargs...)
    end
    data = _ShellFEMSurfaceData(
        ps, p_scat_modes, dpdn_scat_modes, nothing, nothing, nothing, nothing)
    return FEMSolution(s, boundary, k, method, data)
end

# --- `target_strength`/`scattering_amplitude` on `AbstractSolution`s -----------------------
# `scattering_amplitude` is the solution-level replacement for the low-level, unexported
# `form_function(boundary, k, ...)` family: the complex amplitude [m], pre-dB-conversion.

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

# `fem`'s :radial/:meridian solvers only ever compute the final scalar (see `_ScalarFEMData`), so
# `target_strength` here also takes no keywords (see `ModalSolution` above for why), and
# `scattering_amplitude` has no result to return at all — both reject with a message naming the
# actual fix rather than silently dropping keywords or a raw MethodError naming `_ScalarFEMData`.
function target_strength(sol::FEMSolution{_ScalarFEMData}; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "target_strength(::FEMSolution) from a :radial/:meridian solve takes no keywords: the " *
        "observation angle was already fixed when fem(...) was called. Call fem(...) again with " *
        "a different `incidence_angle` to get a different result. (A `Shell`-body fem(...) result " *
        "does support `angle`/`azimuth` here — this method is for the scalar-only :radial/:meridian solvers.)"))
    return sol.data.ts
end
function scattering_amplitude(sol::FEMSolution{_ScalarFEMData}; kwargs...)
    throw(ArgumentError(
        "scattering_amplitude is not available for this FEMSolution: the :radial/:meridian solvers " *
        "only compute the final target strength, not the complex amplitude. Use target_strength(sol) " *
        "instead, or use modal(...)/kirchhoff(...) for this body/boundary combination if you need " *
        "the complex amplitude."))
end

function target_strength(sol::FEMSolution{_ShellFEMSurfaceData}; angle::Real = π, azimuth::Real = 0.0)
    return target_strength(scattering_amplitude(sol; angle = angle, azimuth = azimuth))
end

function scattering_amplitude(sol::FEMSolution{_ShellFEMSurfaceData}; angle::Real = π, azimuth::Real = 0.0)
    d = sol.data
    length(d.p_ext_modes) == 1 &&
        return far_field(d.ps_ext, d.p_ext_modes[1], d.dpdn_ext_modes[1], sol.k, angle)
    return far_field(d.ps_ext, d.p_ext_modes, d.dpdn_ext_modes, sol.k, angle, azimuth)
end

function _axisymmetric_amplitude(k::Real, d::_AxisymmetricSurfaceData; angle::Real, azimuth::Real)
    ps = panels(d.mesh)
    length(d.p_scat_modes) == 1 &&
        return far_field(ps, d.p_scat_modes[1], d.dpdn_scat_modes[1], k, angle)
    return far_field(ps, d.p_scat_modes, d.dpdn_scat_modes, k, angle, azimuth)
end

function target_strength(
        sol::Union{
            BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData}};
        angle::Real = π, azimuth::Real = 0.0)
    return target_strength(scattering_amplitude(sol; angle = angle, azimuth = azimuth))
end

function scattering_amplitude(
        sol::Union{
            BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData}};
        angle::Real = π, azimuth::Real = 0.0)
    _axisymmetric_amplitude(sol.k, sol.data; angle = angle, azimuth = azimuth)
end

function target_strength(
        sol::BEMSolution{_FullBEMSurfaceData}; direction::Union{Nothing, AbstractVector} = nothing)
    return target_strength(scattering_amplitude(sol; direction = direction))
end

function scattering_amplitude(
        sol::BEMSolution{_FullBEMSurfaceData}; direction::Union{Nothing, AbstractVector} = nothing)
    d = sol.data
    xhat = direction === nothing ?
           .-_bem3d_incidence_direction(d.incidence_angle, d.incidence_azimuth) :
           direction
    return far_field(d.quad, xhat, sol.k, d.p_scat, d.dpdn_scat)
end

function target_strength(sol::MFSSolution{_BentMFSSurfaceData})
    target_strength(scattering_amplitude(sol))
end

function scattering_amplitude(sol::MFSSolution{_BentMFSSurfaceData})
    d = sol.data
    β = d.incidence_angle
    q̂ = (-cos(β), 0.0, -sin(β))
    f = zero(ComplexF64)
    for i in eachindex(d.points)
        f += (im * sol.k * _dot3(q̂, d.normals[i]) * d.p_scat[i] + d.dpdn_scat[i]) *
             cis(-sol.k * _dot3(q̂, d.points[i])) * d.areas[i]
    end
    return f / (4π)
end
