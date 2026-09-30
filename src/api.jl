# Unified, short-named public API. Dispatches to the numerical implementations elsewhere in this
# package. Five dispatchers: `modal`, `kirchhoff`, `fem`, `bem`, `mfs`.

# --- Geometry: the `AbstractBody` hierarchy -----------------------------------------

"""
    Sphere(radius)

A sphere of positive `radius` [m]. Monostatic scattering is independent of orientation.
Sphere `modal` and `kirchhoff` calls do not take `incidence_angle`. Numerical BEM/MFS
solvers accept it to set the incident direction for directional field queries.
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

- `radius_curvature` [m]: `Inf` (default) is straight. A finite value bends the cylinder in the
  `xy` plane, with midpoint at the origin and midpoint tangent along `+x`. `modal` applies a
  near-broadside correction. `kirchhoff` and `mfs(body, ...)` describe the lateral surface only.
  Use `bem(...; method=:full)` or `mfs(mesh(...; method=:full), ...)` for closed surfaces.
- `endcap_depth` [m]: `0.0` (default) uses flat end caps. A positive value caps the cylinder
  with half-spheroid domes of that depth instead. Full BEM and straight-cylinder MFS honor
  this field. Axisymmetric BEM and FEM always use flat ends.

Full meshing requires `radius_curvature > radius`, an arc shorter than a full circle, and
nonintersecting ends. See [Closed bent cylinders](@ref bent-cylinder-tutorial) for the
bent-geometry conventions.
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

A structural shell of positive `thickness` [m] inside the outer surface of a
[`Sphere`](@ref) or [`Spheroid`](@ref). For a prolate spheroid, thickness is measured
at the equator and the inner surface is confocal. Solve with [`fem`](@ref) and a
structural [`Shelled`](@ref) material. The thickness must be smaller than the body's radius.
"""
struct Shell <: AbstractBody
    body::AbstractBody
    thickness::Float64
    function Shell(body::AbstractBody, thickness::Real)
        body isa Union{Sphere, Spheroid} ||
            throw(ArgumentError("Shell only supports a Sphere or Spheroid base body"))
        thickness > 0 || throw(ArgumentError("Shell thickness must be positive"))
        thickness < _characteristic_radius(body) || throw(ArgumentError(
            "Shell thickness ($thickness) must be less than the base body's characteristic " *
            "radius ($(_characteristic_radius(body))) — the shell FEM theory assumes a thin " *
            "shell over a solid base body, not a thickness comparable to or larger than the body itself"))
        return new(body, Float64(thickness))
    end
end

# --- Solutions: one concrete type per dispatcher, so a result's own type tells you which solver
# produced it. Internal `data` shapes are reused across sub-methods (e.g. axisymmetric bem/mfs)
# purely as an implementation detail; that reuse never surfaces as a shared public type. -----------

"""
    AbstractSolution

Supertype of results returned by [`modal`](@ref), [`kirchhoff`](@ref), [`fem`](@ref),
[`bem`](@ref) and [`mfs`](@ref). Query [`target_strength`](@ref),
[`scattering_amplitude`](@ref) and [`diagnostics`](@ref) where available.
Construct solutions through their solver rather than their internal storage fields.
"""
abstract type AbstractSolution end

# Reusable axisymmetric surface-field data (bem/mfs axisymmetric solves), an internal component
# type shared by `BEMSolution`/`MFSSolution`, never exposed as its own public type.
struct _AxisymmetricSurfaceData
    mesh::MeridianMesh
    p_scat_modes::Vector{Vector{ComplexF64}}
    dpdn_scat_modes::Vector{Vector{ComplexF64}}
    p_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    dpdn_int_modes::Union{Nothing, Vector{Vector{ComplexF64}}}
    incidence_angle::Float64
    diagnostics::NamedTuple
    source_modes::Union{Nothing, Vector{NamedTuple}}
end

function _AxisymmetricSurfaceData(mesh, p, dp, pi, dpi, beta, report)
    _AxisymmetricSurfaceData(mesh, p, dp, pi, dpi, beta, report, nothing)
end

# Full 3D (non-axisymmetric) BEM surface-field data, `bem(...; method=:full)` only.
struct _FullBEMSurfaceData
    quad::Any
    p_scat::Vector{ComplexF64}
    dpdn_scat::Vector{ComplexF64}
    incidence_angle::Float64
    incidence_azimuth::Float64
    diagnostics::NamedTuple
    single_layer_density::Union{Nothing, Vector{ComplexF64}}
    incident::Union{Nothing, IncidentField}
end

function _FullBEMSurfaceData(quad, p, dp, beta, alpha, report, density)
    _FullBEMSurfaceData(quad, p, dp, beta, alpha, report, density, nothing)
end

function _FullBEMSurfaceData(quad, p, dp, beta, alpha, report)
    _FullBEMSurfaceData(quad, p, dp, beta, alpha, report, nothing)
end

# Non-axisymmetric bent-cylinder MFS surface-field data, `mfs` on a bent `Cylinder` only.
struct _BentMFSSurfaceData
    p_scat::Vector{ComplexF64}
    dpdn_scat::Vector{ComplexF64}
    points::Vector{NTuple{3, Float64}}
    normals::Vector{NTuple{3, Float64}}
    areas::Vector{Float64}
    incidence_angle::Float64
    diagnostics::NamedTuple
end

# FEM paths that retain only target strength.
struct _ScalarFEMData
    ts::Float64
    diagnostics::NamedTuple
end

"""
    _RadialFEMData

Per-degree spherical radial fields for unit incident pressure. `coefficient`
multiplies the outgoing Hankel function and Legendre polynomial, including the
plane-wave factor `(2l+1)i^l`. Fluid shell nodal pressures and regular interior
pressure coefficients describe total pressure. Interior coefficients multiply
`j_l(k_in*r)`. Exterior coefficients describe scattered pressure.

Elastic nodal `longitudinal` and `shear` values are the displacement potentials
scaled by `rho_ext*omega^2/p_inc`, including the same plane-wave factor. Both are
dimensionless. With their Legendre factors restored, the scaled displacement is
`grad(longitudinal) + curl(curl(r*shear*e_r))`. The monopole has no shear potential.
Radii are in meters; material contrasts and the exterior wavenumber are stored
in the enclosing solution. [`pressure`](@ref) evaluates acoustic pressure in fluid
regions; elastic stress and displacement evaluation is unavailable.
"""
struct _RadialFEMData
    modes::Vector{NamedTuple}
    diagnostics::NamedTuple
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
    incidence_angle::Float64
    diagnostics::NamedTuple
end

# Cylinder meridian FEM (rigid/pressure-release/fluid-filled): panels and per-Fourier-mode
# complex surface traces at r=R, mirroring `_ShellFEMSurfaceData`'s exterior fields. Reduces to
# `far_field`/`target_strength(ps, p_modes, dpdn_modes, k, angle, azimuth)` at any observation
# direction, not only the backscatter this path's own `_ScalarFEMData` predecessor retained.
struct _CylinderMeridianFEMData
    ps::Vector{Panel}
    p_modes::Vector{Vector{ComplexF64}}
    dpdn_modes::Vector{Vector{ComplexF64}}
    incidence_angle::Float64
    diagnostics::NamedTuple
end

# Spheroid meridian FEM retains the same exterior traces on its enclosing
# spherical DtN boundary. The traces support complex far-field observations.
struct _SpheroidMeridianFEMData
    ps::Vector{Panel}
    p_modes::Vector{Vector{ComplexF64}}
    dpdn_modes::Vector{Vector{ComplexF64}}
    incidence_angle::Float64
    diagnostics::NamedTuple
end

# Elastic cylinder radial FEM: raw per-azimuthal-mode coefficients from `_raw_bn_radial_fem`,
# and the axial geometry (`length`, `aspect_angle`) needed to reconstruct the backscatter
# amplitude via `_elastic_cylinder_radial_fem_amplitude`. Backscatter only, unlike the
# meridian path above: the Fraunhofer axial envelope this reduction uses has no general
# bistatic form.
struct _CylinderRadialFEMData
    modes::Vector{ComplexF64}
    length::Float64
    aspect_angle::Float64
    diagnostics::NamedTuple
end

struct _SphereModalData
    coefficients::Vector{ComplexF64}
    interior_coefficients::Union{Nothing, Vector{ComplexF64}}
    shell_coefficients::Union{Nothing, Vector{NTuple{2, ComplexF64}}}
    incident::IncidentField
end

"""
    ModalSolution

Result of [`modal`](@ref): the modal-series complex scattering amplitude `f` [m] at the
angle(s) `modal` was called with. Post-process with [`target_strength`](@ref) or
[`scattering_amplitude`](@ref). Supported spheres also retain coefficients for [`pressure`](@ref)
evaluation with a unit plane wave traveling along +x; `pressure` is not available for a
[`SphericalWave`](@ref) or [`BesselBeam`](@ref) incident field. Finite-cylinder and
bent-cylinder paths include approximations.
"""
struct ModalSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
    data::Union{Nothing, _SphereModalData}
end

ModalSolution(body, boundary, k, f) = ModalSolution(body, boundary, k, f, nothing)

"""
    TMatrixSolution

Result of [`tmatrix`](@ref): the complex scattering amplitude `f` [m] of an elastic spheroid or
shell at the incident and observation directions `tmatrix` was called with. Post-process with
[`target_strength`](@ref) or [`scattering_amplitude`](@ref).
"""
struct TMatrixSolution <: AbstractSolution
    body::Spheroid
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
end

"""
    KirchhoffSolution

Result of [`kirchhoff`](@ref): the high-frequency (physical-optics) complex scattering amplitude
`f` [m] at the angle(s) `kirchhoff` was called with. Post-process with [`target_strength`](@ref) or
[`scattering_amplitude`](@ref).
"""
struct KirchhoffSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    f::ComplexF64
end

"""
    FEMSolution

Result of [`fem`](@ref). Supported radial spheres retain complex backscatter
coefficients and radial pressure or elastic potential fields.
Post-process with [`scattering_amplitude`](@ref) or [`target_strength`](@ref).
For supported radial spheres, [`pressure`](@ref) samples acoustic pressure in the
exterior, fluid shells and fluid interiors at Cartesian points.
Structural `Shell` results retain surface traces and support observation `angle`/`azimuth`.
Cylinder radial and meridian paths retain target strength only. Observation keywords
are rejected outside the structural `Shell` case.
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
`Cylinder`) or full 3D (`method=:full`) boundary-element surface data, including supplied
closed surfaces and coupled fluid regions.
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

Result of [`mfs`](@ref). Axisymmetric body solves use
[`target_strength`](@ref)`(sol; angle, azimuth)`. Full closed-surface solves use
`target_strength(sol; direction)`, defaulting to backscatter. The lateral-only bent-body
overload returns a monostatic result, evaluated with `target_strength(sol)`.
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

Modal-series result, returns a [`ModalSolution`](@ref). Dispatches on `body`'s concrete type.
`Sphere` backscatter is angle-independent, with no `incidence_angle` keyword, but accepts
`incident::IncidentField=PlaneWave()` (see [`SphericalWave`](@ref) and [`BesselBeam`](@ref));
`angle` is then measured from the incident field's own axis. A bent `Cylinder` automatically
applies the Fresnel bend-coherence correction (see [`Cylinder`](@ref)).
Post-process with [`target_strength`](@ref)`(sol)` in dB re 1 m² or [`scattering_amplitude`](@ref)`(sol)`,
the complex scattering amplitude in m.
"""
function modal(body::Sphere, boundary::AbstractBoundaryCondition, k::Real; kwargs...)
    f = form_function(boundary, k, body.radius; kwargs...)
    return ModalSolution(body, boundary, k, f)
end

function modal(body::Sphere,
        boundary::Union{Rigid, PressureRelease, Impedance, FluidFilled,
            SolidElastic, Shelled{ElasticLayer, FluidInterior},
            Shelled{FluidLayer, FluidInterior}, Shelled{FluidLayer, VacuumInterior}},
        k::Real;
        angle::Real = π, m_max::Integer = _default_mode_count(k * body.radius),
        incident::IncidentField = PlaneWave())
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    m_max >= 0 || throw(ArgumentError("m_max must be nonnegative"))
    coefficients = ComplexF64[]
    has_interior = boundary isa FluidFilled ||
                   (boundary isa Shelled && boundary.interior isa FluidInterior)
    interior = has_interior ? ComplexF64[] : nothing
    shell = boundary isa Shelled{FluidLayer} ? NTuple{2, ComplexF64}[] : nothing
    total = zero(ComplexF64)
    for l in 0:m_max
        prefactor = (2l + 1) * _incident_coefficient(incident, l, k)
        if boundary isa FluidFilled
            mode = _sphere_fluid_coefficients(boundary, l, k, body.radius)
            coefficient = mode.scattered
            push!(interior, prefactor * mode.interior)
        elseif boundary isa Shelled
            mode = _sphere_shell_coefficients(boundary, l, k, body.radius)
            coefficient = mode.scattered
            interior === nothing || push!(interior, prefactor * mode.interior)
            shell === nothing || push!(shell, prefactor .* mode.shell)
        else
            coefficient = _modal_coefficient(boundary, l, k, body.radius)
        end
        push!(coefficients, prefactor * coefficient)
        total += prefactor * (-im)^(l + 1) * legendre_p(l, cos(angle)) * coefficient
    end
    return ModalSolution(body, boundary, k, total / k,
        _SphereModalData(coefficients, interior, shell, incident))
end

function modal(body::Spheroid, boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, kwargs...)
    f = form_function(boundary, k, body; incidence_angle = incidence_angle, kwargs...)
    return ModalSolution(body, boundary, k, f)
end

function modal(::Spheroid, ::_ElasticSpheroidBoundary, ::Real; kwargs...)
    throw(ArgumentError("elastic spheroids and shells are solved by `tmatrix`, not `modal`"))
end

# --- `tmatrix`: transition-matrix solution for elastic spheroids ------------

"""
    tmatrix(body::Spheroid, boundary, k; incidence_angle=π/2, incidence_azimuth=0,
        scatter_angle=π-incidence_angle, scatter_azimuth=incidence_azimuth+π, m_max, n_max, check=true)

Transition-matrix scattering by a prolate or oblate spheroid, returns a [`TMatrixSolution`](@ref).
`boundary` is a `SolidElastic` solid or `Shelled(ElasticLayer(...), FluidInterior(...) or VacuumInterior(),
radius_ratio)`, an elastic shell whose confocal inner surface has an equatorial semi-axis
`radius_ratio` times the outer one. Angles are in radians from the axis of symmetry, and the defaults
give backscatter. `m_max` and `n_max` truncate the azimuthal orders and degrees, and a shell solve
is repeated with both reduced by 2 and warns when the amplitude changes by more than 1%. Pass
`check = false` to skip the repeat. Post-process with [`target_strength`](@ref) or
[`scattering_amplitude`](@ref).

See [Transition-matrix solutions](@ref tmatrix-theory).
"""
function tmatrix(body::Spheroid, boundary::_ElasticSpheroidBoundary, k::Real;
        incidence_angle::Real = π / 2, kwargs...)
    f = form_function(boundary, k, body; incidence_angle, kwargs...)
    return TMatrixSolution(body, boundary, k, f)
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
dispatch and bend-auto-detection as [`modal`](@ref). Post-process with [`target_strength`](@ref)`(sol)`
or [`scattering_amplitude`](@ref)`(sol)`.
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

function _fem_volume_solution(body, boundary, k::Real; kwargs...)
    return FEMSolution(
        body, boundary, k, :volume, _fem_volume(body, boundary, k; kwargs...))
end

"""
    fem(body::AbstractBody, boundary::AbstractBoundaryCondition, k; method=:radial, R=1.2*characteristic_radius, incidence_angle=π/2, adaptive=false, kwargs...)

Finite-element result, returns a [`FEMSolution`](@ref). Post-process with
[`target_strength`](@ref)`(sol)` in dB re 1 m².

`method`:
- `:radial`: radial FEM with an exact Dirichlet-to-Neumann closure. `Sphere` only, angle
  independent. Covers `Rigid`, `PressureRelease`, `FluidFilled`, `SolidElastic`, and elastic or
  fluid `Shelled` boundaries on a `Sphere`, plus `SolidElastic`/elastic `Shelled` on a `Cylinder`.
  `adaptive=true` refines `n_elements` until `target_tol` is met (`Rigid`/`PressureRelease`/
  `FluidFilled` only).
- `:meridian`: 2D (ρ,z) meridian FEM with the angular part discretized. `Sphere` supports only
  axial incidence. `Cylinder`/`Spheroid` support general `incidence_angle`.

- `:volume`: full-3D finite elements on curved order-2 tetrahedra. Covers `Sphere` and `Spheroid`
  with `Rigid`, `PressureRelease`, `FluidFilled`, `SolidElastic` and elastic `Shelled` boundaries,
  with a fluid or empty interior. The default `closure = :auto` uses the exact Dirichlet-to-Neumann map
  on a sphere or a perfectly matched layer, whichever gives the smaller domain, and `:dtn`, `:pml`,
  `:pml_spherical` and `:pml_spheroidal` force a closure. Accepts `incidence_angle`,
  `incidence_azimuth`, `points_per_wavelength`, `domain_radius`, `clearance`, `pml_thickness`,
  `pml_sigma`, and `solver` (`:auto`, `:direct` or `:iterative`), and supports `angle`/`azimuth` in
  [`scattering_amplitude`](@ref).

`R` is the Dirichlet-to-Neumann truncation radius in m, default `1.2` times the body's
characteristic radius. See `fem(shell::Shell, ...)` for the elastic-shell/fluid-coupling case.

Supported radial spheres also support [`scattering_amplitude`](@ref)`(sol)` in complex meters.
See [FEM and shell coupling](@ref fem-theory) for field normalization and per-boundary support.
"""
function fem(body::Sphere, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        method::Symbol = :radial, R::Real = 1.2body.radius, adaptive::Bool = false, kwargs...)
    method === :volume && return _fem_volume_solution(body, boundary, k; kwargs...)
    reports = _SolveReports()
    if method === :radial
        modes = adaptive ?
                _radial_fem_modes_adaptive(
            boundary, k, body.radius, R; solve_reports = reports, kwargs...) :
                _radial_fem_modes(
            boundary, k, body.radius, R; solve_reports = reports, kwargs...)
        report = _summarize_solves(reports; method, solver_options = (;
            R, adaptive, kwargs...))
        return FEMSolution(body, boundary, k, method, _RadialFEMData(modes, report))
    end
    if method === :meridian
        adaptive &&
            throw(ArgumentError("fem(::Sphere, ...; method=:meridian) has no adaptive variant"))
        ts = meridian_fem_target_strength(
            boundary, k, body.radius, R; solve_reports = reports, kwargs...)
        report = _summarize_solves(reports; method, solver_options = (; R, kwargs...))
        return FEMSolution(body, boundary, k, method, _ScalarFEMData(ts, report))
    end
    throw(ArgumentError("fem(::Sphere, ...) supports method=:radial or :meridian, got $method"))
end

function fem(
        body::Sphere, boundary::SolidElastic, k::Real; method::Symbol = :radial, kwargs...)
    method === :volume && return _fem_volume_solution(body, boundary, k; kwargs...)
    method === :radial || throw(ArgumentError(
        "fem(::Sphere, ::SolidElastic, ...) supports method=:radial or :volume, got $method"))
    reports = _SolveReports()
    modes = _solid_elastic_sphere_radial_fem_modes(
        k, body.radius; solve_reports = reports,
        density_contrast = boundary.density_contrast,
        speed_longitudinal_contrast = boundary.speed_longitudinal_contrast,
        speed_transversal_contrast = boundary.speed_transversal_contrast, kwargs...)
    report = _summarize_solves(reports; method, solver_options = (; kwargs...))
    return FEMSolution(body, boundary, k, method, _RadialFEMData(modes, report))
end

function fem(
        body::Sphere, boundary::Shelled{ElasticLayer, FluidInterior},
        k::Real; method::Symbol = :radial, kwargs...)
    method === :volume && return _fem_volume_solution(body, boundary, k; kwargs...)
    method === :radial || throw(ArgumentError(
        "fem(::Sphere, ::Shelled{ElasticLayer,FluidInterior}, ...) supports method=:radial or :volume, got $method"))
    reports = _SolveReports()
    identical_fluid = boundary.material.interior_coupling === :identical_fluid
    modes = _elastic_shell_sphere_radial_fem_modes(
        k, body.radius; solve_reports = reports,
        density_shell_contrast = boundary.material.density_contrast,
        speed_longitudinal_contrast = boundary.material.speed_longitudinal_contrast,
        speed_transversal_contrast = boundary.material.speed_transversal_contrast,
        radius_ratio = boundary.radius_ratio,
        density_interior_contrast = identical_fluid ? 1.0 :
                                    boundary.interior.density_contrast,
        soundspeed_interior_contrast = identical_fluid ? 1.0 :
                                       boundary.interior.soundspeed_contrast, kwargs...)
    report = _summarize_solves(reports; method, solver_options = (; kwargs...))
    return FEMSolution(body, boundary, k, method, _RadialFEMData(modes, report))
end

function fem(body::Sphere,
        boundary::Union{
            Shelled{FluidLayer, VacuumInterior}, Shelled{FluidLayer, FluidInterior}},
        k::Real; method::Symbol = :radial, kwargs...)
    method === :radial ||
        throw(ArgumentError("fem(::Sphere, ::Shelled{FluidLayer}, ...) only supports method=:radial"))
    reports = _SolveReports()
    modes = _fluid_shell_sphere_radial_fem_modes(boundary, k, body.radius;
        solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method, solver_options = (; kwargs...))
    return FEMSolution(body, boundary, k, method, _RadialFEMData(modes, report))
end

function fem(body::Cylinder, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        method::Symbol = :meridian, R::Real = 1.2body.radius, incidence_angle::Real = π / 2,
        m_max::Integer = _default_mode_count(k * body.radius), kwargs...)
    _isbent(body) && throw(ArgumentError(
        "fem(::Cylinder, ...) has no bent-cylinder implementation in this package yet"))
    method === :meridian ||
        throw(ArgumentError("fem(::Cylinder, ::Union{Rigid,PressureRelease,FluidFilled}, ...) only supports method=:meridian"))
    reports = _SolveReports()
    ps, p_modes, dpdn_modes = _cylinder_meridian_fem_modes(
        boundary, k, body.radius, body.length, R, incidence_angle;
        m_max, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method, solver_options = (;
        R, incidence_angle, m_max, kwargs...))
    data = _CylinderMeridianFEMData(ps, p_modes, dpdn_modes, incidence_angle, report)
    return FEMSolution(body, boundary, k, method, data)
end

function fem(body::Cylinder,
        boundary::Union{SolidElastic, Shelled{ElasticLayer, FluidInterior}}, k::Real;
        method::Symbol = :radial, incidence_angle::Real = π / 2, kwargs...)
    _isbent(body) && throw(ArgumentError(
        "fem(::Cylinder, ...) has no bent-cylinder implementation in this package yet"))
    method === :radial ||
        throw(ArgumentError("fem(::Cylinder, ::Union{SolidElastic,Shelled{ElasticLayer,FluidInterior}}, ...) only supports method=:radial"))
    reports = _SolveReports()
    modes = _elastic_cylinder_radial_fem_modes(boundary, k, body.radius;
        aspect_angle = incidence_angle, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method, solver_options = (;
        incidence_angle, kwargs...))
    data = _CylinderRadialFEMData(modes, body.length, incidence_angle, report)
    return FEMSolution(body, boundary, k, method, data)
end

function fem(body::Spheroid, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        method::Symbol = :meridian, R::Real = 1.2max(body.a, body.b), incidence_angle::Real = π /
                                                                                              2,
        m_max::Integer = _default_mode_count(k * max(body.a, body.b)), kwargs...)
    method === :volume &&
        return _fem_volume_solution(body, boundary, k; incidence_angle, kwargs...)
    method === :meridian || throw(ArgumentError(
        "fem(::Spheroid, ...) supports method=:meridian or :volume, got $method"))
    reports = _SolveReports()
    ps, p_modes, dpdn_modes = _spheroid_meridian_fem_modes(
        boundary, k, body.a, body.b, R, incidence_angle; m_max, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method, solver_options = (;
        R, incidence_angle, m_max, kwargs...))
    data = _SpheroidMeridianFEMData(ps, p_modes, dpdn_modes, incidence_angle, report)
    return FEMSolution(body, boundary, k, method, data)
end

struct _VolumeRegionGeometry <: AbstractBody
    bodies::Vector{AbstractBody}
    centers::Vector{Vector{Float64}}
    orientations::Vector{Vector{Float64}}
    parents::Vector{Int}
end

const _VolumeRegionMaterial = Union{
    FluidFilled, SolidElastic, ViscoelasticSolid, ViscousLayer}

struct _VolumeRegionMaterials <: AbstractBoundaryCondition
    materials::Vector{Any}
end

"""
    fem(bodies::AbstractVector{<:Union{Sphere,Spheroid}}, materials::AbstractVector, k;
        method=:volume, parents=collect(0:length(bodies)-1), centers, orientations,
        incidence_angle=π/2, incidence_azimuth=0, kwargs...)

Full-3D volume FEM of coupled fluid and elastic regions, for example a fish body with a gas-filled
swimbladder and a bony backbone. Region `i` is `bodies[i]` with the contrasts of `materials[i]`
relative to the unbounded exterior. `parents[i]` is the region immediately outside it, with `0` for the
exterior, and the default is a nested chain. `centers[i]` is the region's center and
`orientations[i]` the direction of its symmetry axis, in the frame whose origin is the phase
reference of the far field. Regions may be nested or disjoint but must not partially overlap.
The keywords of the single-body `method = :volume` apply, except `closure = :pml_spheroidal`.

Returns a [`FEMSolution`](@ref). Post-process with `scattering_amplitude(sol; angle, azimuth)`
or `target_strength(sol; angle, azimuth)`.
"""
function fem(bodies::AbstractVector{<:AbstractBody},
        materials::AbstractVector, k::Real; method::Symbol = :volume,
        parents::AbstractVector{<:Integer} = collect(0:(length(bodies) - 1)),
        centers = [zeros(3) for _ in bodies],
        orientations = [[0.0, 0.0, 1.0] for _ in bodies], kwargs...)
    method === :volume || throw(ArgumentError(
        "fem(::AbstractVector, ::AbstractVector, ...) only supports method=:volume"))
    all(b -> b isa Union{Sphere, Spheroid}, bodies) || throw(ArgumentError(
        "region bodies must be Sphere or Spheroid"))
    all(m -> m isa _VolumeRegionMaterial, materials) || throw(ArgumentError(
        "region materials must be FluidFilled, SolidElastic, ViscoelasticSolid or ViscousLayer"))
    data = _fem_volume_regions(
        bodies, materials, k; parents, centers, orientations, kwargs...)
    geometry = _VolumeRegionGeometry(collect(AbstractBody, bodies),
        [Float64.(c) for c in centers], [Float64.(o) for o in orientations],
        collect(Int, parents))
    return FEMSolution(
        geometry, _VolumeRegionMaterials(collect(Any, materials)), k,
        :volume, data)
end

"""
    FreeSurfaceSolution

Result of [`free_surface`](@ref): the volume FEM solutions of a body driven by the direct incident
wave and by its interface-reflected image, held separately with the interface's reflection sign.
Post-process with [`target_strength`](@ref), [`scattering_amplitude`](@ref) or [`pressure`](@ref).
"""
struct FreeSurfaceSolution <: AbstractSolution
    direct::FEMSolution
    reflected::FEMSolution
    sign::Int
    incidence_angle::Float64
    incidence_azimuth::Float64
end

"""
    free_surface(body::Union{Sphere,Spheroid}, boundary, k, depth; condition=:pressure_release,
        incidence_angle=0, incidence_azimuth=0, orientation=[0,0,1], kwargs...)

Volume FEM scattering by `body`, centered on the normal of an infinite planar interface and
submerged a `depth` [m] below it. `condition = :pressure_release` (default) is an air-water free
surface, `condition = :rigid` an idealized seafloor. `incidence_angle = 0` and `incidence_azimuth`
give the direct wave's direction along the interface's outward normal, in the convention of
`fem(...; method = :volume)`. `boundary` is a region material (`FluidFilled`, `SolidElastic`,
`ViscoelasticSolid` or `ViscousLayer`).

Solved exactly by the method of images: `body` and a mirror image across the interface form a
coupled two-body volume FEM in an unbounded fluid, solved once for the direct wave and once for its
specular reflection, and added with the interface's reflection coefficient (`-1` for
`:pressure_release`, `+1` for `:rigid`). Accepts the keywords of `fem(...; method = :volume)` for
coupled regions. Returns a [`FreeSurfaceSolution`](@ref).
"""
function free_surface(
        body::Union{Sphere, Spheroid}, boundary::_VolumeRegionMaterial, k::Real,
        depth::Real; condition::Symbol = :pressure_release, incidence_angle::Real = 0.0,
        incidence_azimuth::Real = 0.0, orientation = [0.0, 0.0, 1.0], kwargs...)
    condition in (:pressure_release, :rigid) || throw(ArgumentError(
        "condition must be :pressure_release or :rigid, got $condition"))
    isfinite(depth) && depth > 0 ||
        throw(ArgumentError("depth must be finite and positive"))
    sign = condition === :pressure_release ? -1 : 1
    centers = [[0.0, 0.0, -depth], [0.0, 0.0, depth]]
    orientations = [Float64.(orientation), [1.0, 1.0, -1.0] .* Float64.(orientation)]
    system = _volume_system_regions([body, body], [boundary, boundary], k;
        parents = [0, 0], centers, orientations, kwargs...)
    geometry = _VolumeRegionGeometry([body, body], centers, orientations, [0, 0])
    materials = _VolumeRegionMaterials([boundary, boundary])
    solve(angle) = FEMSolution(geometry, materials, Float64(k), :volume,
        _volume_solution(system, angle, incidence_azimuth))
    return FreeSurfaceSolution(solve(incidence_angle), solve(π - incidence_angle), sign,
        Float64(incidence_angle), Float64(incidence_azimuth))
end

function fem(body::Sphere, boundary::Shelled{ElasticLayer, VacuumInterior}, k::Real;
        method::Symbol = :volume, kwargs...)
    method === :volume || throw(ArgumentError(
        "fem(::Sphere, ::Shelled{ElasticLayer,VacuumInterior}, ...) only supports method=:volume"))
    return _fem_volume_solution(body, boundary, k; kwargs...)
end

function fem(body::Spheroid,
        boundary::Union{SolidElastic,
            Shelled{ElasticLayer, <:Union{FluidInterior, VacuumInterior}}},
        k::Real; method::Symbol = :volume, incidence_angle::Real = π / 2, kwargs...)
    method === :volume || throw(ArgumentError(
        "fem(::Spheroid, ::Union{SolidElastic,Shelled{ElasticLayer}}, ...) only supports method=:volume"))
    return _fem_volume_solution(body, boundary, k; incidence_angle, kwargs...)
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

A discretized surface mesh, returned by [`mesh`](@ref). The single public mesh type, whether the
underlying representation is an axisymmetric meridian curve (`method=:axisymmetric`) or a full 3D
triangulated surface (`method=:full`). `body`/`method`/`resolution` record what `mesh(...)` was
called with. `resolution` is the maximum corner-edge length in meters. Inspect the surface with
`coordinates`, `normals`, `elements`, and `element_count`.
"""
struct Mesh{D}
    data::D
    body::AbstractBody
    method::Symbol
    resolution::Float64
end

"""
    mesh(body::AbstractBody; resolution=nothing, k=nothing, method=:axisymmetric,
         qorder=4, mesh_order=2)

Build a [`Mesh`](@ref) for `body`, the same construction `bem`/`mfs` use internally. Exactly one
of `resolution` or `k` must be given. `resolution` sets panel count (`:axisymmetric`) or target
element edge length in m (`:full`) directly. `k` in 1/m derives a wavenumber-appropriate default.
For full surfaces, `mesh_order` (1, 2 or 3) controls triangle geometry and `qorder` controls
quadrature. Supplied surfaces use `mesh(path)`, `mesh(generate)` or `mesh(nodes, triangles)`.
"""
function mesh(body::AbstractBody; resolution::Union{Nothing, Real} = nothing,
        k::Union{Nothing, Real} = nothing, method::Symbol = :axisymmetric,
        qorder::Integer = 4, mesh_order::Integer = 2)
    (resolution === nothing) == (k === nothing) &&
        throw(ArgumentError("mesh(...) needs exactly one of `resolution` or `k`"))
    if method === :axisymmetric
        body isa Cylinder && _isbent(body) &&
            throw(ArgumentError(
                "a bent Cylinder requires mesh(...; method=:full)"))
        n = resolution === nothing ? _axisymmetric_default_panels(body, k) : Int(resolution)
        return Mesh(_axisymmetric_mesh(body, n), body, method, Float64(n))
    end
    if method === :full
        body isa Union{Sphere, Spheroid, Cylinder} ||
            throw(ArgumentError("mesh(...; method=:full) has no mesh generator for $(typeof(body)) in this package yet"))
        meshsize = resolution === nothing ? bem3d_elements_per_wavelength(k) :
                   Float64(resolution)
        if body isa Cylinder
            surface = _cylinder_full_mesh(body; meshsize, qorder, mesh_order)
            return Mesh(surface.data, body, method, meshsize)
        end
        quad = body isa Sphere ?
               gmsh_sphere_mesh(body.radius; meshsize, qorder, mesh_order) :
               gmsh_spheroid_mesh(body.a, body.b; meshsize, qorder, mesh_order)
        return Mesh(quad, body, method, meshsize)
    end
    throw(ArgumentError("mesh(...) supports method=:axisymmetric or :full, got $method"))
end

"""
    coordinates(mesh::Mesh)

Element midpoint/quadrature-node positions of `mesh`: `(rho, x)` radial/axial tuples
for `method=:axisymmetric`, Cartesian `(x,y,z)` 3-vectors for `method=:full`.
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

Boundary-element solve, returns a [`BEMSolution`](@ref). `method=:axisymmetric` supports
`Sphere`/`Spheroid`/straight `Cylinder`. `method=:full` supports a generated
`Sphere`/`Spheroid`/`Cylinder` or a supplied `Mesh`, and is required for a bent `Cylinder`.
Post-process with [`target_strength`](@ref)`(sol; angle, azimuth)` (axisymmetric) or
[`target_strength`](@ref)`(sol; direction)` (full 3D).

Full BEM accepts `meshsize` in m, geometry `mesh_order` (1, 2 or 3, default 2), quadrature
`qorder` (default 4), `correction`, `compression` and `gmres_kwargs`. Rigid/soft full BEM
defaults to `formulation=:burton_miller` (`:cbie` selects the conventional equation). Fluid
full BEM defaults to `formulation=:muller` (`:cbie` selects the pressure-only system
with interior traces eliminated).
`equilibrate=true` scales the matrix before factorization. `condition_limit=512` bounds the
optional SVD condition-number calculation.
Single-interface fluid Müller also accepts `compression=(method=:hmatrix, tol=1e-8)`
with density interpolation, using block-preconditioned GMRES. It scales from local blocks,
skips SVD condition estimates and accepts `gmres_kwargs`; dense LU remains the default.

See [BEM and MFS](@ref boundary-theory) for the underlying formulations and
[`diagnostics`](@ref) for convergence and residual checks.
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
            "a bent Cylinder requires bem(...; method=:full)"))
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
    reports = _SolveReports()
    p_scat, dpdn_scat, _ = solve_axial(boundary, k, mesh_outer, mesh_inner;
        solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric, solver_options = (;
        n, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh_outer, [p_scat], [dpdn_scat], nothing, nothing, 0.0, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_axial(body::AbstractBody, boundary::Union{Rigid, PressureRelease, Impedance},
        k::Real, mesh::MeridianMesh; kwargs...)
    reports = _SolveReports()
    p_scat, dpdn_scat, _ = solve_axial(
        boundary, k, mesh; solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(mesh), incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], nothing, nothing, 0.0, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_axial(
        body::AbstractBody, boundary::FluidFilled, k::Real, mesh::MeridianMesh; kwargs...)
    reports = _SolveReports()
    p_scat, dpdn_scat, _, p_int, dpdn_int = solve_axial(
        boundary, k, mesh; solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(mesh), incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], [p_int], [dpdn_int], 0.0, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_oblique(
        body::AbstractBody, boundary::Union{Rigid, PressureRelease, FluidFilled, Impedance},
        k::Real, mesh::MeridianMesh,
        incidence_angle::Real; m_max::Integer, kwargs...)
    reports = _SolveReports()
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique(
        boundary, k, mesh, incidence_angle; m_max, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(mesh), incidence_angle, m_max, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, p_scat_modes, dpdn_scat_modes, nothing, nothing, incidence_angle, report)
    return BEMSolution(body, boundary, k, :axisymmetric, data)
end

function _bem_full(body::Union{Sphere, Spheroid},
        boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        incident = nothing,
        meshsize::Real = bem3d_elements_per_wavelength(k), qorder::Integer = 4,
        mesh_order::Integer = 2, kwargs...)
    incident = _resolve_incident(
        k, incidence_angle, incidence_azimuth; incident)
    quad = body isa Sphere ?
           gmsh_sphere_mesh(body.radius; meshsize, qorder, mesh_order) :
           gmsh_spheroid_mesh(body.a, body.b; meshsize, qorder, mesh_order)
    density = Ref{Union{Nothing, Vector{ComplexF64}}}(nothing)
    capture = boundary isa Rigid ? (; _density = density) : (;)
    p_scat, dpdn_scat, _, diagnostics = solve_full_bem(
        boundary, k, quad; incidence_angle = incidence_angle,
        incidence_azimuth = incidence_azimuth, incident, return_diagnostics = true, capture..., kwargs...)
    diagnostics = merge(diagnostics,
        (;
            meshsize = Float64(meshsize), quadrature_order = qorder, mesh_order,
            illumination = incident === nothing ? :plane_wave : :prescribed))
    data = _FullBEMSurfaceData(quad, p_scat, dpdn_scat, incidence_angle, incidence_azimuth,
        diagnostics, density[], incident)
    return BEMSolution(body, boundary, Float64(k), :full, data)
end
function _bem_full(
        body::Cylinder, boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real;
        meshsize::Real = bem3d_elements_per_wavelength(k), qorder::Integer = 4,
        mesh_order::Integer = 2, kwargs...)
    surface = _cylinder_full_mesh(body; meshsize, qorder, mesh_order)
    solution = bem(surface, boundary, k; kwargs...)
    d = solution.data
    report = merge(d.diagnostics, (;
        meshsize = Float64(meshsize), mesh_order, quadrature_order = qorder))
    data = _FullBEMSurfaceData(d.quad, d.p_scat, d.dpdn_scat,
        d.incidence_angle, d.incidence_azimuth, report, d.single_layer_density, d.incident)
    return BEMSolution(body, boundary, Float64(k), :full, data)
end

# --- `mfs`: axisymmetric and bent-cylinder method of fundamental solutions --

"""
    mfs(body::AbstractBody, boundary::AbstractBoundaryCondition, k; incidence_angle=π/2, offset=0.3*characteristic_radius, kwargs...)

Method-of-fundamental-solutions solve, returns an [`MFSSolution`](@ref). Axisymmetric for
`Sphere`/`Spheroid`/straight `Cylinder`. Lateral-only 3D point sources for a bent `Cylinder`
(omits end caps). Use `mfs(mesh(body; method=:full, ...), ...)` for closed-surface conditions
and general observation directions. `offset` is a maximum source displacement in m from the
surface, reduced near flat-cap rims.

For bent cylinders, `n_s`/`n_phi` (defaults 40/32, at least 3) set the axial and azimuthal
source-grid counts. `oversampling` (integer, default 1) multiplies the collocation budget while
keeping the source grid fixed. Values above 1 give a least-squares solve.

See [BEM and MFS](@ref boundary-theory) for source placement guidance and
[`diagnostics`](@ref) for residuals, conditioning and rank (`condition_limit=512` bounds the
SVD size, 0 skips it).
"""
function mfs(body::Union{Sphere, Spheroid, Cylinder},
        boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, offset::Real = 0.3_characteristic_radius(body),
        n::Integer = _axisymmetric_default_panels(body, k),
        m_max::Integer = _default_mode_count(k * _characteristic_radius(body)),
        oversampling::Integer = 1, condition_limit::Integer = 512, kwargs...)
    oversampling >= 1 || throw(ArgumentError("mfs: oversampling must be at least 1"))
    condition_limit >= 0 || throw(ArgumentError("mfs: condition_limit must be nonnegative"))
    if body isa Cylinder && _isbent(body)
        return _mfs_bent(body, boundary, k; incidence_angle = incidence_angle,
            offset = offset, oversampling = oversampling, condition_limit, kwargs...)
    end
    mesh, source_mesh = _mfs_meridian_meshes(body, n, oversampling)
    iszero(incidence_angle) &&
        return _mfs_axial(
            body, boundary, k, mesh; offset, source_mesh,
            oversampling, condition_limit, kwargs...)
    return _mfs_oblique(
        body, boundary, k, mesh, incidence_angle; offset,
        m_max, source_mesh, oversampling, condition_limit, kwargs...)
end

function _mfs_meridian_meshes(body, n, oversampling)
    source_mesh = (body isa Cylinder && _iscapped(body)) ?
                  cylinder_spheroidal_endcap_mesh(body.radius, body.length, body.endcap_depth, n) :
                  _axisymmetric_mesh(body, n)
    mesh = oversampling == 1 ? source_mesh :
           (body isa Cylinder && _iscapped(body)) ?
           cylinder_spheroidal_endcap_mesh(body.radius, body.length, body.endcap_depth, oversampling *
                                                                                        n) :
           _axisymmetric_mesh(body, oversampling * n)
    return mesh, source_mesh
end

function _mfs_axial(body::AbstractBody, boundary::Union{Rigid, PressureRelease, Impedance},
        k::Real, mesh::MeridianMesh; offset::Real, source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat, dpdn_scat, _ = solve_axial_mfs(boundary, k, mesh;
        offset, source_mesh, source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling,
            offset, incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], nothing, nothing, 0.0, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_axial(
        body::AbstractBody, boundary::FluidFilled, k::Real, mesh::MeridianMesh; offset::Real,
        offset_ext::Real = offset, offset_int::Real = offset,
        source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat, dpdn_scat, _, p_int, dpdn_int = solve_axial_mfs(
        boundary, k, mesh; offset_ext, offset_int, source_mesh,
        source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling, offset_ext, offset_int,
            incidence_angle = 0.0, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, [p_scat], [dpdn_scat], [p_int], [dpdn_int], 0.0, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_oblique(
        body::AbstractBody, boundary::Union{Rigid, PressureRelease, Impedance},
        k::Real, mesh::MeridianMesh, incidence_angle::Real;
        offset::Real, m_max::Integer, source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique_mfs(
        boundary, k, mesh, incidence_angle; m_max, offset,
        source_mesh, source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling,
            offset, incidence_angle, m_max, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, p_scat_modes, dpdn_scat_modes, nothing,
        nothing, incidence_angle, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_oblique(body::AbstractBody, boundary::FluidFilled, k::Real,
        mesh::MeridianMesh, incidence_angle::Real;
        offset::Real, m_max::Integer, offset_ext::Real = offset, offset_int::Real = offset,
        source_mesh = mesh, oversampling = 1, kwargs...)
    reports = _SolveReports()
    source_modes = NamedTuple[]
    p_scat_modes, dpdn_scat_modes, _ = solve_oblique_mfs(
        boundary, k, mesh, incidence_angle;
        m_max, offset_ext, offset_int, source_mesh,
        source_modes, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :axisymmetric,
        solver_options = (; n = npanels(source_mesh), oversampling, offset_ext, offset_int,
            incidence_angle, m_max, kwargs...))
    data = _AxisymmetricSurfaceData(
        mesh, p_scat_modes, dpdn_scat_modes, nothing,
        nothing, incidence_angle, report, source_modes)
    return MFSSolution(body, boundary, k, data)
end

function _mfs_bent(body::Cylinder, boundary::Union{Rigid, PressureRelease}, k::Real;
        incidence_angle::Real = π / 2, offset::Real, n_s::Integer = 40,
        n_phi::Union{Nothing, Integer} = nothing, n_φ::Union{Nothing, Integer} = nothing,
        oversampling::Integer = 1, kwargs...)
    n_phi !== nothing && n_φ !== nothing &&
        throw(ArgumentError("mfs: supply only n_phi, not both n_phi and the legacy n_φ"))
    azimuth_count = something(n_phi, n_φ, 32)
    n_s >= 3 || throw(ArgumentError("mfs: n_s must be at least 3"))
    azimuth_count >= 3 || throw(ArgumentError("mfs: n_phi must be at least 3"))
    reports = _SolveReports()
    p_scat, dpdn_scat, points, normals, areas = solve_bent_cylinder_mfs(
        boundary, k, body.radius, body.length, body.radius_curvature;
        aspect_angle = incidence_angle, offset = offset, n_s = n_s, n_φ = azimuth_count,
        oversampling, solve_reports = reports, kwargs...)
    report = _summarize_solves(reports; method = :bent,
        solver_options = (;
            n_s, n_phi = azimuth_count, offset, incidence_angle, oversampling, kwargs...))
    data = _BentMFSSurfaceData(
        p_scat, dpdn_scat, points, normals, areas, incidence_angle, report)
    return MFSSolution(body, boundary, k, data)
end

# --- `fem` on a `Shell` body: thin and general shell-fluid coupling -----

function _prolate_shell_geometry(s::Shell)
    s.body isa Spheroid ||
        throw(ArgumentError("this shell method requires a Spheroid-based Shell (got $(typeof(s.body)))"))
    return ProlateShellGeometry(s.body.a, s.body.b, s.thickness)
end

"""
    fem(shell::Shell, boundary::Shelled, ext_density, ext_soundspeed, int_density, int_soundspeed, k;
        method=:general, incidence_angle=π/2, kwargs...)

Elastic-shell/fluid coupling, returns a [`FEMSolution`](@ref). Post-process with
[`target_strength`](@ref)`(sol; angle, azimuth)`. `boundary` is built via
[`Shelled`](@ref)`(poisson, density, youngs_modulus)`. `ext_density`/`ext_soundspeed` and
`int_density`/`int_soundspeed` are the absolute density [kg/m³] and sound speed in m/s of the
exterior and interior fluids, not contrasts.

`method=:thin` is an axisymmetric reduction, `shell.body isa Spheroid` only, axial incidence
only (`incidence_angle = 0.0`). Pass `int_density=0` for a vacuum-backed shell.

`method=:general` (default) is a full through-thickness 2D shell FEM, general incidence,
`shell.body isa Union{Sphere,Spheroid}`, always fluid-filled. Use a small `int_density`/
`int_soundspeed` for a near-vacuum limit.

See [FEM and shell coupling](@ref fem-theory) for the shell theory.
"""
function fem(s::Shell, boundary::Shelled{ElasticFEMLayer, Nothing},
        ext_density::Real, ext_soundspeed::Real,
        int_density::Real, int_soundspeed::Real, k::Real;
        method::Symbol = :general, incidence_angle::Real = π / 2,
        m_max::Integer = _default_mode_count(k * _characteristic_radius(s.body)),
        n_eta::Integer = 65, n_t::Integer = 3,
        pole_offset::Real = method === :thin ? 1e-4 : 1e-3, kwargs...)
    material = boundary.material
    freq_hz = k * ext_soundspeed / (2π)
    reports = _SolveReports()
    options = (; incidence_angle, m_max = method === :thin ? 0 : m_max,
        n_eta, n_t, pole_offset, kwargs...)
    if method === :thin
        iszero(incidence_angle) || throw(ArgumentError(
            "fem(::Shell, ...; method=:thin) only supports axial incidence (the axisymmetric " *
            "reduction retains m=0), use method=:general for oblique incidence"))
        geometry = _prolate_shell_geometry(s)
        if iszero(int_density)
            p_scat, dpdn_scat, ps, shell_state, _ = solve_shell_fluid_coupled(
                geometry, material, ext_density, ext_soundspeed,
                freq_hz; n_eta, pole_offset, solve_reports = reports, kwargs...)
            report = _summarize_solves(reports; method, solver_options = options)
            data = _ShellFEMSurfaceData(
                ps, [p_scat], [dpdn_scat], nothing, nothing,
                nothing, shell_state, incidence_angle, report)
            return FEMSolution(s, boundary, k, method, data)
        end
        p_ext, dpdn_ext, ps_ext, p_int, dpdn_int, ps_int,
        shell_state, _ = solve_shell_fluid_filled_coupled(
            geometry, material, ext_density, ext_soundspeed, int_density,
            int_soundspeed, freq_hz; n_eta, pole_offset, solve_reports = reports, kwargs...)
        report = _summarize_solves(reports; method, solver_options = options)
        data = _ShellFEMSurfaceData(
            ps_ext, [p_ext], [dpdn_ext], ps_int, [p_int],
            [dpdn_int], shell_state, incidence_angle, report)
        return FEMSolution(s, boundary, k, method, data)
    end
    method === :general ||
        throw(ArgumentError("fem(::Shell, ...) supports method=:thin or :general, got $method"))
    int_density > 0 || throw(ArgumentError(
        "fem(::Shell, ...; method=:general) requires positive interior density"))
    p_scat_modes, dpdn_scat_modes, ps = if s.body isa Spheroid
        solve_general_shell_fluid_filled_coupled(
            _prolate_shell_geometry(s), material.density,
            material.youngs_modulus, material.poisson,
            ext_density, ext_soundspeed, int_density, int_soundspeed, freq_hz,
            incidence_angle; m_max = m_max, n_eta = n_eta, n_t = n_t,
            pole_offset = pole_offset, solve_reports = reports, kwargs...)
    else
        mesh = build_structured_spherical_shell(s.body.radius, s.thickness, n_eta, n_t;
            pole_offset = pole_offset)
        solve_general_shell_fluid_filled_coupled(
            mesh, material.density, material.youngs_modulus, material.poisson,
            ext_density, ext_soundspeed, int_density, int_soundspeed,
            freq_hz, incidence_angle; m_max = m_max, solve_reports = reports, kwargs...)
    end
    report = _summarize_solves(reports; method, solver_options = options)
    data = _ShellFEMSurfaceData(
        ps, p_scat_modes, dpdn_scat_modes, nothing, nothing,
        nothing, nothing, incidence_angle, report)
    return FEMSolution(s, boundary, k, method, data)
end

# --- `fourier`: conformal-mapping semi-analytic method for bodies of revolution -----

"""
    FMSolution

Result of [`fourier`](@ref). Post-process with [`target_strength`](@ref)`(sol; angle,
azimuth)` or [`scattering_amplitude`](@ref)`(sol; angle, azimuth)` (backscatter by default, same
convention as axisymmetric [`bem`](@ref)/[`mfs`](@ref)). [`diagnostics`](@ref) reports the
mapping's admissibility, the truncation orders and a truncation-consistency check. `b_check` holds
the reduced-truncation coefficients.
"""
struct FMSolution <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    mapping::ConformalMapping
    b::Matrix{ComplexF64}
    incidence_angle::Float64
    b_check::Union{Nothing, Matrix{ComplexF64}}
end

"""
    fourier(body::Irregular, boundary::AbstractBoundaryCondition, k;
           incidence_angle=π/2, continuation_steps=8, mapping_order=length(body.rc),
           m_max=_default_mode_count(k*body.a), n_max=m_max, rtol=1e-6, maxevals=1000)

Fourier-matching result, returns an [`FMSolution`](@ref). Conformally maps `body`'s meridian
profile to a coordinate system where the mapped surface is exactly circular, then matches the
boundary condition using spherical wave functions. See [Fourier matching](@ref
fourier-matching-theory). Supports [`Rigid`](@ref), [`PressureRelease`](@ref) and
[`FluidFilled`](@ref) boundaries. `mapping_order` and `continuation_steps` control the conformal
mapping (see `solve_mapping`). `m_max`/`n_max` truncate the modal series. `rtol` and `maxevals` set the tolerance and maximum
node count of the boundary-matching quadrature. Post-process with
[`target_strength`](@ref)`(sol; angle, azimuth)` or [`scattering_amplitude`](@ref)`(sol; angle,
azimuth)`, defaulting to backscatter.

Each solve is repeated at `n_max` and `m_max` reduced by 2. A warning is emitted when the far-field
amplitude changes by more than `1e-2` of its peak, and the change is reported by
[`diagnostics`](@ref) as `convergence`. See [Fourier matching](@ref fourier-matching-theory).
"""
function fourier(body::Irregular, boundary::AbstractBoundaryCondition, k::Real;
        incidence_angle::Real = π / 2, continuation_steps::Integer = 8,
        mapping_order::Integer = max(length(body.rc), 1),
        m_max::Integer = _default_mode_count(k * body.a), n_max::Integer = m_max,
        rtol::Real = 1e-6, maxevals::Integer = 1000)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive, got $k"))
    mapping = solve_mapping(body, mapping_order; continuation_steps)
    is_admissible(mapping) || throw(ArgumentError(
        "Irregular's conformal mapping is inadmissible (Jacobian vanishes somewhere). " *
        "Try a higher mapping_order or more continuation_steps"))
    transition = _boundary_transition(mapping, k, boundary; m_max, n_max, rtol, maxevals)
    a = _incident_coefficients(n_max, m_max, k, incidence_angle)
    b = _apply_transition(transition, a, n_max, m_max)
    b_check = _check_coefficients(transition, k, incidence_angle, n_max, m_max)
    _warn_fm_convergence(_fm_convergence(b, b_check, k))
    return FMSolution(
        body, boundary, Float64(k), mapping, b, Float64(incidence_angle), b_check)
end

"""
    fourier(body::Sphere, boundary, k; kwargs...)
    fourier(body::Spheroid, boundary, k; mapping_order=..., kwargs...)

Convenience overloads for the canonical bodies, matching [`bem`](@ref)/[`mfs`](@ref)'s body-type
coverage. Both convert `body` to an equivalent [`Irregular`](@ref) internally and solve with
[`fourier`](@ref). [`modal`](@ref) is exact and cheaper for these bodies. These overloads exist
for interface consistency and cross-checking, not as the recommended solver. The returned
[`FMSolution`](@ref) keeps the original `Sphere`/`Spheroid` in its `body` field.

A spheroid with aspect ratio above 2.5 defaults to a higher `mapping_order` and a matching `m_max`/`n_max`.
See [Fourier matching](@ref fourier-matching-theory) for the validated aspect-ratio range.
"""
function fourier(body::Sphere, boundary::AbstractBoundaryCondition, k::Real; kwargs...)
    irregular_body = Irregular(body.radius, Float64[], Float64[])
    sol = fourier(irregular_body, boundary, k; kwargs...)
    return FMSolution(body, sol.boundary, sol.k, sol.mapping, sol.b, sol.incidence_angle,
        sol.b_check)
end

const _FM_ELONGATED_ASPECT = 2.5

# Mapping order and mode count for a spheroid. The usable `n_max` is limited by the mapping order, so elongated bodies take a high order and a matching mode count.
function _spheroid_fourier_defaults(body::Spheroid, mapping_order, options)
    aspect = max(body.a, body.b) / min(body.a, body.b)
    aspect > _FM_ELONGATED_ASPECT || return (
        something(mapping_order, clamp(round(Int, 6aspect), 8, 64)), options)
    order = something(mapping_order, clamp(round(Int, 20aspect), 16, 96))
    modes = round(Int, order / (2aspect)) + 2
    (haskey(options, :m_max) || haskey(options, :n_max)) && return order, options
    return order, (; m_max = modes, options...)
end

function fourier(body::Spheroid, boundary::AbstractBoundaryCondition, k::Real;
        mapping_order::Union{Nothing, Integer} = nothing, kwargs...)
    mapping_order, options = _spheroid_fourier_defaults(body, mapping_order, kwargs)
    irregular_body = Irregular(mapping_order) do theta
        1 / sqrt((cos(theta) / body.a)^2 + (sin(theta) / body.b)^2)
    end
    sol = fourier(irregular_body, boundary, k; mapping_order, options...)
    return FMSolution(body, sol.boundary, sol.k, sol.mapping, sol.b, sol.incidence_angle,
        sol.b_check)
end

function target_strength(sol::FMSolution; angle::Real = π - sol.incidence_angle, azimuth::Real = π)
    return target_strength(scattering_amplitude(sol; angle, azimuth))
end

function scattering_amplitude(
        sol::FMSolution; angle::Real = π - sol.incidence_angle, azimuth::Real = π)
    return fourier_matching_amplitude(sol.b, sol.k, angle, azimuth)
end

function diagnostics(sol::FMSolution)
    convergence = _fm_convergence(sol.b, sol.b_check, sol.k)
    (; admissible = is_admissible(sol.mapping),
        m_max = size(sol.b, 2) - 1, n_max = size(sol.b, 1) - 1, convergence,
        convergence_tolerance = _FM_CONVERGENCE_TOLERANCE,
        converged = isnan(convergence) ? nothing : convergence <= _FM_CONVERGENCE_TOLERANCE)
end

# --- `target_strength`/`scattering_amplitude` on `AbstractSolution`s -----------------------
# `scattering_amplitude` is the solution-level replacement for the low-level, unexported
# `form_function(boundary, k, ...)` family: the complex amplitude [m], pre-dB-conversion.

"""
    scattering_amplitude(solution; kwargs...)

Return the complex far-field scattering amplitude in meters. Axisymmetric BEM/MFS and
structural shell FEM accept observation `angle` and `azimuth` [rad] in body coordinates;
their defaults are backscatter, `angle = pi - incidence_angle`, `azimuth = pi`.
Full BEM accepts a unit-vector `direction`, defaulting to the negative incident direction.
Cartesian body length is x, width is y and height/depth is z. Polar angles are from +x;
azimuth is from +y toward +z, so `(angle,azimuth)=(pi/2,0)` points along +y.
Bent MFS returns backscatter only. Modal/Kirchhoff observation is fixed at solve time and
post-processing keywords throw `ArgumentError`. Supported radial spheres return complex
backscatter amplitude without observation keywords. Cylinder radial and meridian FEM
paths retain only target strength and throw `ArgumentError` for complex amplitude.

# Examples
```julia
solution = modal(Sphere(0.01), Rigid(), 100.0)
amplitude = scattering_amplitude(solution)
```
"""
function scattering_amplitude end

"""
    target_strength(solution::AbstractSolution; kwargs...)

Return target strength in dB re 1 m². Where complex amplitude is retained, this is
`20 * log10(abs(scattering_amplitude(solution; kwargs...)))`, with the same observation
keywords and backscatter defaults. Scalar-only FEM paths return their stored target strength.
Non-structural radial/meridian FEM rejects observation keywords with `ArgumentError`.
"""
target_strength

"""
    diagnostics(solution)

Return a named tuple of solver diagnostics for BEM, MFS, FEM and Fourier matching. Modal and
Kirchhoff solutions return `nothing`.

Fourier matching reports `admissible`, `m_max`, `n_max`, `convergence`, `convergence_tolerance` and
`converged`, which is `nothing` when `n_max` is too small to reduce.

Full-3D BEM reports `converged`, `iterations`, `relative_residual`, `residual_history`,
`unknown_count`, `meshsize`, `quadrature_order`, and the solver settings used. Axisymmetric
BEM/MFS/FEM report per-system entries in `systems`. MFS systems also report source counts,
`condition_number`, `numerical_rank` and a held-out `boundary_residual`.

See [BEM and MFS](@ref boundary-theory) for field definitions and what each residual does and
does not measure.

# Example
```julia
solution = bem(Sphere(0.01), Rigid(), 100.0; method = :full, meshsize = 0.01)
diagnostics(solution).converged
```
"""
diagnostics(::AbstractSolution) = nothing
diagnostics(sol::Union{BEMSolution, MFSSolution, FEMSolution}) = sol.data.diagnostics

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

function target_strength(sol::TMatrixSolution; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "target_strength(::TMatrixSolution) takes no keywords: the directions were already " *
        "fixed when tmatrix(...) was called. Call tmatrix(...) again with different angles."))
    return target_strength(sol.f)
end
function scattering_amplitude(sol::TMatrixSolution; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "scattering_amplitude(::TMatrixSolution) takes no keywords: the directions were already " *
        "fixed when tmatrix(...) was called. Call tmatrix(...) again with different angles."))
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

function scattering_amplitude(sol::FEMSolution{_RadialFEMData}; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "Radial sphere FEM supports backscatter without observation keywords."))
    return _radial_fem_amplitude(sol.data.modes, sol.k)
end

function target_strength(sol::FEMSolution{_RadialFEMData}; kwargs...)
    return target_strength(scattering_amplitude(sol; kwargs...))
end

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
        "scattering_amplitude is not available for this FEMSolution: this FEM path " *
        "retains only target strength. Use target_strength(sol) " *
        "instead, or use modal(...)/kirchhoff(...) for this body/boundary combination if you need " *
        "the complex amplitude."))
end

function scattering_amplitude(sol::FEMSolution{_VolumeFEMData};
        angle::Real = π - sol.data.incidence_angle,
        azimuth::Real = sol.data.incidence_azimuth + π)
    return _volume_amplitude(sol.data, sol.k; angle, azimuth)
end

function target_strength(sol::FEMSolution{_VolumeFEMData};
        angle::Real = π - sol.data.incidence_angle,
        azimuth::Real = sol.data.incidence_azimuth + π)
    return target_strength(scattering_amplitude(sol; angle, azimuth))
end

function scattering_amplitude(sol::FreeSurfaceSolution;
        angle::Real = π - sol.incidence_angle, azimuth::Real = sol.incidence_azimuth + π)
    return scattering_amplitude(sol.direct; angle, azimuth) +
           sol.sign * scattering_amplitude(sol.reflected; angle, azimuth)
end

function target_strength(sol::FreeSurfaceSolution; kwargs...)
    return target_strength(scattering_amplitude(sol; kwargs...))
end

function target_strength(sol::FEMSolution{_ShellFEMSurfaceData};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π)
    return target_strength(scattering_amplitude(sol; angle = angle, azimuth = azimuth))
end

function scattering_amplitude(sol::FEMSolution{_ShellFEMSurfaceData};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π)
    d = sol.data
    length(d.p_ext_modes) == 1 &&
        return far_field(d.ps_ext, d.p_ext_modes[1], d.dpdn_ext_modes[1], sol.k, angle)
    return far_field(d.ps_ext, d.p_ext_modes, d.dpdn_ext_modes, sol.k, angle, azimuth)
end

function target_strength(sol::FEMSolution{T};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π) where {
        T <: Union{_CylinderMeridianFEMData, _SpheroidMeridianFEMData}}
    return target_strength(scattering_amplitude(sol; angle, azimuth))
end

function scattering_amplitude(sol::FEMSolution{T};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π) where {
        T <: Union{_CylinderMeridianFEMData, _SpheroidMeridianFEMData}}
    d = sol.data
    length(d.p_modes) == 1 &&
        return far_field(d.ps, d.p_modes[1], d.dpdn_modes[1], sol.k, angle)
    return far_field(d.ps, d.p_modes, d.dpdn_modes, sol.k, angle, azimuth)
end

function target_strength(sol::FEMSolution{_CylinderRadialFEMData})
    return target_strength(scattering_amplitude(sol))
end

function scattering_amplitude(sol::FEMSolution{_CylinderRadialFEMData})
    d = sol.data
    return _elastic_cylinder_radial_fem_amplitude(d.modes, sol.k, d.length, d.aspect_angle)
end

function _axisymmetric_amplitude(k::Real, d::_AxisymmetricSurfaceData; angle::Real, azimuth::Real)
    d.source_modes === nothing ||
        return _mfs_source_amplitude(k, d.source_modes, angle, azimuth)
    ps = panels(d.mesh)
    length(d.p_scat_modes) == 1 &&
        return far_field(ps, d.p_scat_modes[1], d.dpdn_scat_modes[1], k, angle)
    return far_field(ps, d.p_scat_modes, d.dpdn_scat_modes, k, angle, azimuth)
end

function target_strength(
        sol::Union{
            BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData}};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π)
    return target_strength(scattering_amplitude(sol; angle = angle, azimuth = azimuth))
end

function scattering_amplitude(
        sol::Union{
            BEMSolution{_AxisymmetricSurfaceData}, MFSSolution{_AxisymmetricSurfaceData}};
        angle::Real = π - sol.data.incidence_angle, azimuth::Real = π)
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
    if _uses_edge_quadrature(d)
        return sol.boundary isa FluidFilled ?
               _edge_fluid_far_field(d, sol.k, xhat, sol.boundary) :
               _edge_far_field(d, sol.k, xhat)
    end
    return far_field(d.quad, xhat, sol.k, d.p_scat, d.dpdn_scat)
end

function target_strength(sol::MFSSolution{_BentMFSSurfaceData})
    target_strength(scattering_amplitude(sol))
end

function scattering_amplitude(sol::MFSSolution{_BentMFSSurfaceData})
    d = sol.data
    β = d.incidence_angle
    q̂ = (-cos(β), -sin(β), 0.0)
    f = zero(ComplexF64)
    for i in eachindex(d.points)
        f += (im * sol.k * _dot3(q̂, d.normals[i]) * d.p_scat[i] + d.dpdn_scat[i]) *
             cis(-sol.k * _dot3(q̂, d.points[i])) * d.areas[i]
    end
    return f / (4π)
end
