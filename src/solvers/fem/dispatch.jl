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
- `:meridian`: 2D (ρ,z) meridian FEM with the angular part discretized. `Sphere` supports
  `Rigid` and `PressureRelease` with axial incidence. `Cylinder`/`Spheroid` also support
  `FluidFilled` and general `incidence_angle`.
  For cylinders and spheroids, `mode_threads=1` keeps mode factorizations serial by default.
  A spheroid's default `m_max` follows the equatorial radius with a fluid-interior guard,
  and is zero at axial incidence. Explicit `m_max` overrides this estimate.
  Set `retain_fields=false` for far-field-only sweeps to omit stored nodal pressure.
  This preserves far-field output but disables scattered/total [`pressure`](@ref) queries.
  Cylinders and spheroids accept `order=2:8` for higher-order mapped elements with
  element-interior elimination. The default `order=1` retains the bilinear mesh.
  Higher-order defaults size `n_r` and `n_theta` from the wavelength, including a
  fluid interior. Explicit mesh counts are element counts. `adaptive=true` refines
  individual Fourier modes until their combined mesh-change estimate is below
  `target_tol` dB in backscatter TS (default `0.01`), with at most
  `max_refinements=4` increases per mode. Inspect
  `diagnostics(sol).refinement.converged`. This estimates spatial convergence,
  independently of `m_max` and `l_max`, and does not bound phase or bistatic error.
  `dtn_form=:augmented` keeps the outgoing modal boundary map rectangular instead
  of expanding its dense boundary block. The default remains `:assembled`.
  `quadrature_extra=0` adds integration points for higher-order meridian FEM.
  Increase it to check integration independently of the mesh and element order.

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

Supported radial and meridian spheres also support [`scattering_amplitude`](@ref)`(sol)`
in complex meters.
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
        fields = _sphere_meridian_fem_fields(
            boundary, k, body.radius, R; solve_reports = reports, kwargs...)
        report = _summarize_solves(reports; method, solver_options = (; R, kwargs...))
        data = _SphereMeridianFEMData(
            fields.radii, fields.angles, fields.pressure, fields.outgoing, report)
        return FEMSolution(body, boundary, k, method, data)
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
        m_max::Integer = _default_mode_count(k * body.radius),
        mode_threads::Integer = 1, retain_fields::Bool = true, order::Integer = 1,
        dtn_form::Symbol = :assembled, kwargs...)
    _isbent(body) && throw(ArgumentError(
        "fem(::Cylinder, ...) has no bent-cylinder implementation in this package yet"))
    method === :meridian ||
        throw(ArgumentError("fem(::Cylinder, ::Union{Rigid,PressureRelease,FluidFilled}, ...) only supports method=:meridian"))
    1 <= order <= 8 || throw(ArgumentError("cylinder meridian FEM order must be in 1:8"))
    dtn_form in (:assembled, :augmented) ||
        throw(ArgumentError("dtn_form must be :assembled or :augmented"))
    if order > 1
        return _spheroid_high_order_fem(body, boundary, k;
            R, incidence_angle, m_max, mode_threads,
            retain_fields, order, dtn_form, kwargs...)
    end
    reports = _SolveReports()
    traces = _cylinder_meridian_fem_modes(
        boundary, k, body.radius, body.length, R, incidence_angle;
        m_max, mode_threads, solve_reports = reports, retain_fields, dtn_form, kwargs...)
    ps, p_modes, dpdn_modes = traces[1:3]
    fields = retain_fields ? traces[4] : _MeridianFEMModeField[]
    sort!(reports.systems; by = r -> r.mode)
    report = _summarize_solves(reports;
        method,
        solver_options = (;
            R, incidence_angle, m_max, mode_threads,
            retain_fields, order, dtn_form, kwargs...))
    data = _CylinderMeridianFEMData(
        ps, p_modes, dpdn_modes, fields, incidence_angle, report)
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
        m_max::Integer = _spheroid_meridian_default_mmax(body, boundary, k, incidence_angle),
        mode_threads::Integer = 1, retain_fields::Bool = true,
        order::Integer = 1,
        dtn_form::Symbol = :assembled, kwargs...)
    if method === :volume
        mode_threads == 1 && retain_fields && order == 1 && dtn_form === :assembled ||
            throw(ArgumentError(
                "mode_threads, retain_fields, order and dtn_form controls require method=:meridian"))
        return _fem_volume_solution(body, boundary, k; incidence_angle, kwargs...)
    end
    method === :meridian || throw(ArgumentError(
        "fem(::Spheroid, ...) supports method=:meridian or :volume, got $method"))
    1 <= order <= 8 || throw(ArgumentError("spheroid meridian FEM order must be in 1:8"))
    dtn_form in (:assembled, :augmented) ||
        throw(ArgumentError("dtn_form must be :assembled or :augmented"))
    if order > 1
        return _spheroid_high_order_fem(body, boundary, k;
            R, incidence_angle, m_max, mode_threads,
            retain_fields, order, dtn_form, kwargs...)
    end
    reports = _SolveReports()
    traces = _spheroid_meridian_fem_modes(
        boundary, k, body.a, body.b, R, incidence_angle;
        m_max, mode_threads, solve_reports = reports, retain_fields, dtn_form, kwargs...)
    ps, p_modes, dpdn_modes = traces[1:3]
    fields = retain_fields ? traces[4] : _MeridianFEMModeField[]
    report = _summarize_solves(reports;
        method,
        solver_options = (;
            R, incidence_angle, m_max, mode_threads, retain_fields, dtn_form, kwargs...))
    data = _SpheroidMeridianFEMData(
        ps, p_modes, dpdn_modes, fields, incidence_angle, report)
    return FEMSolution(body, boundary, k, method, data)
end

function fem(body::Union{Sphere, Spheroid}, boundary::SpatialFluid, k::Real;
        method::Symbol = :volume, kwargs...)
    method === :volume || throw(ArgumentError("SpatialFluid requires method=:volume"))
    return _fem_volume_solution(body, boundary, k; kwargs...)
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
`SpatialFluid` supplies positive density and sound-speed contrast profiles at
quadrature points, in the same global coordinate frame as `centers`.

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
        "region materials must be FluidFilled, SpatialFluid, SolidElastic, ViscoelasticSolid or ViscousLayer"))
    data = _fem_volume_regions(
        bodies, materials, k; parents, centers, orientations, kwargs...)
    geometry = _VolumeRegionGeometry(collect(AbstractBody, bodies),
        [Float64.(c) for c in centers], [Float64.(o) for o in orientations],
        collect(Int, parents))
    return FEMSolution(
        geometry, _VolumeRegionMaterials(collect(Any, materials)), k,
        :volume, data)
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

function _layered_fem_layers(boundary::Shelled)
    boundary.interior isa FluidInterior || throw(ArgumentError(
        "layered FEM requires a FluidInterior core"))
    layers, ratios = _layered_materials(boundary.material)
    all(layer -> layer isa Union{FluidLayer, ElasticLayer, ViscousLayer}, layers) ||
        throw(ArgumentError("layered FEM requires FluidLayer, ElasticLayer or ViscousLayer"))
    all(
        layer -> !(layer isa ElasticLayer) ||
                 layer.interior_coupling === :generalized, layers) ||
        throw(ArgumentError("layered FEM requires generalized elastic coupling"))
    viscous = filter(layer -> layer isa ViscousLayer, layers)
    isempty(viscous) ||
        all(
            layer -> layer.soundspeed_exterior == first(viscous).soundspeed_exterior, viscous) ||
        throw(ArgumentError("viscous layers must use the same exterior sound speed"))
    return layers, ratios
end

_layered_fem_interface(body::Sphere, ratio) = Sphere(ratio * body.radius)
function _layered_fem_interface(body::Spheroid, ratio)
    xi = _confocal_inner_xi(body, ratio)
    equatorial, polar = _confocal_axes(body.kind, body.q, xi)
    return Spheroid(polar, equatorial)
end

function _layered_volume_regions(body::Union{Sphere, Spheroid}, boundary::Shelled)
    layers, ratios = _layered_fem_layers(boundary)
    material(layer::FluidLayer) = FluidFilled(
        layer.density_contrast, layer.soundspeed_contrast)
    material(layer::ElasticLayer) = SolidElastic(layer.density_contrast,
        layer.speed_longitudinal_contrast, layer.speed_transversal_contrast)
    material(layer::ViscousLayer) = layer
    interfaces = vcat(ratios[2:end], boundary.radius_ratio)
    bodies = [body]
    for ratio in interfaces
        push!(bodies, _layered_fem_interface(body, ratio))
    end
    materials = vcat([material(layer) for layer in layers],
        FluidFilled(boundary.interior.density_contrast,
            boundary.interior.soundspeed_contrast))
    return bodies, materials
end

"""
    fem(body::Union{Sphere,Spheroid}, boundary::Shelled{<:Union{LayeredMaterial,ViscousLayer}}, k;
        method=:volume, kwargs...)

FEM for concentric spherical or confocal spheroidal `FluidLayer`, `ElasticLayer`
and `ViscousLayer` materials around a `FluidInterior`. For spheres, `method=:radial`
uses quadratic displacement elements with `n_elements=256` per region, including
the core, and supports only `m_max=0`. It retains complex monopole scattering.
Point-field queries require `method=:volume`, which solves the full 3D coupled
field and also supports spheroids. Nested layer ratios scale each enclosing
radius or equatorial semi-axis. The core ratio refers to the outer body, and
spheroidal interfaces share its focal distance. Contrasts refer to the exterior.
"""
function fem(body::Union{Sphere, Spheroid},
        boundary::Shelled{<:Union{LayeredMaterial, ViscousLayer}},
        k::Real; method::Symbol = :volume, kwargs...)
    if method === :radial && body isa Sphere
        data = _layered_sphere_displacement_fem(body, boundary, k; kwargs...)
        return FEMSolution(body, boundary, k, method, data)
    end
    method === :volume || throw(ArgumentError(
        "layered FEM supports method=:volume, or method=:radial for a spherical monopole"))
    bodies, materials = _layered_volume_regions(body, boundary)
    return fem(bodies, materials, k; method, kwargs...)
end
