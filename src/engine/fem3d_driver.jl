# Bodies, materials and the driver of the full-3D volume FEM, see fem3d.jl.

# Geometry and material description of a body for the volume solver.
function _volume_geometry(body::Sphere, boundary, k)
    return _volume_geometry((body.radius, body.radius), body.radius, boundary, k)
end

function _volume_geometry(body::Spheroid, boundary, k)
    return _volume_geometry((body.b, body.a), body.b, boundary, k; spheroid = body)
end

function _volume_geometry(outer_axes, equatorial, boundary::Rigid, k; spheroid = nothing)
    return (; outer_axes, inner_axes = nothing, kinds = (; body = 0, inner = 0),
        model = (; kind = :rigid, solid = nothing, interior = nothing),
        body_wavelength = Inf, thickness = Inf)
end

function _volume_geometry(
        outer_axes, equatorial, boundary::PressureRelease, k; spheroid = nothing)
    return (; outer_axes, inner_axes = nothing, kinds = (; body = 0, inner = 0),
        model = (; kind = :soft, solid = nothing, interior = nothing),
        body_wavelength = Inf, thickness = Inf)
end

function _volume_geometry(
        outer_axes, equatorial, boundary::FluidFilled, k; spheroid = nothing)
    interior = (boundary.density_contrast, k / boundary.soundspeed_contrast)
    return (; outer_axes, inner_axes = nothing,
        kinds = (; body = _REGION_INTERIOR, inner = 0),
        model = (; kind = :fluid, solid = nothing, interior),
        body_wavelength = 2π * boundary.soundspeed_contrast / k, thickness = Inf)
end

function _volume_solid_parameters(density, cL, cT)
    return (density, density * (cL^2 - 2cT^2), density * cT^2)
end

function _volume_geometry(
        outer_axes, equatorial, boundary::SolidElastic, k; spheroid = nothing)
    solid = _volume_solid_parameters(boundary.density_contrast,
        boundary.speed_longitudinal_contrast, boundary.speed_transversal_contrast)
    return (;
        outer_axes, inner_axes = nothing, kinds = (; body = _REGION_SOLID, inner = 0),
        model = (; kind = :solid, solid, interior = nothing),
        body_wavelength = 2π * boundary.speed_transversal_contrast / k, thickness = Inf)
end

function _volume_geometry(outer_axes, equatorial,
        boundary::Shelled{ElasticLayer, <:Union{FluidInterior, VacuumInterior}}, k;
        spheroid = nothing)
    layer = boundary.material
    solid = _volume_solid_parameters(layer.density_contrast,
        layer.speed_longitudinal_contrast, layer.speed_transversal_contrast)
    ratio = boundary.radius_ratio
    inner_equatorial = ratio * equatorial
    inner_axes = if spheroid === nothing
        (ratio * outer_axes[1], ratio * outer_axes[2])
    elseif spheroid.kind === :prolate
        (inner_equatorial, sqrt(inner_equatorial^2 + spheroid.q^2))
    else
        inner_equatorial > spheroid.q || throw(ArgumentError(
            "radius_ratio must exceed $(round(spheroid.q / spheroid.b; sigdigits = 4)) for the confocal inner surface of this oblate spheroid to exist"))
        (inner_equatorial, sqrt(inner_equatorial^2 - spheroid.q^2))
    end
    fluid = boundary.interior isa FluidInterior
    interior = if !fluid
        nothing
    elseif layer.interior_coupling === :identical_fluid
        (1.0, k)
    else
        (boundary.interior.density_contrast, k / boundary.interior.soundspeed_contrast)
    end
    return (; outer_axes, inner_axes,
        kinds = (; body = _REGION_SOLID, inner = fluid ? _REGION_INTERIOR : 0),
        model = (; kind = :shell, solid, interior),
        body_wavelength = 2π * layer.speed_transversal_contrast / k,
        thickness = equatorial - inner_equatorial)
end

# Smallest radius of curvature of a spheroid with equatorial and polar semi-axes (b, a).
_spheroid_curvature(axes) = min(axes[1]^2 / axes[2], axes[2]^2 / axes[1])

# Confocal spheroid with focal half-distance `c` and coordinate `xi`, as equatorial and polar semi-axes.
function _confocal_axes(kind, c, xi)
    return (c * sqrt(kind === :prolate ? xi^2 - 1 : xi^2 + 1), c * xi)
end

# Coordinate xi of the confocal spheroid whose polar (prolate) or equatorial (oblate) semi-axis is `length`.
function _confocal_coordinate(kind, c, length)
    return kind === :prolate ? length / c : sqrt((length / c)^2 - 1)
end

# Nested surfaces of the computational domain. A PML conforms to a spheroid as a confocal spheroid and is a sphere otherwise. With
# `:pml` the confocal shape is used when its domain is under 60% of the spherical volume. `lower` and `upper` are the levels between
# which the far-field cutoff of the confocal shape rises.
function _volume_domain(body, geometry, closure, wavelength, domain_radius, clearance,
        thickness, sigma, rotation)
    axes = geometry.outer_axes
    extent = maximum(axes)
    R_fluid = domain_radius === nothing ? 1.5 * extent : Float64(domain_radius)
    R_fluid > 1.05 * extent || throw(ArgumentError(
        "domain_radius must exceed the body's largest semi-axis by at least 5%"))
    R_out = closure === :dtn ? R_fluid : R_fluid + thickness
    spherical = (; kind = :spherical, c = 0.0, rotation, R = R_fluid,
        interface_axes = closure === :dtn ? nothing : (R_fluid, R_fluid),
        domain_axes = (R_out, R_out),
        pml = (; kind = :spherical, R = R_fluid, thickness, sigma0 = sigma))
    (closure in (:pml, :pml_spheroidal) && body isa Spheroid && axes[1] != axes[2]) ||
        return spherical
    kind = axes[2] > axes[1] ? :prolate : :oblate
    c = sqrt(abs(axes[2]^2 - axes[1]^2))
    gap = clearance === nothing ? max(wavelength, 0.5 * minimum(axes)) : Float64(clearance)
    length_body = kind === :prolate ? axes[2] : axes[1]
    xi_body = axes[2] / c
    xi_interface = _confocal_coordinate(kind, c, length_body + gap)
    xi_outer = _confocal_coordinate(kind, c, length_body + gap + thickness)
    domain_axes = _confocal_axes(kind, c, xi_outer)
    volume(a) = a[1]^2 * a[2]
    (closure === :pml_spheroidal ||
     volume(domain_axes) < 0.6 * volume(spherical.domain_axes)) ||
        return spherical
    return (; kind, c, rotation, interface_axes = _confocal_axes(kind, c, xi_interface),
        domain_axes, pml = (; kind, c, rotation, xi_interface, xi_outer, sigma0 = sigma),
        lower = xi_body + 0.15 * (xi_interface - xi_body),
        upper = xi_body + 0.85 * (xi_interface - xi_body))
end

function _fem_volume(body::Union{Sphere, Spheroid}, boundary, k::Real;
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0, closure::Symbol = :pml,
        domain_radius::Union{Nothing, Real} = nothing, clearance::Union{Nothing, Real} = nothing,
        points_per_wavelength::Real = 8, h::Union{Nothing, Real} = nothing,
        h_body::Union{Nothing, Real} = nothing, pml_thickness::Union{Nothing, Real} = nothing,
        pml_sigma::Real = 6.0, dtn_order::Union{Nothing, Integer} = nothing,
        solver::Symbol = :auto, ilu_tolerance::Real = 1e-3, solver_tolerance::Real = 1e-8)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    closure in (:pml, :pml_spherical, :pml_spheroidal, :dtn) || throw(ArgumentError(
        "closure must be :pml, :pml_spherical, :pml_spheroidal or :dtn, got $closure"))
    geometry = _volume_geometry(body, boundary, k)
    wavelength = 2π / k
    h_fluid = h === nothing ? wavelength / points_per_wavelength : Float64(h)
    curvature = minimum(_spheroid_curvature,
        filter(!isnothing, (geometry.outer_axes, geometry.inner_axes)))
    resolved = min(h_fluid, geometry.body_wavelength / points_per_wavelength,
        minimum(geometry.outer_axes) / 2, geometry.thickness / 1.5, 2curvature)
    h_solid = h_body === nothing ? resolved : Float64(h_body)
    thickness = pml_thickness === nothing ? wavelength : Float64(pml_thickness)
    rotation = _volume_rotation(incidence_angle, incidence_azimuth)
    domain = _volume_domain(body, geometry, closure, wavelength, domain_radius, clearance,
        thickness, Float64(pml_sigma), rotation)
    bodies = [(; axes, center = zeros(3), rotation, size = h_solid)
              for axes in filter(!isnothing, (geometry.outer_axes, geometry.inner_axes))]
    classify = function (centre)
        body_frame = rotation' * centre
        if geometry.inner_axes !== nothing &&
           _body_level(body_frame, geometry.inner_axes) < 1
            return (geometry.kinds.inner, 0)
        end
        _body_level(body_frame, geometry.outer_axes) < 1 && return (geometry.kinds.body, 0)
        return (_REGION_FLUID, 0)
    end
    return _volume_run(k, geometry.model, domain, bodies, classify, x -> Inf, h_fluid;
        closure, dtn_order, thickness, sigma = Float64(pml_sigma), rotation, incidence_angle,
        incidence_azimuth, solver, ilu_tolerance = Float64(ilu_tolerance),
        solver_tolerance = Float64(solver_tolerance))
end

# Meshes, assembles and solves the volume problem, and extracts the far field.
function _volume_run(k, model, domain, bodies, classify, region_size, h_fluid; closure,
        dtn_order, thickness, sigma, rotation, incidence_angle, incidence_azimuth,
        solver = :auto, ilu_tolerance = 1e-3, solver_tolerance = 1e-8)
    # Curved elements can invert near sharp features, so the surface sizes are reduced until every Jacobian is positive.
    local grid, labels, ids
    for shrink in (1.0, 0.75, 0.55, 0.4)
        # Curved boundaries also need a size well below their radius of curvature.
        sizes = (; fluid = h_fluid,
            interface = domain.interface_axes === nothing ? h_fluid :
                        min(h_fluid, _spheroid_curvature(domain.interface_axes) / 4),
            domain = min(h_fluid, _spheroid_curvature(domain.domain_axes) / 4))
        shrunk = [(; b.axes, b.center, b.rotation, size = shrink * b.size) for b in bodies]
        grid, labels, ids = _volume_mesh(shrunk, domain.interface_axes, domain.domain_axes,
            rotation, sizes, classify, region_size)
        _volume_valid_grid(grid) && break
        shrink == 0.4 && throw(ErrorException(
            "the volume mesh has inverted elements. Reduce h_body or increase domain_radius"))
    end
    on_outer(p) = abs(_body_level(rotation' * p, domain.domain_axes) - 1) < 1e-3
    facets = _volume_facets(grid, labels, on_outer)
    K, load, n_nodes, mass = _volume_system(grid, labels, ids, facets, k, model, domain.pml)
    fixed = Dict{Int, ComplexF64}()
    L = 0
    if domain.kind === :spherical
        L = dtn_order === nothing ? ceil(Int, k * domain.R) + 10 : Int(dtn_order)
    end
    if closure === :dtn
        K = K - _dtn_matrix(grid, facets[:outer], n_nodes, k, domain.R, L)
    else
        for node in _facet_nodes(grid, facets[:outer])
            fixed[_pressure_dof(node)] = 0
        end
    end
    if model.kind === :soft
        for node in _facet_nodes(grid, facets[:fluid_bare])
            fixed[_pressure_dof(node)] = -_volume_incident(grid.nodes[node].x, k)
        end
    end
    # The complex-shifted matrix that preconditions the iterative solver damps the wave terms.
    shifted() = K + im * mass
    solver in (:auto, :direct, :iterative) || throw(ArgumentError(
        "solver must be :auto, :direct or :iterative, got $solver"))
    solution, dofs, residual, solve_info = _volume_solve(
        K, load, grid, labels, n_nodes, fixed; solver = closure === :dtn ? :direct : solver,
        shifted, ilu_tolerance, tolerance = solver_tolerance)
    positive, negative, shell = if domain.kind === :spherical
        extraction = closure === :dtn ? facets[:outer] : facets[:pml_interface]
        (_volume_coefficients(solution, grid, extraction, n_nodes, L)..., nothing)
    else
        (ComplexF64[], ComplexF64[],
            _volume_extraction(solution, grid, labels, domain, domain.lower, domain.upper))
    end
    diagnostics = (; method = :volume, closure, shape = domain.kind, dofs,
        cells = count(!=(0), labels), h = h_fluid, h_body = minimum(b.size for b in bodies),
        pml_thickness = closure === :dtn ? 0.0 : thickness, pml_sigma = sigma, residual,
        solver = solve_info.solver, iterations = solve_info.iterations)
    return _VolumeFEMData(
        positive, negative, domain.kind === :spherical ? domain.R : 0.0, L,
        shell, rotation, Float64(incidence_angle), Float64(incidence_azimuth), diagnostics)
end

# Rotation taking the z axis to the unit vector `axis`.
function _axis_rotation(axis)
    u = axis / norm(axis)
    v = cross([0.0, 0.0, 1.0], u)
    c = u[3]
    norm(v) < 1e-12 &&
        return c > 0 ? Matrix{Float64}(I, 3, 3) : Matrix(Diagonal([1.0, -1.0, -1.0]))
    cross_matrix = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
    return I + cross_matrix + cross_matrix^2 * (1 - c) / norm(v)^2
end

# Slowest wave speed of a region material, which sets its mesh size.
_region_speed(material::FluidFilled) = material.soundspeed_contrast
_region_speed(material::SolidElastic) = material.speed_transversal_contrast

# Sphere or spheroid of a region as equatorial and polar semi-axes.
_region_axes(body::Sphere) = (body.radius, body.radius)
_region_axes(body::Spheroid) = (body.b, body.a)

# Solution-frame pose of every region for the incidence rotation.
function _region_poses(bodies, centers, orientations, rotation)
    return [(; axes = _region_axes(bodies[i]), center = rotation * Float64.(centers[i]),
                rotation = rotation * _axis_rotation(Float64.(orientations[i])))
            for i in eachindex(bodies)]
end

# Points on the surface of a posed body, for the containment checks.
function _region_surface_points(pose, count = 60)
    golden = pi * (3 - sqrt(5))
    return map(1:count) do i
        z = 1 - 2 * (i - 0.5) / count
        r = sqrt(1 - z^2)
        local_point = [
            pose.axes[1] * r * cos(golden * i), pose.axes[1] * r * sin(golden * i),
            pose.axes[2] * z]
        pose.rotation * local_point + pose.center
    end
end

# Nested and disjoint fluid and elastic regions: `parents[i]` is the region immediately outside region `i`, with 0 for the exterior.
function _fem_volume_regions(bodies, materials, k::Real;
        parents::AbstractVector{<:Integer} = collect(0:(length(bodies) - 1)),
        centers = [zeros(3) for _ in bodies],
        orientations = [[0.0, 0.0, 1.0] for _ in bodies],
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0, closure::Symbol = :pml,
        domain_radius::Union{Nothing, Real} = nothing, points_per_wavelength::Real = 8,
        h::Union{Nothing, Real} = nothing, h_body::Union{Nothing, Real} = nothing,
        pml_thickness::Union{Nothing, Real} = nothing, pml_sigma::Real = 6.0,
        dtn_order::Union{Nothing, Integer} = nothing,
        solver::Symbol = :auto, ilu_tolerance::Real = 1e-3, solver_tolerance::Real = 1e-8)
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    closure in (:pml, :pml_spherical, :dtn) || throw(ArgumentError(
        "closure must be :pml, :pml_spherical or :dtn for coupled regions, got $closure"))
    n = length(bodies)
    n > 0 &&
    length(materials) == length(parents) == length(centers) ==
    length(orientations) == n || throw(ArgumentError(
        "bodies, materials, parents, centers and orientations must have one entry per region"))
    all(p -> 0 <= p <= n, parents) && all(i -> parents[i] != i, 1:n) ||
        throw(ArgumentError("parents must index regions or be 0 for the exterior"))
    ancestors = map(1:n) do i
        chain, j = Int[], parents[i]
        while j != 0
            j in chain && throw(ArgumentError("parents must not contain a cycle"))
            push!(chain, j)
            j = parents[j]
        end
        chain
    end
    rotation = _volume_rotation(incidence_angle, incidence_azimuth)
    poses = _region_poses(bodies, centers, orientations, rotation)
    level(i, x) = _body_level(poses[i].rotation' * (x - poses[i].center), poses[i].axes)
    for i in 1:n, j in 1:n

        i == j && continue
        inside = all(x -> level(j, x) < 1, _region_surface_points(poses[i]))
        if j in ancestors[i]
            inside || throw(ArgumentError("region $i must lie inside its parent region $j"))
        elseif !(i in ancestors[j])
            any(x -> level(j, x) < 1.0, _region_surface_points(poses[i])) &&
                throw(ArgumentError("regions $i and $j overlap without being nested"))
        end
    end
    wavelength = 2π / k
    h_fluid = h === nothing ? wavelength / points_per_wavelength : Float64(h)
    region_h = [min(h_fluid, wavelength * _region_speed(materials[i]) /
                             points_per_wavelength)
                for i in 1:n]
    surface_size = [h_body === nothing ?
                    min(region_h[i], minimum(poses[i].axes) / 2,
                        2 * _spheroid_curvature(poses[i].axes)) : Float64(h_body)
                    for i in 1:n]
    # A layer between a region and its child needs elements thinner than the layer.
    for i in 1:n
        parents[i] == 0 && continue
        gap = minimum(_region_surface_points(poses[i])) do x
            local_point = poses[parents[i]].rotation' * (x - poses[parents[i]].center)
            norm(local_point) *
            (1 / sqrt(_body_level(local_point, poses[parents[i]].axes)) - 1)
        end
        for j in (i, parents[i])
            region_h[j] = min(region_h[j], gap / 1.5)
            surface_size[j] = min(surface_size[j], gap / 1.5)
        end
    end
    extent = maximum(norm(poses[i].center) + maximum(poses[i].axes) for i in 1:n)
    R_fluid = domain_radius === nothing ? 1.5 * extent : Float64(domain_radius)
    R_fluid > 1.05 * extent || throw(ArgumentError(
        "domain_radius must exceed the largest distance from the origin to a region surface by at least 5%"))
    thickness = pml_thickness === nothing ? wavelength : Float64(pml_thickness)
    R_out = closure === :dtn ? R_fluid : R_fluid + thickness
    domain = (; kind = :spherical, c = 0.0, rotation, R = R_fluid,
        interface_axes = closure === :dtn ? nothing : (R_fluid, R_fluid),
        domain_axes = (R_out, R_out),
        pml = (; kind = :spherical, R = R_fluid, thickness, sigma0 = Float64(pml_sigma)))
    depth = length.(ancestors)
    innermost(x) = begin
        inside = [i for i in 1:n if level(i, x) < 1]
        isempty(inside) ? 0 : inside[argmax(depth[inside])]
    end
    is_solid = [m isa SolidElastic for m in materials]
    classify(x) = (i = innermost(x);
        i == 0 ? (_REGION_FLUID, 0) : (is_solid[i] ? _REGION_SOLID : _REGION_INTERIOR, i))
    region_size(x) = (i = innermost(x); i == 0 ? Inf : region_h[i])
    bodies_run = [(; poses[i].axes, poses[i].center,
                      poses[i].rotation, size = surface_size[i])
                  for i in 1:n]
    model = (; kind = :regions,
        solid = [m isa SolidElastic ?
                 _volume_solid_parameters(
                     m.density_contrast, m.speed_longitudinal_contrast,
                     m.speed_transversal_contrast) : nothing for m in materials],
        interior = [m isa FluidFilled ? (m.density_contrast, k / m.soundspeed_contrast) :
                    nothing for m in materials])
    return _volume_run(k, model, domain, bodies_run, classify, region_size, h_fluid;
        closure, dtn_order, thickness, sigma = Float64(pml_sigma), rotation, incidence_angle,
        incidence_azimuth, solver, ilu_tolerance = Float64(ilu_tolerance),
        solver_tolerance = Float64(solver_tolerance))
end
