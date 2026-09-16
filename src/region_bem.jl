struct _RegionGeometry <: AbstractBody
    surfaces::Vector{Mesh}
    parents::Vector{Int}
end

struct _FluidRegions <: AbstractBoundaryCondition
    materials::Vector{FluidFilled}
end

struct _RegionBEMData
    interfaces::Vector{NamedTuple}
    incidence_angle::Float64
    incidence_azimuth::Float64
    diagnostics::NamedTuple
end

function _region_ancestors(parents, i)
    ancestors = Int[]
    while parents[i] != 0
        i = parents[i]
        push!(ancestors, i)
    end
    return ancestors
end

function _region_patches(quad, offset)
    msh = Inti.mesh(quad)
    patches = _SurfacePatch[]
    for E in Inti.element_types(msh)
        degree = Inti.order(E)
        conversion = _surface_bernstein_conversion(degree)
        for index in eachindex(Inti.elements(msh, E))
            indices = Inti.connectivity(msh, E)[:, index]
            values = [_SurfaceBound.(Inti.nodes(msh)[i]) for i in indices]
            corners = Tuple(indices[collect(Inti.vertices_idxs(E))] .+ offset)
            push!(patches,
                _SurfacePatch(_surface_transform(conversion, values),
                    degree, corners, length(patches) + 1))
        end
    end
    return patches
end

function _region_contains(quad, point)
    _, D = Inti.single_double_layer(; op = Inti.Laplace(; dim = 3),
        target = [point], source = quad, compression = (method = :none,),
        correction = (method = :adaptive, maxdist = Inf, atol = 1e-9))
    winding = -sum(D)
    min(abs(winding), abs(winding - 1)) < 1e-5 ||
        throw(ArgumentError("interface containment is unresolved; refine the surface"))
    return winding > 0.5
end

function _validate_regions(surfaces, parents; maxdepth::Integer = 20, maxwork::Integer = 200000)
    0 <= maxdepth <= 24 ||
        throw(ArgumentError("validation maxdepth must be between 0 and 24"))
    maxwork > 0 || throw(ArgumentError("validation maxwork must be positive"))
    patches = Vector{_SurfacePatch}[]
    offset = 0
    for surface in surfaces
        surface.method == :full && surface.data isa Inti.Quadrature ||
            throw(ArgumentError("each interface must be a full-3D surface mesh"))
        if !(surface.body isa _SurfaceGeometry)
            _validate_surface_patches(Inti.mesh(surface.data); maxdepth, maxwork)
        end
        push!(patches, _region_patches(surface.data, offset))
        offset += length(Inti.nodes(Inti.mesh(surface.data)))
    end
    state = _SurfaceValidation(Dict(n => _surface_bernstein_split(n) for n in 0:4),
        Dict{Tuple{Int, Int}, Int}(), offset, maxdepth, maxwork, 0, 0)
    ancestors = [_region_ancestors(parents, i) for i in eachindex(parents)]
    for i in eachindex(surfaces), j in (i + 1):length(surfaces)

        for a in patches[i], b in patches[j]

            _surface_boxes_separated(a, b) || _surface_pair(a, b, state)
        end
        for (outer, inner) in ((i, j), (j, i))
            contains = _region_contains(surfaces[outer].data, first(surfaces[inner].data).coords)
            contains == (outer in ancestors[inner]) ||
                throw(ArgumentError("interfaces $i and $j do not match the specified parents"))
        end
    end
    return (;
        intersection_check = :adaptive_bernstein, containment_check = :adaptive_winding,
        maxdepth, maxwork, depth = state.depth, work = state.work)
end

"""
    bem(surfaces::AbstractVector{<:Mesh}, materials::AbstractVector{<:FluidFilled}, k;
        parents=collect(0:length(surfaces)-1), incidence_angle=π/2,
        incidence_azimuth=0, equilibrate=true, condition_limit=512,
        correction=(method=:dim,), formulation=:muller, validation=(;))

Coupled fluid transmission across closed, disjoint full-3D interface meshes. Surface `i`
encloses region `i`; `parents[i]` identifies the region immediately outside it, with `0`
denoting the unbounded exterior. Parents must precede their children. The default is a
nested chain; for two inclusions in one body use `parents=[0,1,1]`.
Each normal points from the enclosed region into its parent. Every interface enforces
continuity of pressure and normal velocity. All material density and sound-speed contrasts
are relative to the unbounded exterior, including materials of nested regions.

Meshes may have independent shapes, origins and orientations. Intersecting/touching surfaces,
inconsistent containment and unresolved geometry checks raise `ArgumentError`.
`validation=(maxdepth=20, maxwork=200000)` controls the separation checks.
Only homogeneous, lossless scalar fluids are supported; elastic and viscous walls need
their corresponding structural or constitutive equations.

The default dense Müller system has two unknowns per quadrature node. All interfaces bounding
each region interact. `correction` controls singular and near-singular integration;
density interpolation receives the target's location from the region topology.
Refine geometry and quadrature independently, especially at small gaps and resonances.
For self-interactions with density interpolation, regional `k*radius <= 1` and
`2k*rms_radius <= 1`,
normal derivatives follow from Calderón identities with the pressure operators;
both radius bounds include all boundary nodes of the region and each interface separately.
Radii use the corresponding mean node position, with surface area weights for the RMS. Other
interactions use direct derivative quadrature. Pressure and derivative corrections
share one low-frequency regime throughout a region.
Diagnostics record each derivative route in `derivative_evaluation`.

`formulation=:cbie` instead enforces the pressure representation on both sides of each
interface, with the same shared pressure and density-scaled derivative unknowns. It avoids
hypersingular operators and is useful for low-frequency, strong-contrast fluids. This
conventional formulation has no general protection against fictitious eigenfrequencies;
check against an independent method when extending its frequency range.

Returns a [`BEMSolution`](@ref). `sol.data.interfaces[i]` contains the `surface`, `interior`
and `exterior` region indices, complex total `pressure`, and `normal_derivative_interior`
and `normal_derivative_exterior`, both measured along the stored outward normal.
[`diagnostics`](@ref) includes separate pressure and density-scaled derivative representation
residuals on each side of every interface. Continuity is built into the shared unknowns;
representation residuals measure a different discretization error from the linear residual.
Reconstructed flux residuals depend on the same pressure operators, so they do not
provide an independent check of quadrature accuracy.
For `:cbie`, the pressure representations are the solved equations themselves; their
residuals are algebraic checks, and unsampled flux representation residuals are `nothing`.
Post-process with `scattering_amplitude(sol; direction)` or `target_strength(sol; direction)`.

For concentric spheres, `[FluidFilled(g_shell,h_shell), FluidFilled(g_core,h_core)]`
corresponds to `Shelled(FluidLayer(g_shell,h_shell), FluidInterior(g_core,h_core), b/a)`.
The supplied meshes define `a` and `b`; general interfaces need no radius ratio.
"""
function bem(
        surfaces::AbstractVector{<:Mesh}, materials::AbstractVector{<:FluidFilled}, k::Real;
        parents::AbstractVector{<:Integer} = collect(0:(length(surfaces) - 1)),
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        equilibrate::Bool = true, condition_limit::Integer = 512,
        correction::NamedTuple = (method = :dim,), formulation::Symbol = :muller,
        validation::NamedTuple = (;))
    count = length(surfaces)
    count > 0 || throw(ArgumentError("at least one interface is required"))
    length(materials) == length(parents) == count ||
        throw(ArgumentError("supply one material and parent per interface"))
    all(i -> 0 <= parents[i] < i, 1:count) ||
        throw(ArgumentError("each parent must be zero or an earlier region index"))
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
    all(isfinite, (incidence_angle, incidence_azimuth)) ||
        throw(ArgumentError("incidence angles must be finite"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    formulation in (:muller, :cbie) ||
        throw(ArgumentError("formulation must be :muller or :cbie"))
    all(
        m -> isfinite(m.density_contrast) && m.density_contrast > 0 &&
             isfinite(m.soundspeed_contrast) && m.soundspeed_contrast > 0,
        materials) ||
        throw(ArgumentError("fluid contrasts must be finite and positive"))
    geometry = _validate_regions(surfaces, parents; validation...)
    quads = [surface.data for surface in surfaces]
    source_sizes = _fluid_quadrature_size.(quads)
    region_sizes = map(0:count) do region
        boundaries = [j for j in 1:count if j == region || parents[j] == region]
        combined = _fluid_quadrature_size(Iterators.flatten(quads[boundaries]))
        (;
            radius = max(combined.radius, maximum(source_sizes[j].radius
            for j in boundaries)),
            rms = max(combined.rms, maximum(source_sizes[j].rms for j in boundaries)))
    end
    offsets = cumsum([0; length.(quads)])
    n = last(offsets)
    ranges = [(offsets[i] + 1):offsets[i + 1] for i in 1:count]
    densities = [1.0; getproperty.(materials, :density_contrast)]
    speeds = [1.0; getproperty.(materials, :soundspeed_contrast)]
    direction = _bem3d_incidence_direction(incidence_angle, incidence_azimuth)
    A = formulation === :muller ? Matrix{ComplexF64}(I, 2n, 2n) : zeros(ComplexF64, 2n, 2n)
    inner = formulation === :muller ? 0.5 .* A : nothing
    b = zeros(ComplexF64, 2n)
    derivative_evaluation = NamedTuple[]
    for i in 1:count
        rows = ranges[i]
        if parents[i] == 0
            b[rows] = [cis(k * dot(direction, q.coords)) for q in quads[i]]
            if formulation === :muller
                b[rows .+ n] = [im * k * dot(direction, q.normal) * b[rows[t]]
                                for (t, q) in enumerate(quads[i])]
            end
        end
        for region in (i, parents[i])
            density = densities[region + 1]
            op = Inti.Helmholtz(; k = k / speeds[region + 1], dim = 3)
            regular = formulation === :muller &&
                      _fluid_regular_range(op.k, region_sizes[region + 1])
            equations = region == i ? rows .+ n : rows
            if formulation === :cbie
                A[equations, rows] .+= 0.5 .*
                                       Matrix{ComplexF64}(I, length(rows), length(rows))
            end
            for j in 1:count
                (j == region || parents[j] == region) || continue
                sign = j == region ? 1 : -1
                columns = ranges[j]
                location = i == j ? :on : (j == region ? :inside : :outside)
                options = correction.method == :dim ?
                          merge((; maxdist = Inf), correction, (;
                    target_location = location)) : correction
                S, D = _fluid_layer_operators(op, quads[i], quads[j], options;
                    regular)
                if formulation === :cbie
                    A[equations, columns] .+= sign .* D
                    A[equations, columns .+ n] .-= (sign * density) .* S
                    continue
                end
                K, H, method = _fluid_derivative_operators(
                    op, quads[i], quads[j], S, D, options; regular)
                push!(derivative_evaluation, (; target = i, source = j, region, method))
                for matrix in (region == i ? (A, inner) : (A,))
                    matrix[rows, columns] .+= sign .* D
                    matrix[rows, columns .+ n] .-= (sign * density) .* S
                    matrix[rows .+ n, columns] .+= (sign / density) .* H
                    matrix[rows .+ n, columns .+ n] .-= sign .* K
                end
            end
        end
    end
    if equilibrate
        row_norms = max.(vec(maximum(abs, A; dims = 2)), eps(Float64))
        scaled_A = A ./ row_norms
        col_norms = max.(vec(maximum(abs, scaled_A; dims = 1)), eps(Float64))
        scaled_A ./= transpose(col_norms)
        scaled_b = b ./ row_norms
        scaled_x = scaled_A \ scaled_b
        x = scaled_x ./ col_norms
    else
        scaled_A, scaled_b = A, b
        x = scaled_x = A \ b
    end
    residual = A * x - b
    interior_residual = formulation === :muller ? inner * x : nothing
    exterior_residual = formulation === :muller ? residual - interior_residual : nothing
    interfaces = NamedTuple[]
    interface_residuals = NamedTuple[]
    for i in 1:count
        rows = ranges[i]
        p, v = x[rows], x[rows .+ n]
        pressure_scale = max(norm(p), eps(Float64))
        flux_scale = max(norm(v), k * pressure_scale / densities[i + 1], eps(Float64))
        push!(interfaces,
            (; surface = surfaces[i], interior = i, exterior = parents[i],
                pressure = p, normal_derivative_interior = densities[i + 1] .* v,
                normal_derivative_exterior = densities[parents[i] + 1] .* v))
        representation = if formulation === :muller
            (;
                pressure_interior = norm(interior_residual[rows]) / pressure_scale,
                pressure_exterior = norm(exterior_residual[rows]) / pressure_scale,
                flux_interior = norm(interior_residual[rows .+ n]) / flux_scale,
                flux_exterior = norm(exterior_residual[rows .+ n]) / flux_scale)
        else
            (; pressure_interior = norm(residual[rows .+ n]) / pressure_scale,
                pressure_exterior = norm(residual[rows]) / pressure_scale,
                flux_interior = nothing, flux_exterior = nothing)
        end
        push!(interface_residuals, representation)
    end
    compute_condition = 2n <= condition_limit
    condition_number = compute_condition ? cond(A) : nothing
    scaled_condition_number = compute_condition ?
                              (equilibrate ? cond(scaled_A) : condition_number) : nothing
    scaled_report = _linear_residual(scaled_A, scaled_x, scaled_b)
    report = merge(_linear_residual(A, x, b),
        (; method = :direct, formulation,
            equilibrate, condition_limit, condition_number, scaled_condition_number,
            conditioning = compute_condition ? :svd : :not_computed,
            scaled_relative_residual = scaled_report.relative_residual,
            scaled_absolute_residual = scaled_report.absolute_residual,
            converged = nothing, iterations = nothing, residual_history = Float64[],
            unknown_count = 2n, quadrature_nodes = n, interface_count = count,
            interface_residuals, derivative_evaluation, geometry, correction, compression = (method = :none,),
            solver_options = (; incidence_angle, incidence_azimuth)))
    data = _RegionBEMData(interfaces, Float64(incidence_angle), Float64(incidence_azimuth), report)
    return BEMSolution(_RegionGeometry(collect(surfaces), collect(parents)),
        _FluidRegions(collect(materials)), Float64(k), :full, data)
end

function scattering_amplitude(sol::BEMSolution{_RegionBEMData};
        direction::Union{Nothing, AbstractVector} = nothing)
    incident = _bem3d_incidence_direction(sol.data.incidence_angle, sol.data.incidence_azimuth)
    observation = direction === nothing ? -incident : direction
    length(observation) == 3 && all(isfinite, observation) &&
    isapprox(norm(observation), 1) ||
        throw(ArgumentError("direction must be a finite unit vector with three components"))
    amplitude = zero(ComplexF64)
    for interface in sol.data.interfaces
        interface.exterior == 0 || continue
        quad = interface.surface.data
        p_inc = [cis(sol.k * dot(incident, q.coords)) for q in quad]
        q_inc = [im * sol.k * dot(incident, q.normal) * p_inc[i]
                 for (i, q) in enumerate(quad)]
        amplitude += far_field(quad, observation, sol.k, interface.pressure - p_inc,
            interface.normal_derivative_exterior - q_inc)
    end
    return amplitude
end

function target_strength(sol::BEMSolution{_RegionBEMData}; kwargs...)
    return target_strength(scattering_amplitude(sol; kwargs...))
end
