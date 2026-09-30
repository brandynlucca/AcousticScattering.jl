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
    incident::Union{Nothing, IncidentField}
end

function _RegionBEMData(interfaces, beta, alpha, report)
    _RegionBEMData(interfaces, beta, alpha, report, nothing)
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
        correction=(method=:dim,), formulation=:muller, validation=(;),
        compression=(method=:none,), gmres_kwargs=(;))

Coupled fluid transmission across closed, disjoint full-3D interface meshes. Surface `i`
encloses region `i`. `parents[i]` is the region immediately outside it, with `0` denoting the
unbounded exterior. The default `parents` is a nested chain. `materials[i]` gives that region's
density and sound-speed contrasts relative to the unbounded exterior. Only homogeneous, lossless
scalar fluids are supported.

Meshes may have independent shapes, origins and orientations, but must not intersect or touch.
`validation=(maxdepth=20, maxwork=200000)` controls the geometry checks.

Dense LU is the default. For Müller with density interpolation,
`compression=(method=:hmatrix, tol=1e-8)` compresses self and cross-interface operators.
Close-surface corrections and the low-frequency reconstruction guards are preserved.
Right-preconditioned GMRES uses local cluster blocks and independent degree-two harmonic
coarse spaces on each interface, coupled through the full operator. `gmres_kwargs` controls
iteration tolerances, restart and iteration limit as in single-interface fluid BEM.
The coupled default restart is 600; the other iterative defaults are unchanged.
Equilibration uses local-block maxima; dense SVD condition estimates are skipped.
Interior and exterior representation residuals are evaluated from the compressed blocks,
without storing a second global dense matrix. They do not estimate compression error.

Returns a [`BEMSolution`](@ref). Post-process with `scattering_amplitude(sol; direction)` or
`target_strength(sol; direction)`.

See [Coupled fluid regions](@ref boundary-theory) for the coupling equations,
`formulation` and `correction` tradeoffs, and diagnostics fields.

# Example
```julia
bem(surfaces, materials, k; parents = [0, 1, 1])
```
"""
function bem(
        surfaces::AbstractVector{<:Mesh}, materials::AbstractVector{<:FluidFilled}, k::Real;
        parents::AbstractVector{<:Integer} = collect(0:(length(surfaces) - 1)),
        incidence_angle::Real = π / 2, incidence_azimuth::Real = 0.0,
        incident = nothing, transducer = nothing,
        equilibrate::Bool = true, condition_limit::Integer = 512,
        correction::NamedTuple = (method = :dim,), formulation::Symbol = :muller,
        validation::NamedTuple = (;), compression::NamedTuple = (method = :none,),
        gmres_kwargs::NamedTuple = (;))
    all(isfinite, (incidence_angle, incidence_azimuth)) ||
        throw(ArgumentError("incidence angles must be finite"))
    condition_limit >= 0 || throw(ArgumentError("condition_limit must be nonnegative"))
    system = _assemble_region_bem(surfaces, materials, k; parents, correction,
        formulation, validation, compression)
    factor = _factor_full_fluid(system; equilibrate, condition_limit, gmres_kwargs,
        norm_floor = eps(Float64))
    return _solve_region_bem(
        system, factor; incidence_angle, incidence_azimuth, incident, transducer)
end

function _assemble_region_bem(
        surfaces::AbstractVector{<:Mesh}, materials::AbstractVector{<:FluidFilled}, k::Real;
        parents::AbstractVector{<:Integer} = collect(0:(length(surfaces) - 1)),
        correction::NamedTuple = (method = :dim,), formulation::Symbol = :muller,
        validation::NamedTuple = (;), compression::NamedTuple = (method = :none,))
    count = length(surfaces)
    count > 0 || throw(ArgumentError("at least one interface is required"))
    length(materials) == length(parents) == count ||
        throw(ArgumentError("supply one material and parent per interface"))
    all(i -> 0 <= parents[i] < i, 1:count) ||
        throw(ArgumentError("each parent must be zero or an earlier region index"))
    isfinite(k) && k > 0 || throw(ArgumentError("k must be finite and positive"))
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
    regular = formulation === :muller && all(
        region -> _fluid_regular_range(k / speeds[region], region_sizes[region]), eachindex(speeds))
    reconstruct = all(region -> k / speeds[region] * region_sizes[region].radius <= pi/2,
        eachindex(speeds))
    if compression.method !== :none
        metadata = (; surfaces, materials, parents, k, formulation, correction,
            quads, ranges, n, densities, geometry)
        return _assemble_compressed_regions(
            metadata, speeds; regular, reconstruct, compression)
    end
    A = formulation === :muller ? Matrix{ComplexF64}(I, 2n, 2n) : zeros(ComplexF64, 2n, 2n)
    inner = formulation === :muller ? 0.5 .* A : nothing
    derivative_evaluation = NamedTuple[]
    for i in 1:count
        rows = ranges[i]
        for region in (i, parents[i])
            density = densities[region + 1]
            op = Inti.Helmholtz(; k = k / speeds[region + 1], dim = 3)
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
                    op, quads[i], quads[j], S, D, options; regular, reconstruct)
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
    return (; A, inner, surfaces, materials, parents, k, formulation, correction,
        quads, ranges, n, densities, derivative_evaluation, geometry)
end

function _region_incident_rhs(system, incidence_angle, incidence_azimuth; incident = nothing)
    (; parents, k, formulation, quads, ranges, n) = system
    b = zeros(ComplexF64, 2n)
    for i in eachindex(quads)
        parents[i] == 0 || continue
        rows = ranges[i]
        p, dp = _incident_traces(quads[i], k, incidence_angle, incidence_azimuth, incident)
        b[rows] = p
        if formulation === :muller
            b[rows .+ n] = dp
        end
    end
    return b
end

function _solve_region_bem(system, factor;
        incidence_angle::Real = pi / 2, incidence_azimuth::Real = 0.0,
        incident = nothing, transducer = nothing)
    incident = _resolve_incident(
        system.k, incidence_angle, incidence_azimuth; incident, transducer)
    b = _region_incident_rhs(system, incidence_angle, incidence_azimuth; incident)
    solved = hasproperty(system, :compression) ? _solve_compressed_fluid_system(factor, b) :
             _solve_fluid_system(factor, b)
    return _region_bem_solution(
        system, factor, b, solved, incidence_angle, incidence_azimuth; incident)
end

function _region_bem_solution(
        system, factor, b, solved, incidence_angle, incidence_azimuth;
        incident = nothing)
    (; A, inner, surfaces, materials, parents, k, formulation, correction,
        ranges, n, densities, derivative_evaluation, geometry) = system
    (; x) = solved
    count = length(surfaces)
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
    iterative = hasproperty(solved, :history)
    if iterative && !solved.history.isconverged
        @warn "Compressed coupled-fluid BEM GMRES did not converge" iterations=solved.history.iters residual=norm(residual)/max(
            norm(b), eps(Float64))
    end
    report = merge(
        _fluid_residual_report(residual, b, factor.row_norms), factor.diagnostics,
        (; method = iterative ? :gmres : :direct, formulation,
            illumination = incident === nothing ? :plane_wave : :prescribed,
            converged = iterative ? solved.history.isconverged : nothing,
            iterations = iterative ? solved.history.iters : nothing,
            residual_history = iterative ? solved.history[:resnorm] : Float64[],
            unknown_count = 2n, quadrature_nodes = n, interface_count = count,
            interface_residuals, derivative_evaluation, geometry, correction,
            compression = get(system, :compression, (method = :none,)),
            solver_options = merge(iterative ? factor.options : (;), (;
                incidence_angle, incidence_azimuth))))
    data = _RegionBEMData(
        interfaces, Float64(incidence_angle), Float64(incidence_azimuth), report, incident)
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
        p_inc, q_inc = _incident_traces(quad, sol.k, sol.data.incidence_angle,
            sol.data.incidence_azimuth, sol.data.incident)
        amplitude += far_field(quad, observation, sol.k, interface.pressure - p_inc,
            interface.normal_derivative_exterior - q_inc)
    end
    return amplitude
end

function target_strength(sol::BEMSolution{_RegionBEMData}; kwargs...)
    return target_strength(scattering_amplitude(sol; kwargs...))
end
