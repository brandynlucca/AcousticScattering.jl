# GeometryBasics is a Makie dependency, not a direct dependency of this package; access its
# types through Makie's own binding rather than adding a new project dependency for two names.
const Point3f = Makie.GeometryBasics.Point3f
const TriangleFace = Makie.GeometryBasics.TriangleFace

function _field_values(field_complex::AbstractArray, field::Symbol)
    field === :pressure_magnitude && return abs.(field_complex)
    field === :pressure_phase && return angle.(field_complex)
    field === :pressure_real && return real.(field_complex)
    field === :pressure_imag && return imag.(field_complex)
    throw(ArgumentError(
        "field must be :pressure_magnitude, :pressure_phase, :pressure_real, or :pressure_imag, got $field"))
end

# --- Revolved (axisymmetric) surfaces: BEM/MFS axisymmetric, shell FEM, Mesh{MeridianMesh} ---

@recipe(RevolvedSurfacePlot, x, y, z, values) do scene
    Attributes(colormap = _MAGNITUDE_COLORMAP, colorrange = nothing)
end

function Makie.plot!(plot::RevolvedSurfacePlot)
    x, y, z, values = plot.x[], plot.y[], plot.z[], plot.values[]
    if values === nothing
        surface!(plot, x, y, z; color = fill(RGBAf(0.6, 0.6, 0.6, 1), size(x)))
    else
        colorrange = plot.colorrange[] === nothing ? _default_colorrange(values) : plot.colorrange[]
        surface!(plot, x, y, z; color = values, colormap = plot.colormap, colorrange = colorrange)
    end
    return plot
end

Makie.preferred_axis_type(::RevolvedSurfacePlot) = Axis3
Makie.preferred_axis_attributes(::Type{Axis3}, ::RevolvedSurfacePlot) = (aspect = :data,)

function _revolved_surface_plot_data(ps::Vector{AcousticScattering.Panel},
        modes::Union{Nothing, AbstractVector}, field::Union{Nothing, Symbol})
    dummy_modes = modes === nothing ? [zeros(length(ps))] : modes
    surf = AcousticScattering.revolve_panels(ps, dummy_modes)
    values = field === nothing ? nothing : _field_values(surf.field, field)
    return surf.x, surf.y, surf.z, values
end

# --- Native triangulated surfaces: full 3D BEM, Mesh{<:Inti.Quadrature} ---

@recipe(TriMeshPlot, points, faces, values) do scene
    Attributes(colormap = _MAGNITUDE_COLORMAP, colorrange = nothing)
end

function Makie.plot!(plot::TriMeshPlot)
    points, faces, values = plot.points[], plot.faces[], plot.values[]
    if values === nothing
        mesh!(plot, points, faces; color = RGBAf(0.6, 0.6, 0.6, 1))
    else
        colorrange = plot.colorrange[] === nothing ? _default_colorrange(values) : plot.colorrange[]
        mesh!(plot, points, faces; color = values, colormap = plot.colormap, colorrange = colorrange)
    end
    return plot
end

Makie.preferred_axis_type(::TriMeshPlot) = Axis3
Makie.preferred_axis_attributes(::Type{Axis3}, ::TriMeshPlot) = (aspect = :data,)

function _inti_mesh_points_faces(quad)
    msh = quad.mesh
    points = [Point3f(p...) for p in Inti.nodes(msh)]
    faces = TriangleFace{Int}[]
    for E in Inti.element_types(msh)
        E <: SVector && continue
        idxs = Inti.vertices_idxs(E)
        connec = Inti.connectivity(msh, E)
        for j in axes(connec, 2)
            push!(faces, TriangleFace{Int}(connec[idxs[1], j], connec[idxs[2], j], connec[idxs[3], j]))
        end
    end
    return points, faces
end

# Makie.mesh!'s color array must be per-VERTEX (length(points)), not per-face — confirmed
# empirically, a per-face-length array is silently misindexed as per-vertex without erroring
# whenever there happen to be at least as many faces as vertices (the typical case for a closed
# triangulated surface), producing meaningless scrambled coloring rather than a clear error. A
# quadrature rule generally has several nodes per face (e.g. qorder=4), not one node per vertex,
# so there is no single quadrature node at any given vertex either: first average each face's own
# quadrature nodes, then average those per-face values over every face touching each vertex.
function _inti_mesh_vertex_field(quad, field_complex::AbstractArray, field::Symbol)
    msh = quad.mesh
    per_node = _field_values(field_complex, field)
    npts = length(Inti.nodes(msh))
    sums = zeros(Float64, npts)
    counts = zeros(Int, npts)
    for E in Inti.element_types(msh)
        E <: SVector && continue
        idxs = Inti.vertices_idxs(E)
        connec = Inti.connectivity(msh, E)
        qtags = Inti.etype2qtags(quad, E)
        for j in axes(connec, 2)
            tags = @view qtags[:, j]
            face_avg = sum(per_node[tags]) / length(tags)
            for vi in idxs
                v = connec[vi, j]
                sums[v] += face_avg
                counts[v] += 1
            end
        end
    end
    return sums ./ max.(counts, 1)
end

# --- Point clouds: bent-cylinder MFS surface field (no face connectivity) ---

@recipe(PointCloudPlot, points, values) do scene
    Attributes(colormap = _MAGNITUDE_COLORMAP, colorrange = nothing, markersize = 6)
end

function Makie.plot!(plot::PointCloudPlot)
    points, values = plot.points[], plot.values[]
    if values === nothing
        scatter!(plot, points; color = RGBAf(0.6, 0.6, 0.6, 1), markersize = plot.markersize)
    else
        colorrange = plot.colorrange[] === nothing ? _default_colorrange(values) : plot.colorrange[]
        scatter!(plot, points; color = values, colormap = plot.colormap,
            colorrange = colorrange, markersize = plot.markersize)
    end
    return plot
end

Makie.preferred_axis_type(::PointCloudPlot) = Axis3
Makie.preferred_axis_attributes(::Type{Axis3}, ::PointCloudPlot) = (aspect = :data,)

# --- Solution/Mesh -> geometry tier dispatch (see the plan's 3D scope matrix) ---

const _RevolvableSolution = Union{
    AcousticScattering.BEMSolution{AcousticScattering._AxisymmetricSurfaceData},
    AcousticScattering.MFSSolution{AcousticScattering._AxisymmetricSurfaceData}}

function _solution_render(sol::_RevolvableSolution, field::Union{Nothing, Symbol})
    d = sol.data
    ps = AcousticScattering.panels(d.mesh)
    x, y, z, values = _revolved_surface_plot_data(ps, field === nothing ? nothing : d.p_scat_modes, field)
    return (:revolved, x, y, z, values)
end

function _solution_render(
        sol::AcousticScattering.FEMSolution{AcousticScattering._ShellFEMSurfaceData},
        field::Union{Nothing, Symbol})
    d = sol.data
    x, y, z, values = _revolved_surface_plot_data(
        d.ps_ext, field === nothing ? nothing : d.p_ext_modes, field)
    return (:revolved, x, y, z, values)
end

function _solution_render(
        sol::AcousticScattering.BEMSolution{AcousticScattering._FullBEMSurfaceData},
        field::Union{Nothing, Symbol})
    quad = sol.data.quad
    points, faces = _inti_mesh_points_faces(quad)
    values = field === nothing ? nothing : _inti_mesh_vertex_field(quad, sol.data.p_scat, field)
    return (:trimesh, points, faces, values)
end

function _solution_render(
        sol::AcousticScattering.MFSSolution{AcousticScattering._BentMFSSurfaceData},
        field::Union{Nothing, Symbol})
    if field === nothing
        m = AcousticScattering.mesh(sol.body; k = sol.k)
        return _mesh_render(m)
    end
    points = [Point3f(p...) for p in sol.data.points]
    values = _field_values(sol.data.p_scat, field)
    return (:points, points, values)
end

function _solution_render(
        sol::Union{AcousticScattering.ModalSolution, AcousticScattering.KirchhoffSolution},
        field::Union{Nothing, Symbol})
    field === nothing || throw(ArgumentError(
        "plot(::$(nameof(typeof(sol))); kind=:surface_field) has no surface field data (this " *
        "solution stores a single far-field amplitude, not a surface field) — use kind=:mesh " *
        "for the body shape only."))
    m = AcousticScattering.mesh(sol.body; k = sol.k)
    return _mesh_render(m)
end

function _solution_render(
        sol::AcousticScattering.FEMSolution{AcousticScattering._ScalarFEMData},
        field::Union{Nothing, Symbol})
    field === nothing || throw(ArgumentError(
        "plot(::FEMSolution; kind=:surface_field) is not available for this FEMSolution " *
        "(method=:radial/:meridian): only a scalar target strength was computed, no surface " *
        "field. Use kind=:mesh for the body shape, or bem(...)/mfs(...) on the same " *
        "body/boundary for a solution with surface field data."))
    m = AcousticScattering.mesh(sol.body; k = sol.k)
    return _mesh_render(m)
end

function _mesh_render(m::AcousticScattering.Mesh{AcousticScattering.MeridianMesh})
    ps = AcousticScattering.panels(m.data)
    x, y, z, values = _revolved_surface_plot_data(ps, nothing, nothing)
    return (:revolved, x, y, z, values)
end

function _mesh_render(m::AcousticScattering.Mesh{<:Inti.Quadrature})
    points, faces = _inti_mesh_points_faces(m.data)
    return (:trimesh, points, faces, nothing)
end

function _render_solution_plot(kind::Symbol, x, y, z, values; kwargs...)
    kind === :revolved && return revolvedsurfaceplot(x, y, z, values; kwargs...)
    error("unreachable")
end
function _render_solution_plot(kind::Symbol, a, b, c; kwargs...)
    kind === :trimesh && return trimeshplot(a, b, c; kwargs...)
    error("unreachable")
end
function _render_solution_plot(kind::Symbol, a, b; kwargs...)
    kind === :points && return pointcloudplot(a, b; kwargs...)
    error("unreachable")
end
function _render_solution_plot!(ax, kind::Symbol, x, y, z, values; kwargs...)
    kind === :revolved && return revolvedsurfaceplot!(ax, x, y, z, values; kwargs...)
    error("unreachable")
end
function _render_solution_plot!(ax, kind::Symbol, a, b, c; kwargs...)
    kind === :trimesh && return trimeshplot!(ax, a, b, c; kwargs...)
    error("unreachable")
end
function _render_solution_plot!(ax, kind::Symbol, a, b; kwargs...)
    kind === :points && return pointcloudplot!(ax, a, b; kwargs...)
    error("unreachable")
end

"""
    plot(m::Mesh; kwargs...)

Plot the geometry of a generated or solver-owned [`Mesh`](@ref) (from [`mesh`](@ref) or a
solution's body), no field coloring since no solve is associated with a bare mesh.
"""
function Makie.plot(m::AcousticScattering.Mesh; kwargs...)
    render = _mesh_render(m)
    return _render_solution_plot(render[1], render[2:end]...; kwargs...)
end

function Makie.plot!(ax, m::AcousticScattering.Mesh; kwargs...)
    render = _mesh_render(m)
    return _render_solution_plot!(ax, render[1], render[2:end]...; kwargs...)
end
