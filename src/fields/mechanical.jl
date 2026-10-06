# Field evaluation is exposed through the existing solution object. Keep
# quantity selection separate from pressure's total/scattered/incident field.
function (solution::AbstractSolution)(points; quantity::Symbol = :pressure, kwargs...)
    quantity === :pressure && return pressure(solution, points; kwargs...)
    quantity in (:displacement, :velocity, :stress) || throw(ArgumentError(
        "quantity must be :pressure, :displacement, :velocity or :stress"))
    return _material_field(solution, points, quantity; kwargs...)
end

function _material_field(solution::AbstractSolution, points, quantity; kwargs...)
    throw(ArgumentError("$(quantity) evaluation requires a full-3D volume FEM solution"))
end

function _material_field(solution::FEMSolution{_VolumeFEMData}, points, quantity;
        density_exterior::Union{Nothing, Real} = nothing,
        soundspeed_exterior::Union{Nothing, Real} = nothing,
        pressure_amplitude::Number = 1)
    density_exterior === nothing ||
        (isfinite(density_exterior) && density_exterior > 0) ||
        throw(ArgumentError("density_exterior must be finite and positive in kg/m^3"))
    soundspeed_exterior === nothing ||
        (isfinite(soundspeed_exterior) && soundspeed_exterior > 0) ||
        throw(ArgumentError("soundspeed_exterior must be finite and positive in m/s"))
    scale = if quantity === :stress
        _material_pressure_scale(pressure_amplitude)
    else
        density_exterior !== nothing && soundspeed_exterior !== nothing ||
            throw(ArgumentError("$(quantity) requires density_exterior and soundspeed_exterior"))
        value = _material_displacement_scale(
            density_exterior, soundspeed_exterior, pressure_amplitude)
        quantity === :velocity ? -im * solution.k * soundspeed_exterior * value : value
    end
    return _field_points(
        p -> _material_field_values(
            solution, p, quantity, scale; soundspeed_exterior), points)
end

function _material_displacement_scale(density, speed, amplitude)
    isfinite(density) && density > 0 ||
        throw(ArgumentError("density_exterior must be finite and positive in kg/m³"))
    isfinite(speed) && speed > 0 ||
        throw(ArgumentError("soundspeed_exterior must be finite and positive in m/s"))
    return _material_pressure_scale(amplitude) / (density * speed^2)
end

function _material_pressure_scale(amplitude)
    isfinite(amplitude) ||
        throw(ArgumentError("pressure_amplitude must be finite in Pa"))
    return ComplexF64(amplitude)
end

function _material_field_values(solution::FEMSolution{_VolumeFEMData}, points, kind, scale;
        soundspeed_exterior = nothing)
    all(p -> all(isfinite, p), points) ||
        throw(ArgumentError("point coordinates must be finite"))
    data = solution.data
    system = data.system
    grid = system.grid
    handler = Ferrite.PointEvalHandler(
        grid, [Ferrite.Vec{3}(Float64.(point)) for point in points])
    pv = Ferrite.PointValues(_VOLUME_IP, _VOLUME_GIP)
    values = kind in (:displacement, :velocity) ?
             SVector{3, ComplexF64}[] : Matrix{ComplexF64}[]
    for (i, location) in enumerate(Ferrite.PointIterator(handler))
        location === nothing && throw(ArgumentError(
            "point $(points[i]) lies outside the volume FEM computational domain"))
        cell_id = Ferrite.cellid(location)
        system.labels[cell_id] == _REGION_SOLID || throw(ArgumentError(
            "point $(points[i]) is outside an elastic or viscous material region"))
        material = _volume_field_material(solution, system.ids[cell_id])
        viscous = material isa ViscousLayer
        if viscous && soundspeed_exterior !== nothing
            isapprox(soundspeed_exterior, material.soundspeed_exterior;
                rtol = 1e-12, atol = 0) || throw(ArgumentError(
                "soundspeed_exterior must match the ViscousLayer solve"))
        end
        Ferrite.reinit!(pv, location)
        cell = grid.cells[cell_id]
        nodes = cell.nodes
        n_nodes = system.n_nodes
        u = SVector{3, ComplexF64}(ntuple(
            a -> sum(
                Ferrite.shape_value(pv, 1, j) *
                data.solution[_displacement_dof(n_nodes, nodes[j], a)]
            for j in eachindex(nodes)),
            3))
        if kind in (:displacement, :velocity)
            push!(values, scale * u)
            continue
        end
        gradient = zeros(ComplexF64, 3, 3)
        for j in eachindex(nodes)
            shape_gradient = Ferrite.shape_gradient(pv, 1, j)
            for a in 1:3, b in 1:3

                gradient[a, b] += data.solution[_displacement_dof(n_nodes, nodes[j], a)] *
                                  shape_gradient[b]
            end
        end
        solid = system.model.solid isa Vector ?
                system.model.solid[system.ids[cell_id]] : system.model.solid
        _, lame, mu, reduced = solid
        sigma = mu * (gradient + transpose(gradient))
        dilatation = reduced ? _volume_cell_mean_dilatation(data, cell_id) :
                     sum(gradient[b, b] for b in 1:3)
        for a in 1:3
            sigma[a, a] += lame * dilatation
        end
        push!(values, scale * sigma)
    end
    return values
end

function _volume_cell_mean_dilatation(data::_VolumeFEMData, cell_id)
    system = data.system
    grid = system.grid
    cell = grid.cells[cell_id]
    nodes = cell.nodes
    coords = [grid.nodes[i].x for i in nodes]
    cv = Ferrite.CellValues(_VOLUME_QR_CELL, _VOLUME_IP, _VOLUME_GIP)
    Ferrite.reinit!(cv, cell, coords)
    integral = zero(ComplexF64)
    volume = 0.0
    for q in 1:Ferrite.getnquadpoints(cv)
        dV = Ferrite.getdetJdV(cv, q)
        volume += dV
        for j in eachindex(nodes)
            shape_gradient = Ferrite.shape_gradient(cv, q, j)
            for a in 1:3
                integral += data.solution[_displacement_dof(system.n_nodes, nodes[j], a)] *
                            shape_gradient[a] * dV
            end
        end
    end
    return integral / volume
end

function _volume_field_material(solution, id)
    boundary = solution.boundary
    boundary isa _VolumeRegionMaterials && return boundary.materials[id]
    boundary isa Shelled && return boundary.material
    return boundary
end
