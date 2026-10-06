# Concentric spherical fluid/elastic stacks. Fluid coefficients are divided by
# their exterior-relative density; elastic coefficients use the normalized
# Goodman-Stern potentials. This gives common radial-displacement rows without
# assigning a shear modulus to a fluid.

function _layered_materials(material::AbstractShellMaterial, outer_ratio::Real = 1.0)
    return AbstractShellMaterial[material], Float64[outer_ratio]
end

function _layered_materials(material::LayeredMaterial, outer_ratio::Real = 1.0)
    inner, radii = _layered_materials(
        material.inner, outer_ratio * material.radius_ratio)
    return vcat(AbstractShellMaterial[material.outer], inner),
    vcat(Float64[outer_ratio], radii)
end

function _sphere_mixed_fluid_basis(layer::FluidLayer, m::Integer, k::Real, r::Real,
        kind::Symbol)
    x = k * r / layer.soundspeed_contrast
    radial, derivative = if kind === :outgoing
        (ComplexF64[hs(m, x)], ComplexF64[hsd(m, x)])
    elseif kind === :regular
        (ComplexF64[js(m, x)], ComplexF64[jsd(m, x)])
    else
        (ComplexF64[js(m, x), ys(m, x)],
            ComplexF64[jsd(m, x), ysd(m, x)])
    end
    zeros_field = zeros(ComplexF64, length(radial))
    return (; kind = :fluid,
        pressure = layer.density_contrast .* radial,
        radial = x .* derivative,
        stress = zeros_field, tangential_stress = zeros_field,
        tangential_displacement = zeros_field)
end

function _sphere_mixed_elastic_basis(layer::ElasticLayer, m::Integer, k::Real, r::Real)
    g = layer.density_contrast
    xL = k * r / layer.speed_longitudinal_contrast
    xT = k * r / layer.speed_transversal_contrast
    beta = (layer.speed_transversal_contrast /
            layer.speed_longitudinal_contrast)^2
    ell = m * (m + 1)
    pressure = ComplexF64[]
    radial = ComplexF64[]
    stress = ComplexF64[]
    tangential_stress = ComplexF64[]
    tangential_displacement = ComplexF64[]
    for (f, fd, fdd) in ((js, jsd, jsdd), (ys, ysd, ysdd))
        l, ld, ldd = f(m, xL), fd(m, xL), fdd(m, xL)
        push!(radial, xL * ld)
        push!(stress, g * ((1 - 2beta) * l - 2beta * ldd))
        push!(tangential_stress, g * 2 * (xL * ld - l))
        push!(tangential_displacement, l)
        if m > 0
            t, td, tdd = f(m, xT), fd(m, xT), fdd(m, xT)
            push!(radial, ell * t)
            push!(stress, -g * 2ell / xT^2 * (xT * td - t))
            push!(tangential_stress,
                g * (xT^2 * tdd + (m + 2) * (m - 1) * t))
            push!(tangential_displacement, -(xT * td + t))
        end
    end
    append!(pressure, zeros(ComplexF64, length(radial)))
    return (; kind = :elastic, pressure, radial, stress,
        tangential_stress, tangential_displacement)
end

function _sphere_mixed_basis(layer::FluidLayer, m, k, r, kind = :both)
    _sphere_mixed_fluid_basis(layer, m, k, r, kind)
end
function _sphere_mixed_basis(layer::ElasticLayer, m, k, r, kind = :both)
    _sphere_mixed_elastic_basis(layer, m, k, r)
end

function _sphere_mixed_stack_coefficients(
        bc::Shelled{<:LayeredMaterial}, layers, m::Integer, k::Real, a::Real)
    all(layer -> layer isa Union{FluidLayer, ElasticLayer}, layers) ||
        throw(ArgumentError("mixed spherical layers require FluidLayer or ElasticLayer"))
    all(
        layer -> !(layer isa ElasticLayer) ||
                 layer.interior_coupling === :generalized, layers) ||
        throw(ArgumentError("mixed spherical layers require generalized elastic-fluid coupling"))
    _, ratios = _layered_materials(bc.material)
    radii = a .* vcat(ratios, bc.radius_ratio)
    nlayer = length(layers)
    columns = UnitRange{Int}[]
    next_column = 2
    for layer in layers
        count = layer isa FluidLayer || m == 0 ? 2 : 4
        push!(columns, next_column:(next_column + count - 1))
        next_column += count
    end
    core_is_fluid = bc.interior isa FluidInterior
    nunknown = next_column - 1 + Int(core_is_fluid)
    matrix = zeros(ComplexF64, nunknown, nunknown)
    rhs = zeros(ComplexF64, nunknown)
    exterior = FluidLayer(1.0, 1.0)
    core = core_is_fluid ?
           FluidLayer(bc.interior.density_contrast,
        bc.interior.soundspeed_contrast) : nothing
    row = 1
    for interface in 1:(nlayer + 1)
        r = radii[interface]
        outer = interface == 1 ? exterior : layers[interface - 1]
        inner = interface <= nlayer ? layers[interface] : core
        outer_basis = _sphere_mixed_basis(
            outer, m, k, r, interface == 1 ? :outgoing : :both)
        inner_basis = inner === nothing ? nothing :
                      _sphere_mixed_basis(
            inner, m, k, r, interface == nlayer + 1 ? :regular : :both)
        outer_cols = interface == 1 ? (1:1) : columns[interface - 1]
        inner_cols = interface <= nlayer ? columns[interface] : (nunknown:nunknown)
        incident = interface == 1 ?
                   _sphere_mixed_fluid_basis(exterior, m, k, r, :regular) : nothing

        function equation!(outer_field::Union{Nothing, Symbol}, outer_sign::Int,
                inner_field::Union{Nothing, Symbol}, inner_sign::Int)
            if outer_field !== nothing
                matrix[row, outer_cols] .+= outer_sign .*
                                            getproperty(outer_basis, outer_field)
                incident === nothing ||
                    (rhs[row] -= outer_sign * only(getproperty(incident, outer_field)))
            end
            if inner_field !== nothing
                matrix[row, inner_cols] .+= inner_sign .*
                                            getproperty(inner_basis, inner_field)
            end
            row += 1
        end

        if inner === nothing
            if outer_basis.kind === :fluid
                equation!(:pressure, 1, nothing, 0)
            else
                equation!(:stress, 1, nothing, 0)
                m > 0 && equation!(:tangential_stress, 1, nothing, 0)
            end
        elseif outer_basis.kind === :fluid && inner_basis.kind === :fluid
            equation!(:pressure, 1, :pressure, -1)
            equation!(:radial, 1, :radial, -1)
        elseif outer_basis.kind === :elastic && inner_basis.kind === :elastic
            equation!(:stress, 1, :stress, -1)
            equation!(:radial, 1, :radial, -1)
            if m > 0
                equation!(:tangential_stress, 1, :tangential_stress, -1)
                equation!(:tangential_displacement, 1,
                    :tangential_displacement, -1)
            end
        else
            if outer_basis.kind === :fluid
                equation!(:pressure, 1, :stress, 1)
                equation!(:radial, 1, :radial, 1)
                m > 0 && equation!(nothing, 0, :tangential_stress, 1)
            else
                equation!(:stress, 1, :pressure, 1)
                equation!(:radial, 1, :radial, 1)
                m > 0 && equation!(:tangential_stress, 1, nothing, 0)
            end
        end
    end
    row == nunknown + 1 || error("mixed layer equation count is inconsistent")
    x = _sphere_interface_solution(matrix, rhs)
    shells = [layer isa FluidLayer ?
              layer.density_contrast .* collect(x[cols]) : collect(x[cols])
              for (layer, cols) in zip(layers, columns)]
    interior = core_is_fluid ? core.density_contrast * x[end] : nothing
    return (; scattered = x[1], shell = shells, interior)
end
