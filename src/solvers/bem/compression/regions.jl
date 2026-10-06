# Coupled Müller blocks act on total pressure and density-normalized normal derivative.
function _apply_region_interaction!(yp, yv, interaction, p, v)
    (; S, D, K, H, factor, density, sign, pressure_work, flux_work) = interaction
    mul!(pressure_work, D, p)
    mul!(pressure_work, S, v, -density, 1)
    if factor === nothing
        mul!(flux_work, H, p, inv(density), 0)
        mul!(flux_work, K, v, -1, 1)
    else
        # H*p/rho - K*v = S⁻¹[D*(D*p-rho*S*v)-p/4]/rho.
        mul!(flux_work, D, pressure_work)
        @. flux_work -= 0.25p
        LinearAlgebra.ldiv!(factor, flux_work)
        flux_work ./= density
    end
    @. yp += sign*pressure_work
    @. yv += sign*flux_work
    return nothing
end

function _region_operator(interactions, n; interior = false)
    return LinearMap{ComplexF64}(2n, 2n) do y, x
        y .= (interior ? 0.5 : 1.0) .* x
        for interaction in interactions
            interior && !interaction.interior && continue
            rows, columns = interaction.rows, interaction.columns
            _apply_region_interaction!(view(y, rows), view(y, rows .+ n), interaction,
                view(x, columns), view(x, columns .+ n))
        end
        return y
    end
end

function _region_diagonal_contribution!(blocks, interaction, offset, n)
    (; S, D, K, H, factor, density, sign) = interaction
    singles, doubles = _fluid_diagonal_blocks(S), _fluid_diagonal_blocks(D)
    if factor === nothing
        adjoints, hypersingular = _fluid_diagonal_blocks(K), _fluid_diagonal_blocks(H)
    end
    initialize = isempty(blocks)
    for j in eachindex(singles)
        indices, s = singles[j]
        d = last(doubles[j])
        if factor === nothing
            kk, h = last(adjoints[j]), last(hypersingular[j])
        else
            f = lu(s)
            kk, h = f \ (d*s), f \ (d*d - 0.25I)
        end
        global_indices = indices .+ offset
        indices = [global_indices; global_indices .+ n]
        if initialize
            push!(blocks, (
                indices, Matrix{ComplexF64}(I, length(indices), length(indices))))
        else
            first(blocks[j]) == indices || error("inconsistent interface cluster ordering")
        end
        last(blocks[j]) .+= [sign*d (-sign*density)*s; (sign/density)*h -sign*kk]
    end
    return blocks
end

function _assemble_compressed_regions(metadata, speeds; regular, reconstruct, compression)
    (; quads, parents, k, formulation, correction, ranges, n, densities) = metadata
    compression.method === :hmatrix ||
        throw(ArgumentError("coupled fluid compression supports only :none or :hmatrix"))
    formulation === :muller && correction.method === :dim ||
        throw(ArgumentError("compressed coupled fluid BEM requires formulation=:muller and correction=:dim"))
    tol = get(compression, :tol, 1e-8)
    isfinite(tol) && 0 < tol < 1 ||
        throw(ArgumentError("compression tolerance must lie in (0, 1)"))
    compression = (; method = :hmatrix, tol)
    interactions, derivative_evaluation = NamedTuple[], NamedTuple[]
    diagonal = [Tuple{Vector{Int}, Matrix{ComplexF64}}[] for _ in quads]
    # Kernel blocks depend on source, target and wavenumber; density and interface
    # signs are applied separately below. Share equal-wavenumber material sides.
    operators = Dict{Tuple{Int, Int, Real}, NamedTuple}()
    for i in eachindex(quads), region in (i, parents[i])

        density = densities[region + 1]
        op = Inti.Helmholtz(; k = k/speeds[region + 1], dim = 3)
        for j in eachindex(quads)
            (j == region || parents[j] == region) || continue
            sign = j == region ? 1 : -1
            location = i == j ? :on : (j == region ? :inside : :outside)
            options = merge((; maxdist = Inf), correction, (; target_location = location))
            raw = get!(operators, (i, j, op.k)) do
                S, D = _fluid_layer_operators(
                    op, quads[i], quads[j], options; regular, compression)
                if i == j && reconstruct
                    factor = _fluid_assembly_lu(S; rtol = tol/100)
                    K = H = nothing
                    method = :calderon
                else
                    K, H = _fluid_layer_operators(op, quads[i], quads[j], options;
                        derivative = true, regular, compression)
                    factor = nothing
                    method = :direct
                end
                (; S, D, K, H, factor, method)
            end
            (; S, D, K, H, factor, method) = raw
            interaction = (; S, D, K, H, factor, density, sign,
                rows = ranges[i], columns = ranges[j], interior = region == i,
                pressure_work = zeros(ComplexF64, length(quads[i])),
                flux_work = zeros(ComplexF64, length(quads[i])))
            push!(derivative_evaluation, (; target = i, source = j, region, method))
            i == j &&
                _region_diagonal_contribution!(diagonal[i], interaction, first(ranges[i])-1, n)
            S, D, K, H = map(_fluid_operator_workspace, (S, D, K, H))
            factor = _fluid_inverse_workspace(factor)
            push!(interactions, merge(interaction, (; S, D, K, H, factor)))
        end
    end
    A = _region_operator(interactions, n)
    inner = _region_operator(interactions, n; interior = true)
    blocks = reduce(vcat, diagonal)
    # Share operator/inverse workspaces with A. GMRES applies A and its right
    # preconditioner sequentially; no additional hierarchical factors are built.
    preconditioner_interfaces = reconstruct ?
                                map(eachindex(quads)) do i
        interaction = only(filter(interactions) do interaction
            interaction.rows == interaction.columns == ranges[i] && !interaction.interior
        end)
        (; indices = ranges[i], S = interaction.S, factor = interaction.factor,
            density_exterior = densities[parents[i] + 1], density_interior = densities[i + 1])
    end : nothing
    return merge(metadata,
        (; A, inner, blocks, compression, derivative_evaluation,
            preconditioner_interfaces, coarse_interactions = interactions))
end
