using AcousticScattering

# Closed flat-ended cylinder about z as explicit nodes and outward-oriented triangles, so tests do not depend on Gmsh's platform-specific meshing. `n` azimuthal divisions (16 or more keeps vertical edges below the sharp-edge threshold), `layers` axial layers and `rings` radial cap rings.
function structured_cylinder(n::Integer, layers::Integer, rings::Integer;
        radius::Real = 0.5, height::Real = 1.0)
    theta = [2pi * (i - 1) / n for i in 1:n]
    nodes = Vector{Float64}[]
    triangles = Vector{Int}[]
    next(i) = mod1(i + 1, n)
    add_ring!(z, r) =
        for t in theta
            push!(nodes, [r * cos(t), r * sin(t), z])
        end
    for z in range(0, height; length = layers + 1)
        add_ring!(z, radius)
    end
    for layer in 1:layers, i in 1:n

        a, b = (layer - 1) * n + i, (layer - 1) * n + next(i)
        c, d = layer * n + next(i), layer * n + i
        push!(triangles, [a, b, c], [a, c, d])
    end
    for (z, outer, flip) in ((0.0, 0, true), (height, layers * n, false))
        rows = [[outer + i for i in 1:n]]
        for m in 1:(rings - 1)
            add_ring!(z, radius * (rings - m) / rings)
            push!(rows, [length(nodes) - n + i for i in 1:n])
        end
        push!(nodes, [0.0, 0.0, z])
        center = length(nodes)
        emit(a, b, c) = push!(triangles, flip ? [a, c, b] : [a, b, c])
        for m in 1:(length(rows) - 1), i in 1:n

            a, b = rows[m][i], rows[m][next(i)]
            c, d = rows[m + 1][next(i)], rows[m + 1][i]
            emit(a, b, c)
            emit(a, c, d)
        end
        for i in 1:n
            emit(center, rows[end][i], rows[end][next(i)])
        end
    end
    return reduce(hcat, nodes), reduce(hcat, triangles)
end
