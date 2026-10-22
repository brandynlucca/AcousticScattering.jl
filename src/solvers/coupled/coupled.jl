"""
    Coupling(; interior, exterior)

Assign a numerical solver to each side of an interface. For example,
`Coupling(interior=:fem, exterior=:bem)` assigns interior finite elements and
exterior boundary elements. [`coupled`](@ref) checks the supported pairing,
geometry and material together.
"""
struct Coupling
    interior::Symbol
    exterior::Symbol
    Coupling(; interior::Symbol, exterior::Symbol) = new(interior, exterior)
end

"""
    CoupledSolution

Result of [`coupled`](@ref). `coupling` records the interior and exterior
solvers. Query [`scattering_amplitude`](@ref), [`target_strength`](@ref),
[`diagnostics`](@ref), [`bistatic_sweep`](@ref) and [`bistatic_map`](@ref).
"""
struct CoupledSolution{D} <: AbstractSolution
    body::AbstractBody
    boundary::AbstractBoundaryCondition
    k::Float64
    coupling::Coupling
    data::D
end

struct _SpheroidFEMBEMData
    modes::Vector{_SpheroidSurfaceMode}
    incidence_angle::Float64
    order::Int
    diagnostics::NamedTuple
end

"""
    coupled(body, boundary, k; coupling::Coupling, kwargs...)

Solve an interface problem with explicit interior and exterior numerical
methods. `Coupling(interior=:fem, exterior=:bem)` supports homogeneous
`FluidFilled` prolate spheroids. Other pairings are rejected.

`k` is exterior wavenumber in rad/m. `incidence_angle` is measured from the
symmetry axis and defaults to broadside. `order=6` and `boundary_order=5`
set polynomial degrees. `n_r`, `n_theta` and `n_boundary` control partitions.
`adaptive=true` refines the spatial mesh to `target_tol=0.01` dB by default.
`m_max` controls the Fourier limit and `rtol` the boundary quadrature.
`quadrature_extra=0` adds integration points to the fixed rules without
changing mesh counts or polynomial degrees. Increase these controls separately
and compare complex amplitudes. A mesh-change threshold is not an error bound.

Results retain far-field traces and support observation `angle` and `azimuth`.
Pressure queries, `retain_fields`, exterior `R` and `dtn_form` are unsupported.
"""
function coupled(body::AbstractBody, boundary::AbstractBoundaryCondition, k::Real;
        coupling::Coupling, incidence_angle::Real = pi/2,
        m_max::Union{Nothing, Integer} = nothing, mode_threads::Integer = 1,
        order::Integer = 6, kwargs...)
    coupling.interior===:fem && coupling.exterior===:bem || throw(ArgumentError(
        "unsupported coupling: interior=$(coupling.interior), exterior=$(coupling.exterior)"))
    body isa Spheroid && body.a>body.b && boundary isa FluidFilled || throw(ArgumentError(
        "interior FEM / exterior BEM requires a homogeneous FluidFilled prolate spheroid"))
    allowed=(:n_r, :n_theta, :n_boundary, :boundary_order,
        :rtol, :chief_points, :quadrature_extra,
        :adaptive, :target_tol, :max_refinements)
    all(key->key in allowed, keys(kwargs)) || throw(ArgumentError(
        "unsupported coupled controls: $(setdiff(collect(keys(kwargs)),collect(allowed)))"))
    limit=m_max===nothing ?
          _spheroid_meridian_default_mmax(body, boundary, k, incidence_angle) : m_max
    return _spheroid_fembem(body, boundary, Float64(k); coupling,
        incidence_angle, m_max = limit, mode_threads, order, kwargs...)
end

include("shell_fluid/thin_shell.jl")
include("shell_fluid/meridian_elasticity.jl")
include("shell_fluid.jl")
include("free_surface.jl")
include("postprocessing.jl")

# Interior confocal finite elements coupled to a curved exterior boundary integral.
# This numerical coordinate map uses no spheroidal wave functions.
function _spheroid_fembem_confocal(
        boundary, a, b, nr, nt, p; half = true, assemble_bulk = true,
        quadrature_extra::Integer = 0)
    # rho=b*s*sin(chi), z=sqrt(a^2-b^2+b^2*s^2)*cos(chi).
    # At s=0 the mesh reaches the focal segment, whose nodes remain distinct.
    # Its natural condition applies to m=0, while nonzero modes vanish there.
    focal = sqrt((a-b)*(a+b))
    nrn, ntn = nr*p+1, nt*p+1
    idx(i, j) = (i-1)*ntn+j
    radial_edges = collect(range(0.0, 1.0; length = nr+1))
    angular_edges = collect(range(0.0, half ? pi/2 : pi; length = nt+1))
    lb = [(i-1)*(p+1)+j for i in 1:(p + 1)
          for j in 1:(p + 1) if i in (1, p+1) || j in (1, p+1)]
    li = setdiff(1:((p + 1) ^ 2), lb)
    q, w = _spheroid_hp_rule(p+3+quadrature_extra)
    bases = [_spheroid_hp_basis(p, x) for x in q]
    nloc = (p+1)^2
    cells = _SpheroidFEMElement[]
    # The orthogonal confocal metric separates into radial and angular factors.
    # Form the same tensor-product quadrature with one-dimensional integrals.
    # This avoids repeating an O(p^4) outer product at every 2D quadrature point.
    function factors(edges, radial)
        return Base.map(1:(length(edges) - 1)) do e
            left, right = edges[e], edges[e + 1]
            mass, stiffness, azimuthal, wave = (zeros(p+1, p+1) for _ in 1:4)
            for j in eachindex(q)
                x = (left+right)/2+(right-left)*q[j]/2
                ns, dns = bases[j]
                if radial
                    aa = hypot(focal, b*x)
                    cm, cs = b^2*x/aa, x*aa
                    ca, cw = focal^2/(aa*x), b^4*x^3/aa
                else
                    cm, cs = sin(x), sin(x)
                    ca, cw = inv(sin(x)), focal^2*sin(x)^3
                end
                weight = w[j]*(right-left)/2
                for v in 1:(p + 1), u in 1:(p + 1)

                    product = ns[u]*ns[v]*weight
                    mass[u, v] += cm*product
                    stiffness[u, v] += cs*dns[u]*dns[v]*4weight/(right-left)^2
                    azimuthal[u, v] += ca*product
                    wave[u, v] += cw*product
                end
            end
            (; mass, stiffness, azimuthal, wave)
        end
    end
    radial_factors = factors(radial_edges, true)
    angular_factors = factors(angular_edges, false)
    for et in 1:nt, er in 1:nr

        ids = [idx((er-1)*p+i, (et-1)*p+j) for i in 1:(p + 1) for j in 1:(p + 1)]
        r, t = radial_factors[er], angular_factors[et]
        S, A, M = if assemble_bulk
            (
                (kron(r.stiffness, t.mass)+kron(r.mass, t.stiffness))/boundary.density_contrast,
                (kron(r.azimuthal, t.mass)+kron(r.mass, t.azimuthal))/boundary.density_contrast,
                (kron(r.wave, t.mass)+kron(r.mass, t.wave)) /
                (boundary.density_contrast*boundary.soundspeed_contrast^2))
        else
            (zeros(0, 0), zeros(0, 0), zeros(0, 0))
        end
        push!(cells, _SpheroidFEMElement(ids, S, A, M, true))
    end
    kept = sort!(unique(vcat([cell.ids[lb] for cell in cells]...)))
    map = zeros(Int, nrn*ntn)
    map[kept] = eachindex(kept)
    iface = map[[idx(nrn, j) for j in 1:ntn]]
    axis_nodes = vcat([idx(1, j) for j in 1:ntn], [idx(i, 1) for i in 1:nrn])
    half || append!(axis_nodes, [idx(i, ntn) for i in 1:nrn])
    fixed = unique(map[axis_nodes])
    load_basis = hcat([kron(first(bases[js]), first(bases[jt]))
                       for jt in eachindex(q) for js in eachindex(q)]...)
    load_points = Base.map(1:length(cells)) do e
        er, et = mod1(e, nr), cld(e, nr)
        sl, sr = radial_edges[er:(er + 1)]
        tl, tr = angular_edges[et:(et + 1)]
        points = Matrix{Float64}(undef, 3, length(q)^2)
        for jt in eachindex(q), js in eachindex(q)

            radial = (sl+sr)/2+(sr-sl)*q[js]/2
            theta = (tl+tr)/2+(tr-tl)*q[jt]/2
            aa = hypot(focal, b*radial)
            j = (jt-1)*length(q)+js
            points[1, j] = b*radial*sin(theta)
            points[2, j] = aa*cos(theta)
            points[3, j] = b^2*radial/aa * (focal^2*sin(theta)^2+b^2*radial^2) *
                           sin(theta) * w[js] * w[jt] * (sr-sl) * (tr-tl)/4
        end
        points
    end
    return (; cells, kept, map, iface, fixed, ndof = length(kept),
        lb, li, p, nr, nt, angular_edges, load_basis, load_points,
        radial_factors, angular_factors, a, b, half, quadrature_extra,
        density_contrast = boundary.density_contrast,
        soundspeed_contrast = boundary.soundspeed_contrast,
        coordinate = :confocal)
end

function _spheroid_fembem_trace(mesh, boundary)
    n, nf = length(boundary.chi), length(mesh.iface)
    T = zeros(n, nf)
    function fe_values(chi)
        theta = chi
        e = clamp(searchsortedlast(mesh.angular_edges, theta), 1, mesh.nt)
        left, right = mesh.angular_edges[e:(e + 1)]
        basis = first(_spheroid_hp_basis(mesh.p, (2theta-left-right)/(right-left)))
        return (e-1)*mesh.p .+ (1:(mesh.p + 1)), basis
    end
    for (i, chi) in enumerate(boundary.chi)
        indices, basis = fe_values(chi)
        T[i, indices] = basis
    end
    # Integrate on the overlay of the independent FE and BE partitions.
    fedges = mesh.angular_edges
    edges = sort!(unique(vcat(fedges, boundary.edges)))
    B = zeros(nf, n)
    q, w = _spheroid_hp_rule(mesh.p+boundary.degree+4+mesh.quadrature_extra)
    for segment in 1:(length(edges) - 1)
        left, right = edges[segment:(segment + 1)]
        right-left < 1e-14 && continue
        for j in eachindex(q)
            chi = (left+right)/2+(right-left)*q[j]/2
            e = clamp(searchsortedlast(boundary.edges, chi), 1, boundary.elements)
            bl, br = boundary.edges[e:(e + 1)]
            basis = _spheroid_curved_basis(boundary.nodes, (2chi-bl-br)/(br-bl))
            indices, fe = fe_values(chi)
            jac = boundary.b*sin(chi)*hypot(boundary.a*sin(chi), boundary.b*cos(chi))*(right-left)*w[j]/2
            B[indices, (e - 1) * boundary.degree .+ (1:boundary.degree)] .+= jac .*
                                                                             (fe*transpose(basis))
        end
    end
    return T, B
end

function _spheroid_fembem_bulk(mesh, k, m, beta)
    lb, li = mesh.lb, mesh.li
    nb, ni = length(lb), length(li)
    Is, Js, Vs = Int[], Int[], Float64[]
    for vector in (Is, Js, Vs)
        sizehint!(vector, length(mesh.cells)*nb^2)
    end
    local_matrix = Matrix{Float64}(undef, nb+ni, nb+ni)
    schur = Matrix{Float64}(undef, nb, nb)
    bi = Matrix{Float64}(undef, nb, ni)
    ib = Matrix{Float64}(undef, ni, nb)
    ii = Matrix{Float64}(undef, ni, ni)
    ids = zeros(Int, nb)
    loads = zeros(ComplexF64, mesh.ndof, 2)
    samples = zeros(ComplexF64, size(mesh.load_basis, 2), 2)
    source_scale = k^2*(inv(mesh.soundspeed_contrast^2)-1)/mesh.density_contrast
    for (element, cell) in enumerate(mesh.cells)
        @. local_matrix = cell.S + m^2*cell.A - k^2*cell.M
        copyto!(schur, view(local_matrix, lb, lb))
        copyto!(bi, view(local_matrix, lb, li))
        copyto!(ib, view(local_matrix, li, lb))
        copyto!(ii, view(local_matrix, li, li))
        LinearAlgebra.ldiv!(LinearAlgebra.lu!(ii), ib)
        mul!(schur, bi, ib, -1.0, 1.0)
        for j in eachindex(ids)
            ids[j] = mesh.map[cell.ids[lb[j]]]
        end
        if !iszero(source_scale)
            points = mesh.load_points[element]
            for j in axes(points, 2)
                amplitude = (m==0 ? 1.0 : 2.0)*im^m*besselj(m, k*points[1, j]*sin(beta))
                phase = cis(k*points[2, j]*cos(beta))
                weight = source_scale*points[3, j]
                samples[j, 1] = weight*amplitude*phase
                samples[j, 2] = weight*amplitude*conj(phase)
            end
            local_load = mesh.load_basis*samples
            condensed_load = local_load[lb, :]-transpose(ib)*local_load[li, :]
            loads[ids, :] += condensed_load
        end
        for j in eachindex(ids), i in eachindex(ids)

            push!(Is, ids[i])
            push!(Js, ids[j])
            push!(Vs, schur[i, j])
        end
    end
    return (; matrix = sparse(Is, Js, Vs, mesh.ndof, mesh.ndof), loads)
end

function _spheroid_fembem_mode(mesh, boundary, T, B, K, V, k, m, beta, reports, workers;
        parity::Int = 0, bulk = _spheroid_fembem_bulk(mesh, k, m, beta))
    ndof, nbe = mesh.ndof, length(boundary.chi)
    D = -K
    for i in 1:nbe
        D[i, i] += 0.5
    end
    rawD, rawV = D, V
    if size(V, 1)>nbe
        # Surface rows enforce (I/2-K)p_s+V*q_s=0. CHIEF rows enforce the
        # zero interior representation, -K*p_s+V*q_s=0. QR obtains the
        # least-squares exterior flux map without forming normal equations.
        F = LinearAlgebra.qr(V)
        D = (F.Q' * D)[1:nbe, :]
        V = Matrix(F.R)
    end
    i, j, v = SparseArrays.findnz(sparse(B))
    upper = sparse(mesh.iface[i], j, -v, ndof, nbe)
    i, j, v = SparseArrays.findnz(sparse(D*T))
    lower = sparse(i, mesh.iface[j], v, nbe, ndof)
    matrix = [bulk.matrix upper; lower sparse(V)]
    incident = [_p_inc_mode(m, k, beta, r, z) for (r, z) in zip(boundary.rho, boundary.z)]
    derivative = [_dpdn_inc_mode(m, k, beta, r, z, nr, nz)
                  for (r, z, nr, nz) in zip(boundary.rho, boundary.z, boundary.nrho, boundary.nz)]
    if parity != 0
        # Reflection changes both the axial coordinate and the axial normal.
        # These are total-field even/odd loads on the positive half-domain.
        for i in eachindex(incident)
            incident[i] = (incident[i] +
                           parity*_p_inc_mode(
                m, k, beta, boundary.rho[i], -boundary.z[i]))/2
            derivative[i] = (derivative[i] +
                             parity*_dpdn_inc_mode(m, k, beta,
                boundary.rho[i], -boundary.z[i], boundary.nrho[i], -boundary.nz[i]))/2
        end
    end
    rhs = zeros(ComplexF64, ndof+nbe)
    # Solve for scattered interior pressure. Its volume forcing is proportional
    # to the sound-speed contrast, and its boundary load to the density contrast.
    # Matched media therefore give zero scattering without subtracting two
    # independently discretized O(1) total and incident fields.
    rhs[1:ndof] = parity==0 ? bulk.loads[:, 1] :
                  (bulk.loads[:, 1]+parity*bulk.loads[:, 2])/2
    rhs[mesh.iface] += (1-inv(mesh.density_contrast))*(B*derivative)
    fixed = m>0 ? mesh.fixed : Int[]
    if parity == -1
        # The odd pressure vanishes on the equator, including the focal point.
        ntn = mesh.nt*mesh.p+1
        equator = mesh.map[[i*ntn for i in 1:(mesh.nr * mesh.p + 1)]]
        fixed = unique(vcat(fixed, equator))
    end
    _meridian_fem_dirichlet!(matrix, rhs, fixed, zeros(ComplexF64, length(fixed)))
    solution = matrix\rhs
    ps, qs = T*solution[mesh.iface], solution[(ndof + 1):end]
    representation_residual = _residual_report(rawD*ps+rawV*qs, incident).relative_residual
    _record_solve!(reports, matrix, solution, rhs; mode = m, component = :fem_bem,
        mode_workers = workers, representation_residual, n_r = mesh.nr,
        n_theta = boundary.half ? 2mesh.nt : mesh.nt, n_boundary = boundary.elements,
        chief_points = size(rawV, 1)-nbe, parity)
    return _SpheroidSurfaceMode(boundary, ps, qs)
end

function _spheroid_fembem_tensor_load(mesh, be, B, k, m, beta)
    p, nr, nt = mesh.p, mesh.nr, mesh.nt
    nrn, ntn = nr*p+1, nt*p+1
    g, h = mesh.density_contrast, mesh.soundspeed_contrast
    load = zeros(ComplexF64, ntn, nrn)
    samples = zeros(ComplexF64, size(mesh.load_basis, 2))
    source_scale = k^2*(inv(h^2)-1)/g
    if !iszero(source_scale)
        for (element, cell) in enumerate(mesh.cells)
            points = mesh.load_points[element]
            for j in axes(points, 2)
                phase = cis(k*points[2, j]*cos(beta))
                samples[j] = source_scale * points[3, j] * (m==0 ? 1.0 : 2.0) * im^m *
                             besselj(m, k*points[1, j]*sin(beta)) * phase
            end
            local_load = mesh.load_basis*samples
            for j in eachindex(cell.ids)
                load[cell.ids[j]] += local_load[j]
            end
        end
    end
    derivative = [_dpdn_inc_mode(m, k, beta, r, z, nx, nz)
                  for (r, z, nx, nz) in zip(be.rho, be.z, be.nrho, be.nz)]
    load[:, end] += (1-inv(g))*(B*derivative)
    return load
end

# Return the boundary response to a unit flux and the scattered-field source.
# Element-interior poles trigger the uncondensed one-dimensional Robin solve.
function _spheroid_fembem_radial_response(factors, p, m, ki, g, lambda, radial_load, eta)
    nr = length(factors)
    diagonal = zeros(ComplexF64, nr+1)
    offdiag = zeros(ComplexF64, nr)
    rhs = zeros(ComplexF64, nr+1, 2)
    rhs[end, 1] = 1
    local_matrix(r) = (r.stiffness+m^2*r.azimuthal-ki^2*r.wave+lambda*r.mass)/g
    ends, inner = [1, p+1], 2:p
    for (e, r) in enumerate(factors)
        A = local_matrix(r)
        F = LinearAlgebra.lu(A[inner, inner]; check = false)
        if !LinearAlgebra.issuccess(F) ||
           minimum(abs, diag(F.U)) <=
           sqrt(eps(Float64))*LinearAlgebra.opnorm(A[inner, inner], Inf)
            n = nr*p+1
            matrix = zeros(ComplexF64, n, n)
            full_rhs = hcat(zeros(ComplexF64, n), radial_load)
            full_rhs[end, 1] = 1
            for (element, factor) in enumerate(factors)
                ids = (element-1)*p .+ (1:(p + 1))
                matrix[ids, ids] += local_matrix(factor)
            end
            if m>0
                matrix[1, :] .= 0
                matrix[:, 1] .= 0
                matrix[1, 1] = 1
                full_rhs[1, :] .= 0
            end
            matrix[end, end] += im*eta
            return (matrix \ full_rhs)[end, :]
        end
        extension = F\A[inner, ends]
        S = A[ends, ends]-A[ends, inner]*extension
        diagonal[e] += S[1, 1]
        diagonal[e + 1] += S[2, 2]
        offdiag[e] = (S[1, 2]+S[2, 1])/2
        ids = (e-1)*p .+ (1:(p + 1))
        rhs[e:(e + 1), 2] -= transpose(extension)*radial_load[ids[inner]]
    end
    rhs[:, 2] += radial_load[1:p:end]
    if m>0
        diagonal[1], offdiag[1] = 1, 0
        rhs[1, :] .= 0
    end
    diagonal[end] += im*eta
    return (LinearAlgebra.Tridiagonal(copy(offdiag), diagonal, offdiag) \ rhs)[end, :]
end

# Fast diagonalization of the same tensor-product finite-element equations.
# No analytical spheroidal functions or fitted modal data enter this reduction.
function _spheroid_fembem_tensor_mode(
        mesh, be, T, B, K, V, k, m, beta, reports, workers; parity = 0,
        load = _spheroid_fembem_tensor_load(mesh, be, B, k, m, beta))
    p, nr, nt = mesh.p, mesh.nr, mesh.nt
    nrn, ntn = nr*p+1, nt*p+1
    g, h = mesh.density_contrast, mesh.soundspeed_contrast
    ki = k/h
    mass, angular = zeros(ntn, ntn), zeros(ntn, ntn)
    for (e, t) in enumerate(mesh.angular_factors)
        ids = (e-1)*p .+ (1:(p + 1))
        mass[ids, ids] += t.mass
        angular[ids, ids] += t.stiffness+m^2*t.azimuthal-ki^2*t.wave
    end
    fixed = m>0 ? (mesh.half ? [1] : [1, ntn]) : Int[]
    parity==-1 && push!(fixed, ntn)
    free = setdiff(1:ntn, fixed)
    eigenproblem = LinearAlgebra.eigen(LinearAlgebra.Symmetric(angular[free, free]),
        LinearAlgebra.Symmetric(mass[free, free]))
    E = zeros(ntn, length(free))
    E[free, :] = eigenproblem.vectors
    # Assemble the uncondensed scattered-field load and transform only its
    # angular coordinate. The radial FE equations remain fully discrete.
    load = parity==0 ? load : (load+parity*(-1)^m*conj(load))/2
    incident = [_p_inc_mode(m, k, beta, r, z) for (r, z) in zip(be.rho, be.z)]
    if parity != 0
        incident = (incident+parity*(-1)^m*conj(incident))/2
    end
    transformed = transpose(E)*load
    eta = mesh.a*mesh.b*ki/g
    alpha, beta_response, forcing = (zeros(ComplexF64, length(free)) for _ in 1:3)
    for j in eachindex(free)
        radial_solution = _spheroid_fembem_radial_response(
            mesh.radial_factors, p, m, ki, g,
            eigenproblem.values[j], view(transformed, j, :), eta)
        beta_response[j] = radial_solution[1]
        alpha[j] = 1-im*eta*beta_response[j]
        forcing[j] = radial_solution[2]
    end
    nbe = length(be.chi)
    D = -K
    for j in 1:nbe
        D[j, j] += 0.5
    end
    rawD, rawV = D, V
    if size(V, 1)>nbe
        F = LinearAlgebra.qr(V)
        D = (F.Q' * D)[1:nbe, :]
        V = Matrix(F.R)
    end
    pressure_map = T*E
    flux_map = transpose(E)*B
    exterior_map = D*pressure_map
    # Retain modes near a Neumann pole as auxiliary unknowns. The eliminated
    # modes have a bounded inverse relative to the Robin scale.
    retained = findall(j->abs(alpha[j])<=0.25eta*abs(beta_response[j]), eachindex(alpha))
    regular = setdiff(eachindex(alpha), retained)
    response = beta_response[regular] ./ alpha[regular]
    particular = forcing[regular] ./ alpha[regular]
    matrix = [V+(exterior_map[:, regular] .* transpose(response))*flux_map[regular, :] exterior_map[:, retained];
              -Diagonal(beta_response[retained])*flux_map[retained, :] Matrix(Diagonal(alpha[retained]))]
    rhs = vcat(-exterior_map[:, regular]*particular, forcing[retained])
    solution = matrix\rhs
    qs = solution[1:nbe]
    coefficients = zeros(ComplexF64, length(alpha))
    coefficients[regular] = particular+response .* (flux_map[regular, :]*qs)
    coefficients[retained] = solution[(nbe + 1):end]
    ps = pressure_map*coefficients
    representation_residual = _residual_report(rawD*ps+rawV*qs, incident).relative_residual
    _record_solve!(reports, matrix, solution, rhs; mode = m,
        component = :fem_bem, mode_workers = workers,
        representation_residual, n_r = nr, n_theta = mesh.half ? 2nt : nt, n_boundary = be.elements,
        chief_points = size(rawV, 1)-nbe, parity, interior_reduction = :tensor,
        interior_node_count = nrn*ntn, retained_angular_modes = length(retained))
    return _SpheroidSurfaceMode(be, ps, qs)
end

function _spheroid_fembem_level(body, boundary, k, nr, nt, nb, order, boundary_order;
        half, incidence_angle, m_max, mode_threads, rtol, chief_points, reports,
        split_symmetry::Bool = true, tensor::Bool = true, modes = 0:m_max,
        quadrature_extra::Integer = 0,
        fields = Vector{_SpheroidSurfaceMode}(undef, m_max+1))
    split = split_symmetry && !half && iseven(nt) && iseven(nb) && iseven(chief_points)
    reduced = half || split
    be = _spheroid_curved_boundary(body.a, body.b, reduced ? div(nb, 2) : nb,
        boundary_order+1; half = reduced)
    mesh = _spheroid_fembem_confocal(boundary, body.a, body.b, nr,
        reduced ? div(nt, 2) : nt, order; half = reduced, assemble_bulk = !tensor,
        quadrature_extra)
    T, B = _spheroid_fembem_trace(mesh, be)
    full_boundary = split ?
                    _spheroid_curved_boundary(body.a, body.b, nb,
        boundary_order+1; half = false) : be
    memory_workers = max(1, floor(Int, 4*(50000/(mesh.ndof+length(be.chi)))^1.5))
    workers = min(mode_threads, Threads.nthreads(), tensor ? 4 : memory_workers)
    groups = UnitRange{Int}[]
    for m in modes
        if isempty(groups) || m-first(last(groups))>=16
            push!(groups, m:m)
        else
            groups[end] = first(last(groups)):m
        end
    end
    for span in groups
        selected = filter(m -> m in span, modes)
        operators = _spheroid_curved_operators(be, k, span, Val(16),
            Val(boundary_order+1); rtol, split, quadrature_extra,
            chief_points = split ? div(chief_points, 2) : chief_points)
        Ks, Vs = operators[1:2]
        # Bound simultaneous factorizations independently of BEM row assembly.
        active_workers = min(workers, length(selected))
        @sync for worker in 1:active_workers
            Threads.@spawn for index in worker:active_workers:length(selected)
                m = selected[index]
                operator_index = m-first(span)+1
                if split
                    bulk = tensor ? nothing :
                           _spheroid_fembem_bulk(mesh, k, m, incidence_angle)
                    load = tensor ?
                           _spheroid_fembem_tensor_load(
                        mesh, be, B, k, m, incidence_angle) : nothing
                    even = tensor ?
                           _spheroid_fembem_tensor_mode(
                        mesh, be, T, B, Ks[operator_index], Vs[operator_index],
                        k, m, incidence_angle, reports, active_workers; parity = 1, load) :
                           _spheroid_fembem_mode(
                        mesh, be, T, B, Ks[operator_index], Vs[operator_index],
                        k, m, incidence_angle, reports, active_workers; parity = 1, bulk)
                    odd = tensor ?
                          _spheroid_fembem_tensor_mode(
                        mesh, be, T, B, operators[3][operator_index],
                        operators[4][operator_index],
                        k, m, incidence_angle, reports, active_workers; parity = -1, load) :
                          _spheroid_fembem_mode(mesh, be, T, B,
                        operators[3][operator_index],
                        operators[4][operator_index], k, m, incidence_angle,
                        reports, active_workers; parity = -1, bulk)
                    # Restore traces in full-meridian order for all observation
                    # directions, independently of the illumination symmetry.
                    pressure = vcat(even.pressure+odd.pressure,
                        reverse(even.pressure-odd.pressure))
                    derivative = vcat(even.derivative+odd.derivative,
                        reverse(even.derivative-odd.derivative))
                    fields[m + 1] = _SpheroidSurfaceMode(full_boundary, pressure, derivative)
                else
                    fields[m + 1] = tensor ?
                                    _spheroid_fembem_tensor_mode(
                        mesh, be, T, B, Ks[operator_index], Vs[operator_index],
                        k, m, incidence_angle, reports, active_workers) :
                                    _spheroid_fembem_mode(
                        mesh, be, T, B, Ks[operator_index],
                        Vs[operator_index], k, m, incidence_angle, reports, active_workers)
                end
            end
        end
    end
    return fields
end

function _spheroid_fembem(body, boundary, k;
        coupling, incidence_angle, m_max, mode_threads, order,
        n_r::Union{Nothing, Integer} = nothing, n_theta::Union{Nothing, Integer} = nothing,
        n_boundary::Union{Nothing, Integer} = nothing, boundary_order::Integer = 5,
        rtol::Real = 1e-6, chief_points::Integer = 8, quadrature_extra::Integer = 0,
        adaptive::Bool = true, target_tol::Real = 0.01, max_refinements::Integer = 4)
    body.a > body.b ||
        throw(ArgumentError("interior FEM / exterior BEM requires a prolate spheroid (a > b)"))
    isfinite(k) && k>0 || throw(ArgumentError("wavenumber must be finite and positive"))
    isfinite(incidence_angle) || throw(ArgumentError("incidence_angle must be finite"))
    isfinite(boundary.density_contrast) && boundary.density_contrast>0 &&
    isfinite(boundary.soundspeed_contrast) && boundary.soundspeed_contrast>0 ||
        throw(ArgumentError("fluid contrasts must be finite and positive"))
    2<=order<=8 || throw(ArgumentError("FEM-BEM order must be in 2:8"))
    1<=boundary_order<=8 || throw(ArgumentError("boundary_order must be in 1:8"))
    m_max>=0 || throw(ArgumentError("m_max must be nonnegative"))
    mode_threads>=1 || throw(ArgumentError("mode_threads must be positive"))
    chief_points>=1 || throw(ArgumentError("chief_points must be positive"))
    quadrature_extra>=0 || throw(ArgumentError("quadrature_extra must be nonnegative"))
    isfinite(rtol) && 0<rtol<1 ||
        throw(ArgumentError("rtol must be finite and between zero and one"))
    isfinite(target_tol) && target_tol>0 ||
        throw(ArgumentError("target_tol must be finite and positive"))
    max_refinements>=1 || throw(ArgumentError("max_refinements must be positive"))
    wave = max(k, k/boundary.soundspeed_contrast)
    nr = n_r===nothing ? max(4, ceil(Int, wave*body.b/(1.2order))) : Int(n_r)
    nt = n_theta===nothing ? max(64, 2ceil(Int, wave*body.a/(0.8order))) : Int(n_theta)
    nb = n_boundary===nothing ? max(32, 2ceil(Int, wave*body.a/(1.8*(boundary_order+1)))) :
         Int(n_boundary)
    nr>=1 || throw(ArgumentError("n_r must be positive"))
    nt>=2 || throw(ArgumentError("n_theta must be at least two"))
    nb>=2 || throw(ArgumentError("n_boundary must be at least two"))
    half = abs(float(incidence_angle))==oftype(float(incidence_angle), pi)/2 &&
           iseven(nt) && iseven(nb)
    reports = _SolveReports()
    meshes = [(nr, nt, nb)]
    for level in 1:max_refinements
        r, t, b = last(meshes)
        push!(meshes, (ceil(Int, 4r/3), 2ceil(Int, 2t/3), 2ceil(Int, 2b/3)))
    end
    fields = Vector{_SpheroidSurfaceMode}(undef, m_max+1)
    contributions = zeros(ComplexF64, m_max+1)
    changes = fill(ComplexF64(Inf), m_max+1)
    levels = fill(-1, m_max+1)
    history = NamedTuple[]
    threshold_fraction = -expm1(-target_tol*log(10)/20)
    while true
        estimate = _spheroid_hp_ts_change(contributions, changes)
        adaptive && estimate.change_db<=target_tol && break
        available = findall(level -> level<(adaptive ? max_refinements : 0), levels)
        isempty(available) && break
        threshold = threshold_fraction*estimate.scale/length(contributions)
        selected = filter(index -> estimate.weights[index]>threshold, available)
        if isempty(selected)
            exhausted_changes = [levels[index]==max_refinements ? changes[index] :
                                 zero(eltype(changes)) for index in eachindex(changes)]
            _spheroid_hp_ts_change(contributions, exhausted_changes).change_db>target_tol &&
                break
            index = available[argmax(estimate.weights[available])]
            iszero(estimate.weights[index]) && break
            selected = [index]
        end
        for next_level in sort!(unique(levels[selected] .+ 1))
            indices = filter(index -> levels[index]+1==next_level, selected)
            nr, nt, nb = meshes[next_level + 1]
            _spheroid_fembem_level(body, boundary, k, nr, nt, nb, order, boundary_order;
                half, incidence_angle, m_max, mode_threads, rtol, chief_points, reports,
                quadrature_extra,
                modes = indices .- 1, fields)
            for index in indices
                value = _spheroid_curved_amplitude(
                    fields[index], k, index-1, pi-incidence_angle, pi; quadrature_extra)
                changes[index] = levels[index]<0 ? ComplexF64(Inf) :
                                 value-contributions[index]
                contributions[index] = value
                levels[index] = next_level
            end
            change_db = _spheroid_hp_ts_change(contributions, changes).change_db
            push!(history,
                (; n_r = nr, n_theta = nt, n_boundary = nb,
                    modes = indices .- 1, change_db, amplitude_change = sum(abs, changes)))
        end
    end
    nr, nt, nb = meshes[maximum(levels) + 1]
    change_db = _spheroid_hp_ts_change(contributions, changes).change_db
    amplitude_change = sum(abs, changes)
    if adaptive
        converged = change_db<=target_tol
        _record_refinement!(reports, converged, change_db, target_tol, nr)
        reports.refinement = merge(reports.refinement,
            (; estimator = :fourier_ts_mesh_change, amplitude_change, history,
                mode_refinements = levels, mode_n_r = [meshes[l + 1][1] for l in levels],
                mode_n_theta = [meshes[l + 1][2] for l in levels],
                mode_n_boundary = [meshes[l + 1][3] for l in levels]))
        converged ||
            @warn "FEM-BEM did not reach target_tol=$target_tol dB within $max_refinements refinements, returning the latest solve"
    end
    report = _summarize_solves(reports; method = :coupled,
        coupling = (; interior = coupling.interior, exterior = coupling.exterior),
        solver_options = (;
            incidence_angle, m_max, mode_threads, order, n_r = nr, n_theta = nt,
            n_boundary = nb, boundary_order, rtol, chief_points, quadrature_extra, adaptive, target_tol,
            max_refinements, retain_fields = false), equatorial_symmetry = half,
        equatorial_decomposition = half ? :even :
                                   iseven(nt) && iseven(nb) && iseven(chief_points) ?
                                   :even_odd : :none,
        representation_residual = maximum(r.representation_residual
        for r in reports.systems))
    return CoupledSolution(body, boundary, Float64(k), coupling,
        _SpheroidFEMBEMData(fields, Float64(incidence_angle), Int(order), report))
end
