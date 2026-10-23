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
An integer `max_refinements=4` uses joint mesh refinement. A named tuple with
positive integer `radial`, `angular` and `boundary` budgets requests independent
coordinate checks. These additional solves can cost more than joint refinement.
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
    # Each trace row and coupling column has support only on overlapping
    # elements. Preserve that sparsity when applying the angular eigenvectors.
    return sparse(T), sparse(B)
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

# Solve the complete banded radial Robin system, including element interiors.
# The Robin radial matrix is complex symmetric. Its last inverse column is
# also its last inverse row (without conjugation), so every incident load can
# reuse this one numerically computed boundary response.
function _spheroid_fembem_radial_pencil(factors, p, m, ki, g, eta)
    n = length(factors)*p+1
    base = zeros(ComplexF64, 3p+1, n)
    mass = zeros(Float64, 3p+1, n)
    @inbounds for (e, r) in enumerate(factors), j in 1:(p + 1), i in 1:(p + 1)
        row, col = (e-1)*p+i, (e-1)*p+j
        base[2p + 1 + row - col, col] += (r.stiffness[i, j]+m^2*r.azimuthal[i, j]-ki^2*r.wave[i, j])/g
        mass[2p + 1 + row - col, col] += r.mass[i, j]/g
    end
    if m>0
        @inbounds for j in 1:(p + 1)
            base[2p + 2 - j, j] = mass[2p + 2 - j, j] = 0
            base[2p + j, 1] = mass[2p + j, 1] = 0
        end
        base[2p + 1, 1] = 1
    end
    base[2p + 1, n] += im*eta
    return base, mass
end

function _spheroid_fembem_radial_map!(rhs, band, base, mass, p, lambda)
    @. band = base+lambda*mass
    fill!(rhs, 0)
    rhs[end] = 1
    n = length(rhs)
    factor, pivots = LinearAlgebra.LAPACK.gbtrf!(p, p, n, band)
    return LinearAlgebra.LAPACK.gbtrs!('N', p, p, n, factor, pivots, rhs)
end

# Exact band Cholesky and triangular solves avoid a dense generalized
# eigensystem reduction. The resulting standard problem remains symmetric.
function _spheroid_band_cholesky!(L)
    p, n = size(L, 1)-1, size(L, 2)
    @inbounds for j in 1:n
        d = L[1, j]
        for k in max(1, j - p):(j - 1)
            d -= L[j - k + 1, k]^2
        end
        d>0 || throw(LinearAlgebra.PosDefException(j))
        L[1, j] = sqrt(d)
        for i in (j + 1):min(n, j + p)
            s = L[i - j + 1, j]
            for k in max(1, i - p):(j - 1)
                s -= L[i - k + 1, k]*L[j - k + 1, k]
            end
            L[i - j + 1, j] = s/L[1, j]
        end
    end
    return L
end

function _spheroid_band_ldiv!(L, X; transposed = false)
    p, n = size(L, 1)-1, size(L, 2)
    @inbounds for j in axes(X, 2)
        if transposed
            for i in n:-1:1
                s = X[i, j]
                for k in (i + 1):min(n, i + p)
                    s -= L[k - i + 1, i]*X[k, j]
                end
                X[i, j] = s/L[1, i]
            end
        else
            for i in 1:n
                s = X[i, j]
                for k in max(1, i - p):(i - 1)
                    s -= L[i - k + 1, k]*X[k, j]
                end
                X[i, j] = s/L[1, i]
            end
        end
    end
    return X
end

function _spheroid_fembem_angular(mesh, k, m, parity)
    p, ntn = mesh.p, mesh.nt*mesh.p+1
    first = m>0 ? 2 : 1
    last = parity==-1 || (!mesh.half && m>0) ? ntn-1 : ntn
    free = first:last
    n = length(free)
    mass, angular = zeros(p+1, n), zeros(n, n)
    ki = k/mesh.soundspeed_contrast
    @inbounds for (e, t) in enumerate(mesh.angular_factors), j in 1:(p + 1), i in 1:(p + 1)
        row, col = (e-1)*p+i-first+1, (e-1)*p+j-first+1
        if 1<=row<=n && 1<=col<=n
            angular[row, col] += t.stiffness[i, j]+m^2*t.azimuthal[i, j]-ki^2*t.wave[i, j]
            row>=col && (mass[row - col + 1, col] += t.mass[i, j])
        end
    end
    _spheroid_band_cholesky!(mass)
    _spheroid_band_ldiv!(mass, angular)
    standard = _spheroid_band_ldiv!(mass, copy(transpose(angular)))
    values, vectors = LinearAlgebra.LAPACK.syevd!('V', 'U', standard)
    _spheroid_band_ldiv!(mass, vectors; transposed = true)
    E = zeros(ntn, n)
    E[free, :] = vectors
    return (; values, vectors = E)
end

function _spheroid_fembem_exterior(
        be, K, V, m, parity; reuse = nothing, exterior_key = nothing)
    _reuse_numerical(exterior_key===nothing ? nothing : reuse,
        (:fembem_exterior, exterior_key, m, parity==-1)) do
        nbe=length(be.chi)
        D=-K
        for j in 1:nbe
            D[j, j]+=0.5
        end
        rawD, rawV=D, V
        reducedV=V
        if size(V, 1)>nbe
            factor=LinearAlgebra.qr(V)
            D=(factor.Q' * D)[1:nbe, :]
            reducedV=Matrix(factor.R)
        end
        (; D, V = reducedV, rawD, rawV)
    end
end

function _spheroid_fembem_tensor_operator(mesh, be, T, B, K, V, k, m, parity;
        reuse = nothing, exterior_key = nothing)
    # Geometry and discretization are explicit in every reuse key. Parity zero
    # and even parity have the same angular constraints, but different loads.
    angular_key = (:fembem_angular, mesh.a, mesh.b, mesh.nt, mesh.p, mesh.half,
        mesh.quadrature_extra, mesh.soundspeed_contrast, k, m, parity==-1)
    eigenproblem = _reuse_numerical(reuse, angular_key) do
        _spheroid_fembem_angular(mesh, k, m, parity)
    end
    E = eigenproblem.vectors
    g, ki = mesh.density_contrast, k/mesh.soundspeed_contrast
    eta = mesh.a*mesh.b*ki/g
    radial_key = (:fembem_radial, angular_key, mesh.nr, g)
    radial = _reuse_numerical(reuse, radial_key) do
        responses = Matrix{ComplexF64}(undef, mesh.nr*mesh.p+1, length(eigenproblem.values))
        base, mass = _spheroid_fembem_radial_pencil(
            mesh.radial_factors, mesh.p, m, ki, g, eta)
        band = similar(base)
        for j in eachindex(eigenproblem.values)
            _spheroid_fembem_radial_map!(view(responses, :, j), band, base, mass,
                mesh.p, eigenproblem.values[j])
        end
        responses
    end
    beta_response = vec(radial[end, :])
    alpha = 1 .- im*eta .* beta_response
    nbe = length(be.chi)
    exterior=_spheroid_fembem_exterior(be, K, V, m, parity; reuse, exterior_key)
    D, V, rawD, rawV=exterior.D, exterior.V, exterior.rawD, exterior.rawV
    pressure_map = T*E
    flux_map = transpose(E)*B
    retained = findall(j->abs(alpha[j])<=0.25eta*abs(beta_response[j]), eachindex(alpha))
    regular = setdiff(eachindex(alpha), retained)
    response = beta_response[regular] ./ alpha[regular]
    # Choose the cheaper contraction of the same Schur complement. Applying
    # complex D to the real angular map costs half as much per entry as applying
    # it to the complex boundary response. Neither path truncates angular modes.
    regular_exterior = length(regular)>2nbe ? nothing : D*pressure_map[:, regular]
    schur = if regular_exterior===nothing
        interior_response = (pressure_map[:, regular] .* transpose(response))*flux_map[regular, :]
        V+D*interior_response
    else
        V+(regular_exterior .* transpose(response))*flux_map[regular, :]
    end
    retained_exterior = D*pressure_map[:, retained]
    matrix = [schur retained_exterior;
              -Diagonal(beta_response[retained])*flux_map[retained, :] Matrix(Diagonal(alpha[retained]))]
    return (;
        E, radial, alpha, beta_response, rawD, rawV,
        D, pressure_map, flux_map, regular_exterior,
        retained, regular, response, matrix, factor = LinearAlgebra.lu(matrix))
end

function _spheroid_fembem_factor_matrices(factors, p, k, m, scale)
    rows, cols = Int[], Int[]
    stiffness, mass = Float64[], Float64[]
    for array in (rows, cols, stiffness, mass)
        sizehint!(array, length(factors)*(p+1)^2)
    end
    for (e, f) in enumerate(factors), j in 1:(p + 1), i in 1:(p + 1)
        push!(rows, (e-1)*p+i)
        push!(cols, (e-1)*p+j)
        push!(stiffness, scale*(f.stiffness[i, j]+m^2*f.azimuthal[i, j]-k^2*f.wave[i, j]))
        push!(mass, scale*f.mass[i, j])
    end
    n=length(factors)*p+1
    return sparse(rows, cols, stiffness, n, n), sparse(rows, cols, mass, n, n)
end

# Diagonalize the shorter radial coordinate, then solve every angular system
# as a complete band matrix. The complex Robin term is removed by the boundary
# equation. It changes only the algebraic solver, not the FEM weak form.
function _spheroid_fembem_radial_inverse(mesh, k, m, parity)
    p=mesh.p
    nr, nt=mesh.nr*p+1, mesh.nt*p+1
    radial_free=(m > 0 ? 2 : 1):nr
    angular_free=(m > 0 ? 2 : 1):(parity == -1 || (!mesh.half && m > 0) ? nt - 1 : nt)
    ki=k/mesh.soundspeed_contrast
    Ar, Mr=_spheroid_fembem_factor_matrices(
        mesh.radial_factors, p, ki, m, inv(mesh.density_contrast))
    At, Mt=_spheroid_fembem_factor_matrices(mesh.angular_factors, p, ki, m, 1.0)
    Ar, Mr=Ar[radial_free, radial_free], Mr[radial_free, radial_free]
    At, Mt=At[angular_free, angular_free], Mt[angular_free, angular_free]
    # Match exterior impedance, including for large density contrast.
    eta=mesh.a*mesh.b*k
    robin=complex.(Matrix(Ar))
    robin[end, end]-=im*eta
    L=LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix(Mr))).L
    eigensystem=LinearAlgebra.eigen(L\robin/transpose(L))
    E=transpose(L)\eigensystem.vectors
    for j in axes(E, 2)
        column=view(E, :, j)
        normalization=sum(column .* (Mr*column))
        isfinite(normalization) && !iszero(normalization) || return nothing
        column ./= sqrt(normalization)
    end
    orthogonality=norm(transpose(E)*Mr*E-I, Inf)
    eigen_residual=norm(robin*E-Mr*E*Diagonal(eigensystem.values))/(norm(robin)*norm(E))
    isfinite(orthogonality) && orthogonality<=1e-8 && eigen_residual<=1e-12 ||
        return nothing
    n=length(angular_free)
    factors=map(eigensystem.values) do lambda
        band=zeros(ComplexF64, 3p+1, n)
        for j in 1:n, i in max(1, j - p):min(n, j + p)

            band[2p + 1 + i - j, j]=At[i, j]+lambda*Mt[i, j]
        end
        matrix, pivots=LinearAlgebra.LAPACK.gbtrf!(p, p, n, band)
        (; matrix, pivots)
    end
    return (; E, radial_free, angular_free, factors, Ar, Mr, At, Mt, eta, p,
        boundary_weights = vec(E[end, :]), orthogonality, eigen_residual)
end

function _spheroid_fembem_radial_solve!(work, inverse)
    n=length(inverse.angular_free)
    for j in axes(work, 2)
        factor=inverse.factors[j]
        LinearAlgebra.LAPACK.gbtrs!('N', inverse.p, inverse.p, n,
            factor.matrix, factor.pivots, view(work, :, j))
    end
    return work
end

function _spheroid_fembem_radial_apply!(out, inverse, input, work)
    for j in axes(work, 2)
        @views work[:, j] .= input .* inverse.boundary_weights[j]
    end
    _spheroid_fembem_radial_solve!(work, inverse)
    return mul!(out, work, inverse.boundary_weights)
end

function _spheroid_fembem_radial_operator(
        mesh, be, T, B, K, V, k, m, parity; reuse = nothing, exterior_key = nothing)
    key=(:fembem_radial_inverse, mesh.a, mesh.b, mesh.nr, mesh.nt, mesh.p, mesh.half,
        mesh.quadrature_extra, mesh.density_contrast, mesh.soundspeed_contrast, k, m, parity==-1)
    inverse=_reuse_numerical(reuse, key) do
        _spheroid_fembem_radial_inverse(mesh, k, m, parity)
    end
    inverse===nothing && return nothing
    exterior=_spheroid_fembem_exterior(be, K, V, m, parity; reuse, exterior_key)
    D, V, rawD, rawV=exterior.D, exterior.V, exterior.rawD, exterior.rawV
    pressure_map=T[:, inverse.angular_free]
    flux_map=B[inverse.angular_free, :]
    factor=_reuse_numerical(exterior_key===nothing ? nothing : reuse,
        (:fembem_exterior_factor, exterior_key, m, parity==-1)) do
        LinearAlgebra.lu(V)
    end
    exterior_map=factor\(D*pressure_map)
    return (; inverse, rawD, rawV, pressure_map, flux_map, exterior_map)
end

function _spheroid_fembem_radial_mode(mesh, be, T, B, K, V, k, m, beta, reports, workers;
        parity, load, reuse, reuse_key, exterior_key = nothing, maxiter = 300)
    op=_reuse_numerical(reuse, (:fembem_radial_operator, reuse_key, m, parity==-1)) do
        _spheroid_fembem_radial_operator(
            mesh, be, T, B, K, V, k, m, parity; reuse, exterior_key)
    end
    op===nothing && return (; field = nothing, reason = :radial_eigensystem, iterations = 0)
    inverse=op.inverse
    load=parity==0 ? load : (load+parity*(-1)^m*conj(load))/2
    source=load[inverse.angular_free, inverse.radial_free]
    transformed=source*inverse.E
    work=copy(transformed)
    _spheroid_fembem_radial_solve!(work, inverse)
    rhs=work*inverse.boundary_weights
    n=length(rhs)
    input, mass_image=zeros(ComplexF64, n), zeros(ComplexF64, n)
    exterior_image=zeros(ComplexF64, length(be.chi))
    function apply!(out, u)
        mul!(exterior_image, op.exterior_map, u)
        mul!(input, op.flux_map, exterior_image)
        mul!(mass_image, inverse.Mt, u)
        @. input+=im*inverse.eta*mass_image
        _spheroid_fembem_radial_apply!(out, inverse, input, work)
        out .+= u
        return out
    end
    operator=LinearMaps.LinearMap{ComplexF64}(apply!, n; ismutating = true)
    u, history=IterativeSolvers.gmres(operator, rhs; reltol = 2e-13,
        restart = min(n, 100), maxiter, log = true)
    residual=_linear_residual(operator, u, rhs)
    if !history.isconverged || !(residual.relative_residual<=1e-12)
        return (; field = nothing, reason = :iteration, iterations = history.iters)
    end
    qs=-op.exterior_map*u
    ps=op.pressure_map*u
    # Reconstruct the volume field and check the original, unpreconditioned
    # finite-element equations, including the forcing and interface flux.
    mul!(input, op.flux_map, qs)
    mul!(mass_image, inverse.Mt, u)
    @. input-=im*inverse.eta*mass_image
    work .= transformed
    for j in axes(work, 2)
        @views work[:, j] .+= input .* inverse.boundary_weights[j]
    end
    _spheroid_fembem_radial_solve!(work, inverse)
    interior=work*transpose(inverse.E)
    original_rhs=copy(source)
    original_rhs[:, end] += op.flux_map*qs
    original_residual=inverse.Mt*(interior*inverse.Ar)+inverse.At*(interior*inverse.Mr)-original_rhs
    fem_residual=_residual_report(original_residual, original_rhs)
    trace_residual=norm(interior[:, end]-u)/max(norm(u), floatmin(Float64))
    if !(fem_residual.relative_residual<=1e-9 && trace_residual<=1e-10)
        return (; field = nothing, reason = :original_equations, iterations = history.iters)
    end
    incident=[_p_inc_mode(m, k, beta, r, z) for (r, z) in zip(be.rho, be.z)]
    parity==0 || (incident=(incident+parity*(-1)^m*conj(incident))/2)
    representation_residual=_residual_report(op.rawD*ps+op.rawV*qs, incident).relative_residual
    _record_solve!(
        reports, operator, u, rhs; mode = m, component = :fem_bem, mode_workers = workers,
        method = :gmres, iterations = history.iters, converged = true,
        representation_residual, n_r = mesh.nr, n_theta = mesh.half ? 2mesh.nt : mesh.nt,
        n_boundary = be.elements, chief_points = size(op.rawV, 1)-length(be.chi), parity,
        interior_reduction = :radial, interior_node_count = (mesh.nr*mesh.p+1)*(mesh.nt*mesh.p+1),
        fem_relative_residual = fem_residual.relative_residual, trace_relative_residual = trace_residual,
        radial_eigen_residual = inverse.eigen_residual, radial_orthogonality = inverse.orthogonality)
    return (; field = _SpheroidSurfaceMode(be, ps, qs),
        reason = :accepted, iterations = history.iters)
end

function _spheroid_fembem_use_radial(mesh)
    nr, nt=mesh.nr*mesh.p+1, mesh.nt*mesh.p+1
    storage=sizeof(ComplexF64)*(3mesh.p+1)*nr*nt
    return nt>=1024 && nt>=4nr && storage<=384*1024^2
end

# Fast diagonalization of the same tensor-product finite-element equations.
# No analytical spheroidal functions or fitted modal data enter this reduction.
function _spheroid_fembem_tensor_mode(
        mesh, be, T, B, K, V, k, m, beta, reports, workers; parity = 0,
        load = _spheroid_fembem_tensor_load(mesh, be, B, k, m, beta), reuse = nothing, reuse_key = nothing,
        interior_reduction = :auto, radial_maxiter = 300, exterior_key = nothing)
    radial_attempt = nothing
    if interior_reduction===:radial ||
       (interior_reduction===:auto && _spheroid_fembem_use_radial(mesh))
        radial_attempt=_spheroid_fembem_radial_mode(
            mesh, be, T, B, K, V, k, m, beta, reports, workers;
            parity, load, reuse, reuse_key, exterior_key, maxiter = radial_maxiter)
        radial_attempt.field===nothing || return radial_attempt.field
    end
    op = _reuse_numerical(reuse, (:fembem_operator, reuse_key, m, parity==-1)) do
        _spheroid_fembem_tensor_operator(
            mesh, be, T, B, K, V, k, m, parity; reuse, exterior_key)
    end
    load = parity==0 ? load : (load+parity*(-1)^m*conj(load))/2
    incident = [_p_inc_mode(m, k, beta, r, z) for (r, z) in zip(be.rho, be.z)]
    parity==0 || (incident=(incident+parity*(-1)^m*conj(incident))/2)
    transformed = transpose(op.E)*load
    forcing = [sum(view(op.radial, :, j) .* view(transformed, j, :)) for j in axes(op.E, 2)]
    particular = forcing[op.regular] ./ op.alpha[op.regular]
    exterior_forcing = op.regular_exterior===nothing ?
                       op.D*(op.pressure_map[:, op.regular]*particular) :
                       op.regular_exterior*particular
    rhs = vcat(-exterior_forcing, forcing[op.retained])
    solution = op.factor\rhs
    nbe = length(be.chi)
    qs = solution[1:nbe]
    coefficients = zeros(ComplexF64, length(op.alpha))
    coefficients[op.regular] = particular+op.response .* (op.flux_map[op.regular, :]*qs)
    coefficients[op.retained] = solution[(nbe + 1):end]
    ps = op.pressure_map*coefficients
    representation_residual = _residual_report(op.rawD*ps+op.rawV*qs, incident).relative_residual
    _record_solve!(reports, op.matrix, solution, rhs; mode = m,
        component = :fem_bem, mode_workers = workers,
        representation_residual, n_r = mesh.nr, n_theta = mesh.half ? 2mesh.nt : mesh.nt, n_boundary = be.elements,
        chief_points = size(op.rawV, 1)-nbe, parity, interior_reduction = :tensor,
        interior_node_count = (mesh.nr*mesh.p+1)*(mesh.nt*mesh.p+1), retained_angular_modes = length(op.retained),
        radial_attempt)
    return _SpheroidSurfaceMode(be, ps, qs)
end

function _spheroid_fembem_level(body, boundary, k, nr, nt, nb, order, boundary_order;
        half, incidence_angle, m_max, mode_threads, rtol, chief_points, reports,
        split_symmetry::Bool = true, tensor::Bool = true, modes = 0:m_max,
        interior_reduction = :auto,
        operator_spans = nothing,
        quadrature_extra::Integer = 0,
        fields = Vector{_SpheroidSurfaceMode}(undef, m_max+1))
    reuse = _numerical_reuse_context()
    split = split_symmetry && !half && iseven(nt) && iseven(nb) && iseven(chief_points)
    reduced = half || split
    reuse_key = (
        body.a, body.b, boundary.density_contrast, boundary.soundspeed_contrast, k,
        nr, nt, nb, order, boundary_order, half, split, rtol, chief_points, quadrature_extra)
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
    operator_spans===nothing || (groups=operator_spans)
    for span in groups
        span_key = (reuse_key, first(span), last(span))
        exterior_key=(
            body.a, body.b, k, nb, boundary_order, half, split, rtol, chief_points,
            quadrature_extra, first(span), last(span))
        selected = filter(m -> m in span, modes)
        width = length(span)==1 ? Val(1) :
                length(span)<=4 ? Val(4) : length(span)<=8 ? Val(8) : Val(16)
        operators = _spheroid_curved_operators(be, k, span, width,
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
                        k, m, incidence_angle, reports, active_workers;
                        parity = 1, load, reuse, reuse_key = span_key,
                        interior_reduction, exterior_key) :
                           _spheroid_fembem_mode(
                        mesh, be, T, B, Ks[operator_index], Vs[operator_index],
                        k, m, incidence_angle, reports, active_workers; parity = 1, bulk)
                    odd = tensor ?
                          _spheroid_fembem_tensor_mode(
                        mesh, be, T, B, operators[3][operator_index],
                        operators[4][operator_index],
                        k, m, incidence_angle, reports, active_workers;
                        parity = -1, load, reuse, reuse_key = span_key,
                        interior_reduction, exterior_key) :
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
                        k, m, incidence_angle, reports, active_workers;
                        reuse, reuse_key = span_key, interior_reduction, exterior_key) :
                                    _spheroid_fembem_mode(
                        mesh, be, T, B, Ks[operator_index],
                        Vs[operator_index], k, m, incidence_angle, reports, active_workers)
                end
            end
        end
    end
    return fields
end

function _spheroid_fembem_refinement_limits(value)
    limits = value isa Integer ? (value, value, value) :
             value isa NamedTuple &&
             issetequal(keys(value), (:radial, :angular, :boundary)) ?
             (value.radial, value.angular, value.boundary) : nothing
    limits!==nothing && all(n->n isa Integer && n>=1, limits) ||
        throw(ArgumentError("max_refinements must be a positive integer or " *
                            "(radial=..., angular=..., boundary=...) of positive integers"))
    return Int.(limits)
end

# Compare the current mesh with three independently coarsened meshes and a
# jointly coarsened mesh. The fourth difference measures the interaction that
# would be missed by adding the three coordinate changes. These are observed
# discretization changes, not rigorous bounds on the remaining continuum error.
function _spheroid_fembem_directional_changes(values, levels)
    current = values[levels]
    directions = ntuple(3) do d
        lower = ntuple(j->levels[j]-(j==d), 3)
        current-values[lower]
    end
    joint = current-values[levels .- 1]
    return (directions..., joint-sum(directions))
end

function _spheroid_fembem_mark_directions(weights, levels, limits)
    axis_weights = [weights[d]+weights[4]/3 for d in 1:3]
    available = [d for d in 1:3 if levels[d]<limits[d]]
    isempty(available) && return levels
    largest = maximum(axis_weights)
    iszero(largest) && return levels
    return ntuple(d->levels[d]+(d in available && axis_weights[d]>=largest/2), 3)
end

# Process each common BEM batch across the requested interior meshes before
# moving to the next batch. Bound its size so recent operators fit in the
# adaptive cache even at fine boundary resolutions.
function _spheroid_fembem_refinement_requests(
        requested, counts, m_max, boundary_order, reduced)
    schedule=NamedTuple[]
    for boundary_level in sort!(unique(level[3] for level in keys(requested)))
        nodes=counts[3][boundary_level + 1]*(boundary_order+1)÷(reduced ? 2 : 1)
        width=nodes>1024 ? 1 : nodes>512 ? 4 : nodes>256 ? 8 : 16
        meshes=sort!([level for level in keys(requested) if level[3]==boundary_level];
            by = level->(level[2], level[1]))
        for first_mode in 0:width:m_max
            span=first_mode:min(m_max, first_mode + width - 1)
            for mesh_level in meshes
                indices=filter(i->i-1 in span, requested[mesh_level])
                isempty(indices) || push!(schedule, (; mesh_level, indices, span))
            end
        end
    end
    return schedule
end

function _spheroid_fembem_directional(body, boundary, k, initial, limits;
        half, incidence_angle, m_max, mode_threads, order, boundary_order,
        rtol, chief_points, reports, quadrature_extra, target_tol, separate = true)
    counts = map(1:3) do d
        sizes = [initial[d]]
        for _ in 1:limits[d]
            n = last(sizes)
            push!(sizes, d==1 ? cld(4n, 3) : 2cld(2n, 3))
        end
        sizes
    end
    nm = m_max+1
    levels = fill((1, 1, 1), nm)
    directional = falses(nm)
    values = [Dict{NTuple{3, Int}, ComplexF64}() for _ in 1:nm]
    fields = Vector{_SpheroidSurfaceMode}(undef, nm)
    contributions = zeros(ComplexF64, nm)
    changes = fill(ComplexF64(Inf), nm, 4)
    history = NamedTuple[]
    threshold_fraction = -expm1(-target_tol*log(10)/20)
    while true
        requested = Dict{NTuple{3, Int}, Vector{Int}}()
        for index in 1:nm
            level = levels[index]
            neighbors = [level, level .- 1]
            if directional[index]
                append!(neighbors, [ntuple(j->level[j]-(j==d), 3) for d in 1:3])
            end
            for mesh_level in neighbors
                haskey(values[index], mesh_level) && continue
                push!(get!(requested, mesh_level, Int[]), index)
            end
        end
        reduced=half || (iseven(initial[2]) && iseven(initial[3]) && iseven(chief_points))
        schedule=separate ?
                 _spheroid_fembem_refinement_requests(
            requested, counts, m_max, boundary_order, reduced) :
                 [(; mesh_level, indices = requested[mesh_level], span = nothing)
                  for mesh_level in sort!(collect(keys(requested)))]
        for request in schedule
            mesh_level, indices=request.mesh_level, request.indices
            nr, nt, nb = ntuple(d->counts[d][mesh_level[d] + 1], 3)
            trial = _spheroid_fembem_level(
                body, boundary, k, nr, nt, nb, order, boundary_order;
                half, incidence_angle, m_max, mode_threads, rtol, chief_points, reports,
                quadrature_extra, modes = indices .- 1,
                operator_spans = request.span===nothing ? nothing : [request.span])
            for index in indices
                values[index][mesh_level] = _spheroid_curved_amplitude(
                    trial[index], k, index-1, pi-incidence_angle, pi; quadrature_extra)
                if mesh_level==levels[index]
                    fields[index] = trial[index]
                    contributions[index] = values[index][mesh_level]
                end
            end
            push!(history,
                (; n_r = nr, n_theta = nt, n_boundary = nb,
                    modes = indices .- 1, refinement_levels = mesh_level,
                    change_db = Inf, amplitude_change = Inf))
        end
        for index in 1:nm
            if directional[index]
                changes[index, :] .= _spheroid_fembem_directional_changes(values[index], levels[index])
            else
                # Modes that pass the original joint check do not need three
                # additional solves just to decide where further work belongs.
                level = levels[index]
                changes[index, :] .= (
                    values[index][level]-values[index][level .- 1], 0, 0, 0)
            end
        end
        estimate = _spheroid_hp_ts_change(contributions, vec(changes))
        history[end] = merge(last(history),
            (; change_db = estimate.change_db,
                amplitude_change = sum(abs, changes)))
        estimate.change_db<=target_tol && break
        weights = reshape(estimate.weights, nm, 4)
        threshold = threshold_fraction*estimate.scale/nm
        selected = findall(index->sum(view(weights, index, :))>threshold, 1:nm)
        promoted = false
        for index in selected
            if directional[index]
                next = _spheroid_fembem_mark_directions(view(weights, index, :), levels[index], limits)
                promoted |= next!=levels[index]
                levels[index] = next
            else
                if separate
                    directional[index] = true
                    promoted = true
                elseif all(levels[index][d]<limits[d] for d in 1:3)
                    levels[index] = levels[index] .+ 1
                    promoted = true
                end
            end
        end
        promoted || break
    end
    sizes = [ntuple(d->counts[d][level[d] + 1], 3) for level in levels]
    estimate = _spheroid_hp_ts_change(contributions, vec(changes))
    converged = estimate.change_db<=target_tol
    nr, nt, nb = ntuple(d->maximum(size[d] for size in sizes), 3)
    _record_refinement!(reports, converged, estimate.change_db, target_tol, nr)
    reports.refinement = merge(reports.refinement,
        (;
            estimator = separate ? :directional_fourier_ts_mesh_change :
                        :fourier_ts_mesh_change,
            amplitude_change = sum(abs, changes),
            history, mode_refinements = maximum.(levels),
            mode_refinements_by_axis = (; radial = getindex.(levels, 1),
                angular = getindex.(levels, 2), boundary = getindex.(levels, 3)),
            mode_directional_checks = directional,
            directional_amplitude_changes = [directional[i] ? changes[i, d] :
                                             ComplexF64(NaN)
                                             for i in 1:nm, d in 1:3],
            interaction_amplitude_changes = [directional[i] ? changes[i, 4] :
                                             ComplexF64(NaN)
                                             for i in 1:nm],
            joint_amplitude_changes = [values[i][levels[i]]-values[i][levels[i] .- 1]
                                       for i in 1:nm],
            mode_n_r = getindex.(sizes, 1), mode_n_theta = getindex.(sizes, 2),
            mode_n_boundary = getindex.(sizes, 3)))
    converged ||
        @warn "FEM-BEM did not reach target_tol=$target_tol dB within the refinement budgets, returning the latest solve"
    return fields, nr, nt, nb
end

function _spheroid_fembem(body, boundary, k;
        coupling, incidence_angle, m_max, mode_threads, order,
        n_r::Union{Nothing, Integer} = nothing, n_theta::Union{Nothing, Integer} = nothing,
        n_boundary::Union{Nothing, Integer} = nothing, boundary_order::Integer = 5,
        rtol::Real = 1e-6, chief_points::Integer = 8, quadrature_extra::Integer = 0,
        adaptive::Bool = true, target_tol::Real = 0.01, max_refinements = 4)
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
    limits = _spheroid_fembem_refinement_limits(max_refinements)
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
    fields = if adaptive
        separate = max_refinements isa NamedTuple
        function refine()
            _spheroid_fembem_directional(body, boundary, k, (nr, nt, nb), limits;
                half, incidence_angle, m_max, mode_threads, order, boundary_order,
                rtol, chief_points, reports, quadrature_extra, target_tol, separate)
        end
        # Separate checks revisit exterior and interior operators. Joint
        # refinement advances every mesh count, so a standalone solve has
        # no such reuse. An enclosing angle-sweep cache still applies.
        result, nr, nt, nb = separate ?
                             _with_numerical_reuse(refine; retain_prefix = false) : refine()
        result
    else
        _spheroid_fembem_level(body, boundary, k, nr, nt, nb, order, boundary_order;
            half, incidence_angle, m_max, mode_threads, rtol, chief_points, reports,
            quadrature_extra)
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
    iterative=filter(system->system.method===:gmres, reports.systems)
    if !isempty(iterative)
        report=merge(report,
            (; method = length(iterative)==length(reports.systems) ? :gmres : :mixed,
                iterations = sum(system.iterations for system in iterative),
                converged = all(system.converged for system in iterative)))
    end
    return CoupledSolution(body, boundary, Float64(k), coupling,
        _SpheroidFEMBEMData(fields, Float64(incidence_angle), Int(order), report))
end
