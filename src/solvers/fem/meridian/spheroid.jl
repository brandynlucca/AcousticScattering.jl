# Spheroid geometry and mode sampling for the shared meridian FEM assembly.

"""
    _spheroid_r_inner(θ, a, b)

Distance from the center to a prolate/oblate spheroid's own surface along
the ray at angle `θ` from the x-axis (axis of symmetry, semi-axis `a`), `b` is the equatorial semi-axis, matching [`Spheroid`](@ref)'s convention.
From `(z/a)² + (ρ/b)² = 1` with `z = r cosθ, ρ = r sinθ`:
`r(θ) = 1/√(cos²θ/a² + sin²θ/b²)`.
"""
function _spheroid_r_inner(θ::Real, a::Real, b::Real)
    return 1 / sqrt(cos(θ)^2 / a^2 + sin(θ)^2 / b^2)
end

# Azimuthal order follows the transverse radius, independently of the polar
# degree needed on the enclosing DtN sphere. Slow interiors retain additional
# orders around their angular wave scale. Cap this heuristic at the previous
# enclosing-body estimate, so extreme contrasts retain the existing budget.
function _spheroid_meridian_default_mmax(body::Spheroid, boundary, k, incidence_angle)
    angle = float(incidence_angle)
    (iszero(angle) || abs(angle) == oftype(angle, pi)) && return 0
    legacy = _default_mode_count(k * max(body.a, body.b))
    exterior = _default_mode_count(k * body.b)
    interior = boundary isa FluidFilled ?
               ceil(Int, k * body.b / boundary.soundspeed_contrast) + 4 : 0
    return min(legacy, max(exterior, interior))
end

"""
    _spheroid_meridian_fem_modes(boundary, k, a, b, R, incidence_angle;
        m_max, n_r=30, n_theta=60, l_max=default, mode_threads=1, retain_fields=false)

Complex traces on the enclosing DtN sphere for azimuthal Fourier modes
`m = 0, ..., m_max`. Angles are measured from the symmetry axis. Mode solves
run serially unless `mode_threads` explicitly permits concurrent factorizations.
`retain_fields=true` also returns the nodal fields for pressure evaluation.
"""
function _spheroid_meridian_fem_modes(
        boundary::Union{Rigid, PressureRelease, FluidFilled}, k::Real,
        a::Real, b::Real, R::Real, incidence_angle::Real;
        m_max::Integer, n_r::Integer = 30, n_theta::Integer = 60,
        l_max::Integer = max(_default_mode_count(k * R), m_max),
        mode_threads::Integer = 1, solve_reports = nothing, retain_fields::Bool = false,
        dtn_form::Symbol = :assembled)
    m_max >= 0 || throw(ArgumentError("m_max must be nonnegative"))
    l_max >= m_max || throw(ArgumentError("l_max must be at least m_max"))
    n_r >= 1 || throw(ArgumentError("n_r must be positive"))
    n_theta >= 2 || throw(ArgumentError("n_theta must be at least two"))
    mode_threads >= 1 || throw(ArgumentError("mode_threads must be positive"))
    R > max(a, b) || throw(ArgumentError("R must exceed both spheroid semi-axes"))
    p_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    dpdn_modes = Vector{Vector{ComplexF64}}(undef, m_max + 1)
    fields = Vector{_MeridianFEMModeField}(undef, retain_fields ? m_max + 1 : 0)
    hs_cache, hsd_cache = _spherical_hankel_cache(l_max, k * R)
    function solve_mode(m)
        _, pR, dpdnR, field = _spheroid_fem_mode_trace(
            boundary, m, k, incidence_angle, a, b, R, n_r, n_theta, l_max,
            hs_cache, hsd_cache; solve_reports, retain_fields, dtn_form)
        p_modes[m + 1] = pR
        dpdn_modes[m + 1] = dpdnR
        retain_fields && (fields[m + 1] = field)
        return nothing
    end
    workers = min(mode_threads, Threads.nthreads(), m_max + 1)
    if workers == 1
        for m in 0:m_max
            solve_mode(m)
        end
    else
        @sync for worker in 1:workers
            Threads.@spawn for m in (worker - 1):workers:m_max
                solve_mode(m)
            end
        end
    end
    angles = collect(range(0.0, pi; length = n_theta + 1))
    mesh_R = MeridianMesh(R .* sin.(angles), R .* cos.(angles))
    ps_R = panels(mesh_R)
    p_panel_modes = [ComplexF64[0.5 * (pm[j] + pm[j + 1]) for j in 1:n_theta]
                     for pm in p_modes]
    dpdn_panel_modes = [ComplexF64[0.5 * (dm[j] + dm[j + 1]) for j in 1:n_theta]
                        for dm in dpdn_modes]
    return retain_fields ? (ps_R, p_panel_modes, dpdn_panel_modes, fields) :
           (ps_R, p_panel_modes, dpdn_panel_modes)
end

function _spheroid_fem_mode_trace(boundary::Union{Rigid, PressureRelease, FluidFilled},
        m::Integer, k::Real, incidence_angle::Real, a::Real, b::Real, R::Real,
        n_r::Integer, n_theta::Integer, l_max::Integer,
        hs_cache::Vector{ComplexF64}, hsd_cache::Vector{ComplexF64}; solve_reports = nothing,
        retain_fields::Bool = true, dtn_form::Symbol = :assembled)
    angles = collect(range(0.0, pi; length = n_theta + 1))
    r_inner = [_spheroid_r_inner(t, a, b) for t in angles]
    return _meridian_fem_mode_trace(boundary, m, k, incidence_angle, angles, r_inner, R,
        n_r, n_theta, l_max, hs_cache, hsd_cache; solve_reports, retain_fields, dtn_form)
end

# Higher-order mapped spheroid elements and element-interior condensation.
function _spheroid_hp_rule(q)
    eig = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(q), [j/sqrt(4j^2-1)
                                                                      for j in 1:(q - 1)]))
    return eig.values, 2eig.vectors[1, :] .^ 2
end
function _spheroid_hp_basis(p, x)
    nodes=collect(range(-1.0, 1.0; length = p+1))
    N=ones(p+1)
    D=zeros(p+1)
    for i in 1:(p + 1)
        for j in 1:(p + 1)
            j==i || (N[i] *= (x-nodes[j])/(nodes[i]-nodes[j]))
        end
        for j in 1:(p + 1)
            j==i && continue
            v=1/(nodes[i]-nodes[j])
            for l in 1:(p + 1)
                (l==i || l==j) || (v *= (x-nodes[l])/(nodes[i]-nodes[l]))
            end
            D[i] += v
        end
    end
    return N, D
end
function _spheroid_hp_grid(edges, p)
    return vcat(
        [collect(range(edges[e], edges[e + 1]; length = p+1))[1:(end - 1)]
         for e in 1:(length(edges) - 1)]...,
        last(edges))
end
function _spheroid_hp_profile(t, a, b)
    st, ct = sincos(t)
    r = 1 / sqrt(ct^2 / a^2 + st^2 / b^2)
    return r, r^3 * st * ct * (1 / a^2 - 1 / b^2)
end
function _spheroid_hp_geometry(
        boundary, a, b, R, nr, nt, p; condensed = true, half = false, graded = true,
        profile = t -> _spheroid_hp_profile(t, a, b), angular_edges = nothing)
    fluid=boundary isa FluidFilled
    g=fluid ? boundary.density_contrast : 1.0
    h=fluid ? boundary.soundspeed_contrast : 1.0
    nrn=nr*p+1
    ntn=nt*p+1
    base=fluid ? 1+(nrn-2)*ntn : 0
    extidx(i, j) = base+(i-1)*ntn+j
    intidx(i, j) = i==1 ? 1 : i==nrn ? base+j : 1+(i-2)*ntn+j
    ndof=base+nrn*ntn
    iface=[extidx(1, j) for j in 1:ntn]
    outer=[extidx(nrn, j) for j in 1:ntn]
    custom_edges = angular_edges !== nothing
    angular_edges = custom_edges ? angular_edges :
                    collect(range(0.0, half ? pi/2 : pi; length = nt+1))
    if graded && !custom_edges
        angular_edges=Base.map(angular_edges) do target
            left, right=0.0, half ? pi/2 : pi
            for iteration in 1:55
                mid=(left+right)/2
                u=(mid+atan(a*sin(mid), b*cos(mid)))/2
                if u<target
                    left=mid
                else
                    right=mid
                end
            end
            (left+right)/2
        end
        angular_edges[1]=0.0
        angular_edges[end]=half ? pi/2 : pi
    end
    theta=_spheroid_hp_grid(angular_edges, p)
    extedges=_graded_radial_s(nr)
    intedges=graded ? collect(range(0.0, 1.0; length = nr+1)) :
             _graded_radial_s_interior(nr)
    q, w=_spheroid_hp_rule(p+2)
    Ns=[_spheroid_hp_basis(p, x) for x in q]
    matrices = NTuple{3, SparseMatrixCSC{Float64, Int}}[]
    cells = _SpheroidFEMElement[]
    lb=[(i-1)*(p+1)+j for i in 1:(p + 1)
        for j in 1:(p + 1) if i in (1, p+1) || j in (1, p+1)]
    li=setdiff(1:((p + 1) ^ 2), lb)
    for inside in (fluid ? (true, false) : (false,))
        edges=inside ? intedges : extedges
        gidx=inside ? intidx : extidx
        weight=inside ? 1/g : 1.0
        massweight=inside ? 1/h^2 : 1.0
        Is=Int[]
        Js=Int[]
        Ss=Float64[]
        As=Float64[]
        Ms=Float64[]
        nloc=(p+1)^2
        if !condensed
            for v in (Is, Js, Ss, As, Ms)
                sizehint!(v, nr*nt*nloc^2)
            end
        end
        S=zeros(nloc, nloc)
        A=similar(S)
        M=similar(S)
        N=zeros(nloc)
        ds=similar(N)
        dt=similar(N)
        drho=similar(N)
        dz=similar(N)
        for ej in 1:nt, ei in 1:nr

            fill!(S, 0)
            fill!(A, 0)
            fill!(M, 0)
            ids=[gidx((ei-1)*p+i, (ej-1)*p+j) for i in 1:(p + 1) for j in 1:(p + 1)]
            hs=edges[ei + 1]-edges[ei]
            ht=angular_edges[ej + 1]-angular_edges[ej]
            for jt in eachindex(q), is in eachindex(q)

                s=(edges[ei]+edges[ei + 1])/2+hs*q[is]/2
                t=(angular_edges[ej]+angular_edges[ej + 1])/2+ht*q[jt]/2
                st, ct=sincos(t)
                ri, dri = profile(t)
                r=inside ? s*ri : ri+s*(R-ri)
                rs=inside ? ri : R-ri
                rt=inside ? s*dri : (1-s)*dri
                rho=r*st
                rho_s=rs*st
                z_s=rs*ct
                rho_t=rt*st+r*ct
                z_t=rt*ct-r*st
                detj=-rs*r
                jac=weight*w[is]*w[jt]*abs(detj)*hs*ht/4
                ns, dns=Ns[is]
                ntv, dnt=Ns[jt]
                for i in 1:(p + 1), j in 1:(p + 1)

                    l=(i-1)*(p+1)+j
                    N[l]=ns[i]*ntv[j]
                    ds[l]=2/hs*dns[i]*ntv[j]
                    dt[l]=2/ht*ns[i]*dnt[j]
                    drho[l]=(z_t*ds[l]-z_s*dt[l])/detj
                    dz[l]=(-rho_t*ds[l]+rho_s*dt[l])/detj
                end
                for j in 1:nloc, i in 1:nloc

                    S[i, j]+=jac*rho*(drho[i]*drho[j]+dz[i]*dz[j])
                    A[i, j]+=jac/rho*N[i]*N[j]
                    M[i, j]+=jac*rho*massweight*N[i]*N[j]
                end
            end
            if condensed
                push!(cells, _SpheroidFEMElement(ids, copy(S), copy(A), copy(M), inside))
            else
                for j in 1:nloc, i in 1:nloc

                    push!(Is, ids[i])
                    push!(Js, ids[j])
                    push!(Ss, S[i, j])
                    push!(As, A[i, j])
                    push!(Ms, M[i, j])
                end
            end
        end
        condensed || push!(matrices,
            (sparse(Is, Js, Ss, ndof, ndof), sparse(Is, Js, As, ndof, ndof),
                sparse(Is, Js, Ms, ndof, ndof)))
    end
    axis=half ? (1,) : (1, ntn)
    fixed=[extidx(i, j) for i in 1:nrn for j in axis]
    fluid && append!(fixed, [intidx(i, j) for i in 1:nrn for j in axis])
    if condensed
        kept=sort!(unique(vcat([cell.ids[lb] for cell in cells]...)))
        map=zeros(Int, ndof)
        map[kept]=1:length(kept)
        iface=map[iface]
        outer=map[outer]
        fixed=map[fixed]
        return (; S = nothing, A = nothing, M = nothing,
            lift = nothing, cells, lb, li, map, kept,
            iface, outer, fixed, ndof = length(kept), theta, angular_edges,
            extedges, intedges, nr, nt, p, a, b, R, fluid, condensed, half, profile)
    end
    S, A, M=last(matrices)
    lift=if fluid
        Si, Ai, Mi=first(matrices)
        S+=Si
        A+=Ai
        M+=Mi
        (Si[:, iface], Ai[:, iface], Mi[:, iface])
    else
        nothing
    end
    return (; S, A, M, lift, iface, outer, fixed, ndof, theta, angular_edges,
        extedges, intedges, nr, nt, p, a, b, R, fluid, condensed, half, profile)
end
function _spheroid_hp_dtn(mesh, m, lmax)
    (; nt, p, theta, angular_edges)=mesh
    nl=lmax-m+1
    Q=zeros(nl, length(theta))
    vals=zeros(nl)
    for e in 1:nt
        width=angular_edges[e + 1]-angular_edges[e]
        # Resolve polar oscillations on each graded element. The rule depends
        # on its width, so reflection uses the same quadrature as the full mesh.
        q, w=_spheroid_hp_rule(max(p+2, ceil(Int, (lmax+1)*width/pi)+p+1))
        for i in eachindex(q)
            t=(angular_edges[e]+angular_edges[e + 1])/2+(angular_edges[e + 1]-angular_edges[e])*q[i]/2
            N, _=_spheroid_hp_basis(p, q[i])
            _meridian_legendre_values!(vals, m, cos(t))
            jac=w[i]*(angular_edges[e + 1]-angular_edges[e])/2*sin(t)
            for j in 1:(p + 1), l in 1:nl

                Q[l, (e - 1) * p + j]+=jac*N[j]*vals[l]
            end
        end
    end
    mesh.half && (Q[2:2:end, :].=0)
    return Q
end
function _spheroid_hp_mode(mesh, boundary, k, m, beta = pi/2;
        l_max = max(_default_mode_count(k*mesh.R), m), retain_fields = false,
        solve_reports = nothing, dtn_form::Symbol = :assembled, mode_workers::Integer = 1)
    (; S, A, M, iface, outer, theta, p, a, b, R, ndof, angular_edges, nt)=mesh
    lmax=l_max
    hs, hsd=_spherical_hankel_cache(lmax, k*R)
    Q=_spheroid_hp_dtn(mesh, m, lmax)
    d=[k*hsd[l + 1]/hs[l + 1] for l in m:lmax]
    rhs=zeros(ComplexF64, ndof)
    incident=[_p_inc_mode(m, k, beta, first(mesh.profile(t))*sin(t),
                  first(mesh.profile(t))*cos(t)) for t in theta]
    K=if mesh.condensed
        lb, li=mesh.lb, mesh.li
        Is=Int[]
        Js=Int[]
        Vs=Float64[]
        for v in (Is, Js, Vs)
            sizehint!(v, length(mesh.cells)*length(lb)^2)
        end
        interface_position=zeros(Int, ndof)
        interface_position[iface]=1:length(iface)
        for cell in mesh.cells
            local_matrix=cell.S+m^2*cell.A-k^2*cell.M
            schur=local_matrix[lb, lb]-local_matrix[lb, li]*(local_matrix[li, li]\local_matrix[li, lb])
            ids=mesh.map[cell.ids[lb]]
            if cell.inside
                # The shared interface unknown is scattered pressure. Interior
                # total pressure adds the incident trace, and its known lift
                # enters the condensed boundary load through the same Schur map.
                lift=ComplexF64[interface_position[id]>0 ?
                                incident[interface_position[id]] : 0 for id in ids]
                load=schur*lift
                for j in eachindex(ids)
                    rhs[ids[j]]-=load[j]
                end
            end
            for j in eachindex(ids), i in eachindex(ids)

                push!(Is, ids[i])
                push!(Js, ids[j])
                push!(Vs, schur[i, j])
            end
        end
        ComplexF64.(sparse(Is, Js, Vs, ndof, ndof))
    else
        ComplexF64.(S+m^2*A-k^2*M)
    end
    if dtn_form === :assembled
        dtn=(mesh.half ? 2 : 1)*R^2*(Q'*(d .* Q))
        K+=sparse(repeat(outer, length(outer)),
            repeat(outer; inner = length(outer)), vec(-dtn), ndof, ndof)
    end
    if mesh.fluid && !mesh.condensed
        Ls, La, Lm=mesh.lift
        rhs .-= (Ls+m^2*La-k^2*Lm)*incident
    end
    if !(boundary isa PressureRelease)
        q, w=_spheroid_hp_rule(p+3)
        for e in 1:nt, i in eachindex(q)

            t=(angular_edges[e]+angular_edges[e + 1])/2+(angular_edges[e + 1]-angular_edges[e])*q[i]/2
            N, _=_spheroid_hp_basis(p, q[i])
            st, ct=sincos(t)
            ri, dri = mesh.profile(t)
            rt=dri*st+ri*ct
            zt=dri*ct-ri*st
            speed=hypot(rt, zt)
            value=_dpdn_inc_mode(m, k, beta, ri*st, ri*ct, -zt/speed, rt/speed)*ri*st*speed*w[i]*(angular_edges[e + 1]-angular_edges[e])/2
            for j in 1:(p + 1)
                rhs[iface[(e - 1) * p + j]]+=N[j]*value
            end
        end
    end
    fixed=m>0 ? copy(mesh.fixed) : Int[]
    values=zeros(ComplexF64, length(fixed))
    if boundary isa PressureRelease
        append!(fixed, iface)
        append!(values, -incident)
    end
    K, rhs = _meridian_fem_boundary_system(K, rhs, outer, Q, d, R, dtn_form;
        symmetry_factor = mesh.half ? 2 : 1)
    _meridian_fem_dirichlet!(K, rhs, fixed, values)
    solved=_dtn_fem_solve(K, rhs, m, lmax, k*R; solve_reports,
        n_r = mesh.nr, n_theta = mesh.half ? 2mesh.nt : mesh.nt, order = p,
        condensed = mesh.condensed, equatorial_symmetry = mesh.half, dtn_form, mode_workers)
    field = dtn_form === :assembled ? solved : solved[1:ndof]
    Bl=(mesh.half ? 2 : 1)*(Q*field[outer]) ./ hs[(m + 1):end]
    retained=retain_fields ? _spheroid_hp_recover(mesh, field, incident, k, m) : nothing
    return Bl, retained
end
function _spheroid_hp_mode_amplitude(Bl, k, m, theta, phi)
    # Associated Legendre functions use nonnegative cylindrical radius.
    # A negative sine of the polar angle reverses the azimuthal direction.
    observation_phi=sin(theta)<0 ? phi+pi : phi
    P=_meridian_legendre_values!(zeros(length(Bl)), m, cos(theta))
    return cos(m*observation_phi)*sum(Bl[j]*(-im)^(m+j)*P[j] for j in eachindex(Bl))/k
end
function _spheroid_hp_amplitude(outgoing, k, theta = pi/2, phi = pi)
    return sum(_spheroid_hp_mode_amplitude(Bl, k, index-1, theta, phi)
    for (index, Bl) in enumerate(outgoing))
end

function _spheroid_hp_recover(mesh, solution, incident, k, m)
    if mesh.condensed
        full=zeros(ComplexF64, length(mesh.map))
        full[mesh.kept]=solution
        lifting=zeros(ComplexF64, length(full))
        lifting[mesh.kept[mesh.iface]]=incident
        for cell in mesh.cells
            local_matrix=cell.S+m^2*cell.A-k^2*cell.M
            boundary_values=full[cell.ids[mesh.lb]]
            cell.inside && (boundary_values .+= lifting[cell.ids[mesh.lb]])
            full[cell.ids[mesh.li]]=-(local_matrix[mesh.li, mesh.li] \
                                      (local_matrix[mesh.li, mesh.lb]*boundary_values))
        end
    else
        full=solution
    end
    nrn=mesh.nr*mesh.p+1
    ntn=length(mesh.theta)
    base=mesh.fluid ? 1+(nrn-2)*ntn : 0
    exterior=ComplexF64[full[base + (i - 1) * ntn + j] for i in 1:nrn, j in 1:ntn]
    interior=if mesh.fluid
        ComplexF64[i==nrn ? full[base + j]+incident[j] :
                   full[i==1 ? 1 : 1+(i-2)*ntn+j] for i in 1:nrn, j in 1:ntn]
    else
        nothing
    end
    return _SpheroidHighOrderFEMModeField(mesh.theta,
        _spheroid_hp_grid(mesh.extedges, mesh.p), _spheroid_hp_grid(mesh.intedges, mesh.p),
        Float64(mesh.R), exterior, interior)
end

function _spheroid_hp_modes(mesh, boundary, k, incidence_angle, m_max;
        l_max, mode_threads, retain_fields, solve_reports, modes = 0:m_max,
        dtn_form::Symbol = :assembled,
        outgoing = Vector{Vector{ComplexF64}}(undef, m_max+1),
        fields = Vector{_SpheroidHighOrderFEMModeField}(undef, retain_fields ? m_max+1 : 0))
    # Fine adaptive meshes can otherwise hold several large sparse LU factors
    # at once. mode_threads is an upper bound, including on refined meshes.
    memory_workers=max(1, floor(Int, 4*(50_000/mesh.ndof)^1.5))
    workers=min(mode_threads, Threads.nthreads(), length(modes), memory_workers)
    function solve_mode(m)
        coefficients, field=_spheroid_hp_mode(mesh, boundary, k, m, incidence_angle;
            l_max, retain_fields, solve_reports, dtn_form, mode_workers = workers)
        outgoing[m + 1]=coefficients
        retain_fields && (fields[m + 1]=field)
        return nothing
    end
    if workers==1
        for m in modes
            solve_mode(m)
        end
    else
        @sync for worker in 1:workers
            Threads.@spawn for index in worker:workers:length(modes)
                solve_mode(modes[index])
            end
        end
    end
    return outgoing, fields
end

function _spheroid_hp_ts_change(contributions, changes)
    amplitude=sum(contributions)
    scale=abs(amplitude)
    if any(!isfinite, changes)
        return (; change_db = Inf, weights = fill(Inf, length(changes)), scale)
    end
    if iszero(scale)
        weights=abs.(changes)
        return (; change_db = all(iszero, weights) ? 0.0 : Inf, weights, scale)
    end
    rotated=changes .* (conj(amplitude)/scale)
    radial=abs.(real.(rotated))
    transverse=abs.(imag.(rotated))
    a=sum(radial)/scale
    b=sum(transverse)/scale
    # The rectangle enclosing independently signed modal changes bounds the
    # magnitude between |f|-sum(radial) and hypot(|f|+sum(radial),sum(transverse)).
    # Transverse (phase) changes enter TS quadratically rather than linearly.
    lower=a<1 ? -20/log(10)*log1p(-a) : Inf
    upper=10/log(10)*log1p(2a+a^2+b^2)
    return (; change_db = max(lower, upper), weights = radial .+ b .* transverse, scale)
end

function _meridian_hp_geometry(body::Spheroid, boundary, R, nr, nt, order; half)
    return _spheroid_hp_geometry(boundary, body.a, body.b, R, nr, nt, order;
        condensed = true, half, graded = true)
end

_meridian_hp_dimensions(body::Spheroid) = (body.a, body.b, max(body.a, body.b))
_meridian_hp_radius(body::Spheroid, theta) = _spheroid_r_inner(theta, body.a, body.b)
function _meridian_hp_half(body::Spheroid, nt, beta)
    iseven(nt) && abs(float(beta)) == oftype(float(beta), pi) / 2
end

function _spheroid_hp_adaptive_modes(body, boundary, k, nr, nt, order, half;
        R, incidence_angle, m_max, l_max, mode_threads, retain_fields,
        target_tol, max_refinements, solve_reports, dtn_form::Symbol = :assembled)
    n_modes=m_max+1
    nrs=[nr]
    nts=[nt]
    angular_ratio=nt/nr
    for level in 1:max_refinements
        next_nr=ceil(Int, 4last(nrs)/3)
        push!(nrs, next_nr)
        push!(nts, half ? 2ceil(Int, angular_ratio*next_nr/2) :
                   ceil(Int, angular_ratio*next_nr))
    end
    outgoing=Vector{Vector{ComplexF64}}(undef, n_modes)
    fields=Vector{_SpheroidHighOrderFEMModeField}(undef, retain_fields ? n_modes : 0)
    contributions=zeros(ComplexF64, n_modes)
    changes=fill(ComplexF64(Inf), n_modes)
    levels=fill(-1, n_modes)
    threshold_fraction=-expm1(-target_tol*log(10)/20)
    cached_level=-1
    cached_mesh=nothing
    while true
        estimate=_spheroid_hp_ts_change(contributions, changes)
        estimate.change_db<=target_tol && break
        threshold=threshold_fraction*estimate.scale
        available=findall(level->level<max_refinements, levels)
        isempty(available) && break
        selected=filter(index->estimate.weights[index]>threshold/n_modes, available)
        # A mode at its refinement limit can consume most of the error budget.
        # Continue improving other modes if that can still meet the total gate.
        if isempty(selected)
            exhausted_changes=[levels[index]==max_refinements ? changes[index] :
                               zero(eltype(changes)) for index in eachindex(changes)]
            exhausted=_spheroid_hp_ts_change(contributions, exhausted_changes)
            exhausted.change_db>target_tol && break
            index=available[argmax(estimate.weights[available])]
            iszero(estimate.weights[index]) && break
            selected=[index]
        end
        for next_level in sort!(unique(levels[selected] .+ 1))
            indices=filter(index->levels[index]+1==next_level, selected)
            if cached_level!=next_level
                cached_mesh=_meridian_hp_geometry(body, boundary, R,
                    nrs[next_level + 1], half ? nts[next_level + 1]÷2 : nts[next_level + 1], order;
                    half)
                cached_level=next_level
            end
            _spheroid_hp_modes(cached_mesh, boundary, k, incidence_angle, m_max;
                l_max, mode_threads, retain_fields, solve_reports, dtn_form,
                modes = indices .- 1, outgoing, fields)
            for index in indices
                value=_spheroid_hp_mode_amplitude(
                    outgoing[index], k, index-1, pi-incidence_angle, pi)
                changes[index]=levels[index]<0 ? ComplexF64(Inf) :
                               value-contributions[index]
                contributions[index]=value
                levels[index]=next_level
            end
        end
    end
    change_db=_spheroid_hp_ts_change(contributions, changes).change_db
    amplitude_change=sum(abs, changes)
    converged=change_db<=target_tol
    mode_n_r=nrs[levels .+ 1]
    mode_n_theta=nts[levels .+ 1]
    _record_refinement!(solve_reports, converged, change_db, target_tol, maximum(mode_n_r))
    solve_reports.refinement=merge(solve_reports.refinement,
        (; estimator = :modal_ts_mesh_change, amplitude_change,
            mode_n_r, mode_n_theta, mode_refinements = levels))
    converged ||
        @warn "meridian FEM did not reach target_tol=$target_tol dB within $max_refinements refinements per mode, returning the latest solves"
    return outgoing, fields, maximum(mode_n_r), maximum(mode_n_theta)
end

function _spheroid_high_order_fem(body, boundary, k;
        R, incidence_angle, m_max, mode_threads, retain_fields, order,
        n_r::Union{Nothing, Integer} = nothing, n_theta::Union{Nothing, Integer} = nothing,
        l_max::Integer = max(_default_mode_count(k*R), m_max),
        adaptive::Bool = false, target_tol::Real = 0.01, max_refinements::Integer = 4,
        dtn_form::Symbol = :assembled)
    isfinite(k) && k>0 || throw(ArgumentError("wavenumber must be finite and positive"))
    m_max>=0 || throw(ArgumentError("m_max must be nonnegative"))
    l_max>=m_max || throw(ArgumentError("l_max must be at least m_max"))
    mode_threads>=1 || throw(ArgumentError("mode_threads must be positive"))
    dtn_form in (:assembled, :augmented) ||
        throw(ArgumentError("dtn_form must be :assembled or :augmented"))
    isfinite(incidence_angle) || throw(ArgumentError("incidence_angle must be finite"))
    isfinite(R) || throw(ArgumentError("R must be finite"))
    a, b, extent = _meridian_hp_dimensions(body)
    R>extent || throw(ArgumentError("R must enclose the body"))
    isfinite(target_tol) && target_tol>0 ||
        throw(ArgumentError("target_tol must be finite and positive"))
    max_refinements>=1 || throw(ArgumentError("max_refinements must be positive"))
    internal_wave=if boundary isa FluidFilled
        isfinite(boundary.density_contrast) && boundary.density_contrast>0 &&
        isfinite(boundary.soundspeed_contrast) && boundary.soundspeed_contrast>0 ||
            throw(ArgumentError("fluid density and sound-speed contrasts must be finite and positive"))
        k*extent/boundary.soundspeed_contrast
    else
        0.0
    end
    exterior_wave=2k*(R-min(a, b))
    nr=n_r===nothing ? max(4, ceil(Int, max(internal_wave, exterior_wave)/(1.8order))) :
       Int(n_r)
    nt=n_theta===nothing ? 4nr : Int(n_theta)
    nr>=1 || throw(ArgumentError("n_r must be positive"))
    minimum_theta = body isa Cylinder ? 3 : 2
    nt>=minimum_theta || throw(ArgumentError("n_theta must be at least $minimum_theta"))
    half=_meridian_hp_half(body, nt, incidence_angle)
    reports=_SolveReports()
    if adaptive
        outgoing, fields, nr, nt=_spheroid_hp_adaptive_modes(
            body, boundary, k, nr, nt, order, half;
            R, incidence_angle, m_max, l_max, mode_threads,
            retain_fields, target_tol, max_refinements, dtn_form,
            solve_reports = reports)
    else
        mesh=_meridian_hp_geometry(body, boundary, R, nr, half ? nt÷2 : nt, order; half)
        outgoing, fields=_spheroid_hp_modes(mesh, boundary, k, incidence_angle, m_max;
            l_max, mode_threads, retain_fields, solve_reports = reports, dtn_form)
    end
    report=_summarize_solves(reports; method = :meridian,
        solver_options = (; R, incidence_angle, m_max, mode_threads, retain_fields, order,
            n_r = nr, n_theta = nt, l_max, adaptive, target_tol, max_refinements, dtn_form),
        equatorial_symmetry = half)
    data=_SpheroidHighOrderFEMData(
        outgoing, fields, Float64(incidence_angle), Int(order), report)
    return FEMSolution(body, boundary, k, :meridian, data)
end
