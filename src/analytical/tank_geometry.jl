"""
    ProfiledTank(envelope::Tank; radius, bottom)

Water volume with a radial sidewall `radius(phi,z)` and bottom height `bottom(x,y)`
in meters, within a cylindrical `envelope`. Azimuth is measured about its center.
Callbacks must support ForwardDiff, be finite and smooth, and be safe for
concurrent calls. Radius must remain positive and within the envelope. The bottom
must join the sidewall at the envelope's minimum z, and stay below its surface.
User-supplied profiles are sampled for basic validation, not geometrically certified.

The envelope must have zero bottom image coefficient: use
[`boundary_scattering`](@ref) for the actual bottom instead of double-counting a
planar image. Its surface image remains available through `TankTransducer`.
Geometry affects containment, map masks and reflection positions/normals/areas;
it does not by itself solve a bounded-tank wave problem.
"""
struct ProfiledTank{R, B} <: AbstractTank
    bounds::NamedTuple{(:x, :y, :z), NTuple{3, NTuple{2, Float64}}}
    walls::Vector{Wall}
    radius::Float64
    radius_at::R
    bottom_at::B
end

function ProfiledTank(envelope::Tank; radius, bottom)
    envelope.radius!==nothing || throw(ArgumentError("a cylindrical envelope is required"))
    envelope.walls[1].reflection isa Number && iszero(envelope.walls[1].reflection) ||
        throw(ArgumentError("disable the planar bottom image before specifying a profiled bottom"))
    cx, cy=sum(envelope.bounds.x)/2, sum(envelope.bounds.y)/2
    lo, hi=envelope.bounds.z
    for z in range(lo, hi; length = 17), phi in range(0, 2pi; length = 65)

        r=radius(phi, z)
        r isa Real && isfinite(r) && 0<r<=envelope.radius*(1+64eps()) ||
            throw(ArgumentError("profile radii must be positive and inside the envelope"))
    end
    for phi in range(0, 2pi; length = 65), u in range(0, 1; length = 17)

        r=u*radius(phi, lo)
        b=bottom(cx+r*cos(phi), cy+r*sin(phi))
        b isa Real && isfinite(b) && lo<=b<hi ||
            throw(ArgumentError("bottom must lie between the envelope floor and surface"))
        u==1 && abs(b-lo)>1e-10 &&
            throw(ArgumentError("bottom must join the sidewall at its base"))
    end
    ProfiledTank(envelope.bounds, copy(envelope.walls), envelope.radius, radius, bottom)
end

function _inside_tank(tank::ProfiledTank, point)
    all(d->tank.bounds[d][1]<=point[d]<=tank.bounds[d][2], 1:3) || return false
    x, y=point[1]-sum(tank.bounds.x)/2, point[2]-sum(tank.bounds.y)/2
    hypot(x, y)<=tank.radius_at(atan(y, x), point[3]) &&
        point[3]>=tank.bottom_at(point[1], point[2])
end

function TankTransducer(source::Transducer, tank::ProfiledTank; reference_point = nothing)
    _inside_tank(tank, source.position) ||
        throw(ArgumentError("transducer center is outside the water"))
    rotation=_axis_rotation(collect(source.axis))
    for phi in range(0, 2pi; length = 257)
        point=SVector(source.position)+source.radius*(cos(phi)*SVector{3}(rotation[:, 1]) +
                                                      sin(phi)*SVector{3}(rotation[:, 2]))
        _inside_tank(tank, point) ||
            throw(ArgumentError("transducer aperture is outside the profiled water volume"))
    end
    TankTransducer(source, tank.walls; reference_point)
end

function _field_excluded(tank::ProfiledTank, x)
    cx, cy=sum(tank.bounds.x)/2, sum(tank.bounds.y)/2
    r=hypot(x[1]-cx, x[2]-cy)
    tolerance=64eps(max(tank.radius, norm(x)))
    onside=abs(r-tank.radius_at(atan(x[2]-cy, x[1]-cx), x[3]))<=tolerance &&
           tank.bounds.z[1]<=x[3]<=tank.bounds.z[2]
    onbottom=r<=tank.radius && abs(x[3]-tank.bottom_at(x[1], x[2]))<=tolerance
    onside || onbottom
end

function _tank_boundary_quadrature(tank::ProfiledTank, boundary, counts)
    boundary in (:sidewall, :bottom) ||
        throw(ArgumentError("boundary must be :sidewall or :bottom"))
    length(counts)==2 && all(n->n isa Integer && n>0, counts) ||
        throw(ArgumentError("quadrature needs two positive integer orders"))
    nphi, nv=counts
    lo, hi=tank.bounds.z
    nodes, weights=boundary===:sidewall ? gauss(nv, lo, hi) : gauss(nv, 0.0, 1.0)
    cx, cy=sum(tank.bounds.x)/2, sum(tank.bounds.y)/2
    points=Vector{SVector{3, Float64}}(undef, nphi*nv)
    normals=similar(points)
    areas=Vector{Float64}(undef, length(points))
    Threads.@threads for i in 1:nv
        for j in 1:nphi
            index=(i-1)*nphi+j
            phi=2pi*(j-0.5)/nphi
            sn, cs=sincos(phi)
            if boundary===:sidewall
                z=nodes[i]
                r=tank.radius_at(phi, z)
                rp, rz=ForwardDiff.gradient(v->tank.radius_at(v[1], v[2]), SVector(phi, z))
                cross=SVector(r*cs+rp*sn, r*sn-rp*cs, -r*rz)
                jacobian=norm(cross)
                points[index]=SVector(cx+r*cs, cy+r*sn, z)
                normals[index]=cross/jacobian
                areas[index]=weights[i]*2pi/nphi*jacobian
            else
                rim=tank.radius_at(phi, lo)
                r=nodes[i]*rim
                x, y=cx+r*cs, cy+r*sn
                z=tank.bottom_at(x, y)
                bx, by=ForwardDiff.gradient(v->tank.bottom_at(v[1], v[2]), SVector(x, y))
                cross=SVector(bx, by, -1.0)
                jacobian=norm(cross)
                points[index]=SVector(x, y, z)
                normals[index]=cross/jacobian
                areas[index]=weights[i]*2pi/nphi*nodes[i]*rim^2*jacobian
            end
        end
    end
    all(isfinite, areas) && all(>(0), areas) && all(p->all(isfinite, p), normals) ||
        throw(ArgumentError("profile quadrature has invalid normals or areas"))
    (; points, normals, areas)
end
