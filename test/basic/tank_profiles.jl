using AcousticScattering, Test, LinearAlgebra
using StaticArrays: SVector

@testset "Profiled tank geometry and reflection" begin
    AS=AcousticScattering
    envelope=Tank(0.4, (-0.4, 0.4); reflections = (bottom = 0, surface = 0))
    flat=ProfiledTank(envelope; radius = (phi, z)->0.4, bottom = (x, y)->-0.4)
    taper=ProfiledTank(envelope; radius = (phi, z)->0.3+0.1*(z+0.4)/0.8, bottom = (
        x, y)->-0.4)
    @test_throws ArgumentError ProfiledTank(Tank(0.4, (-0.4, 0.4)); radius = (p, z)->0.4, bottom = (
        x, y)->-0.4)
    @test_throws ArgumentError ProfiledTank(envelope; radius = (p, z)->0.5, bottom = (
        x, y)->-0.4)
    @test AS._inside_tank(taper, (0.35, 0.0, 0.3))
    @test !AS._inside_tank(taper, (0.38, 0.0, -0.3))
    quad=AS._tank_boundary_quadrature(taper, :sidewall, (96, 80))
    @test sum(quad.areas)≈pi*(0.3+0.4)*hypot(0.8, 0.1) rtol=1e-13
    @test all(n->isapprox(norm(n), 1.0; atol = 1e-14), quad.normals)
    @test maximum(abs(n[3]+0.125/sqrt(1+0.125^2)) for n in quad.normals)<1e-14
    floor=AS._tank_boundary_quadrature(taper, :bottom, (96, 80))
    @test sum(floor.areas)≈pi*0.3^2 rtol=1e-13
    @test all(==(SVector(0.0, 0.0, -1.0)), floor.normals)
    raised=ProfiledTank(envelope; radius = (phi, z)->0.4,
        bottom = (x, y)->-0.4+0.01*exp(-((x^2+y^2)/0.1^2)^2))
    @test !AS._inside_tank(raised, (0.0, 0.0, -0.395))
    @test AS._inside_tank(raised, (0.0, 0.0, -0.385))
    k=15.0
    a, b=SVector(-0.1, 0.02, 0.1), SVector(0.15, -0.03, 0.12)
    pointfield(y) = IncidentField(x->cis(k*norm(SVector(x)-y))/(4pi*norm(SVector(x)-y)),
        x->begin
            v=SVector(x)-y
            r=norm(v)
            cis(k*r)/(4pi*r)*(im*k-1/r)*v/r
        end)
    original=sidewall_scattering(envelope, k, pointfield(a); quadrature = (96, 80))
    profiled=sidewall_scattering(flat, k, pointfield(a); quadrature = (96, 80))
    @test pressure(original, b)≈pressure(profiled, b) rtol=1e-13
    forward=sidewall_scattering(taper, k, pointfield(a); quadrature = (96, 80))
    reverse=sidewall_scattering(taper, k, pointfield(b); quadrature = (96, 80))
    @test pressure(forward, b)≈pressure(reverse, a) rtol=1e-12
    forward_floor=boundary_scattering(
        raised, k, pointfield(a); boundary = :bottom, quadrature = (128, 120))
    reverse_floor=boundary_scattering(
        raised, k, pointfield(b); boundary = :bottom, quadrature = (128, 120))
    @test pressure(forward_floor, b)≈pressure(reverse_floor, a) rtol=1e-12
    # Independent finite planar-disk result, source and receiver on its axis.
    source=SVector(0.0, 0.0, 0.1)
    disk=ProfiledTank(Tank(0.4, (0.0, 1.0); reflections = (bottom = 0, surface = 0));
        radius = (phi, z)->0.4, bottom = (x, y)->0.0)
    reflection=boundary_scattering(
        disk, k, pointfield(source); boundary = :bottom, quadrature = (96, 80))
    h=0.1
    edge=hypot(h, 0.4)
    exact=cis(2k*h)/(8pi*h)-h*cis(2k*edge)/(8pi*edge^2)
    @test pressure(reflection, source)≈exact rtol=1e-11
    @test_throws ArgumentError pressure(reflection, (0.0, 0.0, 0.0))
    @test_throws ArgumentError boundary_scattering(disk, k, pointfield(source); boundary = :unknown)
    tx=Transducer((0.0, 0.0, 0.2), (0.0, 0.0, -1.0), 0.02)
    illuminated=TankTransducer(tx, raised)
    @test pressure(illuminated, k, (0.1, 0.0, 0.0))≈pressure(tx, k, (0.1, 0.0, 0.0))
    sample=tank_field(illuminated, k, raised; resolution = (21, 81))
    @test any(==(-2), sample.regions)
    @test all(isfinite, sample.pressure[sample.regions .== 0])
end
