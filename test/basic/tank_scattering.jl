using AcousticScattering, Test, LinearAlgebra
using StaticArrays: SVector

@testset "Thin filament and curved reflection" begin
    AS=AcousticScattering
    k=50.0
    plane=IncidentField(x->cis(k*x[1]), x->SVector(im*k*cis(k*x[1]), 0im, 0im))
    vertices=[(0.0, 0.0, -0.015), (0.0, 0.0, 0.015)]
    line=ThinFilament(vertices, 0.0001; density_ratio = 1.14, sound_speed_ratio = 1.7)
    f=filament_scattering(line, k, plane)
    same=ThinFilament(vertices, 0.0001; density_ratio = 1, sound_speed_ratio = 1)
    @test pressure(filament_scattering(same, k, plane), (0.2, 0.0, 0.0))==0
    @test_throws ArgumentError ThinFilament(vertices, -1; density_ratio = 1, sound_speed_ratio = 1)
    @test_throws ArgumentError filament_scattering(line, 4000.0, plane)
    @test_throws ArgumentError pressure(f, (0.0, 0.0, 0.0))
    @test_throws ArgumentError sidewall_scattering(Tank((-1, 1), (-1, 1), (-1, 1)), k, plane)
    x=SVector(0.13, 0.07, 0.08)
    delta=1e-6
    finite_difference=SVector{3}([(pressure(f, x+delta*SVector{3}(ntuple(i->Float64(i==j), 3))) -
                                   pressure(f, x-delta*SVector{3}(ntuple(i->Float64(i==j), 3))))/(2delta)
                                  for j in 1:3])
    @test IncidentField(f).gradient(x)≈finite_difference rtol=1e-7
    thick=ThinFilament(vertices, 0.0002; density_ratio = 1.14, sound_speed_ratio = 1.7)
    @test pressure(filament_scattering(thick, k, plane), x)≈4pressure(f, x) rtol=1e-13
    refined=filament_scattering(line, k, plane; panels_per_wavelength = 8, order = 6)
    @test pressure(refined, x)≈pressure(f, x) rtol=1e-7
    far=pressure(f, (-100.0, 0.0, 0.0))*100cis(-k*100)
    expected=k^2*line.radius^2*0.03/4*(1/(1.14*1.7^2)-1-2*0.14/2.14)
    @test far≈expected rtol=1e-4
    cylinder=AS.form_function(FluidFilled(1.14, 1.7), k, line.radius, 0.03; m_max = 4)
    @test far≈cylinder rtol=1e-3

    # Exact finite-disk Kirchhoff integral for coincident on-axis source/receiver.
    # As disk radius -> infinity, this tends to the planar image G(2h).
    kp, h, radius=4.0, 1.0, 8.0
    rho, w=AS.gauss(180, 0.0, radius)
    points=SVector{3, Float64}[]
    mono=ComplexF64[]
    dip=SVector{3, ComplexF64}[]
    for (r, weight) in zip(rho, w), j in 1:32

        phi=2pi*j/32
        y=SVector(r*cos(phi), r*sin(phi), 0.0)
        distance=norm(y-SVector(0.0, 0.0, h))
        G=cis(kp*distance)/(4pi*distance)
        dn=(im*kp-1/distance)*h/distance*G
        area=2pi*r*weight/32
        push!(points, y)
        push!(mono, -area*dn)
        push!(dip, area*G*SVector(0.0, 0.0, 1.0))
    end
    planar=ScatteringField(kp, points, mono, dip, nothing)
    rmax=hypot(h, radius)
    exact=cis(2kp*h)/(8pi*h)-h*cis(2kp*rmax)/(8pi*rmax^2)
    @test pressure(planar, (0.0, 0.0, h))≈exact rtol=1e-11

    tank=Tank(0.4, (-0.4, 0.4))
    kw=15.0
    a=SVector(-0.12, 0.03, 0.02)
    b=SVector(0.15, -0.06, -0.03)
    pointfield(y) = IncidentField(x->cis(kw*norm(SVector(x)-y))/(4pi*norm(SVector(x)-y)),
        x->begin
            v=SVector(x)-y
            r=norm(v)
            cis(kw*r)/(4pi*r)*(im*kw-1/r)*v/r
        end)
    wa=sidewall_scattering(tank, kw, pointfield(a); quadrature = (96, 80))
    wb=sidewall_scattering(tank, kw, pointfield(b); quadrature = (96, 80))
    @test pressure(wa, b)≈pressure(wb, a) rtol=1e-12
    wr=sidewall_scattering(tank, kw, pointfield(a); quadrature = (128, 100))
    @test pressure(wr, b)≈pressure(wa, b) rtol=1e-10
    @test_throws ArgumentError pressure(wa, (0.4, 0.0, 0.0))
    @test pressure(
        sidewall_scattering(tank, kw, pointfield(a); reflection = 0,
            quadrature = (16, 12)), b)==0
    # Independent receive-aperture integration fixes the dipole sign/normalization.
    rx=Transducer((0.2, 0.1, 0.12), (0.0, 0.0, -1.0), 0.012)
    received=received_pressure(f, rx)
    unodes, uweights=AS.gauss(8, 0.0, 1.0)
    direct=sum(w/48*pressure(f,
                   (rx.position[1]+rx.radius*sqrt(u)*cos(2pi*j/48),
                       rx.position[2]+rx.radius*sqrt(u)*sin(2pi*j/48), rx.position[3]))
    for (u, w) in zip(unodes, uweights), j in 1:48)
    @test received≈direct rtol=1e-8
    tx=Transducer((-0.2, 0.0, 0.1), (0.0, 0.0, -1.0), 0.015)
    dressed=ScatteringTransducer(tx, f)
    @test pressure(dressed, k, x)≈pressure(tx, k, x)+pressure(f, x)
    @test_throws ArgumentError pressure(dressed, k+1, x)
    coeff=(; density_ratio = 0.92, sound_speed_ratio = 1950/1500,
        backing_density_ratio = 0.0012, backing_sound_speed_ratio = 343/1500)
    @test fluid_wall_reflection(kw, 1.0; thickness = 0.0, coeff...) ≈
          (0.0012*343/1500-1)/(0.0012*343/1500+1)
    @test abs(fluid_wall_reflection(kw, 0.3; thickness = 0.003, coeff...))<=1
    critical=sqrt(1-1/(1950/1500)^2)
    @test isfinite(fluid_wall_reflection(kw, critical; thickness = 0.003, coeff...))
    @test fluid_wall_reflection(kw, 1.0; thickness = 0.1, density_ratio = 1.0,
        sound_speed_ratio = 1.0, backing_density_ratio = 1.0, backing_sound_speed_ratio = 1.0)≈0 atol=1e-14
    # Independent normal-incidence input-impedance solution checks layer phase.
    g1, h1, g2, h2, depth=1.2, 1.5, 0.8, 0.9, 0.037
    z1, z2=g1*h1, g2*h2
    zin=z1*(z2-im*z1*tan(kw*depth/h1))/(z1-im*z2*tan(kw*depth/h1))
    @test fluid_wall_reflection(kw, 1.0; thickness = depth, density_ratio = g1,
        sound_speed_ratio = h1, backing_density_ratio = g2, backing_sound_speed_ratio = h2) ≈
          (zin-1)/(zin+1) rtol=1e-13
    @test fluid_wall_reflection(kw, 1.0; thickness = pi*h1/(2kw), density_ratio = g1,
        sound_speed_ratio = h1, backing_density_ratio = g2, backing_sound_speed_ratio = h2) ≈
          (z1^2-z2)/(z1^2+z2) atol=1e-13
    reflection=(k, c)->fluid_wall_reflection(k, c; thickness = 0.003, coeff...)
    wc=sidewall_scattering(tank, kw, pointfield(a); quadrature = (96, 80),
        reflection, reference_point = (0.0, 0.0, 0.0))
    wd=sidewall_scattering(tank, kw, pointfield(b); quadrature = (96, 80),
        reflection, reference_point = (0.0, 0.0, 0.0))
    @test pressure(wc, b)≈pressure(wd, a) rtol=1e-12
end

@testset "Target environmental pressure and reception" begin
    AS=AcousticScattering
    k=15.0
    tank=Tank(0.4, (-0.4, 0.4); reflections = (bottom = 0, surface = 0))
    tx=Transducer((-0.2, 0.0, 0.1), (1.0, 0.0, 0.0), 0.015)
    rx=Transducer((0.2, 0.1, 0.12), (0.0, 0.0, -1.0), 0.012)
    line=ThinFilament([(0.0, 0.0, 0.05), (0.0, 0.0, 0.25)], 0.001;
        density_ratio = 1.14, sound_speed_ratio = 1.7)
    wall_tx=sidewall_scattering(tank, k, tx; reflection = 0.7, quadrature = (64, 48))
    wall_rx=sidewall_scattering(tank, k, rx; reflection = 0.7, quadrature = (64, 48))
    line_tx=filament_scattering(line, k, tx)
    line_rx=filament_scattering(line, k, rx)
    source=ScatteringTransducer(tx, wall_tx, line_tx)
    receiver=ScatteringTransducer(rx, wall_rx, line_rx)
    surface=mesh(; semiaxes = (0.03, 0.03, 0.03), resolution = 0.6, mesh_order = 3, qorder = 4)
    sol=bem(surface, Rigid(), k; transducer = source, compression = (method = :none,))
    outgoing_wall=sidewall_scattering(tank, k, sol; reflection = 0.7, quadrature = (64, 48))
    outgoing_line=filament_scattering(line, k, sol)
    received=received_pressure(sol, receiver)
    nodes, weights=AS.gauss(6, 0.0, 1.0)
    pts=[(rx.position[1]+rx.radius*sqrt(u)*cos(2pi*j/32),
             rx.position[2]+rx.radius*sqrt(u)*sin(2pi*j/32), rx.position[3])
         for (u, w) in zip(nodes, weights), j in 1:32]
    values=pressure(sol, pts; field = :scattered)+pressure(outgoing_wall, pts)+pressure(outgoing_line, pts)
    # Scattered-only sampling uses solved traces, not incident point evaluations.
    # This matters when the retained incident field integrates an entire wall.
    unavailable=IncidentField(x->error("unneeded incident sample"),
        x->error("unneeded incident gradient"))
    data=sol.data
    isolated=AS._FullBEMSurfaceData(data.quad, data.p_scat, data.dpdn_scat,
        data.incidence_angle, data.incidence_azimuth, data.diagnostics,
        data.single_layer_density, unavailable)
    stored=AS.BEMSolution(sol.body, sol.boundary, sol.k, sol.method, isolated)
    @test pressure(stored, pts; field = :scattered)≈pressure(sol, pts; field = :scattered)
    direct=sum(weights[i]/32*values[i, j] for i in eachindex(weights), j in 1:32)
    @test received≈direct rtol=1e-7
    sample=tank_field(sol, tank; resolution = (17, 33), transducers = (tx = tx, rx = rx),
        scattering_fields = (outgoing_wall, outgoing_line))
    @test any(==(-3), sample.regions)
    @test all(isfinite, sample.pressure[sample.regions .== 0])
    ix, iz=12, 12
    point=(sample.horizontal[ix], 0.0, sample.vertical[iz])
    expected=pressure(sol, point)+pressure(outgoing_wall, point)+pressure(outgoing_line, point)
    @test sample.pressure[ix, iz]≈expected rtol=1e-12
    reverse=bem(surface, Rigid(), k; transducer = receiver, compression = (method = :none,))
    @test received/(tx.radius^2)≈received_pressure(reverse, source)/(rx.radius^2) rtol=0.01
end
