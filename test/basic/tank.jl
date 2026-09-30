using AcousticScattering
using LinearAlgebra
using Test

@testset "Curved surface pressure bounds" begin
    AS = AcousticScattering
    center = [0.03, 0.02, -0.01]
    target = mesh(; semiaxes = (0.025, 0.015, 0.01), center,
        rotation = (axis = (0, 0, 1), angle = pi/7), resolution = 0.8, mesh_order = 3, qorder = 4)
    sol = bem(target, FluidFilled(1.05, 1.02), 80.0; compression = (method = :none,))
    patches = AS._region_patches(target.data, 0)
    points = [center, center+[0.018, 0.004, 0], center+[0.028, 0, 0],
        center+[0, 0, 0.011], center+[0, 0.15, 0], center+[-1.0, 0, 0],
        center+[0, 0, -1.0], collect(first(target.data).coords)]
    reference = [AS._surface_location(patches, p) for p in points]
    @test reference[1] == :inside
    @test all(==(:outside), reference[3:7])
    @test AS._boundary_pressure_regions(sol, points, :total) ==
          [AS._surface_pressure_region(sol.boundary, loc, :total) for loc in reference]
    exterior = points[3:7]
    @test AS._boundary_pressure_regions(sol, exterior, :scattered) == fill(:exterior, 5)
    @test_throws ArgumentError AS._boundary_pressure_regions(sol, exterior, :interior)
    @test_throws ArgumentError AS._boundary_pressure_regions(sol, [center], :scattered)
    @test all(isfinite, pressure(sol, points[1:7]))
end

@testset "Receiver aperture pressure" begin
    AS=AcousticScattering
    tx=Transducer((-0.065, 0.0, 0.37), (sind(10), 0, -cosd(10)), 0.046)
    rx=Transducer((0.065, 0.0, 0.37), (-sind(10), 0, -cosd(10)), 0.04)
    k=2pi*120000/1500
    p=received_pressure(tx, rx, k)
    refined=received_pressure(tx, rx, k; quadrature = (24, 96))
    @test p≈refined rtol=2e-5 atol=1e-9
    @test p/tx.radius^2 ≈ received_pressure(rx, tx, k)/rx.radius^2 rtol=2e-5
    bottom=Wall((0, 0, -0.47), (0, 0, 1), 1.0)
    @test received_pressure(TankTransducer(tx, bottom), rx, k) ≈
          p+received_pressure(AS._image(bottom, tx), rx, k) rtol=1e-9
    @test_throws ArgumentError received_pressure(tx, rx, k; quadrature = (0, 4))
    # Normalize the target reciprocity integral against an independent average
    # of the actual scattered pressure over the receiver aperture.
    surface=mesh(
        Sphere(0.03); method = :full, resolution = 0.025, mesh_order = 3, qorder = 4)
    sol=bem(surface, Rigid(), 12.0; transducer = tx, compression = (method = :none,))
    u, w=AS.gauss(12, 0.0, 1.0)
    rotation=AS._axis_rotation(collect(rx.axis))
    points=[Tuple(collect(rx.position)+rotation*[
                rx.radius*sqrt(r)*cos(2pi*j/48), rx.radius*sqrt(r)*sin(2pi*j/48), 0])
            for r in u for j in 0:47]
    weights=[weight/48 for weight in w for j in 0:47]
    expected=sum(weights .* pressure(sol, points; field = :scattered))
    @test received_pressure(sol, rx)≈expected rtol=2e-5 atol=1e-10
end

@testset "Finite tone burst and oscillogram" begin
    AS=AcousticScattering
    duration=256e-6
    f0=120000.0
    for window in (:rectangular, :hann), f in (10000.0, 116000.0, 120000.0, 137000.0)

        pulse=tone_burst(f0, duration; window, phase = 0.3, start_time = 1e-4)
        expected=AS.quadgk(t->begin
                weight=window===:hann ? (1-cos(2pi*t/duration))/2 : 1.0
                weight*cos(2pi*f0*t+0.3)*cis(2pi*f*(t+1e-4))
            end,
            0.0, duration; rtol = 1e-10)[1]
        @test pulse(f)≈expected rtol=1e-8 atol=1e-14
    end
    @test_throws ArgumentError tone_burst(f0, 0.0)
    @test_throws ArgumentError tone_burst(f0, duration; window = :unknown)
    frequencies=collect(20000.0:250.0:220000.0)
    delay=0.0011
    response=cis.(2pi .* frequencies .* delay)
    pulse=tone_burst(f0, duration; window = :hann)
    analytic=time_synthesis(frequencies, response, pulse; analytic = true, nfft = 8192)
    real_trace=time_synthesis(frequencies, response, pulse; nfft = 8192)
    @test real.(analytic.signal) ≈ real_trace.signal
    @test abs(analytic.times[argmax(abs.(analytic.signal))]-(delay+duration/2))<analytic.timestep
    trace=oscillogram(frequencies, (bottom = response, target = 0.1response), pulse;
        time_reference = duration/2, range_origin = 0.06, nfft = 8192)
    @test trace.signal ≈ 1.1real_trace.signal
    @test trace.envelope ≈ 1.1abs.(analytic.signal)
    @test trace.ranges[argmax(trace.envelope)] ≈ 0.06+1500delay/2 atol=1500trace.timestep/2
    cancellation=oscillogram(frequencies, (first = response, second = -response), pulse)
    @test all(iszero, cancellation.signal) && all(iszero, cancellation.envelope)
    @test_throws ArgumentError oscillogram(frequencies, (;), pulse)
    @test_throws ArgumentError oscillogram(frequencies, (echo = response,), pulse; sound_speed = -1)
end

@testset "Circular piston boundary integral" begin
    AS=AcousticScattering
    function disk_reference(k, a, x)
        radial=rho->rho*AS.quadgk(
            phi->begin
                r=sqrt((x[1]-rho*cos(phi))^2+(x[2]-rho*sin(phi))^2+x[3]^2)
                cis(k*r)/r
            end,
            0.0, 2pi; rtol = 1e-9)[1]
        k/(2pi*im)*AS.quadgk(radial, 0.0, a; rtol = 1e-9)[1]
    end
    a=0.0461
    for k in (12.0, 2pi*120000/1500),
        x in (
            (0.0, 0.0, 0.2), (0.01, 0.0, 0.05), (0.06, 0.03, 0.002),
            (0.2, 0.0, 0.5), (0.04, 0.01, -0.03), (a, 0.0, 0.003))

        value=AS._piston_pressure(k, a, x; rtol = 1e-9)
        @test value≈disk_reference(k, a, x) rtol=1e-7 atol=1e-10
        @test value ≈ AS._piston_pressure(k, a, (x[1], x[2], -x[3]); rtol = 1e-9) rtol=1e-10
    end
    for k in (1e-3, 500.0), z in (0.002, 0.1, 2.0)

        @test AS._piston_pressure(k, a, (0.0, 0.0, z)) ≈ AS._piston_pressure_axial(k, a, z) rtol=1e-8
    end
    k=2pi*120000/1500
    for x in ([0.01, 0.015, 0.1], [a, 0.0, 0.003], [0.0, 0.0, 0.1])
        step=1e-7
        finite_difference=[(AS._piston_pressure(k, a, x+step*I(3)[:, j])-AS._piston_pressure(
                               k, a, x-step*I(3)[:, j]))/(2step) for j in 1:3]
        @test AS._piston_gradient(k, a, x)≈finite_difference rtol=2e-5 atol=1e-7
    end
end

@testset "Cylindrical tank footprint" begin
    AS=AcousticScattering
    tank=Tank(0.18, (-0.45, 0.45))
    @test tank.radius==0.18
    @test length(tank.walls)==2
    @test tank.bounds.x==(-0.18, 0.18)
    @test AS._inside_tank(tank, (0.1, 0.1, 0.0))
    @test !AS._inside_tank(tank, (0.17, 0.17, 0.0))
    @test_throws ArgumentError Tank(-1.0, (-1, 1))
    @test_throws ArgumentError Tank(0.18, (-1, 1); reflections = (
        bottom = 1, surface = -1, side = 1))
    tx=Transducer((-0.06, 0.0, 0.38), (sind(10), 0, -cosd(10)), 0.04)
    @test length(TankTransducer(tx, tank).walls)==2
    @test_throws ArgumentError TankTransducer(Transducer((0.14, 0.14, 0.0), (0, 0, -1), 0.02), tank)
    sampled=tank_field(tx, 5.0, tank; plane = :xy, at = 0.0, resolution = (11, 11))
    @test sampled.regions[1, 1]==-2
    @test isnan(sampled.pressure[1, 1])
    @test isfinite(sampled.pressure[6, 6])
end

@testset "Physical tank pressure planes" begin
    AS=AcousticScattering
    tank=Tank((-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0))
    @test length(tank.walls)==6
    @test tank.walls[end].reflection == -1
    @test_throws ArgumentError Tank((1, 1), (-1, 1), (-1, 1))
    @test_throws ArgumentError Tank((-1, 1), (-1, 1), (-1, 1); reflections = (surface = -1,))
    tx=Transducer((-1.0, 0.0, 0.4), (1.0, 0.0, 0.0), 0.04)
    beam=TankTransducer(tx, tank)
    @test length(beam.walls)==6
    @test_throws ArgumentError TankTransducer(Transducer((0.0, 0.99, 0.0), (1, 0, 0), 0.05), tank)
    k=1.1
    source_map=tank_field(beam, k, tank; resolution = (21, 11), batch_size = 17)
    @test source_map.plane===:xz && source_map.at==0
    @test size(source_map.pressure)==(21, 11)
    @test extrema(source_map.vertical)==(-1.0, 1.0)
    @test source_map.field===:incident
    @test source_map.regions[6, 8]==-1
    @test isnan(source_map.pressure[6, 8])
    @test source_map.pressure[19, 7] ≈
          pressure(beam, k, (source_map.horizontal[19], 0.0, source_map.vertical[7]))
    end_map=tank_field(beam, k, tank; plane = :yz, at = 0.2, resolution = (7, 9))
    @test end_map.pressure[2, 3] ≈
          pressure(beam, k, (0.2, end_map.horizontal[2], end_map.vertical[3]))
    @test_throws ArgumentError tank_field(beam, k, tank; at = 2)
    @test_throws ArgumentError tank_field(beam, k, tank; resolution = (1, 5))
    @test_throws ArgumentError tank_field(beam, k, tank; plane = :bad)

    target=mesh(Sphere(0.2); method = :full, resolution = 0.2, qorder = 4)
    sol=bem(target, Rigid(), k; transducer = beam, compression = (method = :none,))
    original=tank_field(
        sol, tank; resolution = (21, 11), transducers = (transmitter = beam,),
        wall_images = false, batch_size = 17)
    total=tank_field(sol, tank; resolution = (21, 11),
        transducers = (transmitter = beam,), batch_size = 23)
    incident=tank_field(sol, tank; resolution = (21, 11), field = :incident,
        transducers = (transmitter = beam,))
    scattered=tank_field(sol, tank; resolution = (21, 11), field = :scattered,
        transducers = (transmitter = beam,))
    @test original.regions[11, 6]==1 && isnan(original.pressure[11, 6])
    @test isfinite(incident.pressure[11, 6])
    exterior=findall(==(0), total.regions)
    @test total.pressure[exterior] ≈
          incident.pressure[exterior]+scattered.pressure[exterior] rtol=1e-10
    point=(total.horizontal[19], 0.0, total.vertical[7])
    expected=pressure(sol, point)+sum(w.reflection*pressure(
                                          sol, AS._mirror_point(w, point);
                                          field = :scattered)
    for w in tank.walls)
    @test total.pressure[19, 7] ≈ expected rtol=1e-10
    @test original.pressure[19, 7] ≈ pressure(sol, point) rtol=1e-10
    @test total.wall_images && !original.wall_images
    automatic=tank_field(sol, tank; resolution = (21, 11), wall_images = false)
    @test length(automatic.transducers)==1
    @test automatic.regions[6, 8]==-1 && isnan(automatic.pressure[6, 8])
    @test automatic.pressure ≈ original.pressure nans=true
    # A very small receiver approaches a point probe. Its image-weighted
    # reciprocity result checks the outgoing wall-image pressure independently.
    probe=Transducer(point, Tuple(-collect(point)), 0.001)
    probe_factor=-im*k*probe.radius^2/2
    @test received_signal(sol, TankTransducer(probe, tank)) ≈
          -4pi*probe_factor*scattered.pressure[19, 7] rtol=1e-5
    @test_throws ArgumentError tank_field(sol, Tank((-0.1, 0.1), (-1, 1), (-1, 1)); resolution = (
        3, 3))
    @test_throws ArgumentError tank_field(sol, tank; resolution = (3, 3), reflection_weights = ones(5))

    # Single pressure-release wall: direct+image scattered pressure cancels
    # exactly on the wall, independently of the target boundary discretization.
    one_wall=Tank((-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0);
        reflections = (xmin = 0, xmax = 0, ymin = 0, ymax = 0, bottom = 0, surface = -1))
    one_beam=TankTransducer(tx, one_wall)
    one_sol=bem(target, Rigid(), k; transducer = one_beam, compression = (method = :none,))
    wall_map=tank_field(one_sol, one_wall; resolution = (9, 5), transducers = (transmitter = one_beam,))
    @test maximum(abs, wall_map.pressure[:, end]) < 1e-12
end

@testset "Tank incident fields in full BEM" begin
    AS = AcousticScattering
    k, beta, alpha = 1.3, 0.7, 0.2
    direction = AS._bem3d_incidence_direction(beta, alpha)
    p(x) = cis(k*dot(direction, x))
    grad(x) = im*k*direction*p(x)
    incident = IncidentField(p, grad)
    surface = mesh(Sphere(0.3); method = :full, resolution = 0.3, qorder = 4)
    options = (; incidence_angle = beta, incidence_azimuth = alpha,
        compression = (method = :none,))
    for boundary in (Rigid(), PressureRelease(), FluidFilled(1.2, 1.1))
        controls = boundary isa FluidFilled ? (; condition_limit = 0) :
                   (; gmres_kwargs = (reltol = 1e-10, restart = 150, maxiter = 1200))
        baseline = bem(surface, boundary, k; options..., controls...)
        prescribed = bem(surface, boundary, k; options..., controls..., incident)
        @test prescribed.data.p_scat≈baseline.data.p_scat rtol=2e-10 atol=2e-12
        @test prescribed.data.dpdn_scat≈baseline.data.dpdn_scat rtol=2e-10 atol=2e-12
        @test pressure(prescribed, (0.7, 0.2, 0.1); field = :incident) ≈ p((0.7, 0.2, 0.1))
        @test scattering_amplitude(prescribed) ≈ scattering_amplitude(baseline) rtol=2e-10
        complex_scale = 0.4+0.7im
        scaled = bem(surface, boundary, k; options..., controls...,
            incident = IncidentField(x->complex_scale*p(x), x->complex_scale*grad(x)))
        @test scaled.data.p_scat≈complex_scale*baseline.data.p_scat rtol=2e-10 atol=2e-12
        @test scaled.data.dpdn_scat≈complex_scale*baseline.data.dpdn_scat rtol=2e-10 atol=2e-12
        @test pressure(scaled, (0.7, 0.2, 0.1)) ≈
              complex_scale*pressure(baseline, (0.7, 0.2, 0.1)) rtol=2e-10
        if boundary isa FluidFilled
            @test pressure(scaled, (0.05, 0.0, 0.0); field = :interior) ≈
                  complex_scale*pressure(baseline, (0.05, 0.0, 0.0); field = :interior) rtol=2e-10
            compressed = bem(surface, boundary, k; options..., condition_limit = 0,
                incident, compression = (method = :hmatrix, tol = 1e-9))
            @test scattering_amplitude(compressed) ≈ scattering_amplitude(prescribed) rtol=2e-6
        end
    end
    @test_throws ArgumentError bem(surface, Rigid(), k; incident,
        transducer = Transducer((-2, 0, 0), (1, 0, 0), 0.1))
    @test_throws ArgumentError bem(surface, Rigid(), k; options...,
        incident = IncidentField(p, x->[1.0, 2.0]))
    @test_throws ArgumentError bem(surface, Rigid(), k; options...,
        incident = IncidentField(x->NaN, grad))

    # Full spheroid pressure must use the actual surface, not the cylindrical
    # location rule. Supplied meshes and generated bodies share these traces.
    spheroid=mesh(Spheroid(0.3, 0.2); method = :full, resolution = 0.2, qorder = 4)
    spheroid_sol=bem(
        spheroid, FluidFilled(1.0, 1.0), k; options..., condition_limit = 0, incident)
    pts=[(0.7, 0.1, 0.0), (0.0, 0.0, 0.0)]
    @test pressure(spheroid_sol, pts) ≈ p.(pts) rtol=0.02

    # A smooth, source-free superposition with no single propagation direction.
    d2 = AS.SVector(0.0, 0.0, 1.0)
    p2(x) = cis(k*dot(d2, x))
    g2(x) = im*k*d2*p2(x)
    combined = IncidentField(x->p(x)+(0.2-0.3im)*p2(x), x->grad(x)+(0.2-0.3im)*g2(x))
    inner = mesh(Sphere(0.1); method = :full, resolution = 0.1, qorder = 4)
    surfaces, materials = [surface, inner], [FluidFilled(1.1, 1.05), FluidFilled(1.3, 1.2)]
    for formulation in (:muller, :cbie)
        settings = (; formulation, condition_limit = 0)
        a = bem(surfaces, materials, k; settings..., incident)
        b = bem(surfaces, materials, k; settings..., incident = IncidentField(p2, g2))
        c = bem(surfaces, materials, k; settings..., incident = combined)
        for j in 1:2
            @test c.data.interfaces[j].pressure ≈
                  a.data.interfaces[j].pressure +
                  (0.2-0.3im)*b.data.interfaces[j].pressure rtol=2e-10
        end
        @test scattering_amplitude(c; direction = d2) ≈
              scattering_amplitude(a; direction = d2) +
              (0.2-0.3im)*scattering_amplitude(b; direction = d2) rtol=2e-10
        @test pressure(c, (0.7, 0.2, 0.1)) ≈
              pressure(a, (0.7, 0.2, 0.1)) +
              (0.2-0.3im)*pressure(b, (0.7, 0.2, 0.1)) rtol=2e-10
        if formulation===:muller
            compressed=bem(surfaces, materials, k; settings..., incident = combined,
                compression = (method = :hmatrix, tol = 1e-9))
            @test scattering_amplitude(compressed; direction = d2) ≈
                  scattering_amplitude(c; direction = d2) rtol=2e-6
            separated=components(compressed)
            @test all(s->s.data.incident===combined, separated.isolated)
        end
    end
end

@testset "Tank source fields and wall coefficients" begin
    AS=AcousticScattering
    tx=Transducer((-1.0, 0.2, 0.1), (1.0, 0.1, 0.2), 0.07)
    k=12.0
    center=AS.SVector(tx.position)
    axis=AS.SVector(tx.axis)
    perpendicular=normalize(cross(axis, AS.SVector(0.0, 0.0, 1.0)))
    for angle in (0.0, 0.3, 0.9)
        direction=cos(angle)*axis+sin(angle)*perpendicular
        for distance in (20.0, 100.0)
            point=Tuple(center+distance*direction)
            @test pressure(tx, k, point; approximation = :farfield) ≈ pressure(tx, k, point) rtol=0.003
        end
    end
    points=[(0.0, 0.1, 0.2), (0.3, -0.2, 0.1)]
    @test pressure(tx, k, hcat(collect.(points)...)) ≈ pressure(tx, k, points)
    @test_throws ArgumentError pressure(tx, k, tx.position)
    @test_throws ArgumentError Transducer((0, 0, 0), (1, 0), 0.1)
    @test_throws ArgumentError Wall((0, 0, 0), (0, 1), 1.0)
    @test_throws ArgumentError pressure(tx, k, points; approximation = :unknown)
    coefficient=(wave, cosine)->(2cosine-1)/(2cosine+1)*cis(0.03wave)
    wall=Wall((0, 0, -1), (0, 0, 1), coefficient)
    anchor=(0.0, 0.0, 0.0)
    tank=TankTransducer(tx, wall; reference_point = anchor)
    image=Transducer((-1.0, 0.2, -2.1), (tx.axis[1], tx.axis[2], -tx.axis[3]), tx.radius)
    ray=AS.SVector(anchor)-AS.SVector(image.position)
    weight=coefficient(k, abs(ray[3])/norm(ray))
    @test pressure(tank, k, points) ≈
          pressure(tx, k, points)+weight*pressure(image, k, points) rtol=1e-12
    # Reflection is fixed per source/image ray, so its incident field remains a
    # Helmholtz solution and gradients differentiate the same source sum.
    _, g=AS._transducer_field(tank, k)
    x=collect(points[1])
    h=1e-5
    finite_difference=ComplexF64[(pressure(tank, k, x+h*I(3)[:, j])-pressure(tank, k, x-h*I(3)[:, j]))/(2h)
                                 for j in 1:3]
    @test g(x) ≈ finite_difference rtol=2e-7
    @test_throws ArgumentError TankTransducer(tx, wall)
    @test_throws ArgumentError pressure(
        TankTransducer(tx, wall; reference_point = (
            0, 0, -2)), k, points)
    invalid=Wall((0, 0, -1), (0, 0, 1), (wave, cosine)->2)
    @test_throws ArgumentError pressure(TankTransducer(tx, invalid; reference_point = anchor), k, points)
end

@testset "Tank calibration response" begin
    f=collect(100.0:100.0:1000.0)
    delay=0.003
    gain(wave) = (0.7+0.2im+(0.001-0.0002im)*wave)*cis(2pi*wave*delay)
    reference=ComplexF64[(1+0.3im)*cis(0.002wave) for wave in f]
    measured=reference .* gain.(f)
    fit=calibrate_response(f, measured, reference; delay)
    @test fit.response.(f) ≈ gain.(f) rtol=1e-14
    @test maximum(fit.residual)<1e-14
    @test fit.reference_rms ≈ abs.(reference)
    mids=f[1:(end - 1)] .+ 50
    @test fit.response.(mids) ≈ gain.(mids) rtol=1e-14
    repeated=hcat(measured, measured .* (1+0.02im))
    weights=ones(length(f), 2)
    weights[:, 2].=0
    @test calibrate_response(f, repeated, reference; weights, delay).response.(f) ≈ gain.(f) rtol=1e-14
    balanced=calibrate_response(f, repeated, reference; delay)
    @test balanced.response.(f) ≈ (1+0.01im) .* gain.(f) rtol=1e-14
    pulse=gaussian_pulse(500.0, 100.0)
    times=[0.0, 0.003, 0.006]
    @test time_synthesis(f, reference, pulse, times; system_response = fit.response) ≈
          time_synthesis(f, measured, pulse, times) rtol=1e-13
    @test_throws ArgumentError fit.response(99.0)
    @test_throws ArgumentError fit.response(1001.0)
    @test_throws ArgumentError calibrate_response(f, measured, zeros(length(f)))
    @test_throws ArgumentError calibrate_response(f, measured, reference; weights = zeros(length(f)))
    @test_throws ArgumentError calibrate_response(f, measured, reference; reference_floor = 10)
    @test_throws ArgumentError TransferFunction([1.0, 1.0], [1.0, 1.0])
    sphere_sweep=frequency_sweep(k->modal(Sphere(0.02), SolidElastic(5.0, 3.0, 1.5), k),
        collect(1000.0:250.0:3000.0), 1500.0)
    instrument=TransferFunction(sphere_sweep.frequencies,
        (0.8+0.3im) .* cis.(2pi .* sphere_sweep.frequencies .* delay); delay)
    normalized_measurement=instrument.(sphere_sweep.frequencies) .* sphere_sweep.amplitudes
    sphere_fit=calibrate_response(sphere_sweep.frequencies, normalized_measurement,
        sphere_sweep.amplitudes; delay)
    @test sphere_fit.response.(sphere_sweep.frequencies) ≈
          instrument.(sphere_sweep.frequencies)
    @test time_synthesis(sphere_sweep, gaussian_pulse(2000.0, 500.0);
        system_response = sphere_fit.response).signal ≈
          time_synthesis(sphere_sweep.frequencies,
        normalized_measurement, gaussian_pulse(2000.0, 500.0)).signal
end

@testset "Tank incident fields in MFS and volume FEM" begin
    AS=AcousticScattering
    k, beta, alpha=2.0, 0.6, 0.3
    scale=0.4+0.7im
    body=Sphere(0.05)
    surface=mesh(body; method = :full, resolution = 0.04, mesh_order = 2, qorder = 2)
    sources=mesh(body; method = :full, resolution = 0.04, mesh_order = 2, qorder = 1)
    d=AS._bem3d_incidence_direction(beta, alpha)
    p, g=AS._plane_wave_incident(k, d)
    opts=(; source_mesh = sources, offset = 0.02, condition_limit = 0,
        incidence_angle = beta, incidence_azimuth = alpha)
    plane=mfs(surface, Rigid(), k; opts...)
    custom=mfs(surface, Rigid(), k; opts...,
        incident = IncidentField(x->scale*p(x), x->scale*g(x)))
    @test scattering_amplitude(custom) ≈ scale*scattering_amplitude(plane) rtol=1e-9
    @test pressure(custom, (0.1, 0.0, 0.0); field = :incident) ≈ scale*p((0.1, 0.0, 0.0))
    @test diagnostics(custom).illumination===:prescribed
    tx=Transducer((0, 0, -1), (0, 0, 1), 0.02)
    wall=Wall((0, 0, 0.3), (0, 0, -1), 0.6)
    tank=TankTransducer(tx, wall)
    tank_sol=mfs(surface, Rigid(), k; opts..., transducer = tank)
    @test tank_sol.data.incident.source === tank
    mfs_map=tank_field(tank_sol, Tank((-1.2, 1.2), (-0.5, 0.5), (-1.2, 0.5));
        resolution = (5, 5), wall_images = false)
    @test length(mfs_map.transducers)==1
    @test pressure(tank_sol, (0.1, 0.0, 0.0); field = :incident) ≈
          pressure(tank, k, (0.1, 0.0, 0.0))
    rx=Transducer((1, 0, 0), (-1, 0, 0), 0.02)
    @test received_signal(tank_sol, TankTransducer(rx, wall)) ≈
          received_signal(tank_sol, rx)+wall.reflection*received_signal(tank_sol, AS._image(wall, rx))

    # Volume FEM uses +z as its polar axis. Both single and nested-region public
    # APIs must retain the actual field and reproduce linearity with that frame.
    pvol, gvol=AS._plane_wave_incident(k, AS._volume_direction(beta, alpha))
    field=IncidentField(x->scale*pvol(x), x->scale*gvol(x))
    femopts=(; method = :volume, points_per_wavelength = 5, solver = :direct,
        incidence_angle = beta, incidence_azimuth = alpha)
    for (bodies, materials, extra) in ((body, Rigid(), (;)),
        ([body, Sphere(0.02)], [FluidFilled(1.1, 1.2), FluidFilled(1.3, 1.1)],
        (; parents = [0, 1])))
        baseline=fem(bodies, materials, k; femopts..., extra...)
        prescribed=fem(bodies, materials, k; femopts..., extra..., incident = field)
        @test scattering_amplitude(prescribed) ≈ scale*scattering_amplitude(baseline) rtol=1e-9
        @test pressure(prescribed, (0.1, 0.0, 0.0); field = :incident) ≈
              scale*pvol((0.1, 0.0, 0.0))
        @test diagnostics(prescribed).illumination===:prescribed
    end
end

@testset "Tank pulse synthesis" begin
    f = collect(200.0:5.0:1800.0)
    pulse = gaussian_pulse(1000.0, 100.0)
    delay = 0.012
    response = cis.(2pi .* f .* delay)
    fft_result = time_synthesis(f, response, pulse; start_time = -0.01,
        transmit_response = 2im, receive_response = 0.5, system_response = 2)
    direct = time_synthesis(f, response, pulse, fft_result.times;
        transmit_response = 2im, receive_response = 0.5, system_response = 2)
    @test fft_result.signal ≈ direct rtol=2e-12
    sigma=100/(2sqrt(2log(2)))
    # Uniform frequency samples periodize the analytic pulse. Include the tails
    # of adjacent replicas when testing a complete FFT time window.
    expected=[sum(2sigma * sqrt(2pi) * exp(-2pi^2*sigma^2*(t-delay-m*fft_result.period)^2) *
                  real(2im*cis(-2pi*1000*(t-delay-m*fft_result.period))) for m in -1:1)
              for t in fft_result.times]
    @test direct ≈ expected rtol=1e-9
    centered=time_synthesis(f, response, pulse, fft_result.times; reference_delay = delay)
    flat=time_synthesis(f, ones(ComplexF64, length(f)), pulse, fft_result.times)
    @test centered ≈ flat rtol=1e-13
    gated=time_synthesis(f, response, pulse; start_time = -0.01, gate = (0.009, 0.015))
    mask=[0.009<=t<=0.015 for t in gated.times]
    @test all(iszero, gated.signal[.!mask])
    @test gated.signal[mask] ≈ time_synthesis(f, response, pulse, gated.times[mask]) rtol=2e-12
    @test time_synthesis(f, response, pulse, [0.0, delay]; gate = t->0.5) ≈
          0.5time_synthesis(f, response, pulse, [0.0, delay])
    # Non-bin-aligned lower frequency and nonzero time origin.
    shifted=f .+ 0.7
    shifted_fft=time_synthesis(shifted, exp.(0.1im .* shifted), pulse; start_time = 0.00031)
    @test shifted_fft.signal ≈
          time_synthesis(shifted, exp.(0.1im .* shifted), pulse, shifted_fft.times) rtol=2e-12
    @test pulse(1000+50) ≈ 0.5
    @test_throws ArgumentError time_synthesis([1.0, 1.0], ones(2), pulse, [0.0])
    @test_throws ArgumentError time_synthesis([1.0, Inf], ones(2), pulse, [0.0])
    @test_throws ArgumentError time_synthesis([1.0, 2.0], ones(2), pulse, [NaN])
    @test_throws ArgumentError time_synthesis([1.0, 2.0, 4.0], ones(3), pulse)
    @test_throws ArgumentError time_synthesis(f, response, pulse; nfft = 4)
    @test_throws ArgumentError time_synthesis(f, response, pulse; receive_response = [1.0])
    @test_throws ArgumentError time_synthesis(f, response, pulse; gate = (1.0, 0.0))
    @test_throws ArgumentError time_synthesis(f, response, pulse, [0.0]; gate = t->NaN)
end

@testset "Tank BEM reciprocity and enclosing surfaces" begin
    AS=AcousticScattering
    k=1.2
    body=Sphere(0.3)
    target=mesh(body; method = :full, resolution = 0.15, qorder = 4, mesh_order = 3)
    tx=Transducer((-2.0, 0.1, 0.2), (2.0, -0.1, -0.2), 0.08)
    rx=Transducer((1.0, 1.5, 0.2), (-1.0, -1.5, -0.2), 0.07)
    controls=(; compression = (method = :none,),
        gmres_kwargs = (reltol = 1e-11, restart = 150, maxiter = 1200))
    sol_tx=bem(target, Rigid(), k; transducer = tx, controls...)
    sol_rx=bem(target, Rigid(), k; transducer = rx, controls...)
    signal=received_signal(sol_tx, rx)
    @test signal ≈ received_signal(sol_rx, tx) rtol=2e-4
    fluid=FluidFilled(1.2, 1.1)
    fluid_tx=bem(target, fluid, k; transducer = tx,
        compression = (method = :none,), condition_limit = 0)
    fluid_rx=bem(target, fluid, k; transducer = rx,
        compression = (method = :none,), condition_limit = 0)
    @test received_signal(fluid_tx, rx) ≈ received_signal(fluid_rx, tx) rtol=3e-4
    region_tx=bem([target], [fluid], k; transducer = tx, condition_limit = 0)
    @test received_signal(region_tx, rx) ≈ received_signal(fluid_tx, rx) rtol=3e-4
    enclosure=mesh(
        Sphere(0.6); method = :full, resolution = 0.25, qorder = 5, mesh_order = 3)
    @test received_signal(sol_tx, rx; surface = enclosure) ≈ signal rtol=3e-4
    @test_throws ArgumentError received_signal(sol_tx, Transducer((0, 0, 0), (1, 0, 0), 0.05))
    @test_throws ArgumentError received_signal(sol_tx, rx;
        surface = mesh(Sphere(0.1); method = :full, resolution = 0.1, qorder = 4))
    wall=Wall((0, 0, -1.0), (0, 0, 1), 0.4+0.1im)
    tank_rx=TankTransducer(rx, wall)
    @test received_signal(sol_tx, tank_rx) ≈
          signal +
          wall.reflection*received_signal(sol_tx, AS._image(wall, rx)) rtol=1e-12
    @test pressure(sol_tx, (0.7, 0.1, 0.2); field = :incident) ≈
          AS._transducer_field(tx, k)[1]((0.7, 0.1, 0.2))
    direction=AS.SVector(1.0, 0.0, 0.0)
    distance=1000.0
    far_rx=Transducer(Tuple(distance*direction), Tuple(-direction), 0.05)
    C=-im*k*far_rx.radius^2/2
    predicted=-4pi*C*cis(k*distance)/distance*scattering_amplitude(sol_tx; direction)
    @test received_signal(sol_tx, far_rx) ≈ predicted rtol=0.005
end
