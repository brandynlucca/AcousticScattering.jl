using AcousticScattering
using Test

const AS = AcousticScattering

@testset "Far-field-generated mixed spheroid T-matrix" begin
    body = Spheroid(1.5, 1.0)
    boundary = Shelled(
        LayeredMaterial(FluidLayer(1.2, 1.1),
            ElasticLayer(2.7, 4.2, 2.1), 0.8),
        FluidInterior(0.8, 0.9), 0.55)
    nodes, weights = AS.gauss(6)
    nphi = 9

    # Frozen full-3D FEM samples, aspect ratio 1.5, k=1.2, six-point polar rule.
    samples_file = joinpath(@__DIR__, "spheroid_farfield_samples.txt")
    rows = [parse.(Float64, split(line))
            for line in readlines(samples_file)
            if !isempty(line) && !startswith(line, "%")]
    @test length(rows) == 6 * 6 * nphi
    samples = zeros(ComplexF64, 6, 6, nphi)
    index = 1
    for i in 1:6, s in 1:6, p in 1:nphi
        row = rows[index]
        samples[s, i, p] = complex(row[4], row[5])
        index += 1
    end
    @test isapprox(rows[1][1:2], [nodes[1], nodes[1]]; atol = 1e-14)
    @test isapprox(rows[end][1:2], [nodes[end], nodes[end]]; atol = 1e-14)
    blocks = AS._farfield_transition_blocks(samples, nodes, weights, 4)
    transition = AS._farfield_tmatrix_solution(body, boundary, 1.2, nodes, blocks,
        (; method = :fem_farfield, holdout_error = nothing))
    @test transition isa TMatrixSolution
    @test scattering_amplitude(transition) == transition.f
    directed = AS._farfield_tmatrix_solution(body, boundary, 1.2, nodes, blocks,
        diagnostics(transition); incidence_angle = 0.9, incidence_azimuth = 0.2,
        scatter_angle = 0.4, scatter_azimuth = 1.7)
    @test scattering_amplitude(directed) == directed.f
    @test scattering_amplitude(directed) == scattering_amplitude(transition;
        incidence_angle = 0.9, incidence_azimuth = 0.2, angle = 0.4, azimuth = 1.7)
    @test scattering_amplitude(directed; incidence_angle = 0.7) ==
          scattering_amplitude(transition; incidence_angle = 0.7, incidence_azimuth = 0.2,
        angle = pi - 0.7, azimuth = pi + 0.2)
    @test target_strength(directed; angle = 1.1) ==
          target_strength(scattering_amplitude(directed; angle = 1.1))
    scalar = TMatrixSolution(body, boundary, 1, 2)
    @test scalar.k === 1.0
    @test scattering_amplitude(scalar) === 2.0 + 0.0im
    @test diagnostics(scalar) === nothing
    @test_throws ArgumentError scattering_amplitude(scalar; angle = 1.0)
    @test_throws ArgumentError target_strength(scalar; angle = 1.0)

    fixture = split(
        raw"""
% ppw incidence_angle scatter_angle scatter_azimuth real_f imag_f
6 1.5707963267948966 0 0 -0.28003320184576186 0.065822924933570337
6 1.5707963267948966 1.5707963267948966 0 -0.022945060706330872 0.10282220274089578
6 1.5707963267948966 1.5707963267948966 3.1415926535897931 -0.26997688950477344 0.09260974968344278
6 1.5707963267948966 3.1415926535897931 0 -0.28003284332886691 0.065825337003780918
8 1.5707963267948966 0 0 -0.28003874960791397 0.065821944756547943
8 1.5707963267948966 1.5707963267948966 0 -0.022726439194134408 0.10283199018381857
8 1.5707963267948966 1.5707963267948966 3.1415926535897931 -0.26976386616237374 0.092620602440531705
8 1.5707963267948966 3.1415926535897931 0 -0.28003866904300179 0.065822221781649176
""", '\n')
    reference = [parse.(Float64, split(line))
                 for line in fixture
                 if !isempty(line) && !startswith(line, "%")]
    @test length(reference) == 8
    for row in reference
        row[1] == 8 || continue
        actual = scattering_amplitude(transition;
            incidence_angle = row[2], angle = row[3], azimuth = row[4])
        expected = complex(row[5], row[6])
        @test abs(actual - expected) / abs(expected) < 5e-3
    end
    order8_file = split(
        raw"""
% aspect=1.5 k=1.2 ppw=6; polar_order=6,m_max=4 versus polar_order=8,m_max=6
% holdout_error=1.5296249263517034e-5 fem_solves=5 fem_assemblies=1
% incidence_angle scatter_angle scatter_azimuth real_f6 imag_f6 real_f8 imag_f8
1.5707963267948966 0 0 -0.27997480334418268 0.065831657085399176 -0.2800330862405902 0.065824080788674813
1.5707963267948966 1.5707963267948966 0 -0.023011428696568258 0.10281576292350492 -0.02294344928336314 0.10282221613587976
1.5707963267948966 1.5707963267948966 3.1415926535897931 -0.27002996932893164 0.092602026369130175 -0.26997564793847001 0.092609286709100361
1.5707963267948966 3.1415926535897931 0 -0.27997480334418268 0.065831657085399176 -0.2800330862405902 0.065824080788674827
0.34999999999999998 0.34999999999999998 3.1415926535897931 -0.1234243442947159 0.056118179566019936 -0.12344772891549129 0.056116581545809383
0.34999999999999998 1.3999999999999999 0.59999999999999998 -0.23100536745732722 0.070315540035865653 -0.23098619110978794 0.070316320007562286
0.90000000000000002 2 1.7 -0.29449435760859927 0.076502932893050157 -0.2944682084915749 0.076501562124347194
1.2 2.7000000000000002 2.3999999999999999 -0.2306884313524474 0.06828720155437909 -0.23064068430193704 0.068288960053083728
2.3999999999999999 0.80000000000000004 2.8999999999999999 -0.14247584145380376 0.069644266647573852 -0.14254212649636133 0.069644539300234481
2.7999999999999998 1.6000000000000001 0.40000000000000002 -0.24143610418833075 0.07044821939638557 -0.24144429994789046 0.07044910068552708
""",
        '\n')
    order8_rows = [parse.(Float64, split(line))
                   for line in order8_file
                   if !isempty(line) && !startswith(line, "%")]
    @test length(order8_rows) == 10
    for row in order8_rows
        incidence, angle, azimuth = row[1:3]
        saved6, saved8 = complex(row[4], row[5]), complex(row[6], row[7])
        actual6 = scattering_amplitude(transition;
            incidence_angle = incidence, angle, azimuth)
        @test isapprox(actual6, saved6; rtol = 1e-12)
        @test abs(saved8 - saved6) / abs(saved8) < 1e-3
    end
    @test isapprox(
        scattering_amplitude(transition;
            incidence_angle = pi / 2, incidence_azimuth = 0.4,
            angle = pi / 2, azimuth = pi + 0.4),
        scattering_amplitude(transition;
            incidence_angle = pi / 2, angle = pi / 2, azimuth = pi);
        rtol = 1e-12)
    @test isfinite(target_strength(transition))
    @test diagnostics(transition).method === :fem_farfield

    # A finite Legendre/Fourier kernel must survive projection exactly.
    synthetic = zeros(ComplexF64, 6, 6, nphi)
    for s in 1:6, i in 1:6, p in 1:nphi
        xs, xi, phi = nodes[s], nodes[i], 2pi * (p - 1) / nphi
        synthetic[s, i, p] = (1 + 0.2im) *
                             AS._farfield_angular_basis(0, xs, 5)[2] *
                             AS._farfield_angular_basis(0, xi, 5)[3] +
                             (0.3 - 0.4im) * AS._farfield_angular_basis(2, xs, 5)[1] *
                             AS._farfield_angular_basis(2, xi, 5)[2] * cos(2phi)
    end
    synthetic_blocks = AS._farfield_transition_blocks(synthetic, nodes, weights, 4)
    @test maximum(abs(AS._farfield_transition_amplitude(synthetic_blocks,
                      nodes, acos(nodes[i]), 0.0, acos(nodes[s]), 2pi * (p - 1) / nphi) -
                      synthetic[s, i, p]) for s in 1:6, i in 1:6, p in 1:nphi) < 1e-12

    @test_throws ArgumentError tmatrix(Spheroid(1.6, 1.0), boundary, 1.2;
        method = :farfield)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2; method = :farfield,
        polar_order = 5)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2; method = :farfield,
        h_body = -0.1)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2; method = :farfield,
        h = -0.1)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2; method = :farfield,
        domain_radius = 1.0)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2; method = :farfield,
        dtn_order = 0)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2; method = :unknown)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2;
        method = :farfield, incidence_angle = -0.1)
    @test_throws ArgumentError tmatrix(body, boundary, 1.2;
        method = :farfield, scatter_azimuth = NaN)
end

@testset "Matched outer-fluid shell reference" begin
    fixture = split(
        raw"""
% route ppw_or_nmax scatter_angle scatter_azimuth real_f imag_f
% exterior k=1.2 rad/m; broadside incidence pi/2; body (a,b)=(1.5,1.0) m
% matched outer fluid to b=0.85 m; elastic wall to b=0.55 m; fluid core
fem 6 0 0 -0.27778333366418861 0.059350410989692723
fem 6 1.5707963267948966 0 0.010209130984266826 0.098359458709318281
fem 6 1.5707963267948966 3.1415926535897931 -0.27881044759985901 0.084216152510826606
fem 6 3.1415926535897931 0 -0.2777833917169017 0.059349412071865636
fem 8 0 0 -0.27778334061610743 0.059350413611381901
fem 8 1.5707963267948966 0 0.01020929033234546 0.0983594719387276
fem 8 1.5707963267948966 3.1415926535897931 -0.27881028859273788 0.084216197487787511
fem 8 3.1415926535897931 0 -0.27778339895483833 0.059349413621516425
fem_h0.075 6 0 0 -0.2777866131435483 0.059349862166535414
fem_h0.075 6 1.5707963267948966 0 0.010245268448055504 0.098360223229738711
fem_h0.075 6 1.5707963267948966 3.1415926535897931 -0.27877828584369263 0.084220547782421259
fem_h0.075 6 3.1415926535897931 0 -0.27778640331497056 0.05935194418998873
tmatrix 22 0 0 -0.27778564214610535 0.059346015094964372
tmatrix 22 1.5707963267948966 0 0.010361727293042435 0.09836860287290991
tmatrix 22 1.5707963267948966 3.1415926535897931 -0.27866310537775524 0.084221171911321641
tmatrix 22 3.1415926535897931 0 -0.27778564215149815 0.059346015104389797
""", '\n')
    rows = [split(line)
            for line in fixture
            if !isempty(line) && !startswith(line, "%")]
    reference = [row for row in rows if row[1] == "fem" && row[2] == "8"]
    @test length(reference) == 4

    outer = Spheroid(1.5, 1.0)
    wall_body = Spheroid(sqrt(outer.q^2 + 0.85^2), 0.85)
    shell = Shelled(ElasticLayer(2.7, 4.2, 2.1),
        FluidInterior(0.8, 0.9), 0.55 / 0.85)
    for row in reference
        angle, azimuth, real_f, imag_f = parse.(Float64, row[3:6])
        actual = scattering_amplitude(tmatrix(wall_body, shell, 1.2;
            incidence_angle = pi / 2, scatter_angle = angle,
            scatter_azimuth = azimuth, m_max = 8, n_max = 22,
            check = false))
        expected = complex(real_f, imag_f)
        @test abs(actual - expected) / abs(expected) < 5e-3
    end
end
