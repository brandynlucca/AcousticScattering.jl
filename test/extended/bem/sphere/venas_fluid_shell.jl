using AcousticScattering
using Test

@time "Venas layered-fluid sphere" @testset "Venas layered-fluid sphere" begin
    reference_lines = split(raw"""
% Venas-Jenserud three-fluid sphere a=1 b=0.55 k=1.5 rho=[1 1.2 .7] c=[1 1.1 .8]
% columns r mu real(p_scat) imag(p_scat); truncation order 16
1.5 1 -0.025589689329653529 0.01319384135659213
1.5 -1 -0.061669462914307742 0.039414727853982222
1.5 0 -0.024513301514086368 0.034232177719977157
2 0.6 -0.025196926028319629 0.0031994739929484509
% backscatter real imaginary
backscatter 0.10660765367177678 0.0063411389555033037
""", '\n')
    lines = [strip(line)
             for line in reference_lines
             if !isempty(strip(line)) && !startswith(line, "%")]
    rows = [parse.(Float64, split(line)) for line in lines[1:4]]
    points = [(row[1]*row[2], row[1]*sqrt(1-row[2]^2), 0.0) for row in rows]
    expected_pressure = complex.([row[3] for row in rows], [row[4] for row in rows])
    backscatter = split(lines[5])
    expected_amplitude = complex(parse(Float64, backscatter[2]),
        parse(Float64, backscatter[3]))

    body = Sphere(1.0)
    shell = Shelled(FluidLayer(1.2, 1.1), FluidInterior(0.7, 0.8), 0.55)
    modal_solution = modal(body, shell, 1.5; m_max = 16)
    @test pressure(modal_solution, points; field = :scattered)≈expected_pressure rtol=1e-11
    @test scattering_amplitude(modal_solution)≈expected_amplitude rtol=1e-11

    errors = Float64[]
    for n in (128, 256)
        actual = scattering_amplitude(bem(body, shell, 1.5; n))
        push!(errors, abs(actual-expected_amplitude)/abs(expected_amplitude))
        if n == 256
            @test errors[end] < 1e-3
            @test abs(20log10(abs(actual/expected_amplitude))) < 0.01
        end
    end
    @test errors[2] < errors[1]/3
end
