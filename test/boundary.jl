using AcousticScattering
using Test
using LinearAlgebra: norm, I, eigvals, tr, BLAS
using SpecialFunctions: besselj

const AS = AcousticScattering

BLAS.set_num_threads(1)

@testset "Bent-cylinder MFS (method of fundamental solutions)" begin
    radius, length = 0.01, 0.07
    c = 1477.3
    k = 2π * 38000.0 / c
    ρc = 1e6 * length
    cyl_straight = AS.Cylinder(radius, length)
    cyl_bent = AS.Cylinder(radius, length; radius_curvature = ρc)

    @testset "broadside, both boundaries: converges tightly" begin
        for boundary in (AS.Rigid(), AS.PressureRelease())
            ts_exact = AS.target_strength(AS.modal(cyl_straight, boundary, k; m_max = 30))
            ts_mfs = AS.target_strength(AS.mfs(
                cyl_bent, boundary, k; offset = 0.3radius, n_s = 40, n_φ = 32))
            @test ts_mfs ≈ ts_exact atol = 0.3
        end
    end

    @testset "oblique incidence: PressureRelease converges tightly, Rigid more slowly" begin
        angle = 1.2
        ts_pr = AS.target_strength(AS.modal(
            cyl_straight, AS.PressureRelease(), k; incidence_angle = angle, m_max = 30))
        ts_pr_mfs = AS.target_strength(AS.mfs(cyl_bent, AS.PressureRelease(), k;
            incidence_angle = angle, offset = 0.3radius, n_s = 40, n_φ = 32))
        @test ts_pr_mfs ≈ ts_pr atol = 0.3

        ts_rigid = AS.target_strength(AS.modal(
            cyl_straight, AS.Rigid(), k; incidence_angle = angle, m_max = 30))
        ts_rigid_mfs = AS.target_strength(AS.mfs(cyl_bent, AS.Rigid(), k;
            incidence_angle = angle, offset = 0.5radius, n_s = 100, n_φ = 80))
        @test ts_rigid_mfs ≈ ts_rigid atol = 0.7
    end
end

@testset "fluid shell sphere BEM (vs modal series)" begin
    rho_ext, c_ext = 1026.8, 1477.4
    a = 0.05
    rr = 0.9
    k = 2pi * 12000.0 / c_ext

    sphere = AS.Sphere(a)
    bc_g = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext),
        AS.FluidInterior(1.24 / rho_ext, 345.0 / c_ext), rr)
    ts_modal = AS.target_strength(AS.modal(sphere, bc_g, k; m_max = 20))
    diffs_g = Float64[]
    for n in (24, 40)
        sol = AS.bem(sphere, bc_g, k; n = n, rtol = 1e-5)
        push!(diffs_g, abs(AS.target_strength(sol) - ts_modal))
    end
    @test issorted(diffs_g, rev = true)
    @test diffs_g[end] < 0.05

    bc_pr = AS.Shelled(AS.FluidLayer(1028.9 / rho_ext, 1480.3 / c_ext), AS.VacuumInterior(), rr)
    ts_modal_pr = AS.target_strength(AS.modal(sphere, bc_pr, k; m_max = 20))
    diffs_pr = Float64[]
    for n in (24, 40)
        sol = AS.bem(sphere, bc_pr, k; n = n, rtol = 1e-5)
        push!(diffs_pr, abs(AS.target_strength(sol) - ts_modal_pr))
    end
    @test issorted(diffs_pr, rev = true)
    @test diffs_pr[end] < 0.02
end

@testset "AxisymmetricBEM (sphere, axial incidence)" begin
    a = 0.01
    c_water = 1477.4

    mesh = AS.sphere_mesh(a, 10)
    ps = AS.panels(mesh)
    n = length(ps)
    k_static = 1e-6
    row_sums = Vector{ComplexF64}(undef, n)
    for i in 1:n
        xρ, xz = ps[i].rhom, ps[i].zm
        total = zero(ComplexF64)
        for j in 1:n
            pj = ps[j]
            integrand = s -> begin
                ρ2, z2 = AS._panel_point(pj, s)
                AS._azimuthal_dGdn(k_static, xρ, xz, ρ2, z2, pj.nrho, pj.nz) * ρ2 * pj.L
            end
            total += i == j ? AS.quadgk(integrand, 0.0, 0.5, 1.0; rtol = 1e-6)[1] :
                     AS.quadgk(integrand, 0.0, 1.0; rtol = 1e-6)[1]
        end
        row_sums[i] = total
    end
    @test all(rs -> isapprox(rs, -0.5 + 0im; atol = 1e-4), row_sums)

    freq = 38000.0
    k = 2pi * freq / c_water

    sphere = AS.Sphere(a)
    for boundary in (AS.Rigid(), AS.PressureRelease())
        @testset "$(typeof(boundary))" begin
            ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))

            sol16 = AS.bem(
                sphere, boundary, k; n = 16, incidence_angle = 0.0, rtol = 1e-5)
            ts_back_16 = AS.target_strength(sol16; angle = pi)
            ts_fwd_16 = AS.target_strength(sol16; angle = 0.0)

            sol32 = AS.bem(
                sphere, boundary, k; n = 32, incidence_angle = 0.0, rtol = 1e-5)
            ts_back_32 = AS.target_strength(sol32; angle = pi)

            @test ts_back_16 ≈ ts_modal atol = 0.06
            @test ts_back_32 ≈ ts_modal atol = 0.02
            @test abs(ts_back_32 - ts_modal) < abs(ts_back_16 - ts_modal)

            ts_modal_fwd = AS.target_strength(AS.modal(sphere, boundary, k; angle = 0.0))
            @test ts_fwd_16 ≈ ts_modal_fwd atol = 0.06
        end
    end
end

@testset "Axisymmetric MFS (sphere, axial incidence)" begin
    a = 0.01
    c_water = 1477.4
    freq = 38000.0
    k = 2pi * freq / c_water
    sphere = AS.Sphere(a)

    for boundary in (AS.Rigid(), AS.PressureRelease())
        ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))
        for offset_frac in (0.2, 0.5, 0.9)
            ts_mfs = AS.target_strength(AS.mfs(
                sphere, boundary, k; incidence_angle = 0.0, offset = offset_frac * a))
            @test ts_mfs ≈ ts_modal atol = 0.1
        end
    end
end

@testset "Axisymmetric MFS (spheroid, axial incidence)" begin
    a, b = 0.05, 0.02
    c_water = 1477.4
    freq = 20000.0
    k = 2pi * freq / c_water
    body = AS.Spheroid(a, b)

    for boundary in (AS.Rigid(), AS.PressureRelease())
        ts_modal = AS.target_strength(AS.modal(
            body, boundary, k; incidence_angle = 0.0, m_max = 24, n_max = 24))
        for offset_frac in (0.2, 0.5)
            ts_mfs = AS.target_strength(AS.mfs(
                body, boundary, k; incidence_angle = 0.0,
                offset = offset_frac * min(a, b)))
            @test ts_mfs ≈ ts_modal atol = 0.1
        end
    end
end

@testset "Axisymmetric MFS (cylinder with spheroidal endcaps, axial incidence)" begin
    radius, cyl_length, endcap_depth = 0.01, 0.05, 0.01
    c_water = 1477.3
    freq = 38000.0
    k = 2pi * freq / c_water
    mesh = AS.cylinder_spheroidal_endcap_mesh(radius, cyl_length, endcap_depth, 112)
    capped_cyl = AS.Cylinder(radius, cyl_length; endcap_depth = endcap_depth)

    for boundary in (AS.Rigid(), AS.PressureRelease())
        p_bem, dpdn_bem, ps_bem = AS.solve_axial(boundary, k, mesh; rtol = 1e-5)
        ts_bem = AS.target_strength(ps_bem, p_bem, dpdn_bem, k, pi)
        for offset_frac in (0.1, 0.2, 0.3, 0.5)
            ts_mfs = AS.target_strength(AS.mfs(
                capped_cyl, boundary, k; incidence_angle = 0.0,
                offset = offset_frac * radius, n = 112))
            @test ts_mfs ≈ ts_bem atol = 0.05
        end
    end
end

@testset "Axisymmetric MFS (sphere, oblique incidence / rotational symmetry)" begin
    a = 0.01
    c_water = 1477.4
    freq = 38000.0
    k = 2pi * freq / c_water
    sphere = AS.Sphere(a)

    for boundary in (AS.Rigid(), AS.PressureRelease())
        ts_modal = AS.target_strength(AS.modal(sphere, boundary, k))
        for angle_deg in (0.0, 30.0, 60.0, 90.0)
            β = deg2rad(angle_deg)
            sol = AS.mfs(
                sphere, boundary, k; incidence_angle = β, m_max = 15, offset = 0.3a)
            ts_mfs = AS.target_strength(sol; angle = pi - β, azimuth = pi)
            @test ts_mfs ≈ ts_modal atol = 0.1
        end
    end
end

@testset "Axisymmetric MFS fluid-filled/transmission (sphere, axial incidence)" begin
    a = 0.05
    c_water = 1477.4
    freq = 38000.0
    k = 2pi * freq / c_water
    sphere = AS.Sphere(a)

    @testset "rigid limit (gh >> 1)" begin
        stiff = AS.FluidFilled(1e8, 1e8)
        ts_stiff = AS.target_strength(AS.mfs(
            sphere, stiff, k; incidence_angle = 0.0, offset = 0.3a, n = 24))
        ts_rigid = AS.target_strength(AS.mfs(
            sphere, AS.Rigid(), k; incidence_angle = 0.0, offset = 0.3a, n = 24))
        @test ts_stiff ≈ ts_rigid atol = 1e-4
    end

    @testset "pressure-release limit (g << 1)" begin
        soft = AS.FluidFilled(1e-8, 1.0)
        ts_soft = AS.target_strength(AS.mfs(
            sphere, soft, k; incidence_angle = 0.0, offset = 0.3a, n = 24))
        ts_pr = AS.target_strength(AS.mfs(sphere, AS.PressureRelease(), k;
            incidence_angle = 0.0, offset = 0.3a, n = 24))
        @test ts_soft ≈ ts_pr atol = 1e-4
    end

    @testset "converges to the analytical FluidFilled sphere modal series" begin
        g, h = 1.05, 1.02
        bc = AS.FluidFilled(g, h)
        ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
        ts_mfs = AS.target_strength(AS.mfs(
            sphere, bc, k; incidence_angle = 0.0, offset = 0.3a, n = 48))
        @test abs(ts_mfs - ts_modal) < 0.08
    end
end

@testset "Axisymmetric MFS fluid-filled/transmission (sphere, oblique incidence)" begin
    a = 0.05
    c_water = 1477.4
    freq = 38000.0
    k = 2pi * freq / c_water
    sphere = AS.Sphere(a)

    @testset "rigid limit (gh >> 1)" begin
        stiff = AS.FluidFilled(1e8, 1e8)
        for angle_deg in (0.0, 30.0, 60.0, 90.0)
            β = deg2rad(angle_deg)
            sol_stiff = AS.mfs(sphere, stiff, k; incidence_angle = β,
                m_max = 15, offset = 0.3a, n = 24)
            ts_stiff = AS.target_strength(sol_stiff; angle = pi - β, azimuth = pi)

            sol_rigid = AS.mfs(sphere, AS.Rigid(), k; incidence_angle = β,
                m_max = 15, offset = 0.3a, n = 24)
            ts_rigid = AS.target_strength(sol_rigid; angle = pi - β, azimuth = pi)
            @test ts_stiff ≈ ts_rigid atol = 1e-4
        end
    end

    @testset "pressure-release limit (g << 1)" begin
        soft = AS.FluidFilled(1e-8, 1.0)
        for angle_deg in (0.0, 30.0, 60.0, 90.0)
            β = deg2rad(angle_deg)
            sol_soft = AS.mfs(
                sphere, soft, k; incidence_angle = β, m_max = 15, offset = 0.3a, n = 24)
            ts_soft = AS.target_strength(sol_soft; angle = pi - β, azimuth = pi)

            sol_pr = AS.mfs(sphere, AS.PressureRelease(), k; incidence_angle = β,
                m_max = 15, offset = 0.3a, n = 24)
            ts_pr = AS.target_strength(sol_pr; angle = pi - β, azimuth = pi)
            @test ts_soft ≈ ts_pr atol = 1e-4
        end
    end

    @testset "sphere rotational symmetry against the analytical modal series" begin
        g, h = 1.05, 1.02
        bc = AS.FluidFilled(g, h)
        ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
        for angle_deg in (0.0, 30.0, 60.0, 90.0)
            β = deg2rad(angle_deg)
            sol = AS.mfs(
                sphere, bc, k; incidence_angle = β, m_max = 15, offset = 0.3a, n = 48)
            ts_mfs = AS.target_strength(sol; angle = pi - β, azimuth = pi)
            @test abs(ts_mfs - ts_modal) < 0.08
        end
    end
end

@testset "AxisymmetricBEM fluid-filled/transmission (sphere, axial incidence)" begin
    a = 0.05
    c_water = 1477.4
    freq = 38000.0
    k = 2pi * freq / c_water
    sphere = AS.Sphere(a)

    @testset "rigid limit (gh >> 1)" begin
        stiff = AS.FluidFilled(1e8, 1e8)
        ts_stiff = AS.target_strength(AS.bem(
            sphere, stiff, k; incidence_angle = 0.0, n = 24))
        ts_rigid = AS.target_strength(AS.bem(
            sphere, AS.Rigid(), k; incidence_angle = 0.0, n = 24))
        @test ts_stiff ≈ ts_rigid atol = 0.25
    end

    @testset "pressure-release limit (g << 1)" begin
        soft = AS.FluidFilled(1e-8, 1.0)
        ts_soft = AS.target_strength(AS.bem(
            sphere, soft, k; incidence_angle = 0.0, n = 24))
        ts_pr = AS.target_strength(AS.bem(
            sphere, AS.PressureRelease(), k; incidence_angle = 0.0, n = 24))
        @test ts_soft ≈ ts_pr atol = 0.1
    end

    @testset "converges to the analytical FluidFilled sphere modal series" begin
        g, h = 1.05, 1.02
        bc = AS.FluidFilled(g, h)
        ts_modal = AS.target_strength(AS.modal(sphere, bc, k))
        ts_bem = AS.target_strength(AS.bem(
            sphere, bc, k; incidence_angle = 0.0, n = 48))
        @test abs(ts_bem - ts_modal) < 0.15
    end
end
