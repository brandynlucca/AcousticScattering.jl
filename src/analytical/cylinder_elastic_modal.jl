# Elastic cylindrical shell and solid elastic cylinder modal coefficients (ECMS), reusing
# ElasticLayer/SolidElastic from sphere_modal.jl. Stanton (1988) / Doolittle & Überall (1966).

function _elastic_cylinder_shell_matrix(
        m::Integer, ω::Real, ρ_shell::Real, cL::Real, cT::Real,
        b::Real, ρ_int::Real, c_int::Real)
    μ = ρ_shell * cT^2
    kL = ω / cL
    kT = ω / cT
    k3 = ω / c_int
    xLa, xTa = kL, kT
    xLb, xTb = kL * b, kT * b
    x3b = k3 * b

    M = zeros(ComplexF64, 6, 6)
    rhs = zeros(ComplexF64, 6)

    # Row 1: σrr(shell, a) + p1(a) = 0  (σrr = -p at the fluid-solid interface)
    M[1, 1] = besselh(m, 1, ω)
    M[1, 2] = -ρ_shell * ω^2 * besselj(m, xLa) - 2μ * kL * jcd(m, xLa) +
              2μ * m^2 * besselj(m, xLa)
    M[1, 3] = -ρ_shell * ω^2 * bessely(m, xLa) - 2μ * kL * ycd(m, xLa) +
              2μ * m^2 * bessely(m, xLa)
    M[1, 4] = 2μ * m * kT * jcd(m, xTa) - 2μ * m * besselj(m, xTa)
    M[1, 5] = 2μ * m * kT * ycd(m, xTa) - 2μ * m * bessely(m, xTa)
    rhs[1] = -besselj(m, ω)

    # Row 2: radial displacement continuity at a
    M[2, 1] = -(1 / ω) * hcd(m, ω)
    M[2, 2] = kL * jcd(m, xLa)
    M[2, 3] = kL * ycd(m, xLa)
    M[2, 4] = m * besselj(m, xTa)
    M[2, 5] = m * bessely(m, xTa)
    rhs[2] = (1 / ω) * jcd(m, ω)

    # Row 3: σrθ(shell, a) = 0
    M[3, 2] = μ * (-2m * kL * jcd(m, xLa) + 2m * besselj(m, xLa))
    M[3, 3] = μ * (-2m * kL * ycd(m, xLa) + 2m * bessely(m, xLa))
    M[3, 4] = μ * (kT^2 * besselj(m, xTa) - 2m^2 * besselj(m, xTa) + 2kT * jcd(m, xTa))
    M[3, 5] = μ * (kT^2 * bessely(m, xTa) - 2m^2 * bessely(m, xTa) + 2kT * ycd(m, xTa))

    # Row 4: σrr(shell, b) + p3(b) = 0
    M[4, 2] = -ρ_shell * ω^2 * besselj(m, xLb) - 2μ * kL * jcd(m, xLb) +
              2μ * m^2 * besselj(m, xLb)
    M[4, 3] = -ρ_shell * ω^2 * bessely(m, xLb) - 2μ * kL * ycd(m, xLb) +
              2μ * m^2 * bessely(m, xLb)
    M[4, 4] = 2μ * m * kT * jcd(m, xTb) - 2μ * m * besselj(m, xTb)
    M[4, 5] = 2μ * m * kT * ycd(m, xTb) - 2μ * m * bessely(m, xTb)
    M[4, 6] = besselj(m, x3b)

    # Row 5: radial displacement continuity at b
    M[5, 2] = kL * jcd(m, xLb)
    M[5, 3] = kL * ycd(m, xLb)
    M[5, 4] = m * besselj(m, xTb)
    M[5, 5] = m * bessely(m, xTb)
    M[5, 6] = -(1 / (ω * c_int * ρ_int)) * jcd(m, x3b)

    # Row 6: σrθ(shell, b) = 0
    M[6, 2] = μ * (-2m * kL * jcd(m, xLb) + 2m * besselj(m, xLb))
    M[6, 3] = μ * (-2m * kL * ycd(m, xLb) + 2m * bessely(m, xLb))
    M[6, 4] = μ * (kT^2 * besselj(m, xTb) - 2m^2 * besselj(m, xTb) + 2kT * jcd(m, xTb))
    M[6, 5] = μ * (kT^2 * bessely(m, xTb) - 2m^2 * bessely(m, xTb) + 2kT * ycd(m, xTb))

    return M, rhs
end

function _raw_bn(bc::Shelled{ElasticLayer, FluidInterior}, m::Integer, k1a::Real)
    M, rhs = _elastic_cylinder_shell_matrix(m, Float64(k1a), bc.material.density_contrast,
        bc.material.speed_longitudinal_contrast, bc.material.speed_transversal_contrast,
        bc.radius_ratio, bc.interior.density_contrast, bc.interior.soundspeed_contrast)
    M_num = copy(M)
    M_num[:, 1] = rhs
    return det(M_num) / det(M)
end

function _raw_bn(bc::SolidElastic, m::Integer, k1a::Real)
    ω = Float64(k1a)
    ρ = bc.density_contrast
    cL, cT = bc.speed_longitudinal_contrast, bc.speed_transversal_contrast
    μ = ρ * cT^2
    kL, kT = ω / cL, ω / cT
    xL, xT = kL, kT

    M = zeros(ComplexF64, 3, 3)
    rhs = zeros(ComplexF64, 3)

    M[1, 1] = besselh(m, 1, ω)
    M[1, 2] = -ρ * ω^2 * besselj(m, xL) - 2μ * kL * jcd(m, xL) + 2μ * m^2 * besselj(m, xL)
    M[1, 3] = 2μ * m * kT * jcd(m, xT) - 2μ * m * besselj(m, xT)
    rhs[1] = -besselj(m, ω)

    M[2, 1] = -(1 / ω) * hcd(m, ω)
    M[2, 2] = kL * jcd(m, xL)
    M[2, 3] = m * besselj(m, xT)
    rhs[2] = (1 / ω) * jcd(m, ω)

    M[3, 2] = μ * (-2m * kL * jcd(m, xL) + 2m * besselj(m, xL))
    M[3, 3] = μ * (kT^2 * besselj(m, xT) - 2m^2 * besselj(m, xT) + 2kT * jcd(m, xT))

    M_num = copy(M)
    M_num[:, 1] = rhs
    return det(M_num) / det(M)
end

function _cylinder_modal_coefficient(
        bc::Union{Shelled{ElasticLayer, FluidInterior}, SolidElastic}, m::Integer, k1a::Real)
    return -neumann_factor(m) * (-1)^m * _raw_bn(bc, m, k1a)
end
