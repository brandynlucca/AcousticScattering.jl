# Viscoelastic Scattering Model (VESM), Feuillade & Nero (1998), JASA 103(6), 3245-3255. Generalizes
# `ElasticLayer` with a viscous flesh layer. Monopole (m=0) only, see `_modal_coefficient`'s `m >= 1` branch.
# Boundary condition type is `Shelled{LayeredMaterial{ViscousLayer,ElasticLayer},FluidInterior}` (sphere_modal.jl).

const _VESMShell = Shelled{LayeredMaterial{ViscousLayer, ElasticLayer}, FluidInterior}

# β-normalized normal-stress/displacement radial terms shared by both shell layers, β = kL²/kT².
function _vesm_solid_stress_terms(kL::Number, β::Number, r::Real)
    x = kL * r
    return (1 - 2β) * js(0, x) - 2β * jsdd(0, x), (1 - 2β) * ys(0, x) - 2β * ysdd(0, x)
end
function _vesm_solid_displacement_terms(kL::Number, r::Real)
    x = kL * r
    return x * jsd(0, x), x * ysd(0, x)
end

# Flesh's complex compressional wavenumber (Eq. 6) and β2 = kL2²/kT2² (Eq. 7), computed via the
# direct closed form kT2² = iω/ν2 rather than sqrt-then-square, to keep the ν2 → 0 limit exact.
function _vesm_flesh_wavenumbers(bc::_VESMShell, ω::Real)
    flesh = bc.material.outer
    c2 = flesh.soundspeed_contrast * flesh.soundspeed_exterior
    kL2 = (ω / c2) /
          sqrt(complex(1 - im * ω * flesh.kinematic_viscosity_compressional / c2^2))
    β2 = kL2^2 * flesh.kinematic_viscosity_shear / (im * ω)
    return kL2, β2
end

# General-m (m >= 1) compressional-potential radial terms, Feuillade & Nero (1998) Eqs. 19-22.
# Returns (stress_j, disp_j, tang_j, tangdisp_j, stress_y, disp_y, tang_y, tangdisp_y).
function _vesm_compressional_terms(m::Integer, kL::Number, β::Number, r::Real)
    x = kL * r
    j, jd, jdd = js(m, x), jsd(m, x), jsdd(m, x)
    y, yd, ydd = ys(m, x), ysd(m, x), ysdd(m, x)
    stress_j = (1 - 2β) * j - 2β * jdd
    stress_y = (1 - 2β) * y - 2β * ydd
    disp_j = x * jd
    disp_y = x * yd
    tang_j = 2 * (x * jd - j)
    tang_y = 2 * (x * yd - y)
    return stress_j, disp_j, tang_j, j, stress_y, disp_y, tang_y, y
end

# General-m (m >= 1) shear-potential radial terms, same 8-tuple shape as `_vesm_compressional_terms`.
function _vesm_shear_terms(m::Integer, kT::Number, r::Real)
    x = kT * r
    j, jd, jdd = js(m, x), jsd(m, x), jsdd(m, x)
    y, yd, ydd = ys(m, x), ysd(m, x), ysdd(m, x)
    mm1 = m * (m + 1)
    stress_j = -2mm1 * x^(-2) * (x * jd - j)
    stress_y = -2mm1 * x^(-2) * (x * yd - y)
    disp_j = mm1 * j
    disp_y = mm1 * y
    tang_j = x^2 * jdd + (m + 2) * (m - 1) * j
    tang_y = x^2 * ydd + (m + 2) * (m - 1) * y
    tangdisp_j = -(x * jd + j)
    tangdisp_y = -(x * yd + y)
    return stress_j, disp_j, tang_j, tangdisp_j, stress_y, disp_y, tang_y, tangdisp_y
end

# General-m (m >= 1) modal coefficient: a 10x10 boundary-condition system across three interfaces.
# NOT validated and NOT exposed (see the module comment), fails a physical sanity check at low ka.
function _vesm_general_modal_coefficient(
        bc::_VESMShell, m::Integer, k::Real, a::Real,
        Rv::Real, Re::Real, R::Real,
        kL2::Number, β2::Number, kT2::Number,
        kL3::Number, kT3::Number, β3::Number, k4::Number)
    flesh, wall, core = bc.material.outer, bc.material.inner, bc.interior
    inv_rho_flesh = 1 / flesh.density_contrast
    x1 = k * Rv
    a1 = js(m, x1) * inv_rho_flesh
    a2 = x1 * jsd(m, x1)
    a11 = hs(m, x1) * inv_rho_flesh
    a21 = x1 * hsd(m, x1)

    sj2v, dj2v, tj2v, tdj2v, sy2v, dy2v, ty2v, tdy2v = _vesm_compressional_terms(m, kL2, β2, Rv)
    ssj2v, sdj2v, stj2v, stdj2v, ssy2v, sdy2v, sty2v, stdy2v = _vesm_shear_terms(m, kT2, Rv)

    sj2e, dj2e, tj2e, tdj2e, sy2e, dy2e, ty2e, tdy2e = _vesm_compressional_terms(m, kL2, β2, Re)
    ssj2e, sdj2e, stj2e, stdj2e, ssy2e, sdy2e, sty2e, stdy2e = _vesm_shear_terms(m, kT2, Re)
    sj3e, dj3e, tj3e, tdj3e, sy3e, dy3e, ty3e, tdy3e = _vesm_compressional_terms(m, kL3, β3, Re)
    ssj3e, sdj3e, stj3e, stdj3e, ssy3e, sdy3e, sty3e, stdy3e = _vesm_shear_terms(m, kT3, Re)

    sj3R, dj3R, tj3R, _, sy3R, dy3R, ty3R, _ = _vesm_compressional_terms(m, kL3, β3, R)
    ssj3R, sdj3R, stj3R, _, ssy3R, sdy3R, sty3R, _ = _vesm_shear_terms(m, kT3, R)

    ρf, ρw = flesh.density_contrast, wall.density_contrast
    x4 = k4 * R
    air_stress = js(m, x4) * (core.density_contrast / wall.density_contrast)
    air_disp = x4 * jsd(m, x4)

    A_numerator = zeros(ComplexF64, 10, 10)
    # 1: outer normal stress.  2: outer normal displacement.
    # 3: outer, flesh's own tangential stress = 0.
    A_numerator[1, 1:5] = [a1, sj2v, sy2v, ssj2v, ssy2v]
    A_numerator[2, 1:5] = [a2, dj2v, dy2v, sdj2v, sdy2v]
    A_numerator[3, 2:5] = [tj2v, ty2v, stj2v, sty2v]
    # 4: middle normal stress (ρ-weighted).  5: middle normal displacement.
    # 6: middle tangential stress (ρ-weighted).  7: middle tangential displacement.
    A_numerator[4, 2:9] = [ρf * sj2e, ρf * sy2e, ρf * ssj2e, ρf * ssy2e,
        -ρw * sj3e, -ρw * sy3e, -ρw * ssj3e, -ρw * ssy3e]
    A_numerator[5, 2:9] = [dj2e, dy2e, sdj2e, sdy2e, -dj3e, -dy3e, -sdj3e, -sdy3e]
    A_numerator[6, 2:9] = [ρf * tj2e, ρf * ty2e, ρf * stj2e, ρf * sty2e,
        -ρw * tj3e, -ρw * ty3e, -ρw * stj3e, -ρw * sty3e]
    A_numerator[7, 2:9] = [tdj2e, tdy2e, stdj2e, stdy2e, -tdj3e, -tdy3e, -stdj3e, -stdy3e]
    # 8: inner normal stress.  9: inner normal displacement.
    # 10: inner, wall's own tangential stress = 0.
    A_numerator[8, 6:10] = [sj3R, sy3R, ssj3R, ssy3R, -air_stress]
    A_numerator[9, 6:10] = [dj3R, dy3R, sdj3R, sdy3R, -air_disp]
    A_numerator[10, 6:9] = [tj3R, ty3R, stj3R, sty3R]

    A_denominator = copy(A_numerator)
    A_denominator[1:2, 1] = [a11, a21]

    return det(A_numerator) / det(A_denominator)
end

function _modal_coefficient(bc::_VESMShell, m::Integer, k::Real, a::Real)
    flesh, wall, core = bc.material.outer, bc.material.inner, bc.interior
    Rv = a
    Re = bc.material.radius_ratio * a
    R = bc.radius_ratio * a
    ω = k * flesh.soundspeed_exterior

    kL2, β2 = _vesm_flesh_wavenumbers(bc, ω)
    kL3 = (k / wall.speed_longitudinal_contrast)
    kT3 = (k / wall.speed_transversal_contrast)
    β3 = (wall.speed_transversal_contrast / wall.speed_longitudinal_contrast)^2
    k4 = k / core.soundspeed_contrast

    if m >= 1
        # `_vesm_general_modal_coefficient` fails a physical sanity check, see the module comment.
        throw(ArgumentError(
            "the VESM Shelled boundary only implements the monopole (m=0) term. A general-m " *
            "extension exists (_vesm_general_modal_coefficient) but fails a physical " *
            "sanity check (dominant-monopole assumption badly violated at low ka) and is " *
            "not exposed, see this function's own comment. Use " *
            "target_strength(...; m_max=0) explicitly."))
    end

    # Outer interface (water | flesh), r = Rv, identical in form to ElasticLayer's own outer-boundary terms.
    inv_rho_flesh = 1 / flesh.density_contrast
    x1 = k * Rv
    a1 = js(0, x1) * inv_rho_flesh
    a2 = x1 * jsd(0, x1)
    a11 = hs(0, x1) * inv_rho_flesh
    a21 = x1 * hsd(0, x1)
    a12_v, a14_v = _vesm_solid_stress_terms(kL2, β2, Rv)
    a22_v, a24_v = _vesm_solid_displacement_terms(kL2, Rv)

    # Middle interface (flesh | wall), r = Re, both sides solid-like, tangential terms vanish at m=0.
    a12_v_mid, a14_v_mid = _vesm_solid_stress_terms(kL2, β2, Re)
    a22_v_mid, a24_v_mid = _vesm_solid_displacement_terms(kL2, Re)
    a12_e_mid, a14_e_mid = _vesm_solid_stress_terms(kL3, β3, Re)
    a22_e_mid, a24_e_mid = _vesm_solid_displacement_terms(kL3, Re)
    ρf, ρw = flesh.density_contrast, wall.density_contrast

    # Inner interface (wall | air), r = R, identical to ElasticLayer's own inner-boundary terms.
    a42_e, a44_e = _vesm_solid_stress_terms(kL3, β3, R)
    a52_e, a54_e = _vesm_solid_displacement_terms(kL3, R)
    a46_air = js(0, k4 * R) * (core.density_contrast / wall.density_contrast)
    a56_air = (k4 * R) * jsd(0, k4 * R)

    # Columns: [A1 (water, scattered), A2, B2 (flesh), A3, B3 (wall), A4 (air)].
    A_numerator = zeros(ComplexF64, 6, 6)
    A_numerator[1, :] = [a1, a12_v, a14_v, 0, 0, 0]
    A_numerator[2, :] = [a2, a22_v, a24_v, 0, 0, 0]
    A_numerator[3, :] = [
        0, ρf * a12_v_mid, ρf * a14_v_mid, -ρw * a12_e_mid, -ρw * a14_e_mid, 0]
    A_numerator[4, :] = [0, a22_v_mid, a24_v_mid, -a22_e_mid, -a24_e_mid, 0]
    A_numerator[5, :] = [0, 0, 0, a42_e, a44_e, -a46_air]
    A_numerator[6, :] = [0, 0, 0, a52_e, a54_e, -a56_air]

    A_denominator = copy(A_numerator)
    A_denominator[1:2, 1] = [a11, a21]

    return det(A_numerator) / det(A_denominator)
end

"""
    form_function(boundary::Shelled{LayeredMaterial{ViscousLayer,ElasticLayer},FluidInterior}, k, a; angle=π, m_max=0)

Far-field scattering amplitude of the Feuillade & Nero (1998) viscous-
elastic swimbladder model. `m_max` defaults to `0` (not this file's usual
`ka`-dependent default): see [`Shelled`](@ref)'s docstring and
`_modal_coefficient`'s error message for why higher modes aren't exposed, a general-`m` extension exists but fails a physical sanity check and is
not trusted; pass a nonzero `m_max` only once that's fixed.
"""
function form_function(
        boundary::_VESMShell, k::Real, a::Real; angle::Real = π, m_max::Integer = 0)
    x = cos(angle)
    total = zero(ComplexF64)
    for m in 0:m_max
        Am = _modal_coefficient(boundary, m, k, a)
        total += (2m + 1) * legendre_p(m, x) * Am
    end
    return -im / k * total
end
