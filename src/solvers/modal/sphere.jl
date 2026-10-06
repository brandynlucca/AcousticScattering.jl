function _default_mode_count(ka::Real)
    return max(20, ceil(Int, 2.5 * ka + 4 * cbrt(ka + 1) + 10))
end

function _modal_coefficient(::Rigid, m::Integer, k::Real, a::Real)
    ka = k * a
    return -jsd(m, ka) / hsd(m, ka)
end

function _modal_coefficient(::PressureRelease, m::Integer, k::Real, a::Real)
    ka = k * a
    return -js(m, ka) / hs(m, ka)
end

function _modal_coefficient(bc::Impedance, m::Integer, k::Real, a::Real)
    ka = k * a
    return -(jsd(m, ka) + im * js(m, ka) / bc.zeta) /
           (hsd(m, ka) + im * hs(m, ka) / bc.zeta)
end

function _modal_coefficient(bc::FluidFilled, m::Integer, k::Real, a::Real)
    return _sphere_fluid_coefficients(bc, m, k, a).scattered
end

function _sphere_fluid_coefficients(bc::FluidFilled, m::Integer, k::Real, a::Real)
    ka = k * a
    ka_interior = ka / bc.soundspeed_contrast
    gh = bc.density_contrast * bc.soundspeed_contrast

    jd_int = jsd(m, ka_interior)
    jd_ext = jsd(m, ka)
    j_int = js(m, ka_interior)
    j_ext = js(m, ka)
    h_ext = hs(m, ka)
    hd_ext = hsd(m, ka)
    denominator = gh * j_int * hd_ext - h_ext * jd_int
    scattered = (j_ext * jd_int - gh * j_int * jd_ext) / denominator
    interior = gh * (j_ext * hd_ext - jd_ext * h_ext) / denominator
    return (; scattered, interior)
end

function _sphere_interface_solution(matrix, rhs)
    columns = maximum(abs, matrix; dims = 1)
    scaled = matrix ./ columns
    rows = maximum(abs, scaled; dims = 2)
    return (scaled ./ rows \ (rhs ./ vec(rows))) ./ vec(columns)
end

function _modal_coefficient(
        bc::Union{Shelled{FluidLayer, VacuumInterior},
            Shelled{FluidLayer, FluidInterior}, Shelled{ElasticLayer, FluidInterior}},
        m::Integer, k::Real, a::Real)
    return _sphere_shell_coefficients(bc, m, k, a).scattered
end

function _modal_coefficient(bc::Shelled{<:LayeredMaterial}, m::Integer, k::Real, a::Real)
    return _sphere_shell_coefficients(bc, m, k, a).scattered
end

function _sphere_shell_coefficients(bc::Shelled{<:LayeredMaterial}, m::Integer,
        k::Real, a::Real)
    layers, _ = _layered_materials(bc.material)
    all(layer -> layer isa FluidLayer, layers) &&
        return _sphere_fluid_stack_coefficients(bc, m, k, a)
    return _sphere_mixed_stack_coefficients(bc, layers, m, k, a)
end

function _sphere_fluid_stack_coefficients(bc::Shelled{<:LayeredMaterial}, m::Integer,
        k::Real, a::Real)
    layers, ratios = _fluid_layers(bc.material)
    nlayer = length(layers)
    radii = a .* vcat(ratios, bc.radius_ratio)
    nunknown = 1 + 2nlayer + (bc.interior isa FluidInterior ? 1 : 0)
    matrix = zeros(ComplexF64, nunknown, nunknown)
    rhs = zeros(ComplexF64, nunknown)
    for interface in 1:nlayer
        r = radii[interface]
        kr = k * r
        row = 2interface - 1
        if interface == 1
            matrix[row, 1] = hs(m, kr)
            matrix[row + 1, 1] = hsd(m, kr)
            rhs[row] = -js(m, kr)
            rhs[row + 1] = -jsd(m, kr)
        else
            outer = layers[interface - 1]
            outer_kr = kr / outer.soundspeed_contrast
            outer_gh = outer.density_contrast * outer.soundspeed_contrast
            col = 2interface - 2
            matrix[row, col:(col + 1)] = [js(m, outer_kr), ys(m, outer_kr)]
            matrix[row + 1, col:(col + 1)] = [jsd(m, outer_kr), ysd(m, outer_kr)] ./
                                             outer_gh
        end
        inner = layers[interface]
        inner_kr = kr / inner.soundspeed_contrast
        inner_gh = inner.density_contrast * inner.soundspeed_contrast
        col = 2interface
        matrix[row, col:(col + 1)] = [-js(m, inner_kr), -ys(m, inner_kr)]
        matrix[row + 1, col:(col + 1)] = [-jsd(m, inner_kr), -ysd(m, inner_kr)] ./ inner_gh
    end
    r = radii[end]
    kr = k * r / layers[end].soundspeed_contrast
    col = 2nlayer
    row = 2nlayer + 1
    matrix[row, col:(col + 1)] = [js(m, kr), ys(m, kr)]
    if bc.interior isa FluidInterior
        core_kr = k * r / bc.interior.soundspeed_contrast
        matrix[row, end] = -js(m, core_kr)
        gh = layers[end].density_contrast * layers[end].soundspeed_contrast
        core_gh = bc.interior.density_contrast * bc.interior.soundspeed_contrast
        matrix[row + 1, col:(col + 1)] = [jsd(m, kr), ysd(m, kr)] ./ gh
        matrix[row + 1, end] = -jsd(m, core_kr) / core_gh
    end
    x = _sphere_interface_solution(matrix, rhs)
    shells = [(x[2i], x[2i + 1]) for i in 1:nlayer]
    return (; scattered = x[1], shell = shells,
        interior = bc.interior isa FluidInterior ? x[end] : nothing)
end

function _sphere_shell_coefficients(bc::Shelled{FluidLayer}, m::Integer, k::Real, a::Real)
    ka = k * a
    k2a = ka / bc.material.soundspeed_contrast
    k2b = k2a * bc.radius_ratio
    gh_shell = bc.material.density_contrast * bc.material.soundspeed_contrast
    n = bc.interior isa FluidInterior ? 4 : 3
    matrix = zeros(ComplexF64, n, n)
    rhs = zeros(ComplexF64, n)
    matrix[1, 1:3] = [hs(m, ka), -js(m, k2a), -ys(m, k2a)]
    matrix[2, 1:3] = [hsd(m, ka), -jsd(m, k2a) / gh_shell, -ysd(m, k2a) / gh_shell]
    matrix[3, 2:3] = [js(m, k2b), ys(m, k2b)]
    rhs[1:2] = [-js(m, ka), -jsd(m, ka)]
    if bc.interior isa FluidInterior
        k3b = ka * bc.radius_ratio / bc.interior.soundspeed_contrast
        gh_int = bc.interior.density_contrast * bc.interior.soundspeed_contrast
        matrix[3, 4] = -js(m, k3b)
        matrix[4, 2:4] = [
            jsd(m, k2b) / gh_shell, ysd(m, k2b) / gh_shell, -jsd(m, k3b) / gh_int]
    end
    x = _sphere_interface_solution(matrix, rhs)
    return (; scattered = x[1], shell = (x[2], x[3]), interior = n == 4 ? x[4] : nothing)
end

# λ/(λ+2G) = 1 - 2β, 2G/(λ+2G) = 2β, where β = (c_T/c_L)²
function _sphere_shell_coefficients(bc::Shelled{ElasticLayer, FluidInterior}, m::Integer, k::Real, a::Real)
    a_in = bc.radius_ratio * a
    ka1s = k * a
    kLs = (k / bc.material.speed_longitudinal_contrast) * a
    kTs = (k / bc.material.speed_transversal_contrast) * a
    kLf = (k / bc.material.speed_longitudinal_contrast) * a_in
    kTf = (k / bc.material.speed_transversal_contrast) * a_in
    β = (bc.material.speed_transversal_contrast / bc.material.speed_longitudinal_contrast)^2
    inv_rho_shell = 1 / bc.material.density_contrast

    # See ElasticLayer's docstring for interior_coupling.
    if bc.material.interior_coupling === :generalized
        k3f = (k / bc.interior.soundspeed_contrast) * a_in
        rho_int_over_shell = bc.interior.density_contrast / bc.material.density_contrast
    else
        k3f = k * a_in
        rho_int_over_shell = inv_rho_shell
    end

    j(x) = js(m, x)
    jd(x) = jsd(m, x)
    jdd(x) = jsdd(m, x)
    y(x) = ys(m, x)
    yd(x) = ysd(m, x)
    ydd(x) = ysdd(m, x)

    a1 = j(ka1s) * inv_rho_shell
    a2 = ka1s * jd(ka1s)
    a11 = hs(m, ka1s) * inv_rho_shell
    a21 = ka1s * hsd(m, ka1s)

    a12 = (1 - 2β) * j(kLs) - 2β * jdd(kLs)
    a22 = kLs * jd(kLs)
    a32 = 2 * (kLs * jd(kLs) - j(kLs))
    a42 = (1 - 2β) * j(kLf) - 2β * jdd(kLf)
    a52 = kLf * jd(kLf)
    a62 = 2 * (kLf * jd(kLf) - j(kLf))

    a13 = -2m * (m + 1) * kTs^(-2) * (kTs * jd(kTs) - j(kTs))
    a23 = m * (m + 1) * j(kTs)
    a33 = kTs^2 * jdd(kTs) + (m + 2) * (m - 1) * j(kTs)
    a43 = -2m * (m + 1) * kTf^(-2) * (kTf * jd(kTf) - j(kTf))
    a53 = m * (m + 1) * j(kTf)
    a63 = kTf^2 * jdd(kTf) + (m + 2) * (m - 1) * j(kTf)

    a14 = (1 - 2β) * y(kLs) - 2β * ydd(kLs)
    a24 = kLs * yd(kLs)
    a34 = 2 * (kLs * yd(kLs) - y(kLs))
    a44 = (1 - 2β) * y(kLf) - 2β * ydd(kLf)
    a54 = kLf * yd(kLf)
    a64 = 2 * (kLf * yd(kLf) - y(kLf))

    a15 = -2m * (m + 1) * kTs^(-2) * (kTs * yd(kTs) - y(kTs))
    a25 = m * (m + 1) * y(kTs)
    a35 = kTs^2 * ydd(kTs) + (m + 2) * (m - 1) * y(kTs)
    a45 = -2m * (m + 1) * kTf^(-2) * (kTf * yd(kTf) - y(kTf))
    a55 = m * (m + 1) * y(kTf)
    a65 = kTf^2 * ydd(kTf) + (m + 2) * (m - 1) * y(kTf)

    # Interior fluid's own regular solution, at its own wavenumber k3f.
    a46 = j(k3f) * rho_int_over_shell
    a56 = k3f * jd(k3f)

    if m == 0
        A_numerator = zeros(ComplexF64, 4, 4)
        A_numerator[1, 1:3] = [a1, a12, a14]
        A_numerator[2, 1:3] = [a2, a22, a24]
        A_numerator[3, 2:4] = [a42, a44, a46]
        A_numerator[4, 2:4] = [a52, a54, a56]
    else
        A_numerator = zeros(ComplexF64, 6, 6)
        A_numerator[1, 1:5] = [a1, a12, a13, a14, a15]
        A_numerator[2, 1:5] = [a2, a22, a23, a24, a25]
        A_numerator[3, 2:5] = [a32, a33, a34, a35]
        A_numerator[4, 2:6] = [a42, a43, a44, a45, a46]
        A_numerator[5, 2:6] = [a52, a53, a54, a55, a56]
        A_numerator[6, 2:5] = [a62, a63, a64, a65]
    end
    A_denominator = copy(A_numerator)
    A_denominator[1:2, 1] = [a11, a21]

    x = _sphere_interface_solution(A_denominator, -A_numerator[:, 1])
    density = bc.material.interior_coupling === :generalized ?
              bc.interior.density_contrast : 1.0
    return (; scattered = x[1], interior = density * x[end])
end

# Resonance form, see SolidElastic's docstring.
function _modal_coefficient(bc::SolidElastic, m::Integer, k::Real, a::Real)
    ka_sw = k * a
    ka_l = (k / bc.speed_longitudinal_contrast) * a
    ka_t = (k / bc.speed_transversal_contrast) * a
    g = bc.density_contrast

    js_sw = js(m, ka_sw)
    jsd_sw = jsd(m, ka_sw)
    ys_sw = ys(m, ka_sw)
    ysd_sw = ysd(m, ka_sw)
    js_l = js(m, ka_l)
    jsd_l = jsd(m, ka_l)
    js_t = js(m, ka_t)
    jsd_t = jsd(m, ka_t)

    tan_sw = -ka_sw * jsd_sw / js_sw
    tan_l = -ka_l * jsd_l / js_l
    tan_t = -ka_t * jsd_t / js_t
    tan_beta = -ka_sw * ysd_sw / ys_sw
    tan_diff = -js_sw / ys_sw

    along_m = m * (m + 1)
    tan_l_add = tan_l + 1
    tan_t_div = along_m - 1 - ka_t^2 / 2 + tan_t
    numerator = tan_l / tan_l_add - along_m / tan_t_div
    denominator1 = (along_m - ka_t^2 / 2 + 2 * tan_l) / tan_l_add
    denominator2 = along_m * (tan_t + 1) / tan_t_div
    denominator = denominator1 - denominator2
    ratio = -0.5 * ka_t^2 * numerator / denominator
    phi = -ratio / g

    eta_tan = tan_diff * (phi + tan_sw) / (phi + tan_beta)
    cos_eta = 1 / sqrt(1 + eta_tan^2)
    sin_eta = eta_tan * cos_eta

    return sin_eta * (-im * cos_eta - sin_eta)
end

"""
    form_function(boundary::AbstractBoundaryCondition, k, a; angle=π, m_max=default)

Far-field scattering amplitude f(θ) in m of a sphere of radius `a` in m in a
medium with wavenumber `k` in 1/m, evaluated at scattering angle `angle`
in rad (default π, backscatter). `m_max` truncates the modal sum. The
default grows with `ka` and can be overridden for tighter/looser
convergence control.
"""
function form_function(boundary::AbstractBoundaryCondition, k::Real, a::Real;
        angle::Real = π, m_max::Integer = _default_mode_count(k * a))
    x = cos(angle)
    total = zero(ComplexF64)
    for m in 0:m_max
        Am = _modal_coefficient(boundary, m, k, a)
        total += (2m + 1) * legendre_p(m, x) * Am
    end
    return -im / k * total
end

"""
    target_strength(boundary::AbstractBoundaryCondition, k, a; angle=π, m_max=default)

Target strength [dB re 1 m²] of a sphere under modal-series scattering
(rigid, pressure-release, or fluid/gas-filled boundary conditions).
"""
function target_strength(boundary::AbstractBoundaryCondition, k::Real, a::Real; kwargs...)
    return target_strength(form_function(boundary, k, a; kwargs...))
end
