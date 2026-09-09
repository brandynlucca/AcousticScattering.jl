# Hayek & Boisvert (2003) axisymmetric nontorsional elastic-shell dynamic-stiffness operator for
# a confocal prolate spheroidal shell, displacement triplet [u, w, β_η] at each meridional node.

"""
    ProlateShellGeometry(semimajor_length, semiminor_length, shell_thickness)

Confocal prolate spheroidal shell geometry: `semimajor_length` and
`semiminor_length` are the *outer* shell surface's semi-axes [m],
`shell_thickness` [m] is the equatorial thickness. The inner surface is
the confocal (same interfocal distance) surface matching that thickness
at the equator; the shell midsurface is the `ξ`-midpoint between the two.
"""
struct ProlateShellGeometry
    semimajor_length::Float64
    semiminor_length::Float64
    shell_thickness::Float64
    focal_radius::Float64
    semimajor_inner::Float64
    semiminor_inner::Float64
    xi_outer::Float64
    xi_inner::Float64
    xi_mid::Float64
    semimajor_mid::Float64
    semiminor_mid::Float64
    a_shape::Float64
    aspect_ratio::Float64
    bending_epsilon::Float64

    function ProlateShellGeometry(semimajor_length::Real, semiminor_length::Real, shell_thickness::Real)
        semimajor_length > 0 || throw(ArgumentError("semimajor_length must be positive"))
        semiminor_length > 0 || throw(ArgumentError("semiminor_length must be positive"))
        semimajor_length > semiminor_length ||
            throw(ArgumentError("Prolate shell requires semimajor_length > semiminor_length"))
        shell_thickness > 0 || throw(ArgumentError("shell_thickness must be positive"))
        shell_thickness < semiminor_length ||
            throw(ArgumentError("shell_thickness must be smaller than the outer semiminor axis"))

        focal_radius = sqrt(semimajor_length^2 - semiminor_length^2)
        semiminor_inner = semiminor_length - shell_thickness
        semimajor_inner = sqrt(focal_radius^2 + semiminor_inner^2)
        xi_outer = semimajor_length / focal_radius
        xi_inner = semimajor_inner / focal_radius
        xi_mid = 0.5 * (xi_outer + xi_inner)
        semimajor_mid = focal_radius * xi_mid
        semiminor_mid = focal_radius * sqrt(xi_mid^2 - 1.0)
        a_shape = semimajor_mid / focal_radius
        aspect_ratio = semimajor_mid / semiminor_mid
        bending_epsilon = (shell_thickness / semimajor_mid)^2 / 12.0

        return new(
            Float64(semimajor_length), Float64(semiminor_length), Float64(shell_thickness),
            focal_radius, semimajor_inner, semiminor_inner, xi_outer, xi_inner, xi_mid,
            semimajor_mid, semiminor_mid, a_shape, aspect_ratio, bending_epsilon)
    end
end

# Isotropic elastic shell material accessors, `m` is an `ElasticFEMLayer` (sphere_modal.jl),
# constructed via `Shelled(poisson, density, youngs_modulus)`.
shell_F(m::ElasticFEMLayer) = 0.5 * (1.0 - m.poisson)
shell_g(::ElasticFEMLayer) = 3.0 / 7.0
shell_k(::ElasticFEMLayer) = pi^2 / 12.0
function extensional_plate_speed(m::ElasticFEMLayer)
    sqrt(m.youngs_modulus / (m.density * (1.0 - m.poisson^2)))
end

"""
    uniform_eta_grid(n_eta; pole_offset=1e-4)

Uniform meridional grid on the open interval (-1, 1), offset from the
shell apexes (`η = ±1`) by `pole_offset` to avoid the coordinate
singularity there.
"""
function uniform_eta_grid(n_eta::Integer; pole_offset::Real = 1e-4)
    n_eta >= 5 || throw(ArgumentError("n_eta must be at least 5"))
    0 < pole_offset < 0.5 || throw(ArgumentError("pole_offset must lie in (0, 0.5)"))
    return collect(range(-1.0 + pole_offset, 1.0 - pole_offset; length = n_eta))
end

"""
    finite_difference_matrices(eta)

Second-order-accurate first- and second-derivative matrices `(D1, D2)` on
a uniform 1-D grid (one-sided differences at both endpoints).
"""
function finite_difference_matrices(eta::AbstractVector{<:Real})
    n = length(eta)
    n >= 5 || throw(ArgumentError("eta must have at least 5 points"))
    step = eta[2] - eta[1]

    d1 = zeros(n, n)
    d2 = zeros(n, n)
    for i in 2:(n - 1)
        d1[i, i - 1] = -0.5 / step
        d1[i, i + 1] = 0.5 / step
        d2[i, i - 1] = 1.0 / step^2
        d2[i, i] = -2.0 / step^2
        d2[i, i + 1] = 1.0 / step^2
    end
    d1[1, 1:3] = [-3.0, 4.0, -1.0] ./ (2step)
    d1[n, (n - 2):n] = [1.0, -4.0, 3.0] ./ (2step)
    d2[1, 1:4] = [2.0, -5.0, 4.0, -1.0] ./ step^2
    d2[n, (n - 3):n] = [-1.0, 4.0, -5.0, 2.0] ./ step^2
    return d1, d2
end

function _shell_coefficients(geometry::ProlateShellGeometry, eta::AbstractVector{<:Real})
    a2 = geometry.a_shape^2
    A = fill(a2 - 1.0, length(eta))
    B = 1.0 .- eta .^ 2
    C = a2 .- eta .^ 2
    D = sqrt.(A ./ C)
    return A, B, C, D
end

function _shell_fd_block(D1::AbstractMatrix, D2::AbstractMatrix,
        d2_coef::Union{Nothing, AbstractVector}, d1_coef::Union{Nothing, AbstractVector},
        diag_coef::Union{Nothing, AbstractVector})
    n = size(D1, 1)
    block = zeros(n, n)
    d2_coef === nothing || (block .+= d2_coef .* D2)
    d1_coef === nothing || (block .+= d1_coef .* D1)
    diag_coef === nothing || (block[diagind(block)] .+= diag_coef)
    return block
end

"""
    ShellSystem

Assembled shell-only axisymmetric dynamic system: `eta` (grid), the 9
stiffness blocks and 5 mass blocks (each `n_eta × n_eta`, keyed the same
way as the Python prototype: `:K_uu`, `:K_uw`, `:K_u_beta`, `:K_wu`,
`:K_ww`, `:K_w_beta`, `:K_beta_u`, `:K_beta_w`, `:K_beta_beta` and
`:M_uu`, `:M_u_beta`, `:M_ww`, `:M_beta_u`, `:M_beta_beta`), the assembled
`(3n_eta) × (3n_eta)` `dynamic_matrix` (stiffness + `ω̂² *` mass, DOF order
`[u; w; β]`), the surface-load scale factors, and the nondimensional
frequency `ω̂`.
"""
struct ShellSystem
    eta::Vector{Float64}
    structural_blocks::Dict{Symbol, Matrix{Float64}}
    mass_blocks::Dict{Symbol, Matrix{Float64}}
    dynamic_matrix::Matrix{Float64}
    load_scale_q::Vector{Float64}
    load_scale_m::Vector{Float64}
    nondimensional_frequency::Float64
end

"""
    assemble_shell_system(geometry, material, frequency_hz; n_eta=129, pole_offset=1e-4)

Assemble the Hayek & Boisvert axisymmetric nontorsional shell dynamic-
stiffness system for a confocal prolate spheroidal shell at `frequency_hz`
[Hz], on a uniform meridional grid of `n_eta` nodes.
"""
function assemble_shell_system(
        geometry::ProlateShellGeometry, material::ElasticFEMLayer, frequency_hz::Real;
        n_eta::Integer = 129, pole_offset::Real = 1e-4)
    eta = uniform_eta_grid(n_eta; pole_offset = pole_offset)
    D1, D2 = finite_difference_matrices(eta)
    A, B, C, D = _shell_coefficients(geometry, eta)

    a = geometry.a_shape
    nu = material.poisson
    eps = geometry.bending_epsilon
    F = shell_F(material)
    gamma = shell_g(material)
    kappa = shell_k(material)
    omega_hat = 2pi * frequency_hz * geometry.semimajor_mid /
                extensional_plate_speed(material)

    sqrtA, sqrtB, sqrtC = sqrt.(A), sqrt.(B), sqrt.(C)

    K_uu = _shell_fd_block(D1, D2,
        D .* B .- eps .* a^4 .* B .^ 2 .* D ./ C .^ 3,
        -eta .* D .* (1.0 .+ D .^ 2) .-
        eps .* a^4 .* B .* D .* eta .* (3.0 .- 7.0 .* D .^ 2) ./ C .^ 3,
        -eta .^ 2 .* D ./ B .- nu .* a^2 .* D ./ C .- F * kappa * a^2 .* D .^ 3 ./ C .+
        eps .* (-a^4 .* eta .^ 2 .* D ./ (A .* C .^ 2) .+
         F * gamma * kappa * a^6 .* D .^ 3 .* B ./ C .^ 4))

    K_uw = _shell_fd_block(D1, D2,
        nothing,
        -a .* D .* sqrtB ./ sqrtC .* ((1.0 + F * kappa) .* D .^ 2 .+ nu) .-
        eps .* a^5 .* A .* B .^ 1.5 .* (1.0 + F * kappa * gamma) ./ C .^ 4.5,
        a .* eta .* sqrtB .* (4.0 .* A .+ B) ./ C .^ 2.5 .+
        eps .* a^5 .* A .* sqrtB .* eta .*
        (1.0 ./ A .^ 2 .- 6.0 ./ C .^ 2 .+ 9.0 .* A ./ C .^ 3) ./ C .^ 2.5)

    K_u_beta = _shell_fd_block(D1, D2,
        eps .* a^2 .* B .^ 2 ./ C .^ 2,
        -4.0 .* eps .* a^2 .* eta .* A .* B ./ C .^ 3,
        F * kappa .* D .^ 2 .+
        eps .* (a^2 .* eta .^ 2 ./ C .^ 2 .- F * gamma * kappa * a^4 .* B ./ C .^ 4))

    K_wu = _shell_fd_block(D1, D2,
        nothing,
        -a .* sqrtB .* (A .* (1.0 + F * kappa) .+ nu .* C) ./ C .^ 1.5 .+
        eps .* a^5 .* A .* B .^ 1.5 .* (1.0 + F * kappa * gamma) ./ C .^ 4.5,
        a .* eta .*
        (1.0 .+ (nu + F * kappa) .* D .^ 2 .- 3.0 * F * kappa .* A .* B ./ C .^ 2) ./
        sqrt.(B .* C) .+
        eps .* eta .* a^5 .* sqrtB .*
        (1.0 .+ 3.0 * F * kappa * gamma .* D .^ 4 .* (2.0 .- 3.0 .* D .^ 2)) ./
        (A .* C .^ 2.5))

    K_ww = _shell_fd_block(D1, D2,
        F * kappa .* D .* B .- eps * F * kappa * gamma .* a^4 .* D .* B .^ 2 ./ C .^ 3,
        -F * kappa .* D .* (1.0 .+ D .^ 2) .* eta .-
        eps * F * kappa * gamma .* a^4 .* D .* B .* eta .* (3.0 .- 7.0 .* D .^ 2) ./ C .^ 3,
        -a^2 .* D .* (1.0 .+ D .^ 4 .+ 2.0 * nu .* D .^ 2) ./ A .+
        eps .* a^6 .* D .^ 3 .* B ./ C .* (1.0 ./ C .^ 3 .- 1.0 ./ A .^ 3))

    K_w_beta = _shell_fd_block(D1, D2,
        nothing,
        F * kappa .* sqrt.(A .* B) ./ a .-
        eps * a^3 .* sqrt.(A .* B) .* B .* (F * kappa * gamma + 1.0) ./ C .^ 3,
        -F * kappa .* eta .* sqrt.(A ./ B) ./ a .-
        eps * a^3 .* sqrtB .* eta .*
        (1.0 .+ 3.0 * F * kappa * gamma .* D .^ 2 .* (1.0 .- 2.0 .* D .^ 2)) ./
        (C .^ 2 .* sqrtA))

    K_beta_u = _shell_fd_block(D1, D2,
        eps .* a^2 .* B .^ 2 ./ C .^ 2,
        -4.0 .* eps .* a^2 .* A .* B .* eta ./ C .^ 3,
        F * kappa .* D .^ 2 .+
        eps .* (a^2 .* eta .^ 2 ./ C .^ 2 .- F * kappa * gamma * a^4 .* A .* B ./ C .^ 4))

    K_beta_w = _shell_fd_block(D1, D2,
        nothing,
        -F * kappa .* sqrt.(A .* B) ./ a .+
        eps * a^3 .* sqrtB .* (1.0 + F * kappa * gamma) .* A .* B ./ (C .^ 3 .* sqrtA),
        eps * a^3 .* sqrtB .* eta .* C .*
        (-1.0 .+ 3.0 .* D .^ 2 .* (1.0 .- 2.0 .* D .^ 2)) ./ (C .^ 3 .* sqrtA))

    K_beta_beta = _shell_fd_block(D1, D2,
        eps .* D .* B .^ 2,
        -eps .* D .* eta .* (1.0 .+ D .^ 2),
        -F * kappa .* D .* C ./ a^2 .+
        eps .* D .* (eta .^ 2 ./ B .- a^2 ./ C .+ F * kappa * gamma * a^2 .* B ./ C .^ 2))

    M_uu = 1.0 .+ eps * a^4 ./ C .^ 2
    M_u_beta = eps .* (1.0 .+ D .^ 2)
    M_ww = sqrtA .* sqrtC ./ a^2 .* (1.0 .+ eps * a^4 ./ C .^ 2)
    M_beta_u = M_u_beta
    M_beta_beta = eps .* sqrtA .* sqrtC ./ a^2

    n = length(eta)
    zero_n = zeros(n, n)
    structural = [K_uu K_uw K_u_beta; K_wu K_ww K_w_beta; K_beta_u K_beta_w K_beta_beta]
    mass = [Diagonal(M_uu) zero_n Diagonal(M_u_beta)
            zero_n Diagonal(M_ww) zero_n
            Diagonal(M_beta_u) zero_n Diagonal(M_beta_beta)]
    dynamic_matrix = structural + omega_hat^2 * mass

    load_scale_q = @. -(1.0 - nu^2) * geometry.semimajor_mid^2 * sqrt(A * C) /
                      (material.youngs_modulus * geometry.shell_thickness * a^2)
    load_scale_m = @. -(1.0 - nu^2) * sqrt(A * C) /
                      (material.youngs_modulus * geometry.shell_thickness * a^2)

    structural_blocks = Dict(:K_uu => K_uu, :K_uw => K_uw, :K_u_beta => K_u_beta,
        :K_wu => K_wu, :K_ww => K_ww, :K_w_beta => K_w_beta,
        :K_beta_u => K_beta_u, :K_beta_w => K_beta_w, :K_beta_beta => K_beta_beta)
    mass_blocks = Dict(
        :M_uu => Diagonal(M_uu) |> Matrix, :M_u_beta => Diagonal(M_u_beta) |> Matrix,
        :M_ww => Diagonal(M_ww) |> Matrix, :M_beta_u => Diagonal(M_beta_u) |> Matrix,
        :M_beta_beta => Diagonal(M_beta_beta) |> Matrix)

    return ShellSystem(eta, structural_blocks, mass_blocks, dynamic_matrix,
        load_scale_q, load_scale_m, omega_hat)
end

"""
    prolate_confocal_mesh(geometry, eta_nodes; surface=:outer)

Meridian mesh (see [`MeridianMesh`](@ref)) for the confocal prolate
spheroidal surface (`:outer`, `:inner`, or `:mid`) of `geometry`, at the
same meridional coordinates as `eta_nodes` (as used by
[`assemble_shell_system`](@ref)). Traversed north pole to south pole (the
reverse of `eta_nodes`' natural increasing order) so panel outward normals
follow the same `(-Δz, Δρ)/L` convention as [`sphere_mesh`](@ref).
"""
function prolate_confocal_mesh(geometry::ProlateShellGeometry,
        eta_nodes::AbstractVector{<:Real}; surface::Symbol = :outer)
    xi = surface === :outer ? geometry.xi_outer :
         surface === :inner ? geometry.xi_inner :
         surface === :mid ? geometry.xi_mid :
         throw(ArgumentError("surface must be :outer, :inner, or :mid"))
    f = geometry.focal_radius
    eta_rev = reverse(eta_nodes)
    ρ = [f * sqrt(xi^2 - 1.0) * sqrt(max(0.0, 1.0 - η^2)) for η in eta_rev]
    z = [f * xi * η for η in eta_rev]
    return MeridianMesh(ρ, z)
end

"""
    shell_to_bem_interpolation(n_shell)

`(n_shell - 1) × n_shell` matrix linearly interpolating values at the
`n_shell` shell nodes (natural increasing-`η` order) onto the midpoints of
the `n_shell - 1` panels of the corresponding
[`prolate_confocal_mesh`](@ref) (reversed order): panel `i`'s midpoint
sits exactly halfway between shell nodes `n_shell - i` and
`n_shell - i + 1`.
"""
function shell_to_bem_interpolation(n_shell::Integer)
    n = n_shell
    L = zeros(n - 1, n)
    for i in 1:(n - 1)
        L[i, n - i] = 0.5
        L[i, n - i + 1] = 0.5
    end
    return L
end

"""
    bem_to_shell_interpolation(n_shell)

`n_shell × (n_shell - 1)` matrix mapping panel-constant values on the
`(n_shell - 1)`-panel [`prolate_confocal_mesh`](@ref) back onto the
`n_shell` shell nodes: interior shell nodes average their two adjacent
panels, the two shell apex nodes take their single adjacent panel.
"""
function bem_to_shell_interpolation(n_shell::Integer)
    n = n_shell
    L = zeros(n, n - 1)
    L[1, n - 1] = 1.0
    L[n, 1] = 1.0
    for j in 2:(n - 1)
        L[j, n - j] = 0.5
        L[j, n - j + 1] = 0.5
    end
    return L
end
