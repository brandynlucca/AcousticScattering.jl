# [FEM and shell coupling](@id fem-theory)

The FEM paths discretize acoustic or elastic differential equations and couple them to outgoing
acoustic fields. Radial and meridian formulations are available. Full 3D volume FEM is not supported.

## Radial acoustic FEM and DtN

Separating spherical harmonics gives a radial equation for degree `l`:

```math
\frac{d}{dr}\left(r^2\frac{du_l}{dr}\right)
+\left(k^2r^2-l(l+1)\right)u_l=0.
```

On the artificial outer sphere of radius `R`, an outgoing Hankel wave gives the exact
Dirichlet-to-Neumann relation for each retained mode:

```math
u_l'(R)=
k\frac{{h_l^{(1)}}'(kR)}{h_l^{(1)}(kR)}u_l(R).
```

The finite-element weak form follows by multiplying by a test function and integrating by
parts. Rigid/soft conditions apply at the body. Fluid transmission couples interior and
exterior fields. This discretizes the same boundary-value problem as the sphere modal series.

Radial FEM supports linear/quadratic elements for applicable rigid/soft paths.
Fluid and elastic variants have their own discretizations and controls. `R > radius` is
required for an exterior annulus. Moving `R` farther away is not the main accuracy control
when the modal DtN condition is exact. Check element refinement and modal cutoff separately.

See [Numerical convergence](@ref convergence-tutorial). Ordinary radial and meridian solutions
currently retain target strength only. Complex amplitude is unavailable through those results.

## Meridian acoustic FEM

Meridian FEM discretizes radial/axial dependence and uses azimuthal Fourier modes. For mode
`m`, cylindrical-coordinate operators contain the term `-m^2 / rho^2`. Regularity on the
axis and the transformed volume weighting are essential. Meridian FEM supports spheres,
straight cylinders, and spheroids.

Increase mesh resolution and angular/modal orders separately. Oblique cylinder/spheroid
incidence requires the corresponding Fourier content. These methods do not model bend curvature.

## Elastic and coupled shell FEM

Isotropic solid displacement satisfies

```math
\nabla\cdot\boldsymbol\sigma+
\rho_s\omega^2\boldsymbol u=0,
\qquad
\boldsymbol\sigma=\lambda\operatorname{tr}(\boldsymbol\epsilon)\boldsymbol I
+2\mu\boldsymbol\epsilon.
```

Normal displacement and traction couple to fluid pressure. Coupled shell implementations combine
structural finite elements with exterior and, where supported, interior boundary elements.
For example:

```julia
body = Shell(Spheroid(0.05, 0.02), 0.001) # thickness in m
material = Shelled(0.3, 7800.0, 2e11) # Poisson ratio, kg/m³, Pa
solution = fem(body, material, 1026.8, 1477.4, 1000.0, 1480.0, 100.0;
    method = :thin, incidence_angle = 0.0)
strength = target_strength(solution)
```

This expensive structural example is illustrative and is not executed in the ordinary docs
build. The four fluid arguments are exterior density/speed followed by interior density/speed.
The final positional input is exterior wavenumber.

`:thin` uses a midsurface shell formulation and is limited
to axial prolate-spheroid calculations. It supports a no-interior-coupling branch with zero
interior density. `:general` uses a through-thickness meridian elastic discretization for
sphere/spheroid shells and requires a fluid interior. It has no dedicated vacuum branch.
A small positive density is an approximation, not an exact vacuum boundary.
