# [Choosing a solver](@id solver-selection)

Start with the least costly method that supports both your physics and your desired output.
The table lists supported combinations. All six solver families return solution objects. Hover
a column header for what that solver is. Hover the ℹ️ icon next to a geometry for its caveats, or
follow a linked geometry name to the page covering it.

```@raw html
<table class="solver-table">
  <thead>
    <tr>
      <th>Boundary condition</th>
      <th tabindex="0" data-tooltip="Analytical series solution. Exact for the sphere and spheroid, an approximation for finite cylinders.">Modal</th>
      <th tabindex="0" data-tooltip="High-frequency physical-optics approximation. Not established as valid at low frequency.">Kirchhoff</th>
      <th tabindex="0" data-tooltip="No full 3D volume implementation. Only radial, meridian and shell reductions.">FEM</th>
      <th tabindex="0" data-tooltip="Boundary element method. Discretizes boundary integral equations over an axisymmetric or supplied mesh.">BEM</th>
      <th tabindex="0" data-tooltip="Method of fundamental solutions. Fits fields produced by fictitious point sources.">MFS</th>
      <th tabindex="0" data-tooltip="Fourier matching between a canonical body and a general numerical solver.">Fourier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Rigid</td>
      <td>
        Sphere, Spheroid,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders use an exact finite-length reduction. Bent cylinders use a near-broadside correction. Neither is an exact closed-finite-cylinder solution.">ℹ️</span>
      </td>
      <td>
        Sphere, Spheroid,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders use the lateral surface and caps. Bent cylinders use a curved-surface integral.">ℹ️</span>
      </td>
      <td>
        <a href="../fem/#fem-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial or axial-meridian reduction.">ℹ️</span>,
        <a href="../fem/#fem-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Meridian reduction only.">ℹ️</span>,
        <a href="../fem/#fem-theory">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Meridian reduction only, for straight cylinders. FEM does not model bend curvature and always uses flat caps.">ℹ️</span>
      </td>
      <td>
        <a href="../boundary_methods/#boundary-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric or full 3D. Uses Burton–Miller coupling by default.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric or full 3D. Uses Burton–Miller coupling by default.">ℹ️</span>,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders are axisymmetric and always use flat caps. Bent cylinders need the full 3D closed-surface solve, where endcap_depth shapes domed caps. Uses Burton–Miller coupling by default.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Arbitrary</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Arbitrary means a user-supplied surface mesh of any shape, forming one connected boundary. Uses Burton–Miller coupling by default.">ℹ️</span>
      </td>
      <td>
        <a href="../boundary_methods/#boundary-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric.">ℹ️</span>,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders are axisymmetric, where endcap_depth shapes domed caps for reliable source placement. Bent cylinders use a closed surface, or the lateral surface only.">ℹ️</span>
      </td>
      <td>
        Sphere, Spheroid,
        <a href="../fourier_matching/#fourier-matching-theory">Irregular</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Irregular means a smooth axisymmetric body of revolution solved by Fourier matching.">ℹ️</span>
      </td>
    </tr>
    <tr>
      <td>Pressure-release</td>
      <td>
        Sphere, Spheroid,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders use an exact finite-length reduction. Bent cylinders use a near-broadside correction. Neither is an exact closed-finite-cylinder solution.">ℹ️</span>
      </td>
      <td>
        Sphere, Spheroid,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders use the lateral surface and caps. Bent cylinders use a curved-surface integral.">ℹ️</span>
      </td>
      <td>
        <a href="../fem/#fem-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial or axial-meridian reduction.">ℹ️</span>,
        <a href="../fem/#fem-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Meridian reduction only.">ℹ️</span>,
        <a href="../fem/#fem-theory">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Meridian reduction only, for straight cylinders. FEM does not model bend curvature and always uses flat caps.">ℹ️</span>
      </td>
      <td>
        <a href="../boundary_methods/#boundary-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric or full 3D. Uses Burton–Miller coupling by default.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric or full 3D. Uses Burton–Miller coupling by default.">ℹ️</span>,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders are axisymmetric and always use flat caps. Bent cylinders need the full 3D closed-surface solve, where endcap_depth shapes domed caps. Uses Burton–Miller coupling by default.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Arbitrary</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Arbitrary means a user-supplied surface mesh of any shape, forming one connected boundary. Uses Burton–Miller coupling by default.">ℹ️</span>
      </td>
      <td>
        <a href="../boundary_methods/#boundary-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric.">ℹ️</span>,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders are axisymmetric, where endcap_depth shapes domed caps for reliable source placement. Bent cylinders use a closed surface, or the lateral surface only.">ℹ️</span>
      </td>
      <td>
        Sphere, Spheroid,
        <a href="../fourier_matching/#fourier-matching-theory">Irregular</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Irregular means a smooth axisymmetric body of revolution solved by Fourier matching.">ℹ️</span>
      </td>
    </tr>
    <tr>
      <td>Fluid or gas interior</td>
      <td>
        Sphere, Spheroid,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders use an exact finite-length reduction. Bent cylinders use a near-broadside correction. Neither is an exact closed-finite-cylinder solution.">ℹ️</span>
      </td>
      <td>
        Sphere, Spheroid,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders use the lateral surface and caps. Bent cylinders use a curved-surface approximation.">ℹ️</span>
      </td>
      <td>
        <a href="../fem/#fem-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial or axial-meridian reduction.">ℹ️</span>,
        <a href="../fem/#fem-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Meridian reduction only.">ℹ️</span>,
        <a href="../fem/#fem-theory">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Meridian reduction only, for straight cylinders. FEM does not model bend curvature and always uses flat caps.">ℹ️</span>
      </td>
      <td>
        <a href="../boundary_methods/#boundary-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric, or full 3D with a dense Müller system, which needs fewer unknowns than the four-trace conventional alternative. A limited optional CHIEF augmentation is available for the axisymmetric case.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric.">ℹ️</span>,
        <a href="../../tutorials/bent_cylinder/#bent-cylinder-tutorial">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders are axisymmetric and always use flat caps. Bent cylinders need the full 3D closed-surface solve with a dense Müller system, where endcap_depth shapes domed caps.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Arbitrary</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Arbitrary means a user-supplied surface mesh of any shape, forming one connected boundary. Full 3D fluid transmission uses a dense Müller system.">ℹ️</span>
      </td>
      <td>
        <a href="../boundary_methods/#boundary-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric.">ℹ️</span>,
        <a href="../boundary_methods/#boundary-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Axisymmetric.">ℹ️</span>,
        Cylinder&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders only, axisymmetric, where endcap_depth shapes domed caps for reliable source placement. Bent cylinders are not supported for fluid transmission.">ℹ️</span>
      </td>
      <td>
        Sphere, Spheroid,
        <a href="../fourier_matching/#fourier-matching-theory">Irregular</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Irregular means a smooth axisymmetric body of revolution solved by Fourier matching.">ℹ️</span>
      </td>
    </tr>
    <tr>
      <td>Solid elastic</td>
      <td>
        Sphere,
        Cylinder&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders only, using a finite-length approximation. Not an exact closed-finite-cylinder solution.">ℹ️</span>
      </td>
      <td></td>
      <td>
        <a href="../fem/#fem-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial reduction.">ℹ️</span>,
        <a href="../fem/#fem-theory">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial reduction, for straight cylinders only. Not an exact closed-finite-cylinder solution. FEM does not model bend curvature.">ℹ️</span>
      </td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Elastic shell, fluid interior</td>
      <td>
        Sphere,
        Cylinder&nbsp;<span class="hint" tabindex="0" data-tooltip="Straight cylinders only, using a finite-length approximation. Not an exact closed-finite-cylinder solution.">ℹ️</span>
      </td>
      <td></td>
      <td>
        <a href="../fem/#fem-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial reduction, or coupled thin/general structural shell FEM.">ℹ️</span>,
        <a href="../fem/#fem-theory">Spheroid</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Coupled thin/general structural shell FEM only. The thin method supports axial incidence.">ℹ️</span>,
        <a href="../fem/#fem-theory">Cylinder</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial reduction only, for straight cylinders. Not an exact closed-finite-cylinder solution. FEM does not model bend curvature.">ℹ️</span>
      </td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Fluid shell, fluid or vacuum interior</td>
      <td>Sphere</td>
      <td>
        <a href="../kirchhoff/#kirchhoff-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Layer-reflection approximation.">ℹ️</span>
      </td>
      <td>
        <a href="../fem/#fem-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Radial reduction.">ℹ️</span>
      </td>
      <td>
        <a href="../boundary_methods/#boundary-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Two-surface axisymmetric BEM, axial incidence only.">ℹ️</span>
      </td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Viscoelastic layered shell, fluid interior</td>
      <td>
        <a href="../modal/#modal-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Monopole-only reduction.">ℹ️</span>
      </td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Coupled fluid regions, nested or disjoint</td>
      <td>
        <a href="../modal/#modal-theory">Sphere</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Concentric-shell reference approximation only.">ℹ️</span>
      </td>
      <td></td>
      <td></td>
      <td>
        <a href="../boundary_methods/#coupled-fluid-regions">Arbitrary</a>&nbsp;<span class="hint" tabindex="0" data-tooltip="Arbitrary means a user-supplied surface mesh of any shape. Every interface is a supplied mesh, each forming one connected boundary. Contrasts are relative to the unbounded exterior.">ℹ️</span>
      </td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
```

## Choosing by question

- Use sphere or spheroid modal methods as analytical references, while checking series
  convergence and special-function conditioning.
- Use Kirchhoff to explore physical-optics behavior. A numerically converged surface integral
  does not establish validity of physical optics at low frequency.
- Use FEM or BEM when checking analytical reductions, geometry discretization, or interface
  coupling. Compare against a canonical case first.
- Use MFS when source placement is well controlled and its supported geometry suits the problem.
- Use BEM surface results for repeated observation-angle queries. Supported radial FEM spheres
  also provide complex backscatter. Cylinder radial and meridian FEM paths retain target strength
  only.
- Use Fourier matching for a smooth irregular body of revolution between the canonical (sphere/
  spheroid) and general numerical (BEM/MFS/FEM) solvers. See [Fourier matching](@ref
  fourier-matching-theory).

See [BEM and MFS](@ref boundary-theory) for the underlying boundary integral and source-fitting
theory.
