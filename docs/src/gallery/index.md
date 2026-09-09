# [Visualization Gallery](@id gallery)

Explore frequency sweeps, scattering patterns, meshes, and surface fields. Select a figure
to enlarge it, or open its example code.

```@raw html
<div class="scattering-gallery">
  <figure>
    <a href="../tutorials/rigid_sphere_frequency.png" aria-label="Enlarge frequency sweep">
      <img src="../tutorials/rigid_sphere_frequency.png" alt="Target strength of a rigid sphere across 12 to 200 kHz, showing frequency-dependent oscillations." loading="lazy">
    </a>
    <figcaption>
      <h2>Frequency sweep</h2>
      <p>Modal backscatter from a rigid 1 cm sphere. Target strength in dB re 1 m².</p>
```

[Example and data](@ref first-sweep)

```@raw html
    </figcaption>
  </figure>
  <figure>
    <a href="../tutorials/cylinder_incidence.png" aria-label="Enlarge incidence-angle sweep">
      <img src="../tutorials/cylinder_incidence.png" alt="Finite-cylinder target strength from near end-on incidence to broadside at 12 kHz." loading="lazy">
    </a>
    <figcaption>
      <h2>Incidence-angle sweep</h2>
      <p>Rigid cylinder, 1 cm radius and 7 cm length, at 12 kHz. The modal approximation omits cap scattering.</p>
```

[Example code](@ref geometry-tutorial)

```@raw html
    </figcaption>
  </figure>
  <figure>
    <a href="examples/bistatic_polar.png" aria-label="Enlarge polar scattering pattern">
      <img src="examples/bistatic_polar.png" alt="Polar cut through the bistatic scattering pattern of a rigid spheroid at 38 kHz." loading="lazy">
    </a>
    <figcaption>
      <h2>Polar scattering pattern</h2>
      <p>Rigid spheroid at 38 kHz and 45° incidence. Radius shows target strength above the cut's minimum, in dB.</p>
```

[Example code](@ref gallery-polar)

```@raw html
    </figcaption>
  </figure>
  <figure>
    <a href="examples/bistatic_map.png" aria-label="Enlarge angular scattering map">
      <img src="examples/bistatic_map.png" alt="Heatmap of spheroid target strength over observation polar angle and azimuth." loading="lazy">
    </a>
    <figcaption>
      <h2>Angular scattering map</h2>
      <p>The same BEM solution viewed over polar angle and azimuth. Color shows target strength in dB re 1 m².</p>
```

[Example code](@ref gallery-map)

```@raw html
    </figcaption>
  </figure>
  <figure>
    <a href="examples/spheroid_mesh.png" aria-label="Enlarge surface mesh">
      <img src="examples/spheroid_mesh.png" alt="Triangular surface mesh of a prolate spheroid, with edges visible and coordinates in meters." loading="lazy">
    </a>
    <figcaption>
      <h2>Surface mesh</h2>
      <p>Prolate spheroid with 2 cm axial and 1 cm equatorial semi-axes. Target edge length: 6 mm.</p>
```

[Example code](@ref gallery-mesh)

```@raw html
    </figcaption>
  </figure>
  <figure>
    <a href="examples/surface_phase.png" aria-label="Enlarge surface pressure phase">
      <img src="examples/surface_phase.png" alt="Scattered-pressure phase on the spheroid surface, colored from minus pi to pi radians." loading="lazy">
    </a>
    <figcaption>
      <h2>Surface pressure phase</h2>
      <p>Scattered-pressure phase from the spheroid BEM solution. Color shows phase in radians.</p>
```

[Example code](@ref gallery-field)

```@raw html
    </figcaption>
  </figure>
</div>
```
