<h1 align="center">
  <img src="docs/src/assets/logo.svg" alt="AcousticScattering.jl" width="300">
</h1>

[![AcousticScattering](https://img.shields.io/badge/dynamic/xml?url=https%3A%2F%2Fplatform.juliahub.com%2Fdocs%2FGeneral%2FAcousticScattering%2Fstable%2Fversion.svg&query=concat%28%2F%2F%2A%5Blocal-name%28%29%3D%27text%27%5D%5B2%5D%2C+%27+%27%2C+%2F%2F%2A%5Blocal-name%28%29%3D%27text%27%5D%5Blast%28%29%5D%29&label=AcousticScattering&color=32B32E&cacheSeconds=3600)](https://platform.juliahub.com/ui/Packages/General/AcousticScattering)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22776330.svg)](https://doi.org/10.5281/zenodo.22776330)

[![Documentation](https://img.shields.io/badge/docs-latest-blue?label=Package%20documentation)](https://brandynlucca.github.io/AcousticScattering.jl)
[![Build status (basic)](https://img.shields.io/github/actions/workflow/status/brandynlucca/AcousticScattering.jl/CI.yml?label=Build%20status%20(basic)&logo=github&labelColor=24292e)](https://github.com/brandynlucca/AcousticScattering.jl/actions/workflows/CI.yml)
[![Build status (extended)](https://img.shields.io/github/actions/workflow/status/brandynlucca/AcousticScattering.jl/ExtendedCI.yml?label=Build%20status%20(extended)&logo=github&labelColor=24292e)](https://github.com/brandynlucca/AcousticScattering.jl/actions/workflows/ExtendedCI.yml)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL3-green.svg?label=License)](LICENSE)
[![codecov](https://codecov.io/gh/brandynlucca/AcousticScattering.jl/graph/badge.svg?token=FYNNGQSCDB)](https://codecov.io/gh/brandynlucca/AcousticScattering.jl)

AcousticScattering.jl models how individual objects scatter sound in a fluid. It provides access to modal series solutions, Kirchhoff approximations in physical optics, boundary- and finite-element methods, the method of fundamental solutions, and Fourier matching methods for calculating scattering amplitude and target strength. Supported geometries include spheres, spheroids, straight and bent cylinders, shells, and general bodies of revolution. Material options include rigid, pressure-release, fluid-filled, and elastic configurations. Support varies by solver. See the [model guide](https://brandynlucca.github.io/AcousticScattering.jl/stable/models/selection/) for available combinations and limitations.

## Installation

Requires Julia 1.10 or later. Install the development version from GitHub:

```julia
import Pkg
Pkg.add("AcousticScattering.jl")
```

Installation runs an optional precompile workload to reduce first-call latency. [Getting Started](https://brandynlucca.github.io/AcousticScattering.jl/stable/getting_started/#opt-out-before-the-first-precompile) explains how to disable it before installation.

## Examples

Calculate backscatter target strength for a rigid sphere with a 1 cm radius at 38 kHz using the modal series solution:

```julia
using AcousticScattering

# Parameters
frequency = 38000.0 # Hz
sound_speed = 1477.4 # m/s
wavenumber = 2pi * frequency / sound_speed

solution = modal(Sphere(0.01), Rigid(), wavenumber)
pressure(solution, (0.02, 0.0, 0.0)) # total pressure relative to the incident wave
target_strength(solution) # dB re 1 m²
```

Calculate the scattering amplitude and target strength for an aluminum shell on a spheroid with a 2 mm wall at 12 kHz using numerical methods:

```julia
# Parameters
exterior_density = 1000.0 # kg/m³
exterior_sound_speed = 1500.0 # m/s
interior_density = 1000.0 # kg/m³
interior_sound_speed = 1500.0 # m/s
wavenumber = 2pi * 12000.0 / exterior_sound_speed

# Boundaries/material
shell = Shell(Spheroid(0.02, 0.01), 0.002)
elastic_material = Shelled(0.33, 2700.0, 70e9) # Poisson ratio, density (kg/m³), Young's modulus (Pa)

solution = fem(shell, elastic_material, exterior_density, exterior_sound_speed,
    interior_density, interior_sound_speed, wavenumber)

scattering_amplitude(solution) # complex m
target_strength(solution) # dB re 1 m²
```

## Documentation

- [Getting Started](https://brandynlucca.github.io/AcousticScattering.jl/stable/getting_started/):   installation and your first calculation.
- [First frequency sweep](https://brandynlucca.github.io/AcousticScattering.jl/stable/tutorials/):   plot target strength and save the data.
- [Models and theory](https://brandynlucca.github.io/AcousticScattering.jl/stable/models/):   conventions, assumptions, and solver selection.
- [API reference](https://brandynlucca.github.io/AcousticScattering.jl/stable/api/):   geometry, materials, solvers, and result queries.
- [Visualization gallery](https://brandynlucca.github.io/AcousticScattering.jl/stable/gallery/):   figures with links to their examples.

## Citation

If you use this package in research, cite the archived version you used:

> Lucca, B. (2026). *AcousticScattering.jl* (v0.1.1) [Computer software]. > Zenodo. https://doi.org/10.5281/zenodo.22776330

Please also cite the relevant model papers listed in the [documentation references](https://brandynlucca.github.io/AcousticScattering.jl/stable/models/references/).

## License

[GPL-3.0-only license](LICENSE) for the package. The [logo artwork](docs/src/assets/LICENSE) incorporates Julia's dots and is licensed separately under CC BY-NC-SA 4.0.
